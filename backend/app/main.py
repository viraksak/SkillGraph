"""SkillGraph Week 3 Backend Service (`backend/app/main.py`).

Provides a clean layered architecture separating:
1. Configuration & PostgreSQL/SQLite Database Access Layer (`DatabaseConfig`, `DatabaseLayer`)
2. Domain Skill Gap & Explainability Service (`SkillGapService`)
3. Pydantic v2 Request/Response Schemas
4. FastAPI-compatible ASGI Web Application & OpenAPI 3.1 Specification

Notes on offline `.venv` compatibility:
- Uses `fastapi.FastAPI` automatically when `fastapi` is installed.
- Uses a pure-Python ASGI 3.0 + Pydantic v2 fallback router (`SkillGraphASGIApp`)
  when `fastapi` is absent in the research environment, preserving real Pydantic
  input/output validation, HTTP status codes (200/201/400/404/422), OpenAPI 3.1
  JSON generation, and real relational SQL persistence.
"""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.build_graph import (  # noqa: E402
    BASELINE_WEIGHT_PRESETS,
    PriorityWeights,
    RelationalDatabaseManager,
    SkillGapAnalysisEngine,
    SkillGraphBuilder,
    inspect_postgresql_environment,
)

FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


# ---------------------------------------------------------------------------
# 1. Environment Configuration (Zero Hardcoded Credentials)
# ---------------------------------------------------------------------------

class DatabaseConfig(BaseModel):
    """Runtime database configuration loaded exclusively from environment variables."""

    model_config = ConfigDict(extra="ignore")

    host: str = Field(default_factory=lambda: os.environ.get("SKILLGRAPH_DB_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.environ.get("SKILLGRAPH_DB_PORT", "5432")))
    database: str = Field(default_factory=lambda: os.environ.get("SKILLGRAPH_DB_NAME", "skillgraph"))
    user: str = Field(default_factory=lambda: os.environ.get("SKILLGRAPH_DB_USER", ""))
    password_configured: bool = Field(default_factory=lambda: bool(os.environ.get("SKILLGRAPH_DB_PASSWORD") or os.environ.get("PGPASSWORD")))
    offline_sqlite_path: str = Field(default_factory=lambda: os.environ.get("SKILLGRAPH_SQLITE_PATH", ":memory:"))

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls()


# ---------------------------------------------------------------------------
# 2. Domain Exceptions & HTTP Status Mapping
# ---------------------------------------------------------------------------

class ApiDomainError(Exception):
    """Structured API error carrying HTTP status code and error category."""

    def __init__(self, status_code: int, error_code: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail


class ResourceNotFoundError(ApiDomainError):
    def __init__(self, detail: str) -> None:
        super().__init__(404, "RESOURCE_NOT_FOUND", detail)


class InvalidInputError(ApiDomainError):
    def __init__(self, detail: str) -> None:
        super().__init__(400, "INVALID_INPUT", detail)


# ---------------------------------------------------------------------------
# 3. Pydantic v2 Request & Response Schemas
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error_code: str
    detail: str
    status_code: int


class HealthResponse(BaseModel):
    status: str
    service: str
    api_framework_mode: str
    fastapi_installed: bool
    graph_nodes: int
    graph_edges: int
    database_backend_active: str
    postgresql_env: dict[str, Any]
    table_counts: dict[str, int]
    research_disclaimer: str


class OccupationSummary(BaseModel):
    occupation_id: str
    occupation: str
    industry_id: str
    industry: str
    department_label: str | None = None
    isco08_code_candidate: str | None = None
    isco08_label_candidate: str | None = None
    posting_count: int


class SkillSummary(BaseModel):
    skill_id: str
    canonical_skill: str
    skill_category: str
    skill_type: str
    taxonomy_source: str
    mapping_status: str


class OccupationSkillItem(BaseModel):
    skill_id: str
    canonical_skill: str
    skill_category: str
    skill_type: str
    job_count: int
    required_count: int
    preferred_count: int
    occupation_total_postings: int
    demand_share: float


class OccupationSkillsResponse(BaseModel):
    occupation_id: str
    occupation: str
    industry: str
    isco08_code_candidate: str | None = None
    isco08_label_candidate: str | None = None
    total_skills: int
    required_skills: list[OccupationSkillItem]
    preferred_only_skills: list[OccupationSkillItem]
    all_skills: list[OccupationSkillItem]


class StudentSkillInput(BaseModel):
    skill: str | None = None
    skill_id: str | None = None
    canonical_skill: str | None = None
    proficiency_level: str | None = None
    proficiency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: str | None = None

    @model_validator(mode="after")
    def _ensure_skill_identifier(self) -> "StudentSkillInput":
        if not any(v and str(v).strip() for v in (self.skill, self.skill_id, self.canonical_skill)):
            raise ValueError("Each skill entry must include one of 'skill', 'skill_id', or 'canonical_skill'.")
        return self


class PriorityWeightsInput(BaseModel):
    alpha: float = Field(default=0.70, ge=0.0, le=10.0)
    beta: float = Field(default=0.30, ge=0.0, le=10.0)
    gamma: float = Field(default=0.00, ge=0.0, le=10.0)
    delta: float = Field(default=0.00, ge=0.0, le=10.0)
    preset_name: str = "custom_request_weights"

    @model_validator(mode="after")
    def _ensure_positive_signal(self) -> "PriorityWeightsInput":
        if (self.alpha + self.beta + self.gamma) <= 0.0:
            raise ValueError("At least one positive weight (alpha, beta, or gamma) must exceed 0.0.")
        return self


class SkillGapAnalyzeRequest(BaseModel):
    student_id: str = Field(..., min_length=1, max_length=64)
    target_occupation: str = Field(..., min_length=1, max_length=128)
    skills: list[StudentSkillInput | str] = Field(default_factory=list)
    candidate_occupations: list[str] = Field(default_factory=list)
    is_synthetic: bool = True
    strict_validation: bool = False
    top_k: int | None = Field(default=None, ge=1, le=230)
    weights: PriorityWeightsInput | None = None
    run_sensitivity: bool = True


class StructuredExplanationModel(BaseModel):
    skill: str
    skill_id: str
    why_identified: str
    occupation_requirement_role: str
    student_profile_status: str
    importance_assessment: str
    multi_occupation_relevance: str
    priority_calculation: dict[str, Any]
    supporting_evidence: dict[str, Any]


class SkillRecommendationItem(BaseModel):
    rank: int
    skill_id: str
    canonical_skill: str
    skill_category: str
    skill_type: str
    requirement_type: str
    profile_status: str
    gap_factor: float
    required_count: int
    preferred_count: int
    occupation_total_postings: int
    demand_share: float
    required_share: float
    preferred_share: float
    priority_score: float
    priority_tier: str
    learning_stage: str
    scoring: dict[str, Any]
    opportunity: dict[str, Any]
    prerequisite_info: dict[str, Any]
    learning_resources_status: str
    summary_reason: str
    explanation: StructuredExplanationModel


class SkillGapAnalysisResponse(BaseModel):
    analysis_id: str
    created_at: str
    student_id: str
    is_synthetic_profile: bool
    target_occupation: dict[str, Any]
    candidate_occupations: list[str]
    supplied_skill_count: int
    unrecognized_skills: list[dict[str, Any]]
    duplicate_skill_notes: list[str]
    coverage_metrics: dict[str, Any]
    possessed_skills: list[dict[str, Any]]
    partially_demonstrated_skills: list[dict[str, Any]]
    missing_required_skills: list[dict[str, Any]]
    missing_preferred_skills: list[dict[str, Any]]
    priority_recommendations: list[SkillRecommendationItem]
    learning_path: dict[str, Any]
    weight_sensitivity: dict[str, Any] | None = None
    research_integrity_notes: list[str]


class StudentRecommendationsResponse(BaseModel):
    analysis_id: str
    student_id: str
    is_synthetic_profile: bool
    target_occupation: dict[str, Any]
    coverage_metrics: dict[str, Any]
    ordering_strategy: str
    learning_resources_status: str
    limitation_notice: str
    recommendations: list[SkillRecommendationItem]
    learning_path: dict[str, Any]


class StudentExplanationsResponse(BaseModel):
    analysis_id: str
    student_id: str
    target_occupation: dict[str, Any]
    explanations_count: int
    explanations: list[StructuredExplanationModel]


# ---------------------------------------------------------------------------
# 4. Database Abstraction & Service Layer
# ---------------------------------------------------------------------------

class DatabaseLayer:
    """Separates SQL persistence and graph queries from HTTP routes."""

    def __init__(
        self,
        builder: SkillGraphBuilder | None = None,
        config: DatabaseConfig | None = None,
    ) -> None:
        self.config = config or DatabaseConfig.from_env()
        self.builder = builder or SkillGraphBuilder(include_broad_country_mappings=False, min_co_occurrence=5)
        if not self.builder._built:
            self.builder.build()
        self.db_manager = RelationalDatabaseManager(self.builder, db_path=self.config.offline_sqlite_path)
        self.pg_env = inspect_postgresql_environment()

    def health_snapshot(self) -> dict[str, Any]:
        summary = self.builder.summary()
        return {
            "status": "ok",
            "service": "SkillGraph Career Skill Gap & Recommendation API",
            "api_framework_mode": "fastapi" if FASTAPI_AVAILABLE else "asgi_pydantic_fallback",
            "fastapi_installed": FASTAPI_AVAILABLE,
            "graph_nodes": summary["total_active_nodes"],
            "graph_edges": summary["total_active_edges"],
            "database_backend_active": f"sqlite3({self.config.offline_sqlite_path}) [PostgreSQL DDL & constraints mirrored]",
            "postgresql_env": self.pg_env,
            "table_counts": self.db_manager.list_tables_and_counts(),
            "research_disclaimer": (
                "Market skill demand derives from 2,000 synthetic postings; "
                "student gap profiles remain synthetic/anonymized and are not validated user studies."
            ),
        }

    def list_occupations(self, industry: str | None = None, search: str | None = None) -> list[dict[str, Any]]:
        return self.db_manager.fetch_occupations(industry=industry, search=search)

    def list_skills(
        self,
        category: str | None = None,
        skill_type: str | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.db_manager.fetch_skills(category=category, skill_type=skill_type, search=search)

    def get_occupation_skills(self, occupation_id: str) -> dict[str, Any]:
        res = self.db_manager.fetch_occupation_skills(occupation_id)
        if res is None:
            raise ResourceNotFoundError(f"Occupation not found: {occupation_id!r}")
        return res

    def persist_analysis(
        self,
        analysis: dict[str, Any],
        engine: SkillGapAnalysisEngine,
        profile: dict[str, Any],
    ) -> None:
        self.db_manager.save_analysis_result(analysis, engine, profile)

    def get_analysis(self, analysis_id: str) -> dict[str, Any]:
        res = self.db_manager.fetch_analysis(analysis_id)
        if res is None:
            raise ResourceNotFoundError(f"Skill gap analysis not found: {analysis_id!r}")
        return res

    def get_latest_student_recommendations(self, student_id: str) -> dict[str, Any]:
        res = self.db_manager.fetch_latest_student_recommendations(student_id)
        if res is None:
            raise ResourceNotFoundError(f"No skill gap analysis persisted for student_id={student_id!r}")
        return res


class SkillGapService:
    """Coordinates domain gap calculations, weight validation, and DB persistence."""

    def __init__(self, db_layer: DatabaseLayer) -> None:
        self.db_layer = db_layer
        self.engine = SkillGapAnalysisEngine(db_layer.builder)

    def analyze_and_persist(self, req: SkillGapAnalyzeRequest) -> SkillGapAnalysisResponse:
        profile_dict: dict[str, Any] = {
            "student_id": req.student_id,
            "target_occupation": req.target_occupation,
            "skills": [
                s.model_dump() if isinstance(s, BaseModel) else s
                for s in req.skills
            ],
            "candidate_occupations": req.candidate_occupations,
            "is_synthetic": req.is_synthetic,
        }
        weights = (
            PriorityWeights(
                alpha=req.weights.alpha,
                beta=req.weights.beta,
                gamma=req.weights.gamma,
                delta=req.weights.delta,
                preset_name=req.weights.preset_name,
            )
            if req.weights is not None
            else BASELINE_WEIGHT_PRESETS["baseline_balanced_heuristic"]
        )
        try:
            raw_result = self.engine.analyze(
                profile=profile_dict,
                weights=weights,
                strict_validation=req.strict_validation,
                top_k=req.top_k,
                run_sensitivity=req.run_sensitivity,
            )
        except ValueError as exc:
            msg = str(exc)
            if "Unknown target occupation" in msg:
                raise ResourceNotFoundError(msg) from exc
            raise InvalidInputError(msg) from exc

        self.db_layer.persist_analysis(raw_result, self.engine, profile_dict)
        return SkillGapAnalysisResponse.model_validate(raw_result)

    def get_analysis_response(self, analysis_id: str) -> SkillGapAnalysisResponse:
        return SkillGapAnalysisResponse.model_validate(self.db_layer.get_analysis(analysis_id))

    def get_recommendations_response(self, student_id: str) -> StudentRecommendationsResponse:
        payload = self.db_layer.get_latest_student_recommendations(student_id)
        return StudentRecommendationsResponse(
            analysis_id=payload["analysis_id"],
            student_id=payload["student_id"],
            is_synthetic_profile=payload["is_synthetic_profile"],
            target_occupation=payload["target_occupation"],
            coverage_metrics=payload["coverage_metrics"],
            ordering_strategy=payload["learning_path"]["ordering_strategy"],
            learning_resources_status=payload["learning_path"]["learning_resources_status"],
            limitation_notice=payload["learning_path"]["limitation_notice"],
            recommendations=[SkillRecommendationItem.model_validate(r) for r in payload["priority_recommendations"]],
            learning_path=payload["learning_path"],
        )

    def get_explanations_response(self, student_id: str) -> StudentExplanationsResponse:
        payload = self.db_layer.get_latest_student_recommendations(student_id)
        exps = [StructuredExplanationModel.model_validate(r["explanation"]) for r in payload["priority_recommendations"]]
        return StudentExplanationsResponse(
            analysis_id=payload["analysis_id"],
            student_id=payload["student_id"],
            target_occupation=payload["target_occupation"],
            explanations_count=len(exps),
            explanations=exps,
        )


# ---------------------------------------------------------------------------
# 5. Pure-Python ASGI + OpenAPI 3.1 Fallback Router & TestClient
# ---------------------------------------------------------------------------

class SkillGraphASGIApp:
    """ASGI 3.0 FastAPI-compatible router used when `fastapi` is not installed."""

    def __init__(
        self,
        title: str,
        version: str,
        description: str,
        db_layer: DatabaseLayer,
        service: SkillGapService,
    ) -> None:
        self.title = title
        self.version = version
        self.description = description
        self.db_layer = db_layer
        self.service = service
        self.routes: list[tuple[str, re.Pattern[str], Callable[..., Any], str]] = []
        self._register_default_routes()

    def _register(self, method: str, path_template: str, handler: Callable[..., Any]) -> None:
        regex_str = "^" + re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", r"(?P<\1>[^/]+)", path_template) + "$"
        self.routes.append((method.upper(), re.compile(regex_str), handler, path_template))

    def _register_default_routes(self) -> None:
        self._register("GET", "/health", self._h_health)
        self._register("GET", "/occupations", self._h_occupations)
        self._register("GET", "/skills", self._h_skills)
        self._register("GET", "/occupations/{occupation_id}/skills", self._h_occ_skills)
        self._register("POST", "/skill-gap/analyze", self._h_analyze)
        self._register("GET", "/skill-gap/{analysis_id}", self._h_get_analysis)
        self._register("GET", "/recommendations/{student_id}", self._h_get_recs)
        self._register("GET", "/recommendations/{student_id}/explanations", self._h_get_exps)
        self._register("GET", "/openapi.json", self._h_openapi)

    def _h_health(self, params: dict[str, str], query: dict[str, list[str]], body: dict[str, Any] | None) -> tuple[int, Any]:
        return 200, HealthResponse.model_validate(self.db_layer.health_snapshot()).model_dump()

    def _h_occupations(self, params: dict[str, str], query: dict[str, list[str]], body: dict[str, Any] | None) -> tuple[int, Any]:
        industry = query.get("industry", [None])[0]
        search = query.get("search", [None])[0]
        items = [OccupationSummary.model_validate(r).model_dump() for r in self.db_layer.list_occupations(industry, search)]
        return 200, items

    def _h_skills(self, params: dict[str, str], query: dict[str, list[str]], body: dict[str, Any] | None) -> tuple[int, Any]:
        category = query.get("category", [None])[0]
        skill_type = query.get("skill_type", [None])[0]
        search = query.get("search", [None])[0]
        items = [SkillSummary.model_validate(r).model_dump() for r in self.db_layer.list_skills(category, skill_type, search)]
        return 200, items

    def _h_occ_skills(self, params: dict[str, str], query: dict[str, list[str]], body: dict[str, Any] | None) -> tuple[int, Any]:
        occ_id = unquote(params["occupation_id"])
        data = self.db_layer.get_occupation_skills(occ_id)
        return 200, OccupationSkillsResponse.model_validate(data).model_dump()

    def _h_analyze(self, params: dict[str, str], query: dict[str, list[str]], body: dict[str, Any] | None) -> tuple[int, Any]:
        if body is None or not isinstance(body, dict):
            return 422, {"detail": [{"loc": ["body"], "msg": "JSON request object is required."}]}
        try:
            req = SkillGapAnalyzeRequest.model_validate(body)
        except ValidationError as exc:
            return 422, {"detail": exc.errors(include_url=False)}
        res = self.service.analyze_and_persist(req)
        return 200, res.model_dump()

    def _h_get_analysis(self, params: dict[str, str], query: dict[str, list[str]], body: dict[str, Any] | None) -> tuple[int, Any]:
        res = self.service.get_analysis_response(unquote(params["analysis_id"]))
        return 200, res.model_dump()

    def _h_get_recs(self, params: dict[str, str], query: dict[str, list[str]], body: dict[str, Any] | None) -> tuple[int, Any]:
        res = self.service.get_recommendations_response(unquote(params["student_id"]))
        return 200, res.model_dump()

    def _h_get_exps(self, params: dict[str, str], query: dict[str, list[str]], body: dict[str, Any] | None) -> tuple[int, Any]:
        res = self.service.get_explanations_response(unquote(params["student_id"]))
        return 200, res.model_dump()

    def openapi(self) -> dict[str, Any]:
        """Builds an OpenAPI 3.1.0 specification directly from Pydantic v2 models."""
        schemas = {
            m.__name__: m.model_json_schema(ref_template="#/components/schemas/{model}")
            for m in (
                HealthResponse,
                OccupationSummary,
                SkillSummary,
                OccupationSkillsResponse,
                StudentSkillInput,
                PriorityWeightsInput,
                SkillGapAnalyzeRequest,
                SkillRecommendationItem,
                StructuredExplanationModel,
                SkillGapAnalysisResponse,
                StudentRecommendationsResponse,
                StudentExplanationsResponse,
                ErrorResponse,
            )
        }
        return {
            "openapi": "3.1.0",
            "info": {"title": self.title, "version": self.version, "description": self.description},
            "paths": {
                "/health": {"get": {"summary": "Service & DB Health Check", "responses": {"200": {"description": "OK"}}}},
                "/occupations": {"get": {"summary": "List Occupations", "responses": {"200": {"description": "OK"}}}},
                "/skills": {"get": {"summary": "List Canonical Skills", "responses": {"200": {"description": "OK"}}}},
                "/occupations/{occupation_id}/skills": {"get": {"summary": "Get Occupation Skill Profile", "responses": {"200": {"description": "OK"}, "404": {"description": "Not Found"}}}},
                "/skill-gap/analyze": {
                    "post": {
                        "summary": "Analyze & Persist Student Skill Gap",
                        "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SkillGapAnalyzeRequest"}}}},
                        "responses": {"200": {"description": "OK"}, "400": {"description": "Invalid Skill Input"}, "404": {"description": "Unknown Occupation"}, "422": {"description": "Validation Error"}},
                    }
                },
                "/skill-gap/{analysis_id}": {"get": {"summary": "Fetch Persisted Gap Analysis", "responses": {"200": {"description": "OK"}, "404": {"description": "Not Found"}}}},
                "/recommendations/{student_id}": {"get": {"summary": "Get Ordered Skill Recommendations", "responses": {"200": {"description": "OK"}, "404": {"description": "Not Found"}}}},
                "/recommendations/{student_id}/explanations": {"get": {"summary": "Get Structured Explanations", "responses": {"200": {"description": "OK"}, "404": {"description": "Not Found"}}}},
            },
            "components": {"schemas": schemas},
        }

    def _h_openapi(self, params: dict[str, str], query: dict[str, list[str]], body: dict[str, Any] | None) -> tuple[int, Any]:
        return 200, self.openapi()

    def dispatch(
        self,
        method: str,
        path: str,
        query_string: str = "",
        raw_body: bytes | None = None,
    ) -> tuple[int, Any]:
        """Dispatches an HTTP request deterministically and enforces schema/domain validation."""
        body_obj: dict[str, Any] | None = None
        if raw_body:
            try:
                body_obj = json.loads(raw_body.decode("utf-8"))
            except Exception:
                return 422, {"detail": [{"loc": ["body"], "msg": "Malformed JSON request body."}]}

        query = parse_qs(query_string)
        upper_m = method.upper()
        for r_method, r_regex, handler, _ in self.routes:
            if r_method != upper_m:
                continue
            m = r_regex.match(path)
            if m:
                try:
                    return handler(m.groupdict(), query, body_obj)
                except ApiDomainError as exc:
                    return exc.status_code, ErrorResponse(
                        error_code=exc.error_code,
                        detail=exc.detail,
                        status_code=exc.status_code,
                    ).model_dump()
                except Exception as exc:
                    return 500, ErrorResponse(
                        error_code="INTERNAL_SERVER_ERROR",
                        detail=str(exc),
                        status_code=500,
                    ).model_dump()

        return 404, {"error_code": "ROUTE_NOT_FOUND", "detail": f"No route for {upper_m} {path}", "status_code": 404}

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        """Standard ASGI 3.0 entrypoint compatible with Uvicorn and `httpx.ASGITransport`."""
        if scope.get("type") != "http":
            return
        chunks: list[bytes] = []
        while True:
            event = await receive()
            chunks.append(event.get("body", b""))
            if not event.get("more_body", False):
                break
        status, payload = self.dispatch(
            method=scope.get("method", "GET"),
            path=scope.get("path", "/"),
            query_string=(scope.get("query_string") or b"").decode("utf-8"),
            raw_body=b"".join(chunks),
        )
        encoded = json.dumps(payload).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(encoded)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": encoded})


class SkillGraphTestClient:
    """Lightweight synchronous HTTP test client compatible with FastAPI TestClient & ASGI fallback."""

    def __init__(self, app_instance: Any) -> None:
        self.app = app_instance
        self._native_client: Any = None
        if FASTAPI_AVAILABLE:
            from fastapi.testclient import TestClient  # type: ignore
            self._native_client = TestClient(app_instance)

    def request(self, method: str, url: str, json_body: Any = None) -> "_MockableResponse":
        if self._native_client is not None:
            resp = self._native_client.request(method, url, json=json_body)
            return _MockableResponse(resp.status_code, resp.json())

        path, _, q = url.partition("?")
        raw = json.dumps(json_body).encode("utf-8") if json_body is not None else None
        status, data = self.app.dispatch(method, path, q, raw)
        return _MockableResponse(status, data)

    def get(self, url: str) -> "_MockableResponse":
        return self.request("GET", url)

    def post(self, url: str, json: Any = None) -> "_MockableResponse":
        return self.request("POST", url, json_body=json)


class _MockableResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


# ---------------------------------------------------------------------------
# 6. Application Factory (FastAPI Native when Installed, ASGI Fallback Otherwise)
# ---------------------------------------------------------------------------

def create_app(
    builder: SkillGraphBuilder | None = None,
    config: DatabaseConfig | None = None,
) -> tuple[Any, DatabaseLayer, SkillGapService]:
    """Initializes database abstraction, gap service, and FastAPI/ASGI app."""
    db_layer = DatabaseLayer(builder=builder, config=config)
    service = SkillGapService(db_layer)

    title = "SkillGraph API"
    version = "0.3.0-week3"
    description = (
        "Explainable Knowledge Graph-Based Skill Gap Analysis & Learning Path Recommendation API "
        "(Synthetic 2,000-posting corpus foundation; no real user/course data fabricated)."
    )

    if FASTAPI_AVAILABLE:
        from fastapi import FastAPI, HTTPException, Query  # type: ignore

        f_app = FastAPI(title=title, version=version, description=description)

        @f_app.get("/health", response_model=HealthResponse)
        def health() -> HealthResponse:
            return HealthResponse.model_validate(db_layer.health_snapshot())

        @f_app.get("/occupations", response_model=list[OccupationSummary])
        def list_occupations(industry: str | None = Query(default=None), search: str | None = Query(default=None)) -> list[OccupationSummary]:
            return [OccupationSummary.model_validate(r) for r in db_layer.list_occupations(industry=industry, search=search)]

        @f_app.get("/skills", response_model=list[SkillSummary])
        def list_skills(
            category: str | None = Query(default=None),
            skill_type: str | None = Query(default=None),
            search: str | None = Query(default=None),
        ) -> list[SkillSummary]:
            return [SkillSummary.model_validate(r) for r in db_layer.list_skills(category=category, skill_type=skill_type, search=search)]

        @f_app.get("/occupations/{occupation_id}/skills", response_model=OccupationSkillsResponse)
        def get_occupation_skills(occupation_id: str) -> OccupationSkillsResponse:
            try:
                return OccupationSkillsResponse.model_validate(db_layer.get_occupation_skills(occupation_id))
            except ApiDomainError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        @f_app.post("/skill-gap/analyze", response_model=SkillGapAnalysisResponse)
        def analyze_skill_gap(req: SkillGapAnalyzeRequest) -> SkillGapAnalysisResponse:
            try:
                return service.analyze_and_persist(req)
            except ApiDomainError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        @f_app.get("/skill-gap/{analysis_id}", response_model=SkillGapAnalysisResponse)
        def get_skill_gap(analysis_id: str) -> SkillGapAnalysisResponse:
            try:
                return service.get_analysis_response(analysis_id)
            except ApiDomainError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        @f_app.get("/recommendations/{student_id}", response_model=StudentRecommendationsResponse)
        def get_recommendations(student_id: str) -> StudentRecommendationsResponse:
            try:
                return service.get_recommendations_response(student_id)
            except ApiDomainError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        @f_app.get("/recommendations/{student_id}/explanations", response_model=StudentExplanationsResponse)
        def get_explanations(student_id: str) -> StudentExplanationsResponse:
            try:
                return service.get_explanations_response(student_id)
            except ApiDomainError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        return f_app, db_layer, service

    asgi_app = SkillGraphASGIApp(
        title=title,
        version=version,
        description=description,
        db_layer=db_layer,
        service=service,
    )
    return asgi_app, db_layer, service


app, _default_db_layer, _default_service = create_app()


def main() -> None:
    """Runs deterministic API & OpenAPI verification smoke suite without network ports."""
    client = SkillGraphTestClient(app)
    r_health = client.get("/health")
    r_occs = client.get("/occupations?search=Data")
    r_occ_sk = client.get("/occupations/SG-OC-002/skills")
    r_post = client.post(
        "/skill-gap/analyze",
        json={
            "student_id": "SYNTHETIC_API_DEMO_01",
            "target_occupation": "Data Analyst",
            "skills": [
                {"skill": "Python", "proficiency_level": "intermediate"},
                {"skill": "Communication", "proficiency_score": 0.85},
            ],
            "top_k": 5,
        },
    )
    analysis = r_post.json()
    aid = analysis["analysis_id"]
    r_get = client.get(f"/skill-gap/{aid}")
    r_rec = client.get("/recommendations/SYNTHETIC_API_DEMO_01")
    r_exp = client.get("/recommendations/SYNTHETIC_API_DEMO_01/explanations")
    r_openapi = client.get("/openapi.json")

    print("=== SkillGraph FastAPI Backend Smoke Verification ===")
    print(f"GET /health -> HTTP {r_health.status_code} | mode={r_health.json()['api_framework_mode']} | nodes={r_health.json()['graph_nodes']} | edges={r_health.json()['graph_edges']}")
    print(f"GET /occupations?search=Data -> HTTP {r_occs.status_code} | count={len(r_occs.json())}")
    print(f"GET /occupations/SG-OC-002/skills -> HTTP {r_occ_sk.status_code} | required={len(r_occ_sk.json()['required_skills'])} | preferred={len(r_occ_sk.json()['preferred_only_skills'])}")
    print(f"POST /skill-gap/analyze -> HTTP {r_post.status_code} | analysis_id={aid} | gap_score={analysis['coverage_metrics']['skill_gap_score']}")
    print(f"GET /skill-gap/{aid} -> HTTP {r_get.status_code} | matched={r_get.json()['analysis_id'] == aid}")
    print(f"GET /recommendations/SYNTHETIC_API_DEMO_01 -> HTTP {r_rec.status_code} | count={len(r_rec.json()['recommendations'])}")
    print(f"GET /recommendations/SYNTHETIC_API_DEMO_01/explanations -> HTTP {r_exp.status_code} | explanations={r_exp.json()['explanations_count']}")
    print(f"GET /openapi.json -> HTTP {r_openapi.status_code} | paths={len(r_openapi.json()['paths'])} | schemas={len(r_openapi.json()['components']['schemas'])}")


if __name__ == "__main__":
    main()

