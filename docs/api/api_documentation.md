# SkillGraph REST API Documentation (Week 3)

**Service:** `SkillGraph Career Skill Gap & Recommendation API` (`version: 0.3.0-week3`)  
**Implementation:** [backend/app/main.py](file:///D:/Downloads/SkillGraph/backend/app/main.py)  
**OpenAPI Specification:** `GET /openapi.json` (OpenAPI `3.1.0`, `8` documented paths, `13` Pydantic v2 component schemas)

> **Runtime & Environment Status (`Implemented` / `Tested`):**  
> - When `fastapi` is installed, `create_app()` constructs a native `fastapi.FastAPI` instance serving identical Pydantic v2 schemas and endpoints.  
> - In the offline research `.venv` (where `fastapi`, `uvicorn`, `psycopg2`, `psycopg`, `asyncpg`, and `sqlalchemy` are not installed), `create_app()` runs `SkillGraphASGIApp`, a pure-Python ASGI 3.0 + Pydantic v2 router with `SkillGraphTestClient`, preserving full Pydantic input/output validation, HTTP `200`/`400`/`404`/`422` handling, OpenAPI `3.1.0` generation, and an in-memory/SQLite `PRAGMA foreign_keys = ON` relational mirror of the 17-table PostgreSQL schema.  
> - Live database host audit on `127.0.0.1:5432` reports PostgreSQL 17 accepting TCP connections (`postgresql_env.port_5432_status = "127.0.0.1:5432 - accepting connections"`), while `live_pg_execution_ready = false` until Python PostgreSQL drivers and `.env` credentials (`SKILLGRAPH_DB_USER`, `SKILLGRAPH_DB_PASSWORD`) are supplied.

---

## 1. Endpoints Overview

| Method | Path | Summary | Status Codes | Status |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service, graph summary, DB backend & PostgreSQL audit snapshot | `200` | Implemented & Tested |
| `GET` | `/occupations` | List/filter canonical occupations (`?industry=...&search=...`) | `200` | Implemented & Tested |
| `GET` | `/skills` | List/filter canonical taxonomy skills (`?category=...&skill_type=...&search=...`) | `200` | Implemented & Tested |
| `GET` | `/occupations/{occupation_id}/skills` | Retrieve required & preferred-only skill demands for an occupation | `200`, `404` | Implemented & Tested |
| `POST` | `/skill-gap/analyze` | Analyze student profile vs target occupation, compute `P(s)` & persist | `200`, `400`, `404`, `422` | Implemented & Tested |
| `GET` | `/skill-gap/{analysis_id}` | Fetch a previously persisted skill gap analysis by ID | `200`, `404` | Implemented & Tested |
| `GET` | `/recommendations/{student_id}` | Fetch ordered gap recommendations & learning-path stages for latest analysis | `200`, `404` | Implemented & Tested |
| `GET` | `/recommendations/{student_id}/explanations` | Fetch 7-part structured explanations for latest student recommendations | `200`, `404` | Implemented & Tested |
| `GET` | `/openapi.json` | Auto-generated OpenAPI `3.1.0` JSON schema | `200` | Implemented & Tested |

---

## 2. Error Handling Contract (`ErrorResponse` / Pydantic `422`)

Domain errors return `ErrorResponse`:

```json
{
  "error_code": "RESOURCE_NOT_FOUND",
  "detail": "Unknown target occupation: 'Astral Wizard'",
  "status_code": 404
}
```

- **`400 INVALID_INPUT`**: Raised when `strict_validation=true` and supplied skills contain unrecognized terms not in the 230-skill taxonomy, or malformed priority weights (`delta < 0`, etc.).
- **`404 RESOURCE_NOT_FOUND`**: Raised for unknown `occupation_id` / target occupation, missing `analysis_id`, or unknown `student_id` with no persisted analysis.
- **`422 Validation Error`**: Raised by Pydantic v2 when required fields are missing (e.g., empty `student_id`), numeric bounds are violated (`proficiency_score` outside `[0.0, 1.0]`, `top_k < 1` or `> 230`, `alpha + beta + gamma <= 0`), or JSON body is malformed.

---

## 3. Endpoint Reference & Observed Payloads

### 3.1 `GET /health`

Returns service mode (`asgi_pydantic_fallback` or `fastapi`), active Knowledge Graph node/edge counts (`2,336` nodes, `31,503` edges), PostgreSQL host audit, and row counts across all 17 relational tables.

**Observed Response (`200 OK`):**
```json
{
  "status": "ok",
  "service": "SkillGraph Career Skill Gap & Recommendation API",
  "api_framework_mode": "asgi_pydantic_fallback",
  "fastapi_installed": false,
  "graph_nodes": 2336,
  "graph_edges": 31503,
  "database_backend_active": "sqlite3(:memory:) [PostgreSQL DDL & constraints mirrored]",
  "postgresql_env": {
    "psql_exists": true,
    "pg_isready_exists": true,
    "port_5432_status": "127.0.0.1:5432 - accepting connections",
    "live_pg_execution_ready": false,
    "offline_sql_fallback": "sqlite3_with_foreign_keys_enabled"
  },
  "table_counts": {
    "skill_categories": 18,
    "skills": 230,
    "skill_aliases": 325,
    "industries": 15,
    "occupations": 53,
    "countries": 20,
    "job_postings": 2000,
    "job_skill_mentions": 21588,
    "occupation_skill_demand": 1645,
    "skill_co_occurrences": 3987,
    "university_majors": 0,
    "university_courses": 0,
    "course_skills": 0
  }
}
```

### 3.2 `GET /occupations`

Query parameters (all optional):
- `industry` (exact/case-insensitive industry filter)
- `search` (case-insensitive substring match against `occupation`, `occupation_id`, or `department_label`)

**Example `GET /occupations?search=Data` (`200 OK`, 3 items):**
```json
[
  {
    "occupation_id": "SG-OC-002",
    "occupation": "Data Analyst",
    "industry_id": "IND_INFORMATION_TECHNOLOGY",
    "industry": "Information Technology",
    "department_label": "Data Analytics",
    "isco08_code_candidate": "2511",
    "isco08_label_candidate": "Systems analysts",
    "posting_count": 16
  }
]
```

### 3.3 `GET /skills`

Query parameters (all optional):
- `category`: Filter by `skill_category`
- `skill_type`: Filter by `skill_type`
- `search`: Substring filter across `canonical_skill`, `skill_id`, or aliases

**Example `GET /skills?search=SQL` (`200 OK`, 2 items):**
```json
[
  {
    "skill_id": "SG-SK-0006",
    "canonical_skill": "SQL",
    "skill_category": "Database technologies",
    "skill_type": "Technology",
    "taxonomy_source": "SKILLGRAPH_LOCAL_V1",
    "mapping_status": "external_unmapped_download_required"
  },
  {
    "skill_id": "SG-SK-0015",
    "canonical_skill": "PostgreSQL",
    "skill_category": "Database technologies",
    "skill_type": "Technology",
    "taxonomy_source": "SKILLGRAPH_LOCAL_V1",
    "mapping_status": "external_unmapped_download_required"
  }
]
```

### 3.4 `GET /occupations/{occupation_id}/skills`

Returns required skills (`required_skills`), preferred-only skills (`preferred_only_skills`), and combined `all_skills` for `occupation_id` (`SG-OC-002` returns `21` required and `5` preferred-only skills, `26` total). Returns `404` if `occupation_id` is unknown.

### 3.5 `POST /skill-gap/analyze`

Evaluates a synthetic/supplied student profile (`SkillGapAnalyzeRequest`) against the target occupation, runs priority scoring `P(s)` + multi-occupation opportunity `O(s)` + optional weight sensitivity analysis, builds honest heuristic learning-path stages, and persists the run + per-recommendation explanations into the relational store.

**Request Schema (`SkillGapAnalyzeRequest`):**
```json
{
  "student_id": "SYNTHETIC_DOC_DEMO",
  "target_occupation": "Data Analyst",
  "skills": [
    {"skill": "Python", "proficiency_level": "intermediate"},
    {"skill": "SQL", "proficiency_level": "beginner"}
  ],
  "candidate_occupations": [],
  "is_synthetic": true,
  "strict_validation": false,
  "top_k": 5,
  "weights": {
    "alpha": 0.7,
    "beta": 0.3,
    "gamma": 0.0,
    "delta": 0.0,
    "preset_name": "baseline_balanced_heuristic"
  },
  "run_sensitivity": true
}
```

**Observed Coverage Block (`200 OK`, `top_k: 2`):**
```json
{
  "analysis_id": "GAP-F4E961BF597F",
  "student_id": "SYNTHETIC_DOC_DEMO",
  "is_synthetic_profile": true,
  "target_occupation": {
    "occupation_id": "SG-OC-002",
    "occupation": "Data Analyst",
    "industry": "Information Technology",
    "total_postings_analyzed": 16
  },
  "coverage_metrics": {
    "required_skill_coverage": 0.0714,
    "overall_weighted_coverage": 0.1137,
    "skill_gap_score": 0.8863,
    "possessed_count": 1,
    "partially_demonstrated_count": 1,
    "missing_required_count": 19,
    "missing_preferred_count": 5,
    "target_total_demanded_skills": 26
  }
}
```

### 3.6 `GET /skill-gap/{analysis_id}`

Retrieves the full persisted `SkillGapAnalysisResponse` JSON from `skill_gap_analyses` + `recommendation_explanations`. Returns `404` if `analysis_id` does not exist.

### 3.7 `GET /recommendations/{student_id}`

Returns `StudentRecommendationsResponse` for the latest persisted analysis of `student_id` (`ORDER BY created_at DESC, rowid DESC LIMIT 1`), including `recommendations`, `learning_path`, `ordering_strategy`, `learning_resources_status` (`"UNAVAILABLE_NO_VERIFIED_COURSE_CATALOG"`), and explicit `limitation_notice`.

### 3.8 `GET /recommendations/{student_id}/explanations`

Returns `StudentExplanationsResponse` containing the 7-part `StructuredExplanationModel` for every recommended skill:
1. `why_identified`
2. `occupation_requirement_role`
3. `student_profile_status` (strictly writes *"not recorded in the supplied student profile (absence of record does not prove lack of competence)"* for unrecorded skills)
4. `importance_assessment`
5. `multi_occupation_relevance`
6. `priority_calculation` (`priority_score`, `formula`, `weights`, `components` including explicit `D_status: UNAVAILABLE_NO_VERIFIED_PREREQUISITE_GRAPH` and `C_status: UNAVAILABLE_NO_EMPIRICAL_COST_DATA`)
7. `supporting_evidence` (postings count, external required/preferred occupations, associative `co_occurring_skills` tagged `"statistical_co_occurrence_not_prerequisite"`, dataset provenance)

---

## 4. Pydantic v2 Component Schemas (`13`)

`HealthResponse`, `OccupationSummary`, `SkillSummary`, `OccupationSkillsResponse`, `StudentSkillInput`, `PriorityWeightsInput`, `SkillGapAnalyzeRequest`, `SkillRecommendationItem`, `StructuredExplanationModel`, `SkillGapAnalysisResponse`, `StudentRecommendationsResponse`, `StudentExplanationsResponse`, `ErrorResponse`.

