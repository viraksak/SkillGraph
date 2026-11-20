"""SkillGraph Week 4 / Member 01 Module: University Curriculum & Syllabus Ingestion Pipeline.

Parses verified university major and course syllabus catalogs (CSV or JSON),
enforces data provenance and prerequisite-DAG acyclicity, extracts skills using
the offset-preserving negation-aware NLP pipeline, derives evidence-backed
``verified_prerequisites`` for ``SkillGapAnalysisEngine`` and
``LearningPathRecommender``, and persists records into ``university_majors``,
``university_courses``, and ``course_skills``.

Research Integrity Guardrail:
    This module NEVER fabricates university courses, syllabi, or prerequisites,
    and NEVER converts market ``CO_OCCURS_WITH`` edges into prerequisite links.
    When no real curriculum files are supplied, curriculum tables remain empty.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.skill_extraction import SpacySkillExtractor  # noqa: E402
from scripts.preprocess_data import TextPreprocessor  # noqa: E402
from scripts.build_graph import RelationalDatabaseManager  # noqa: E402

DEFAULT_TAXONOMY_CSV = ROOT_DIR / "data" / "processed" / "global_skill_taxonomy.csv"

EXTRACTION_METHOD_LABEL = "spacy_dependency_negation_filtered"


class CurriculumValidationError(ValueError):
    """Raised when curriculum metadata, provenance, or prerequisite DAG is invalid."""


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _parse_prereq_list(raw: Any) -> list[str]:
    """Parses prerequisite course IDs from list, JSON array string, or delimiter string."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text or text.lower() in {"none", "null", "nan", "[]"}:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise CurriculumValidationError(f"Expected JSON list for prerequisite_course_ids, got: {text!r}")
        return [str(x).strip() for x in parsed if str(x).strip()]
    sep = ";" if ";" in text else ","
    return [part.strip() for part in text.split(sep) if part.strip()]


class CurriculumIngestionPipeline:
    """Validates, extracts skills from, and persists university curriculum catalogs."""

    def __init__(
        self,
        taxonomy_path: str | Path = DEFAULT_TAXONOMY_CSV,
        extractor: SpacySkillExtractor | None = None,
        strict_provenance: bool = True,
    ) -> None:
        self.taxonomy_path = Path(taxonomy_path)
        self.preprocessor = TextPreprocessor()
        self.extractor = extractor or SpacySkillExtractor(
            self.taxonomy_path,
            use_fast_tokenizer=True,
        )
        self.strict_provenance = strict_provenance

    @staticmethod
    def load_records(path: str | Path) -> list[dict[str, Any]]:
        """Loads major or course records from CSV or JSON without modifying source files."""
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Curriculum file not found: {src}")
        suffix = src.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(src)
            return df.to_dict(orient="records")
        if suffix == ".json":
            payload = json.loads(src.read_text(encoding="utf-8-sig"))
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                return payload["data"]
            raise CurriculumValidationError(f"Unsupported JSON layout in {src}; expected list or {{'data': [...]}}.")
        raise CurriculumValidationError(f"Unsupported curriculum file format {suffix!r}; use .csv or .json.")

    def validate_records(
        self,
        majors: list[dict[str, Any]],
        courses: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Validates required schema fields, foreign keys, provenance, and DAG acyclicity."""
        clean_majors: list[dict[str, Any]] = []
        major_ids: set[str] = set()
        major_names: set[str] = set()

        for idx, raw in enumerate(majors):
            mid = _clean_str(raw.get("major_id"))
            mname = _clean_str(raw.get("major_name"))
            inst = _clean_str(raw.get("institution_name"))
            deg = _clean_str(raw.get("degree_level"))
            prov = _clean_str(raw.get("data_provenance")) or "verified_curriculum_ingestion"
            source_url = _clean_str(raw.get("source_url"))
            collected_at = _clean_str(raw.get("collected_at") or raw.get("collection_date"))

            if not (mid and mname and inst and deg):
                raise CurriculumValidationError(
                    f"Major row {idx} missing required fields (major_id, major_name, institution_name, degree_level)."
                )
            if mid in major_ids:
                raise CurriculumValidationError(f"Duplicate major_id detected: {mid!r}")
            if mname in major_names:
                raise CurriculumValidationError(f"Duplicate major_name detected: {mname!r}")
            if self.strict_provenance and not (source_url and collected_at):
                raise CurriculumValidationError(
                    f"Major {mid!r} missing required provenance (source_url and collected_at)."
                )
            major_ids.add(mid)
            major_names.add(mname)
            clean_majors.append({
                "major_id": mid,
                "major_name": mname,
                "institution_name": inst,
                "degree_level": deg,
                "data_provenance": prov,
                "source_url": source_url,
                "collected_at": collected_at,
            })

        clean_courses: list[dict[str, Any]] = []
        course_ids: set[str] = set()
        major_code_pairs: set[tuple[str, str]] = set()

        for idx, raw in enumerate(courses):
            cid = _clean_str(raw.get("course_id"))
            ccode = _clean_str(raw.get("course_code"))
            ctitle = _clean_str(raw.get("course_title"))
            mid = _clean_str(raw.get("major_id"))
            syllabus = _clean_str(raw.get("syllabus_text"))
            outcomes = _clean_str(raw.get("learning_outcomes"))
            full_text = f"{syllabus}\n{outcomes}".strip() if outcomes else syllabus
            prov = _clean_str(raw.get("data_provenance")) or "verified_curriculum_ingestion"
            source_url = _clean_str(raw.get("source_url"))
            collected_at = _clean_str(raw.get("collected_at") or raw.get("collection_date"))
            target_level = _clean_str(raw.get("target_proficiency_level")) or None
            prereq_ids = _parse_prereq_list(raw.get("prerequisite_course_ids"))

            if not (cid and ccode and ctitle and mid):
                raise CurriculumValidationError(
                    f"Course row {idx} missing required fields (course_id, course_code, course_title, major_id)."
                )
            if mid not in major_ids:
                raise CurriculumValidationError(
                    f"Course {cid!r} references unknown major_id {mid!r}."
                )
            if cid in course_ids:
                raise CurriculumValidationError(f"Duplicate course_id detected: {cid!r}")
            if (mid, ccode) in major_code_pairs:
                raise CurriculumValidationError(f"Duplicate (major_id, course_code) detected: {(mid, ccode)!r}")
            if self.strict_provenance and not (source_url and collected_at):
                raise CurriculumValidationError(
                    f"Course {cid!r} missing required provenance (source_url and collected_at)."
                )

            course_ids.add(cid)
            major_code_pairs.add((mid, ccode))
            clean_courses.append({
                "course_id": cid,
                "course_code": ccode,
                "course_title": ctitle,
                "major_id": mid,
                "syllabus_text": full_text,
                "data_provenance": prov,
                "source_url": source_url,
                "collected_at": collected_at,
                "target_proficiency_level": target_level,
                "prerequisite_course_ids": prereq_ids,
            })

        self._validate_prerequisite_dag(clean_courses)
        return clean_majors, clean_courses

    @staticmethod
    def _validate_prerequisite_dag(courses: list[dict[str, Any]]) -> None:
        """Ensures course prerequisite edges reference known courses and contain no cycles."""
        known_ids = {c["course_id"] for c in courses}
        adj: dict[str, list[str]] = {}
        for c in courses:
            cid = c["course_id"]
            prereqs = c["prerequisite_course_ids"]
            for pid in prereqs:
                if pid == cid:
                    raise CurriculumValidationError(f"Self-loop prerequisite detected on course {cid!r}.")
                if pid not in known_ids:
                    raise CurriculumValidationError(
                        f"Course {cid!r} declares unknown prerequisite_course_id {pid!r}."
                    )
            adj[cid] = list(prereqs)

        # Standard 3-color DFS cycle detection (0=unvisited, 1=visiting, 2=visited)
        state: dict[str, int] = {cid: 0 for cid in known_ids}

        def dfs(node: str) -> None:
            state[node] = 1
            for dep in adj.get(node, []):
                if state[dep] == 1:
                    raise CurriculumValidationError(
                        f"Prerequisite cycle detected involving course {node!r} -> {dep!r}."
                    )
                if state[dep] == 0:
                    dfs(dep)
            state[node] = 2

        for cid in sorted(known_ids):
            if state[cid] == 0:
                dfs(cid)

    def extract_course_skills(
        self,
        courses: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extracts canonical taxonomy skills from course syllabi while filtering negated mentions.

        Returns:
            ``(course_skill_rows, mention_audit_rows)`` where ``course_skill_rows`` is deduplicated
            on ``(course_id, skill_id)`` for relational DB insertion.
        """
        course_skill_rows: list[dict[str, Any]] = []
        mention_audit_rows: list[dict[str, Any]] = []

        for course in courses:
            cid = course["course_id"]
            raw_text = course.get("syllabus_text") or ""
            if not raw_text.strip():
                continue
            norm = self.preprocessor.normalize_preserving_offsets(raw_text)
            spans = self.extractor.extract(norm["normalized_text"], include_negated=False)

            seen_skills: set[str] = set()
            for sp in spans:
                sid = sp.get("skill_id")
                if not sid:
                    continue
                mention_audit_rows.append({
                    "course_id": cid,
                    "skill_id": sid,
                    "canonical_skill": sp["canonical_skill"],
                    "skill_mention": sp["skill_mention"],
                    "character_start": sp["character_start"],
                    "character_end": sp["character_end"],
                    "modality": sp.get("modality", "mentioned"),
                })
                if sid not in seen_skills:
                    seen_skills.add(sid)
                    course_skill_rows.append({
                        "course_id": cid,
                        "skill_id": sid,
                        "extraction_method": EXTRACTION_METHOD_LABEL,
                        "target_proficiency_level": course.get("target_proficiency_level"),
                    })

        return course_skill_rows, mention_audit_rows

    @staticmethod
    def derive_verified_skill_prerequisites(
        courses: list[dict[str, Any]],
        course_skill_rows: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Derives evidence-backed ``skill_id -> [prereq_skill_ids]`` strictly from course prerequisites."""
        skills_by_course: dict[str, set[str]] = defaultdict(set)
        for row in course_skill_rows:
            skills_by_course[row["course_id"]].add(row["skill_id"])

        prereq_map: dict[str, set[str]] = defaultdict(set)
        for course in courses:
            cid = course["course_id"]
            target_skills = skills_by_course.get(cid, set())
            if not target_skills:
                continue
            for prereq_cid in course.get("prerequisite_course_ids", []):
                for p_sid in skills_by_course.get(prereq_cid, set()):
                    for t_sid in target_skills:
                        if p_sid != t_sid:
                            prereq_map[t_sid].add(p_sid)

        return {sid: sorted(reqs) for sid, reqs in sorted(prereq_map.items())}

    @staticmethod
    def load_into_database(
        db_manager: RelationalDatabaseManager,
        majors: list[dict[str, Any]],
        courses: list[dict[str, Any]],
        course_skill_rows: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Persists validated curriculum records cleanly into the relational database."""
        with db_manager.conn:
            db_manager.conn.executemany(
                """
                INSERT INTO university_majors (
                    major_id, major_name, institution_name, degree_level, data_provenance
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(major_id) DO UPDATE SET
                    major_name = excluded.major_name,
                    institution_name = excluded.institution_name,
                    degree_level = excluded.degree_level,
                    data_provenance = excluded.data_provenance
                """,
                [
                    (
                        m["major_id"],
                        m["major_name"],
                        m["institution_name"],
                        m["degree_level"],
                        m["data_provenance"],
                    )
                    for m in majors
                ],
            )
            db_manager.conn.executemany(
                """
                INSERT INTO university_courses (
                    course_id, course_code, course_title, major_id, syllabus_text, data_provenance
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(course_id) DO UPDATE SET
                    course_code = excluded.course_code,
                    course_title = excluded.course_title,
                    major_id = excluded.major_id,
                    syllabus_text = excluded.syllabus_text,
                    data_provenance = excluded.data_provenance
                """,
                [
                    (
                        c["course_id"],
                        c["course_code"],
                        c["course_title"],
                        c["major_id"],
                        c["syllabus_text"],
                        c["data_provenance"],
                    )
                    for c in courses
                ],
            )
            db_manager.conn.executemany(
                """
                INSERT INTO course_skills (
                    course_id, skill_id, extraction_method, target_proficiency_level
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(course_id, skill_id) DO UPDATE SET
                    extraction_method = excluded.extraction_method,
                    target_proficiency_level = excluded.target_proficiency_level
                """,
                [
                    (
                        r["course_id"],
                        r["skill_id"],
                        r["extraction_method"],
                        r["target_proficiency_level"],
                    )
                    for r in course_skill_rows
                ],
            )

        counts = db_manager.list_tables_and_counts()
        return {
            "university_majors": counts["university_majors"],
            "university_courses": counts["university_courses"],
            "course_skills": counts["course_skills"],
        }

    def ingest(
        self,
        majors: list[dict[str, Any]],
        courses: list[dict[str, Any]],
        db_manager: RelationalDatabaseManager | None = None,
    ) -> dict[str, Any]:
        """Runs end-to-end validation, skill extraction, prerequisite derivation, and optional DB load."""
        clean_majors, clean_courses = self.validate_records(majors, courses)
        course_skill_rows, mention_audit_rows = self.extract_course_skills(clean_courses)
        verified_prereqs = self.derive_verified_skill_prerequisites(clean_courses, course_skill_rows)

        db_counts: dict[str, int] | None = None
        if db_manager is not None:
            db_counts = self.load_into_database(
                db_manager=db_manager,
                majors=clean_majors,
                courses=clean_courses,
                course_skill_rows=course_skill_rows,
            )

        return {
            "majors_validated": len(clean_majors),
            "courses_validated": len(clean_courses),
            "course_skills_extracted": len(course_skill_rows),
            "skill_mentions_audited": len(mention_audit_rows),
            "verified_skill_prerequisites": verified_prereqs,
            "database_counts": db_counts,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkillGraph University Curriculum & Syllabus Ingestion Pipeline (real catalogs only; no fake data)."
    )
    parser.add_argument("--majors", type=str, default=None, help="Path to real university_majors CSV or JSON.")
    parser.add_argument("--courses", type=str, default=None, help="Path to real university_courses CSV or JSON.")
    parser.add_argument(
        "--non-strict-provenance",
        action="store_true",
        help="Allow rows without source_url/collected_at (for local pilot testing only).",
    )
    args = parser.parse_args()

    if not args.majors or not args.courses:
        print("=== SkillGraph Curriculum Ingestion Pipeline (Ready — Awaiting Real Data) ===")
        print("Status: No curriculum files provided; 0 synthetic courses fabricated.")
        print("Required major columns : major_id, major_name, institution_name, degree_level, source_url, collected_at")
        print("Required course columns: course_id, course_code, course_title, major_id, syllabus_text, source_url, collected_at")
        print("Optional course columns: learning_outcomes, prerequisite_course_ids, target_proficiency_level")
        print("Usage: python scripts/ingest_curriculum.py --majors <path> --courses <path>")
        return

    pipeline = CurriculumIngestionPipeline(strict_provenance=not args.non_strict_provenance)
    majors = pipeline.load_records(args.majors)
    courses = pipeline.load_records(args.courses)
    db = RelationalDatabaseManager()
    result = pipeline.ingest(majors=majors, courses=courses, db_manager=db)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
