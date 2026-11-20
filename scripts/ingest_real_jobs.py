"""SkillGraph Week 4 / Member 01 Module: Real Job-Posting Ingestion & Quality Audit Pipeline.

Validates real job postings collected into ``data/raw/real_job_postings.csv`` (or JSON),
enforces provenance (``source_url``, ``source_name``, ``collected_at``) and strict
separation from synthetic postings (``is_synthetic`` must be false), deduplicates
postings, runs offset-preserving preprocessing and negation-aware skill extraction,
and generates a reproducible quality & EDA summary.

Research Integrity Guardrail:
    This module NEVER fabricates real job postings and NEVER injects synthetic
    postings into real-data evaluation summaries.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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

DEFAULT_REAL_JOBS_PATH = ROOT_DIR / "data" / "raw" / "real_job_postings.csv"
DEFAULT_TAXONOMY_CSV = ROOT_DIR / "data" / "processed" / "global_skill_taxonomy.csv"
DEFAULT_OCCUPATIONS_CSV = ROOT_DIR / "data" / "processed" / "global_occupation_mapping.csv"

OPTIONAL_AUDIT_COLUMNS = (
    "country",
    "industry",
    "company_name",
    "education_level",
    "salary_min",
    "salary_max",
    "posted_at",
)


class RealJobValidationError(ValueError):
    """Raised when real job-posting records violate provenance or synthetic-isolation rules."""


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _is_truthy_synthetic(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    return str(value).strip().lower() in {"1", "true", "yes", "synthetic"}


class RealJobPostingPipeline:
    """Validates, cleans, extracts skills from, and audits real job postings."""

    def __init__(
        self,
        taxonomy_path: str | Path = DEFAULT_TAXONOMY_CSV,
        occupations_path: str | Path = DEFAULT_OCCUPATIONS_CSV,
        extractor: SpacySkillExtractor | None = None,
        strict_provenance: bool = True,
    ) -> None:
        self.taxonomy_path = Path(taxonomy_path)
        self.occupations_path = Path(occupations_path)
        self.preprocessor = TextPreprocessor()
        self.extractor = extractor or SpacySkillExtractor(
            self.taxonomy_path,
            use_fast_tokenizer=True,
        )
        self.strict_provenance = strict_provenance
        self.occupation_lookup = self._load_occupation_lookup()

    def _load_occupation_lookup(self) -> dict[str, dict[str, str]]:
        if not self.occupations_path.exists():
            return {}
        df = pd.read_csv(self.occupations_path)
        lookup: dict[str, dict[str, str]] = {}
        for row in df.to_dict(orient="records"):
            occ = _clean_str(row.get("occupation"))
            oid = _clean_str(row.get("occupation_id"))
            if occ and oid:
                lookup[occ.lower()] = {
                    "occupation_id": oid,
                    "occupation": occ,
                }
        return lookup

    @staticmethod
    def load_records(path: str | Path) -> list[dict[str, Any]]:
        """Loads job postings from CSV or JSON cleanly without altering source files."""
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Real job postings file not found: {src}")
        suffix = src.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(src)
            return df.to_dict(orient="records")
        if suffix == ".json":
            raw = src.read_text(encoding="utf-8-sig").strip()
            if not raw:
                return []
            payload = json.loads(raw)
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                return payload["data"]
            raise RealJobValidationError(f"Unsupported JSON layout in {src}; expected list or {{'data': [...]}}.")
        raise RealJobValidationError(f"Unsupported file format {suffix!r}; use .csv or .json.")

    def validate_and_deduplicate(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Enforces provenance, blocks synthetic leakage, and removes duplicate postings."""
        valid_records: list[dict[str, Any]] = []
        seen_job_ids: set[str] = set()
        seen_content_hashes: set[str] = set()
        duplicate_job_ids: list[str] = []
        duplicate_content_job_ids: list[str] = []
        missingness: Counter[str] = Counter()

        for idx, raw in enumerate(records):
            if _is_truthy_synthetic(raw.get("is_synthetic")):
                raise RealJobValidationError(
                    f"Row {idx} is marked is_synthetic=True; synthetic records are forbidden in real_job_postings."
                )
            data_type = _clean_str(raw.get("data_type")).lower()
            if data_type == "synthetic":
                raise RealJobValidationError(
                    f"Row {idx} has data_type='synthetic'; synthetic records are forbidden in real_job_postings."
                )

            job_id = _clean_str(raw.get("job_id"))
            occ = _clean_str(raw.get("occupation") or raw.get("job_title"))
            desc = _clean_str(raw.get("description"))
            source_url = _clean_str(raw.get("source_url"))
            source_name = _clean_str(raw.get("source_name"))
            collected_at = _clean_str(raw.get("collected_at") or raw.get("collection_date"))

            if not (job_id and occ and desc):
                raise RealJobValidationError(
                    f"Row {idx} missing required fields (job_id, occupation/job_title, description)."
                )
            if self.strict_provenance and not (source_url and source_name and collected_at):
                raise RealJobValidationError(
                    f"Real posting {job_id!r} missing required provenance (source_url, source_name, collected_at)."
                )

            if job_id in seen_job_ids:
                duplicate_job_ids.append(job_id)
                continue

            norm_key = f"{occ.lower()}||{' '.join(desc.lower().split())}"
            content_hash = hashlib.sha256(norm_key.encode("utf-8")).hexdigest()[:16]
            if content_hash in seen_content_hashes:
                duplicate_content_job_ids.append(job_id)
                continue

            seen_job_ids.add(job_id)
            seen_content_hashes.add(content_hash)

            for col in OPTIONAL_AUDIT_COLUMNS:
                if not _clean_str(raw.get(col)):
                    missingness[col] += 1

            occ_match = self.occupation_lookup.get(occ.lower())
            valid_records.append({
                "job_id": job_id,
                "occupation": occ,
                "mapped_occupation_id": occ_match["occupation_id"] if occ_match else "UNMAPPED_OCCUPATION",
                "mapped_occupation_name": occ_match["occupation"] if occ_match else None,
                "country": _clean_str(raw.get("country")) or None,
                "industry": _clean_str(raw.get("industry")) or None,
                "description": desc,
                "source_name": source_name or None,
                "source_url": source_url or None,
                "collected_at": collected_at or None,
            })

        return {
            "total_input_rows": len(records),
            "valid_unique_rows": len(valid_records),
            "duplicate_job_ids_dropped": duplicate_job_ids,
            "duplicate_content_dropped": duplicate_content_job_ids,
            "optional_field_missing_counts": dict(sorted(missingness.items())),
            "records": valid_records,
        }

    def process_and_audit(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Runs validation, offset-preserving cleaning, negation-aware extraction, and EDA summary."""
        val = self.validate_and_deduplicate(records)
        clean_recs = val["records"]

        if not clean_recs:
            return {
                "status": "AWAITING_REAL_DATA",
                "total_input_rows": val["total_input_rows"],
                "valid_unique_rows": 0,
                "duplicate_job_ids_dropped": val["duplicate_job_ids_dropped"],
                "duplicate_content_dropped": val["duplicate_content_dropped"],
                "extracted_skill_mentions": 0,
                "eda_summary": {
                    "occupations": {},
                    "countries": {},
                    "top_skills": [],
                    "modality_counts_including_negated": {},
                    "unmapped_occupations": [],
                    "length_preserved_count": 0,
                },
                "extracted_rows": [],
            }

        occ_counts: Counter[str] = Counter()
        country_counts: Counter[str] = Counter()
        skill_counts: Counter[str] = Counter()
        modality_counts: Counter[str] = Counter()
        unmapped_occs: set[str] = set()
        length_preserved = 0
        extracted_rows: list[dict[str, Any]] = []

        for rec in clean_recs:
            occ_counts[rec["occupation"]] += 1
            if rec["country"]:
                country_counts[rec["country"]] += 1
            if rec["mapped_occupation_id"] == "UNMAPPED_OCCUPATION":
                unmapped_occs.add(rec["occupation"])

            norm = self.preprocessor.normalize_preserving_offsets(rec["description"])
            if norm["length_preserved"]:
                length_preserved += 1
            sections = self.preprocessor.segment_job_sections(norm["normalized_text"])

            all_spans = self.extractor.extract(norm["normalized_text"], include_negated=True)
            for sp in all_spans:
                modality_counts[str(sp.get("modality", "mentioned"))] += 1
                if not sp.get("is_negated"):
                    skill_counts[sp["canonical_skill"]] += 1
                    sec = self.preprocessor.get_section_for_span(
                        sections,
                        sp["character_start"],
                        sp["character_end"],
                    )
                    extracted_rows.append({
                        "job_id": rec["job_id"],
                        "occupation": rec["occupation"],
                        "mapped_occupation_id": rec["mapped_occupation_id"],
                        "skill_id": sp["skill_id"],
                        "canonical_skill": sp["canonical_skill"],
                        "skill_mention": sp["skill_mention"],
                        "character_start": sp["character_start"],
                        "character_end": sp["character_end"],
                        "source_section": sec,
                        "modality": sp.get("modality", "mentioned"),
                    })

        return {
            "status": "REAL_DATA_AUDITED",
            "total_input_rows": val["total_input_rows"],
            "valid_unique_rows": len(clean_recs),
            "duplicate_job_ids_dropped": val["duplicate_job_ids_dropped"],
            "duplicate_content_dropped": val["duplicate_content_dropped"],
            "optional_field_missing_counts": val["optional_field_missing_counts"],
            "extracted_skill_mentions": len(extracted_rows),
            "eda_summary": {
                "occupations": dict(occ_counts.most_common()),
                "countries": dict(country_counts.most_common()),
                "top_skills": [
                    {"canonical_skill": k, "mention_count": v}
                    for k, v in skill_counts.most_common(15)
                ],
                "modality_counts_including_negated": dict(sorted(modality_counts.items())),
                "unmapped_occupations": sorted(unmapped_occs),
                "length_preserved_count": length_preserved,
            },
            "extracted_rows": extracted_rows,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkillGraph Real Job-Posting Ingestion & Quality Audit Pipeline (no fake data)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_REAL_JOBS_PATH),
        help="Path to real job postings CSV or JSON (defaults to data/raw/real_job_postings.csv).",
    )
    parser.add_argument(
        "--non-strict-provenance",
        action="store_true",
        help="Allow rows without source_url/source_name/collected_at (for local pilot testing only).",
    )
    args = parser.parse_args()

    pipeline = RealJobPostingPipeline(strict_provenance=not args.non_strict_provenance)
    records = pipeline.load_records(args.input)
    report = pipeline.process_and_audit(records)
    if report["status"] == "AWAITING_REAL_DATA":
        print("=== SkillGraph Real Job-Posting Pipeline (Ready — Awaiting Real Data) ===")
        print(f"Input file: {args.input} (0 rows; 0 synthetic rows fabricated)")
        print("Required fields: job_id, occupation, description, source_url, source_name, collected_at (is_synthetic=False)")
        return
    summary = {k: v for k, v in report.items() if k != "extracted_rows"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
