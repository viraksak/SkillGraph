
"""Load and validate SkillGraph's two-pass human annotation dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_annotation_dataset(
    path: str | Path,
) -> dict[str, Any]:
    """Load two-pass annotations and prepare evaluation-ready records.

    Canonical evaluation requires:
    - A COMPLETED record
    - An APPROVED Pass 1 span
    - An APPROVED Pass 2 mapping
    - Valid character offsets matching the original text
    """

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError("Invalid annotation dataset: expected a records list.")

    ground_truth_by_job: dict[str, list[dict[str, Any]]] = {}
    unresolved_by_job: dict[str, list[dict[str, Any]]] = {}

    summary: Counter[str] = Counter()
    issues: list[dict[str, Any]] = []

    for record in data["records"]:
        job_id = record.get("job_id")
        text = record.get("text", "")

        if not job_id or not isinstance(text, str):
            issues.append({
                "job_id": job_id,
                "issue": "MISSING_JOB_ID_OR_INVALID_TEXT",
            })
            continue

        if record.get("record_status") != "COMPLETED":
            summary["records_not_completed"] += 1
            continue

        summary["completed_records"] += 1

        pass1_spans = record.get("pass_1", {}).get("spans", [])
        pass2_mappings = record.get("pass_2", {}).get(
            "canonical_mappings", []
        )

        pass1_by_id = {}
        pass2_by_id = {}

        for span in pass1_spans:
            span_id = span.get("span_id")

            if not span_id or span_id in pass1_by_id:
                issues.append({
                    "job_id": job_id,
                    "span_id": span_id,
                    "issue": "MISSING_OR_DUPLICATE_PASS1_ID",
                })
                continue

            pass1_by_id[span_id] = span

        for mapping in pass2_mappings:
            span_id = mapping.get("span_id")

            if not span_id or span_id in pass2_by_id:
                issues.append({
                    "job_id": job_id,
                    "span_id": span_id,
                    "issue": "MISSING_OR_DUPLICATE_PASS2_ID",
                })
                continue

            pass2_by_id[span_id] = mapping

        summary["pass1_spans"] += len(pass1_spans)
        summary["pass2_mappings"] += len(pass2_mappings)

        for span_id, span in pass1_by_id.items():
            mapping = pass2_by_id.get(span_id)

            if mapping is None:
                issues.append({
                    "job_id": job_id,
                    "span_id": span_id,
                    "issue": "MISSING_PASS2_MAPPING",
                })
                continue

            summary["pass1_" + str(
                span.get("annotation_status", "MISSING")
            ).lower()] += 1

            summary["pass2_" + str(
                mapping.get("annotation_status", "MISSING")
            ).lower()] += 1

            mention = span.get("skill_mention", "")
            start = span.get("character_start")
            end = span.get("character_end")

            # Validate character offsets.
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or end > len(text)
            ):
                issues.append({
                    "job_id": job_id,
                    "span_id": span_id,
                    "issue": "INVALID_CHARACTER_OFFSETS",
                })
                summary["invalid_offsets"] += 1
                continue

            if text[start:end] != mention:
                issues.append({
                    "job_id": job_id,
                    "span_id": span_id,
                    "issue": "TEXT_SPAN_MISMATCH",
                    "expected": mention,
                    "actual": text[start:end],
                })
                summary["text_span_mismatches"] += 1
                continue

            pass1_status = span.get("annotation_status")
            pass2_status = mapping.get("annotation_status")

            # Only approved pairs enter canonical evaluation.
            if pass1_status == "APPROVED" and pass2_status == "APPROVED":
                canonical = mapping.get("canonical_skill")

                if not canonical:
                    issues.append({
                        "job_id": job_id,
                        "span_id": span_id,
                        "issue": "APPROVED_MAPPING_WITHOUT_CANONICAL_SKILL",
                    })
                    continue

                ground_truth_by_job.setdefault(job_id, []).append({
                    "span_id": span_id,
                    "skill_mention": mention,
                    "character_start": start,
                    "character_end": end,
                    "canonical_skill": canonical,
                    "skill_id": mapping.get("skill_id"),
                    "skill_category": mapping.get("skill_category"),
                    "source_section": span.get("source_section"),
                })

                summary["eligible_canonical_spans"] += 1

            # Preserve approved mentions with unresolved mappings separately.
            elif (
                pass1_status == "APPROVED"
                and pass2_status == "UNRESOLVED_CANONICAL"
            ):
                unresolved_by_job.setdefault(job_id, []).append({
                    "span_id": span_id,
                    "skill_mention": mention,
                    "character_start": start,
                    "character_end": end,
                    "canonical_skill": None,
                    "source_section": span.get("source_section"),
                })

                summary["eligible_unresolved_mentions"] += 1

            else:
                summary["excluded_spans"] += 1

        # Detect mappings that have no corresponding Pass 1 span.
        for span_id in pass2_by_id:
            if span_id not in pass1_by_id:
                issues.append({
                    "job_id": job_id,
                    "span_id": span_id,
                    "issue": "PASS2_MAPPING_WITHOUT_PASS1_SPAN",
                })

    return {
        "dataset_name": data.get("dataset_name"),
        "evaluation_type": data.get("evaluation_type"),
        "ground_truth_by_job": ground_truth_by_job,
        "unresolved_by_job": unresolved_by_job,
        "summary": dict(summary),
        "issues": issues,
        "provenance_note": data.get("provenance_note"),
        "evaluation_limitation_statement": data.get(
            "evaluation_limitation_statement"
        ),
    }
