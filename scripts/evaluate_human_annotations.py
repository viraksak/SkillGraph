"""SkillGraph Week 4 / Member 01 Module: Human-Annotation Importer, Cohen's Kappa & Evaluator.

Validates completed human decisions from ``data/candidates_review_sheet_all50.csv``,
computes inter-annotator agreement (Cohen's kappa) on shared overlap records when
a second annotator sheet is supplied, imports verified human decisions into
``data/annotations/human_annotation_sample_50.json`` (only when explicitly requested
with ``--write-json`` and 100% of rows are reviewed), and evaluates
``DictionarySkillExtractor`` and ``SpacySkillExtractor`` against the human reference.

Research Integrity Guardrail:
    This module NEVER auto-confirms candidate rows, NEVER fabricates annotator IDs,
    and REFUSES to run human-reference evaluation if any row lacks a valid
    ``human_decision`` or ``annotator_id``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.skill_extraction import (  # noqa: E402
    DictionarySkillExtractor,
    SkillExtractionEvaluator,
    SpacySkillExtractor,
)

DEFAULT_REVIEW_CSV = ROOT_DIR / "data" / "candidates_review_sheet_all50.csv"
DEFAULT_SAMPLE_JSON = ROOT_DIR / "data" / "annotations" / "human_annotation_sample_50.json"
DEFAULT_TAXONOMY_CSV = ROOT_DIR / "data" / "processed" / "global_skill_taxonomy.csv"

VALID_HUMAN_DECISIONS = {
    "CONFIRM",
    "CONFIRMED",
    "REJECT",
    "EDIT",
    "UNCERTAIN",
    "UNRESOLVED_CANONICAL",
}
FORBIDDEN_ANNOTATOR_IDS = {
    "HUMAN_ANNOTATOR_TIER_A",
    "AUTO",
    "MACHINE",
    "AI",
    "SYNTHETIC",
    "BOT",
}


class HumanAnnotationValidationError(ValueError):
    """Raised when review rows are incomplete, unverified, or violate provenance rules."""


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def validate_review_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validates that every candidate row carries an authentic human_decision and annotator_id."""
    if not rows:
        raise HumanAnnotationValidationError("Review sheet contains 0 rows.")

    unreviewed_indices: list[int] = []
    annotators: Counter[str] = Counter()
    decisions: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        decision = _clean_str(row.get("human_decision")).upper()
        annotator = _clean_str(row.get("annotator_id"))
        notes = _clean_str(row.get("human_edit_notes"))

        if not decision or not annotator:
            unreviewed_indices.append(idx)
            continue
        if annotator.upper() in FORBIDDEN_ANNOTATOR_IDS:
            raise HumanAnnotationValidationError(
                f"Row {idx} uses forbidden/non-human annotator_id {annotator!r}."
            )
        if decision not in VALID_HUMAN_DECISIONS:
            raise HumanAnnotationValidationError(
                f"Row {idx} has invalid human_decision {decision!r}; expected one of {sorted(VALID_HUMAN_DECISIONS)}."
            )
        if decision == "EDIT" and not notes:
            raise HumanAnnotationValidationError(
                f"Row {idx} is marked EDIT without explanatory human_edit_notes."
            )
        decisions[decision] += 1
        annotators[annotator] += 1

    if unreviewed_indices:
        raise HumanAnnotationValidationError(
            f"Human review incomplete: {len(unreviewed_indices)}/{len(rows)} rows lack human_decision or annotator_id."
        )

    return {
        "total_rows": len(rows),
        "decision_counts": dict(sorted(decisions.items())),
        "annotator_counts": dict(sorted(annotators.items())),
    }


def compute_cohens_kappa(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
) -> dict[str, Any]:
    """Computes Cohen's kappa across shared spans matched by (record_id, job_id, start, end)."""
    def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, str, int, int], str]:
        idx_map: dict[tuple[int, str, int, int], str] = {}
        for r in rows:
            dec = _clean_str(r.get("human_decision")).upper()
            if dec not in VALID_HUMAN_DECISIONS:
                continue
            key = (
                int(r["record_id"]),
                _clean_str(r["job_id"]),
                int(r["character_start"]),
                int(r["character_end"]),
            )
            idx_map[key] = dec
        return idx_map

    map_a = _index_rows(rows_a)
    map_b = _index_rows(rows_b)
    shared_keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not shared_keys:
        raise HumanAnnotationValidationError("No shared reviewed spans found between annotator A and annotator B.")

    n = len(shared_keys)
    labels = sorted(VALID_HUMAN_DECISIONS)
    agree = sum(1 for k in shared_keys if map_a[k] == map_b[k])
    p_o = agree / n

    counts_a = Counter(map_a[k] for k in shared_keys)
    counts_b = Counter(map_b[k] for k in shared_keys)
    p_e = sum((counts_a[lbl] / n) * (counts_b[lbl] / n) for lbl in labels)

    if abs(1.0 - p_e) < 1e-12:
        kappa = 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    disagreements = [
        {
            "record_id": k[0],
            "job_id": k[1],
            "span": [k[2], k[3]],
            "annotator_a_decision": map_a[k],
            "annotator_b_decision": map_b[k],
        }
        for k in shared_keys
        if map_a[k] != map_b[k]
    ]

    return {
        "shared_spans": n,
        "observed_agreement": round(p_o, 6),
        "expected_chance_agreement": round(p_e, 6),
        "cohens_kappa": round(kappa, 6),
        "disagreement_count": len(disagreements),
        "disagreements_sample": disagreements[:15],
    }


def build_human_reviewed_payload(
    review_rows: list[dict[str, Any]],
    base_json_payload: dict[str, Any],
    taxonomy_df: pd.DataFrame,
    dual_annotator: bool = False,
) -> dict[str, Any]:
    """Builds a validated human-reference JSON structure without mutating original job text."""
    summary = validate_review_rows(review_rows)
    tax_by_id = {str(r["skill_id"]).strip(): r for r in taxonomy_df.to_dict(orient="records")}
    tax_by_name = {str(r["canonical_skill"]).strip().lower(): r for r in taxonomy_df.to_dict(orient="records")}

    # Group review rows by (record_id, job_id)
    rows_by_rec: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in review_rows:
        key = (int(row["record_id"]), _clean_str(row["job_id"]))
        rows_by_rec.setdefault(key, []).append(row)

    updated = json.loads(json.dumps(base_json_payload))
    total_confirmed_spans = 0
    unresolved_count = 0

    for rec in updated["records"]:
        rid = int(rec["record_id"])
        jid = _clean_str(rec["job_id"])
        text = rec["text"]
        rec_rows = rows_by_rec.get((rid, jid), [])
        rec_annotators = sorted({_clean_str(r["annotator_id"]) for r in rec_rows})

        new_p1: list[dict[str, Any]] = []
        new_p2: list[dict[str, Any]] = []

        for idx, row in enumerate(rec_rows, start=1):
            dec = _clean_str(row["human_decision"]).upper()
            if dec == "REJECT":
                continue

            start = int(row["character_start"])
            end = int(row["character_end"])
            mention = str(row["skill_mention"])
            if text[start:end] != mention:
                raise HumanAnnotationValidationError(
                    f"Offset mismatch in record {rid} ({jid}): text[{start}:{end}]={text[start:end]!r} != {mention!r}."
                )

            sid = _clean_str(row.get("candidate_skill_id"))
            canon = _clean_str(row.get("candidate_canonical_skill"))
            notes = _clean_str(row.get("human_edit_notes"))

            # Allow EDIT notes formatted like "SG-SK-0048" or canonical skill lookup if specified
            if dec == "EDIT" and notes:
                if notes in tax_by_id:
                    sid = notes
                elif notes.lower() in tax_by_name:
                    sid = str(tax_by_name[notes.lower()]["skill_id"])

            if dec == "UNRESOLVED_CANONICAL":
                sid = ""
                canon = ""
                tax_rec = None
                skill_cat = None
                skill_type = None
                map_status = "UNRESOLVED_CANONICAL"
                p1_status = "APPROVED"
                map_conf = "human_confirmed_oov"
                unresolved_count += 1
            else:
                tax_rec = tax_by_id.get(sid) if sid else None
                if tax_rec is not None:
                    canon = str(tax_rec["canonical_skill"])
                    skill_cat = str(tax_rec["skill_category"])
                    skill_type = str(tax_rec["skill_type"])
                    map_status = "UNCERTAIN" if dec == "UNCERTAIN" else "APPROVED"
                else:
                    canon = canon or None
                    skill_cat = None
                    skill_type = None
                    map_status = "UNRESOLVED_CANONICAL"
                    unresolved_count += 1
                p1_status = "UNCERTAIN" if dec == "UNCERTAIN" else "APPROVED"
                map_conf = "human_flagged_uncertain" if dec == "UNCERTAIN" else "human_confirmed"

            span_id = f"R{rid:02d}-S{idx:02d}"
            new_p1.append({
                "span_id": span_id,
                "skill_mention": mention,
                "character_start": start,
                "character_end": end,
                "source_section": _clean_str(row.get("source_section")) or "unknown",
                "annotation_status": p1_status,
                "reviewer_notes": notes or None,
            })
            new_p2.append({
                "span_id": span_id,
                "skill_mention": mention,
                "character_start": start,
                "character_end": end,
                "canonical_skill": canon,
                "skill_id": sid or None,
                "skill_category": skill_cat,
                "skill_type": skill_type,
                "annotation_status": map_status,
                "mapping_confidence": map_conf,
                "is_oov": tax_rec is None,
                "reviewer_notes": notes or None,
            })
            total_confirmed_spans += 1

        rec["pass_1"]["spans"] = new_p1
        rec["pass_1"]["status"] = "COMPLETED"
        rec["pass_2"]["canonical_mappings"] = new_p2
        rec["pass_2"]["status"] = "COMPLETED"
        rec["record_status"] = "COMPLETED"
        rec["annotator_id"] = rec_annotators[0] if len(rec_annotators) == 1 else ", ".join(rec_annotators)

    updated["total_spans"] = total_confirmed_spans
    updated["unresolved_canonical_spans"] = unresolved_count
    updated["evaluation_type"] = (
        "dual_annotator_human_reference" if dual_annotator else "single_annotator_human_reference"
    )
    updated["provenance_note"] = (
        f"Independently reviewed by {', '.join(summary['annotator_counts'].keys())} "
        f"(decisions: {summary['decision_counts']})."
    )
    return updated


def evaluate_against_human_reference(
    human_payload: dict[str, Any],
    taxonomy_csv_path: str | Path = DEFAULT_TAXONOMY_CSV,
) -> dict[str, Any]:
    """Evaluates Dictionary and spaCy extractors against verified human-reference spans."""
    texts_by_job: dict[str, str] = {}
    gt_by_job: dict[str, list[dict[str, Any]]] = {}

    for rec in human_payload["records"]:
        if rec.get("record_status") != "COMPLETED" or not rec.get("annotator_id"):
            raise HumanAnnotationValidationError(
                f"Record {rec.get('record_id')} is not COMPLETED with a verified annotator_id."
            )
        jid = rec["job_id"]
        texts_by_job[jid] = rec["text"]
        gt_by_job[jid] = [
            {
                "character_start": m["character_start"],
                "character_end": m["character_end"],
                "canonical_skill": m["canonical_skill"] or m["skill_mention"],
                "skill_mention": m["skill_mention"],
            }
            for m in rec["pass_2"]["canonical_mappings"]
            if m.get("canonical_skill")
        ]

    dict_ext = DictionarySkillExtractor(taxonomy_csv_path)
    spacy_ext = SpacySkillExtractor(taxonomy_csv_path, use_fast_tokenizer=True)

    dict_preds = {jid: dict_ext.extract(txt, include_negated=False) for jid, txt in texts_by_job.items()}
    spacy_preds = {jid: spacy_ext.extract(txt, include_negated=False) for jid, txt in texts_by_job.items()}

    dict_strict = SkillExtractionEvaluator.evaluate_exact_spans(dict_preds, gt_by_job)
    spacy_strict = SkillExtractionEvaluator.evaluate_exact_spans(spacy_preds, gt_by_job)

    return {
        "evaluation_type": human_payload.get("evaluation_type", "single_annotator_human_reference"),
        "records_evaluated": len(texts_by_job),
        "dictionary_strict": {
            "tp": dict_strict["total_tp"],
            "fp": dict_strict["total_fp"],
            "fn": dict_strict["total_fn"],
            "precision": round(dict_strict["micro_precision"], 6),
            "recall": round(dict_strict["micro_recall"], 6),
            "f1": round(dict_strict["micro_f1"], 6),
        },
        "spacy_strict": {
            "tp": spacy_strict["total_tp"],
            "fp": spacy_strict["total_fp"],
            "fn": spacy_strict["total_fn"],
            "precision": round(spacy_strict["micro_precision"], 6),
            "recall": round(spacy_strict["micro_recall"], 6),
            "f1": round(spacy_strict["micro_f1"], 6),
        },
        "research_disclaimer": (
            "Synthetic-text single/dual-annotator human-reference evaluation; "
            "not a real-world external labor-market benchmark."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, import, and evaluate completed human reviews from candidates_review_sheet_all50.csv."
    )
    parser.add_argument("--review-csv", type=str, default=str(DEFAULT_REVIEW_CSV))
    parser.add_argument("--second-csv", type=str, default=None, help="Optional second-annotator review CSV for Cohen's kappa.")
    parser.add_argument("--sample-json", type=str, default=str(DEFAULT_SAMPLE_JSON))
    parser.add_argument("--taxonomy-csv", type=str, default=str(DEFAULT_TAXONOMY_CSV))
    parser.add_argument("--write-json", action="store_true", help="Write verified annotations back to human_annotation_sample_50.json.")
    args = parser.parse_args()

    rows = pd.read_csv(args.review_csv).to_dict(orient="records")
    try:
        summary = validate_review_rows(rows)
    except HumanAnnotationValidationError as exc:
        print("=== SkillGraph Human-Annotation Evaluator (BLOCKED_AWAITING_HUMAN_REVIEW) ===")
        print(f"Reason: {exc}")
        print("Action: Fill human_decision (CONFIRM/REJECT/EDIT) and annotator_id in data/candidates_review_sheet_all50.csv, then re-run.")
        return

    base_json = json.loads(Path(args.sample_json).read_text(encoding="utf-8-sig"))
    tax_df = pd.read_csv(args.taxonomy_csv)

    kappa_report = None
    if args.second_csv:
        rows_b = pd.read_csv(args.second_csv).to_dict(orient="records")
        validate_review_rows(rows_b)
        kappa_report = compute_cohens_kappa(rows, rows_b)

    human_payload = build_human_reviewed_payload(
        review_rows=rows,
        base_json_payload=base_json,
        taxonomy_df=tax_df,
        dual_annotator=bool(args.second_csv),
    )
    if args.write_json:
        Path(args.sample_json).write_bytes(
            (json.dumps(human_payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        )

    eval_report = evaluate_against_human_reference(human_payload, args.taxonomy_csv)
    print(json.dumps({
        "review_summary": summary,
        "inter_annotator_agreement": kappa_report,
        "human_reference_evaluation": eval_report,
        "json_updated": bool(args.write_json),
    }, indent=2))


if __name__ == "__main__":
    main()
