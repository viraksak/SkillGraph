"""SkillGraph Member 01 (Milestone 3 Step 1): Real-Data NLP Evaluation & Curriculum-Gap Audit.

Executes Member 01's reproducible evaluation strictly on verified real/human-reviewed
datasets without touching Member 02's backend/frontend/PostgreSQL modules:
1. Human-reference evaluation & FP breakdown on ``human_annotation_sample_50.json`` (50 records).
2. Real-world negation filtering audit on ``data/raw/real_job_postings.csv`` (100 real jobs).
3. Real curriculum-to-market skill coverage & gap analysis comparing the 13 real
   Stanford CS courses against the 100 real job postings.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.skill_extraction import (  # noqa: E402
    DictionarySkillExtractor,
    SkillExtractionEvaluator,
    SpacySkillExtractor,
)
from scripts.ingest_curriculum import CurriculumIngestionPipeline  # noqa: E402
from scripts.ingest_real_jobs import DEFAULT_REAL_JOBS_PATH, RealJobPostingPipeline  # noqa: E402

TAXONOMY_CSV = ROOT_DIR / "data" / "processed" / "global_skill_taxonomy.csv"
HUMAN_JSON = ROOT_DIR / "data" / "annotations" / "human_annotation_sample_50.json"
MAJORS_CSV = ROOT_DIR / "data" / "raw" / "real_university_majors.csv"
COURSES_CSV = ROOT_DIR / "data" / "raw" / "real_university_courses.csv"
OUTPUT_JSON = ROOT_DIR / "outputs" / "reports" / "member1_real_data_evaluation.json"


def run_member1_evaluation(save_path: Path | None = OUTPUT_JSON) -> dict[str, Any]:
    """Runs Member 01's human-reference error analysis and real-data curriculum gap audit."""
    dict_ext = DictionarySkillExtractor(TAXONOMY_CSV)
    spacy_ext = SpacySkillExtractor(TAXONOMY_CSV, use_fast_tokenizer=True)

    # 1. Human-reference evaluation & false-positive audit (50 records)
    human_payload = json.loads(HUMAN_JSON.read_text(encoding="utf-8-sig"))
    texts_by_job: dict[str, str] = {}
    gt_by_job: dict[str, list[dict[str, Any]]] = {}
    unresolved_by_job: dict[str, set[tuple[int, int]]] = {}
    decision_counts: Counter[str] = Counter()

    for rec in human_payload["records"]:
        jid = rec["job_id"]
        texts_by_job[jid] = rec["text"]
        gt_spans: list[dict[str, Any]] = []
        unres_spans: set[tuple[int, int]] = set()
        for m in rec["pass_2"]["canonical_mappings"]:
            status = m.get("annotation_status", "UNCERTAIN")
            decision_counts[status] += 1
            if m.get("canonical_skill"):
                gt_spans.append({
                    "character_start": m["character_start"],
                    "character_end": m["character_end"],
                    "canonical_skill": m["canonical_skill"],
                    "skill_mention": m["skill_mention"],
                })
            else:
                unres_spans.add((m["character_start"], m["character_end"]))
        gt_by_job[jid] = gt_spans
        unresolved_by_job[jid] = unres_spans

    spacy_preds_50 = {jid: spacy_ext.extract(txt, include_negated=False) for jid, txt in texts_by_job.items()}
    dict_preds_50 = {jid: dict_ext.extract(txt, include_negated=False) for jid, txt in texts_by_job.items()}

    spacy_strict_50 = SkillExtractionEvaluator.evaluate_exact_spans(spacy_preds_50, gt_by_job)
    dict_strict_50 = SkillExtractionEvaluator.evaluate_exact_spans(dict_preds_50, gt_by_job)

    # Classify FPs on the 50-record human sample
    fp_reasons: Counter[str] = Counter()
    fp_examples: list[dict[str, Any]] = []
    for jid, preds in spacy_preds_50.items():
        gt_keys = {(g["character_start"], g["character_end"], g["canonical_skill"]) for g in gt_by_job[jid]}
        unres_bounds = unresolved_by_job[jid]
        for p in preds:
            k = (p["character_start"], p["character_end"], p["canonical_skill"])
            if k not in gt_keys:
                if (p["character_start"], p["character_end"]) in unres_bounds:
                    reason = "human_marked_unresolved_canonical"
                else:
                    reason = "unannotated_in_human_reference"
                fp_reasons[reason] += 1
                if len(fp_examples) < 15:
                    fp_examples.append({
                        "job_id": jid,
                        "mention": p["skill_mention"],
                        "predicted_canonical": p["canonical_skill"],
                        "reason": reason,
                    })

    # 2. Real Job Postings (100 real postings) extraction & negation ablation
    job_pipe = RealJobPostingPipeline(TAXONOMY_CSV, extractor=spacy_ext, strict_provenance=True)
    real_records = job_pipe.load_records(DEFAULT_REAL_JOBS_PATH)
    real_jobs_report = job_pipe.process_and_audit(real_records)

    neg_examples: list[dict[str, Any]] = []
    ablation_real_100: dict[str, Any] = {}
    for ext_label, ext_obj in [("SpacySkillExtractor", spacy_ext), ("DictionarySkillExtractor", dict_ext)]:
        total_raw = 0
        total_pos = 0
        total_neg = 0
        for rec in real_records:
            txt = str(rec.get("description") or "")
            raw_spans = ext_obj.extract(txt, include_negated=True)
            pos_spans = ext_obj.extract(txt, include_negated=False)
            total_raw += len(raw_spans)
            total_pos += len(pos_spans)
            for sp in raw_spans:
                if sp.get("is_negated"):
                    total_neg += 1
                    if ext_label == "SpacySkillExtractor" and len(neg_examples) < 10:
                        neg_examples.append({
                            "job_id": rec.get("job_id"),
                            "canonical_skill": sp.get("canonical_skill"),
                            "mention": sp.get("skill_mention"),
                            "modality": sp.get("modality"),
                        })
        ablation_real_100[ext_label] = {
            "raw_mentions_without_negation_filter": total_raw,
            "positive_mentions_with_negation_filter": total_pos,
            "negated_or_prohibited_filtered": total_neg,
        }

    real_job_skill_counts: Counter[str] = Counter()
    real_job_skill_ids: dict[str, str] = {}
    for row in real_jobs_report["extracted_rows"]:
        canon = row["canonical_skill"]
        real_job_skill_counts[canon] += 1
        real_job_skill_ids[canon] = row["skill_id"]

    # 3. Real Curriculum (13 Stanford CS courses) skill extraction & coverage vs real jobs
    curr_pipe = CurriculumIngestionPipeline(TAXONOMY_CSV, extractor=spacy_ext, strict_provenance=True)
    majors = curr_pipe.load_records(MAJORS_CSV)
    courses = curr_pipe.load_records(COURSES_CSV)
    clean_majors, clean_courses = curr_pipe.validate_records(majors, courses)
    course_skill_rows, _ = curr_pipe.extract_course_skills(clean_courses)
    verified_prereqs = curr_pipe.derive_verified_skill_prerequisites(clean_courses, course_skill_rows)

    tax_id_to_name = {
        r["skill_id"]: r["canonical_skill"]
        for r in real_jobs_report["extracted_rows"]
    }
    import pandas as pd
    tax_df = pd.read_csv(TAXONOMY_CSV)
    id_to_canon = {str(r["skill_id"]): str(r["canonical_skill"]) for r in tax_df.to_dict(orient="records")}
    id_to_type = {str(r["skill_id"]): str(r["skill_type"]) for r in tax_df.to_dict(orient="records")}

    curriculum_skill_ids = {r["skill_id"] for r in course_skill_rows}
    curriculum_skill_names = sorted(id_to_canon[sid] for sid in curriculum_skill_ids if sid in id_to_canon)

    real_job_canons = set(real_job_skill_counts.keys())
    covered_real_skills = sorted(real_job_canons & set(curriculum_skill_names))
    missing_real_skills = [
        {
            "canonical_skill": skill,
            "skill_id": real_job_skill_ids[skill],
            "skill_type": id_to_type.get(real_job_skill_ids[skill], "Unknown"),
            "real_job_mentions": count,
        }
        for skill, count in real_job_skill_counts.most_common()
        if skill not in set(curriculum_skill_names)
    ]

    coverage_ratio = round(len(covered_real_skills) / len(real_job_canons), 4) if real_job_canons else 0.0

    result = {
        "member": "Member 01 (hustsak / viraksak)",
        "milestone_step": "Milestone 3 Step 1 — Real-Data NLP & Curriculum-Gap Evaluation",
        "human_reference_50_records": {
            "records_evaluated": len(texts_by_job),
            "human_annotation_status_counts": dict(sorted(decision_counts.items())),
            "spacy_strict": {
                "tp": spacy_strict_50["total_tp"],
                "fp": spacy_strict_50["total_fp"],
                "fn": spacy_strict_50["total_fn"],
                "precision": round(spacy_strict_50["micro_precision"], 6),
                "recall": round(spacy_strict_50["micro_recall"], 6),
                "f1": round(spacy_strict_50["micro_f1"], 6),
            },
            "dictionary_strict": {
                "tp": dict_strict_50["total_tp"],
                "fp": dict_strict_50["total_fp"],
                "fn": dict_strict_50["total_fn"],
                "precision": round(dict_strict_50["micro_precision"], 6),
                "recall": round(dict_strict_50["micro_recall"], 6),
                "f1": round(dict_strict_50["micro_f1"], 6),
            },
            "false_positive_breakdown": dict(sorted(fp_reasons.items())),
            "fp_examples": fp_examples,
        },
        "real_jobs_100_audit": {
            "valid_unique_postings": real_jobs_report["valid_unique_rows"],
            "positive_skill_mentions": real_jobs_report["extracted_skill_mentions"],
            "unique_demanded_skills": len(real_job_canons),
            "modality_counts_including_negated": real_jobs_report["eda_summary"]["modality_counts_including_negated"],
            "negation_ablation": ablation_real_100,
            "filtered_negation_examples": neg_examples,
        },
        "real_curriculum_vs_real_jobs_gap": {
            "majors_count": len(clean_majors),
            "courses_count": len(clean_courses),
            "curriculum_unique_skills_count": len(curriculum_skill_names),
            "curriculum_skills": curriculum_skill_names,
            "verified_skill_prerequisite_nodes": len(verified_prereqs),
            "covered_real_job_skills_count": len(covered_real_skills),
            "covered_real_job_skills": covered_real_skills,
            "curriculum_skill_coverage_ratio": coverage_ratio,
            "top_15_missing_real_job_skills": missing_real_skills[:15],
        },
    }

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return result


if __name__ == "__main__":
    report = run_member1_evaluation()
    print(json.dumps(report, indent=2))
