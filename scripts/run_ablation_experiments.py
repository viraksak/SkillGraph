"""SkillGraph Week 4 Evaluation Script: Controlled Ablation Experiments.

Runs four reproducible ablations on synthetic / fixture inputs and saves
``outputs/reports/ablation_summary.json`` cleanly:
1. Negation filtering ablation (22-doc technical benchmark: include_negated=True vs False).
2. Domain-vs-language skill filtering ablation (exclude_skill_types=None vs ['Language']).
3. Curriculum prerequisite DAG ablation (gamma=0.0 vs gamma=0.3 with verified prerequisites).
4. Priority weight preset sensitivity across BASELINE_WEIGHT_PRESETS.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.skill_extraction import DictionarySkillExtractor, SpacySkillExtractor  # noqa: E402
from scripts import generate_synthetic_benchmark as tech_benchmark  # noqa: E402
from scripts.build_graph import (  # noqa: E402
    BASELINE_WEIGHT_PRESETS,
    PriorityWeights,
    SkillGapAnalysisEngine,
    SkillGraphBuilder,
)
from scripts.ingest_curriculum import CurriculumIngestionPipeline  # noqa: E402

OUTPUT_PATH = ROOT_DIR / "outputs" / "reports" / "ablation_summary.json"


def run_ablations(save_path: Path | None = OUTPUT_PATH) -> dict[str, Any]:
    """Executes all 4 SkillGraph ablations and returns structured metrics."""
    tax_json = ROOT_DIR / "outputs" / "exports" / "skill_taxonomy.json"
    tax_csv = ROOT_DIR / "data" / "processed" / "global_skill_taxonomy.csv"

    # 1. Negation ablation on 22-doc technical benchmark
    negation_ablation: dict[str, Any] = {}
    for label, ext in [
        ("DictionarySkillExtractor", DictionarySkillExtractor(tax_json)),
        ("SpacySkillExtractor", SpacySkillExtractor(tax_json, use_fast_tokenizer=True)),
    ]:
        row: dict[str, Any] = {}
        for mode_label, incl in [("without_negation_filter", True), ("with_negation_filter", False)]:
            tp = fp = fn = 0
            for job in tech_benchmark.SYNTHETIC_JOBS:
                gt = set(job["ground_truth_skills"])
                preds = {sp["canonical_skill"] for sp in ext.extract(job["description"], include_negated=incl)}
                tp += len(preds & gt)
                fp += len(preds - gt)
                fn += len(gt - preds)
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            row[mode_label] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(p, 6),
                "recall": round(r, 6),
                "f1": round(f1, 6),
            }
        negation_ablation[label] = row

    # Build KG & Engine
    builder = SkillGraphBuilder(include_broad_country_mappings=False, min_co_occurrence=5)
    builder.build()
    engine = SkillGapAnalysisEngine(builder)

    # 2. Domain-vs-Language filtering ablation
    demo_profile = {
        "student_id": "SYNTHETIC_ABLATION_DA",
        "is_synthetic": True,
        "target_occupation": "Data Analyst",
        "skills": [
            {"skill": "Python", "proficiency_level": "intermediate"},
            {"skill": "SQL", "proficiency_level": "beginner"},
        ],
    }
    base_da = engine.analyze(demo_profile, top_k=5, run_sensitivity=True)
    domain_da = engine.analyze(demo_profile, top_k=5, exclude_skill_types=["Language"], run_sensitivity=False)

    domain_ablation = {
        "unfiltered_top5": [
            {"rank": r["rank"], "skill": r["canonical_skill"], "skill_type": r["skill_type"], "P": r["priority_score"]}
            for r in base_da["priority_recommendations"]
        ],
        "exclude_language_top5": [
            {"rank": r["rank"], "skill": r["canonical_skill"], "skill_type": r["skill_type"], "P": r["priority_score"]}
            for r in domain_da["priority_recommendations"]
        ],
        "coverage_metrics_unchanged": base_da["coverage_metrics"] == domain_da["coverage_metrics"],
    }

    # 3. Verified Prerequisite DAG (D(s)) ablation using in-memory curriculum fixture
    curr_pipe = CurriculumIngestionPipeline(tax_csv, strict_provenance=False)
    curr_res = curr_pipe.ingest(
        majors=[{
            "major_id": "ABL_MAJ_DS",
            "major_name": "Data Science (Ablation Fixture)",
            "institution_name": "Fixture University",
            "degree_level": "Bachelor",
        }],
        courses=[
            {
                "course_id": "ABL_C1",
                "course_code": "DS101",
                "course_title": "Intro Python & Statistics",
                "major_id": "ABL_MAJ_DS",
                "syllabus_text": "Covers Python and Statistics.",
                "prerequisite_course_ids": [],
            },
            {
                "course_id": "ABL_C2",
                "course_code": "DS201",
                "course_title": "Applied Machine Learning",
                "major_id": "ABL_MAJ_DS",
                "syllabus_text": "Covers Machine Learning.",
                "prerequisite_course_ids": ["ABL_C1"],
            },
        ],
    )
    prereq_dag = curr_res["verified_skill_prerequisites"]
    ds_profile = {
        "student_id": "SYNTHETIC_ABLATION_DS",
        "is_synthetic": True,
        "target_occupation": "Data Scientist",
        "skills": ["Python"],
    }
    ds_no_dag = engine.analyze(
        ds_profile,
        weights=BASELINE_WEIGHT_PRESETS["baseline_balanced_heuristic"],
        run_sensitivity=False,
    )
    ds_with_dag = engine.analyze(
        ds_profile,
        weights=PriorityWeights(alpha=0.5, beta=0.2, gamma=0.3, preset_name="ablation_with_dag"),
        verified_prerequisites=prereq_dag,
        run_sensitivity=False,
    )
    ml_no = next(r for r in ds_no_dag["priority_recommendations"] if r["skill_id"] == "SG-SK-0027")
    ml_yes = next(r for r in ds_with_dag["priority_recommendations"] if r["skill_id"] == "SG-SK-0027")

    prerequisite_ablation = {
        "derived_skill_prerequisites": prereq_dag,
        "without_dag": {
            "D_status": ml_no["scoring"]["components"]["D_status"],
            "D_value": ml_no["scoring"]["components"]["D_dependency_importance"],
            "P_score": ml_no["priority_score"],
            "rank": ml_no["rank"],
        },
        "with_dag": {
            "D_status": ml_yes["scoring"]["components"]["D_status"],
            "D_value": ml_yes["scoring"]["components"]["D_dependency_importance"],
            "P_score": ml_yes["priority_score"],
            "rank": ml_yes["rank"],
            "unmet_prerequisites": ml_yes["prerequisite_info"].get("unmet_prerequisites", []),
        },
    }

    # 4. Weight sensitivity summary
    ws = base_da["weight_sensitivity"]
    weight_sensitivity = {
        "top_k_jaccard_vs_baseline": ws["top_k_jaccard_vs_baseline"],
        "top_k_by_preset": ws["top_k_by_preset"],
    }

    payload = {
        "provenance_disclaimer": "synthetic_and_fixture_ablations_only",
        "ablation_1_negation_22_docs": negation_ablation,
        "ablation_2_domain_vs_language_filtering": domain_ablation,
        "ablation_3_prerequisite_dag_d_score": prerequisite_ablation,
        "ablation_4_weight_sensitivity": weight_sensitivity,
    }
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return payload


if __name__ == "__main__":
    out = run_ablations()
    print(json.dumps(out, indent=2))
