"""SkillGraph Member 01 (Milestone 4 Step 3): Statistical Analysis & Confidence Intervals.

Computes reproducible statistical comparisons for Member 01's research paper:
1. 1,000-resample document-level Bootstrap 95% Confidence Intervals for strict
   Precision, Recall, and F1 on the 50-record human-reviewed sample.
2. Paired McNemar / negation-filtering disagreement test comparing
   ``SpacySkillExtractor`` vs ``DictionarySkillExtractor`` across the 100 real postings.
3. Real-job skill-type breakdown (Technology, Hard skill, Soft skill, Language,
   Knowledge area, Tool) and Spearman/Pearson rank correlation between extractors.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.skill_extraction import DictionarySkillExtractor, SpacySkillExtractor  # noqa: E402
from scripts.ingest_real_jobs import DEFAULT_REAL_JOBS_PATH, RealJobPostingPipeline  # noqa: E402

TAXONOMY_CSV = ROOT_DIR / "data" / "processed" / "global_skill_taxonomy.csv"
HUMAN_JSON = ROOT_DIR / "data" / "annotations" / "human_annotation_sample_50.json"
OUTPUT_JSON = ROOT_DIR / "outputs" / "reports" / "member1_statistical_analysis.json"


def _bootstrap_f1_ci(
    per_doc_counts: list[tuple[int, int, int]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Computes percentile 95% bootstrap CIs over (tp, fp, fn) per document."""
    rng = random.Random(seed)
    n = len(per_doc_counts)
    precs: list[float] = []
    recs: list[float] = []
    f1s: list[float] = []

    for _ in range(n_resamples):
        tp = fp = fn = 0
        for _ in range(n):
            dtp, dfp, dfn = per_doc_counts[rng.randrange(n)]
            tp += dtp
            fp += dfp
            fn += dfn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precs.append(p)
        recs.append(r)
        f1s.append(f1)

    precs.sort()
    recs.sort()
    f1s.sort()
    lo = int(0.025 * n_resamples)
    hi = int(0.975 * n_resamples) - 1
    return {
        "n_docs": n,
        "n_resamples": n_resamples,
        "precision_ci_95": [round(precs[lo], 4), round(precs[hi], 4)],
        "recall_ci_95": [round(recs[lo], 4), round(recs[hi], 4)],
        "f1_ci_95": [round(f1s[lo], 4), round(f1s[hi], 4)],
    }


def _mcnemar_chi2(b: int, c: int) -> dict[str, Any]:
    """Continuity-corrected McNemar chi-square statistic for paired binary disagreements."""
    if (b + c) == 0:
        return {"b": b, "c": c, "chi2_corrected": 0.0, "p_value_approx": 1.0}
    chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c) if abs(b - c) >= 1 else 0.0
    # 1-df chi-square survival function: erfc(sqrt(x / 2))
    p_val = math.erfc(math.sqrt(chi2 / 2.0))
    return {
        "spacy_only_filtered": b,
        "dict_only_filtered": c,
        "chi2_corrected": round(chi2, 4),
        "p_value": round(p_val, 8),
        "statistically_significant_alpha_0_05": p_val < 0.05,
    }


def run_member1_statistics(save_path: Path | None = OUTPUT_JSON) -> dict[str, Any]:
    """Runs bootstrap CIs, McNemar negation test, and skill-type statistics."""
    dict_ext = DictionarySkillExtractor(TAXONOMY_CSV)
    spacy_ext = SpacySkillExtractor(TAXONOMY_CSV, use_fast_tokenizer=True)

    # 1. Bootstrap 95% CI on the 50-record human-reviewed sample
    human_payload = json.loads(HUMAN_JSON.read_text(encoding="utf-8-sig"))
    per_doc_spacy: list[tuple[int, int, int]] = []
    for rec in human_payload["records"]:
        txt = rec["text"]
        gt_keys = {
            (m["character_start"], m["character_end"], m["canonical_skill"])
            for m in rec["pass_2"]["canonical_mappings"]
            if m.get("canonical_skill")
        }
        pred_keys = {
            (p["character_start"], p["character_end"], p["canonical_skill"])
            for p in spacy_ext.extract(txt, include_negated=False)
        }
        tp = len(pred_keys & gt_keys)
        fp = len(pred_keys - gt_keys)
        fn = len(gt_keys - pred_keys)
        per_doc_spacy.append((tp, fp, fn))

    bootstrap_ci = _bootstrap_f1_ci(per_doc_spacy, n_resamples=1000, seed=42)

    # 2. Paired McNemar test on negation filtering + skill-type stats across 100 real jobs
    job_pipe = RealJobPostingPipeline(TAXONOMY_CSV, extractor=spacy_ext, strict_provenance=True)
    real_records = job_pipe.load_records(DEFAULT_REAL_JOBS_PATH)
    tax_df = pd.read_csv(TAXONOMY_CSV)
    type_by_id = {str(r["skill_id"]): str(r["skill_type"]) for r in tax_df.to_dict(orient="records")}

    skill_type_counts: Counter[str] = Counter()
    spacy_counts: Counter[str] = Counter()
    dict_counts: Counter[str] = Counter()
    spacy_neg_set: set[tuple[str, int, int]] = set()
    dict_neg_set: set[tuple[str, int, int]] = set()
    all_raw_set: set[tuple[str, int, int]] = set()

    for rec in real_records:
        jid = str(rec["job_id"])
        txt = str(rec.get("description") or "")
        s_all = spacy_ext.extract(txt, include_negated=True)
        d_all = dict_ext.extract(txt, include_negated=True)

        for sp in s_all:
            key = (jid, sp["character_start"], sp["character_end"])
            all_raw_set.add(key)
            if sp.get("is_negated"):
                spacy_neg_set.add(key)
            else:
                spacy_counts[sp["canonical_skill"]] += 1
                skill_type_counts[type_by_id.get(sp["skill_id"], "Unknown")] += 1

        for sp in d_all:
            key = (jid, sp["character_start"], sp["character_end"])
            all_raw_set.add(key)
            if sp.get("is_negated"):
                dict_neg_set.add(key)
            else:
                dict_counts[sp["canonical_skill"]] += 1

    b_spacy_only = len(spacy_neg_set - dict_neg_set)
    c_dict_only = len(dict_neg_set - spacy_neg_set)
    mcnemar_neg = _mcnemar_chi2(b_spacy_only, c_dict_only)

    # Pearson & Spearman correlation of skill counts across shared taxonomy skills
    all_skills = sorted(set(spacy_counts.keys()) | set(dict_counts.keys()))
    s_vec = np.array([spacy_counts[k] for k in all_skills], dtype=float)
    d_vec = np.array([dict_counts[k] for k in all_skills], dtype=float)
    pearson_r = float(np.corrcoef(s_vec, d_vec)[0, 1]) if len(all_skills) > 1 else 1.0
    spearman_rho = float(pd.Series(s_vec).corr(pd.Series(d_vec), method="spearman")) if len(all_skills) > 1 else 1.0

    result = {
        "member": "Member 01 (hustsak / viraksak)",
        "milestone_step": "Milestone 4 Step 3 — Statistical Analysis & Extractor Comparison",
        "human_sample_bootstrap_ci": bootstrap_ci,
        "real_100_negation_mcnemar": mcnemar_neg,
        "real_100_extractor_frequency_agreement": {
            "unique_skills_compared": len(all_skills),
            "pearson_r": round(pearson_r, 6),
            "spearman_rho": round(spearman_rho, 6),
        },
        "real_100_positive_mentions_by_skill_type": dict(skill_type_counts.most_common()),
    }
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return result


if __name__ == "__main__":
    print(json.dumps(run_member1_statistics(), indent=2))
