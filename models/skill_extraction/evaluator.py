"""Skill Extraction Evaluation Suite.

Computes exact span-level, relaxed overlap, and document-level metrics
(Precision, Recall, F1) with comprehensive error categorization.
"""

from __future__ import annotations

from typing import Any
import pandas as pd


class SkillExtractionEvaluator:
    """Evaluates predicted skill spans against annotated ground truth."""

    @staticmethod
    def evaluate_exact_spans(
        predictions_by_job: dict[str, list[dict[str, Any]]],
        ground_truth_by_job: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Calculates strict span-level Precision, Recall, and F1.

        A match is a True Positive if and only if:
        1. character_start matches exactly
        2. character_end matches exactly
        3. canonical_skill matches exactly
        """
        doc_records = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        fp_examples = []
        fn_examples = []

        for job_id, gt_spans in ground_truth_by_job.items():
            preds = predictions_by_job.get(job_id, [])

            # Canonical representation: (start, end, canonical_skill)
            gt_tuples = {
                (s["character_start"], s["character_end"], s["canonical_skill"]): s
                for s in gt_spans
            }
            pred_tuples = {
                (s["character_start"], s["character_end"], s["canonical_skill"]): s
                for s in preds
            }

            gt_keys = set(gt_tuples.keys())
            pred_keys = set(pred_tuples.keys())

            tp_keys = pred_keys & gt_keys
            fp_keys = pred_keys - gt_keys
            fn_keys = gt_keys - pred_keys

            tp = len(tp_keys)
            fp = len(fp_keys)
            fn = len(fn_keys)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            doc_records.append({
                "job_id": job_id,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            })

            for k in fp_keys:
                fp_examples.append({
                    "job_id": job_id,
                    "span": (k[0], k[1]),
                    "canonical_skill": k[2],
                    "predicted_mention": pred_tuples[k].get("skill_mention", "")
                })

            for k in fn_keys:
                fn_examples.append({
                    "job_id": job_id,
                    "span": (k[0], k[1]),
                    "canonical_skill": k[2],
                    "expected_mention": gt_tuples[k].get("skill_mention", "")
                })

        df_docs = pd.DataFrame(doc_records)

        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

        macro_prec = float(df_docs["precision"].mean()) if not df_docs.empty else 0.0
        macro_rec = float(df_docs["recall"].mean()) if not df_docs.empty else 0.0
        macro_f1 = float(df_docs["f1"].mean()) if not df_docs.empty else 0.0

        return {
            "evaluation_mode": "strict_exact_span",
            "micro_precision": micro_prec,
            "micro_recall": micro_rec,
            "micro_f1": micro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "fp_examples": fp_examples[:25],
            "fn_examples": fn_examples[:25],
            "per_doc_df": df_docs
        }

    @staticmethod
    def evaluate_relaxed_spans(
        predictions_by_job: dict[str, list[dict[str, Any]]],
        ground_truth_by_job: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Calculates relaxed overlap Precision, Recall, and F1.

        A match is a True Positive if:
        1. canonical_skill matches exactly
        2. character spans overlap: max(start1, start2) < min(end1, end2)
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for job_id, gt_spans in ground_truth_by_job.items():
            preds = predictions_by_job.get(job_id, [])

            matched_gt = set()
            matched_pred = set()

            for p_idx, p in enumerate(preds):
                for g_idx, g in enumerate(gt_spans):
                    if g_idx in matched_gt:
                        continue
                    if p["canonical_skill"] == g["canonical_skill"]:
                        # Overlap check
                        if max(p["character_start"], g["character_start"]) < min(p["character_end"], g["character_end"]):
                            matched_gt.add(g_idx)
                            matched_pred.add(p_idx)
                            break

            tp = len(matched_pred)
            fp = len(preds) - tp
            fn = len(gt_spans) - len(matched_gt)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

        return {
            "evaluation_mode": "relaxed_overlap",
            "micro_precision": micro_prec,
            "micro_recall": micro_rec,
            "micro_f1": micro_f1,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
        }

