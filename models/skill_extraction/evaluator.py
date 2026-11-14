"""Skill Extraction Evaluation & 9-Category Error Classification Suite.

Computes exact span-level, boundary-only, and relaxed-overlap metrics
(Micro and Macro Precision, Recall, F1), inter-extractor agreement, and
systematic error classification across all 9 required error categories.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.preprocess_data import TextPreprocessor  # noqa: E402
from .dictionary_extractor import AMBIGUOUS_SHORT_TOKENS, COUNTRY_SPECIFIC_BROAD_SURFACE_MAP  # noqa: E402

ERROR_CATEGORIES: tuple[str, ...] = (
    "false_positive",
    "false_negative",
    "incorrect_boundary",
    "duplicate_match",
    "overlapping_match",
    "incorrect_normalization",
    "ambiguous_skill",
    "tokenization_error",
    "contextual_interpretation_error",
)

COUNTRY_SURFACE_LOWER: frozenset[str] = frozenset(
    item["phrase"].lower() for item in COUNTRY_SPECIFIC_BROAD_SURFACE_MAP
)

AMBIGUOUS_CANONICAL_SET: frozenset[str] = frozenset({
    "R",
    "R (programming language)",
    "C",
    "Go",
    "SIEM",
    "SQL",
    "GIS",
    "SEO",
    "Local Accounting Standards",
})


class SkillExtractionEvaluator:
    """Evaluates predicted skill spans and diagnoses structural error modes."""

    @staticmethod
    def evaluate_exact_spans(
        predictions_by_job: dict[str, list[dict[str, Any]]],
        ground_truth_by_job: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Calculates strict span-level Precision, Recall, and F1."""
        doc_records = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        fp_examples = []
        fn_examples = []

        for job_id, gt_spans in ground_truth_by_job.items():
            preds = predictions_by_job.get(job_id, [])

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

            for k in sorted(fp_keys):
                fp_examples.append({
                    "job_id": job_id,
                    "span": (k[0], k[1]),
                    "canonical_skill": k[2],
                    "predicted_mention": pred_tuples[k].get("skill_mention", ""),
                    "source_section": pred_tuples[k].get("source_section", "unknown"),
                })

            for k in sorted(fn_keys):
                fn_examples.append({
                    "job_id": job_id,
                    "span": (k[0], k[1]),
                    "canonical_skill": k[2],
                    "expected_mention": gt_tuples[k].get("skill_mention", ""),
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
            "per_doc_df": df_docs,
        }

    @staticmethod
    def evaluate_boundary_only(
        predictions_by_job: dict[str, list[dict[str, Any]]],
        ground_truth_by_job: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Evaluates `(character_start, character_end)` alignment regardless of label."""
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for job_id, gt_spans in ground_truth_by_job.items():
            preds = predictions_by_job.get(job_id, [])
            gt_bounds = {(s["character_start"], s["character_end"]) for s in gt_spans}
            pred_bounds = {(s["character_start"], s["character_end"]) for s in preds}

            total_tp += len(gt_bounds & pred_bounds)
            total_fp += len(pred_bounds - gt_bounds)
            total_fn += len(gt_bounds - pred_bounds)

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "evaluation_mode": "boundary_only",
            "micro_precision": prec,
            "micro_recall": rec,
            "micro_f1": f1,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
        }

    @staticmethod
    def evaluate_relaxed_spans(
        predictions_by_job: dict[str, list[dict[str, Any]]],
        ground_truth_by_job: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Calculates relaxed overlap Precision, Recall, and F1."""
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
                        if max(p["character_start"], g["character_start"]) < min(p["character_end"], g["character_end"]):
                            matched_gt.add(g_idx)
                            matched_pred.add(p_idx)
                            break

            tp = len(matched_pred)
            total_tp += tp
            total_fp += len(preds) - tp
            total_fn += len(gt_spans) - len(matched_gt)

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

    @staticmethod
    def compute_extractor_agreement(
        preds_a_by_job: dict[str, list[dict[str, Any]]],
        preds_b_by_job: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Quantifies span-level agreement and divergences between two extractors."""
        all_job_ids = sorted(set(preds_a_by_job.keys()) | set(preds_b_by_job.keys()))
        shared = 0
        only_a = 0
        only_b = 0
        divergences: list[dict[str, Any]] = []

        for jid in all_job_ids:
            set_a = {(s["character_start"], s["character_end"], s["canonical_skill"]): s for s in preds_a_by_job.get(jid, [])}
            set_b = {(s["character_start"], s["character_end"], s["canonical_skill"]): s for s in preds_b_by_job.get(jid, [])}
            keys_a = set(set_a.keys())
            keys_b = set(set_b.keys())

            shared += len(keys_a & keys_b)
            diff_a = keys_a - keys_b
            diff_b = keys_b - keys_a
            only_a += len(diff_a)
            only_b += len(diff_b)

            for k in sorted(diff_a):
                divergences.append({"job_id": jid, "side": "only_in_A", "span": (k[0], k[1]), "canonical_skill": k[2], "mention": set_a[k].get("skill_mention", "")})
            for k in sorted(diff_b):
                divergences.append({"job_id": jid, "side": "only_in_B", "span": (k[0], k[1]), "canonical_skill": k[2], "mention": set_b[k].get("skill_mention", "")})

        union = shared + only_a + only_b
        jaccard = shared / union if union > 0 else 1.0
        return {
            "shared_spans": shared,
            "only_in_a": only_a,
            "only_in_b": only_b,
            "span_jaccard_agreement": jaccard,
            "divergences": divergences[:50],
        }

    @staticmethod
    def classify_extraction_errors(
        ground_truth_by_job: dict[str, list[dict[str, Any]]],
        predictions_by_job: dict[str, list[dict[str, Any]]],
        texts_by_id: dict[str, str] | None = None,
        peer_predictions_by_job: dict[str, list[dict[str, Any]]] | None = None,
        preprocessor: TextPreprocessor | None = None,
        max_examples_per_category: int = 10,
    ) -> dict[str, Any]:
        """Classifies extraction anomalies across all 9 required error categories."""
        prep = preprocessor or TextPreprocessor()
        texts_by_id = texts_by_id or {}
        peer_predictions_by_job = peer_predictions_by_job or {}

        counts: dict[str, int] = {cat: 0 for cat in ERROR_CATEGORIES}
        sub_counts: dict[str, Counter[str]] = {cat: Counter() for cat in ERROR_CATEGORIES}
        examples: dict[str, list[dict[str, Any]]] = {cat: [] for cat in ERROR_CATEGORIES}

        def record(cat: str, sub: str, item: dict[str, Any]) -> None:
            counts[cat] += 1
            sub_counts[cat][sub] += 1
            if len(examples[cat]) < max_examples_per_category:
                examples[cat].append({"sub_category": sub, **item})

        for job_id, gt_spans in ground_truth_by_job.items():
            preds = predictions_by_job.get(job_id, [])
            text = texts_by_id.get(job_id, "")
            sections = prep.segment_job_sections(text) if text else []

            # Category 4: duplicate_match (identical coordinates inside predictions)
            seen_coords: dict[tuple[int, int], dict[str, Any]] = {}
            for p in preds:
                coord = (p["character_start"], p["character_end"])
                if coord in seen_coords:
                    record("duplicate_match", "identical_span_coordinates", {
                        "job_id": job_id, "span": coord, "mention": p.get("skill_mention", ""),
                        "canonical_skill": p.get("canonical_skill", ""),
                    })
                else:
                    seen_coords[coord] = p

            # Category 5: overlapping_match (partial or nested overlaps in predictions)
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    p1, p2 = preds[i], preds[j]
                    if max(p1["character_start"], p2["character_start"]) < min(p1["character_end"], p2["character_end"]):
                        record("overlapping_match", "unmerged_overlapping_spans", {
                            "job_id": job_id,
                            "span_a": (p1["character_start"], p1["character_end"], p1.get("canonical_skill", "")),
                            "span_b": (p2["character_start"], p2["character_end"], p2.get("canonical_skill", "")),
                        })

            gt_by_key = {(g["character_start"], g["character_end"], g["canonical_skill"]): g for g in gt_spans}
            pred_by_key = {(p["character_start"], p["character_end"], p["canonical_skill"]): p for p in preds}
            gt_by_bound = {(g["character_start"], g["character_end"]): g for g in gt_spans}
            pred_by_bound = {(p["character_start"], p["character_end"]): p for p in preds}

            # Evaluate False Positives + Incorrect Boundary + Incorrect Normalization + Ambiguous + Context
            for key, p in pred_by_key.items():
                if key in gt_by_key:
                    continue
                ps, pe, p_canon = key
                mention = p.get("skill_mention", text[ps:pe] if text else "")
                sec = p.get("source_section") or (prep.get_section_for_span(text, ps, pe, sections) if text else "unknown")
                snippet = text[max(0, ps - 25):min(len(text), pe + 25)] if text else ""

                # Category 6: incorrect_normalization (same boundary, wrong canonical label)
                if (ps, pe) in gt_by_bound and gt_by_bound[(ps, pe)]["canonical_skill"] != p_canon:
                    record("incorrect_normalization", "canonical_label_mismatch", {
                        "job_id": job_id, "span": (ps, pe), "mention": mention,
                        "predicted_canonical": p_canon,
                        "expected_canonical": gt_by_bound[(ps, pe)]["canonical_skill"],
                    })

                # Category 3: incorrect_boundary (overlaps a GT span with different boundaries)
                for g in gt_spans:
                    gs, ge = g["character_start"], g["character_end"]
                    if max(ps, gs) < min(pe, ge) and (ps, pe) != (gs, ge):
                        record("incorrect_boundary", "partial_span_boundary_mismatch", {
                            "job_id": job_id, "pred_span": (ps, pe), "gt_span": (gs, ge),
                            "pred_mention": mention, "gt_mention": g.get("skill_mention", ""),
                            "canonical_skill": p_canon,
                        })

                # Category 1: false_positive + Category 7 (ambiguous) + Category 9 (contextual)
                if p_canon.isupper() and not mention.isupper() and len(p_canon) >= 2:
                    fp_sub = "case_insensitive_acronym_collision"
                    record("ambiguous_skill", "case_collision_proper_noun", {
                        "job_id": job_id, "span": (ps, pe), "mention": mention,
                        "canonical_skill": p_canon, "snippet": snippet,
                    })
                    record("contextual_interpretation_error", "proper_noun_acronym_confusion", {
                        "job_id": job_id, "span": (ps, pe), "mention": mention,
                        "canonical_skill": p_canon, "snippet": snippet,
                    })
                elif sec == "intro_or_title":
                    fp_sub = "title_or_department_bleed"
                    record("contextual_interpretation_error", "title_or_department_non_qualification_context", {
                        "job_id": job_id, "span": (ps, pe), "mention": mention,
                        "canonical_skill": p_canon, "section": sec, "snippet": snippet,
                    })
                elif sec == "responsibilities":
                    fp_sub = "responsibility_duty_bleed"
                    record("contextual_interpretation_error", "duty_clause_non_qualification_context", {
                        "job_id": job_id, "span": (ps, pe), "mention": mention,
                        "canonical_skill": p_canon, "section": sec, "snippet": snippet,
                    })
                else:
                    neg_window = text[max(0, ps - 25):ps].lower() if text else ""
                    if any(neg in neg_window for neg in ("not ", "n't ", "without ", "never ", "no ")):
                        fp_sub = "negated_context_match"
                        record("contextual_interpretation_error", "syntactic_negation_ignored", {
                            "job_id": job_id, "span": (ps, pe), "mention": mention,
                            "canonical_skill": p_canon, "snippet": snippet,
                        })
                    else:
                        fp_sub = "unannotated_lexical_match"

                if mention.lower() in AMBIGUOUS_SHORT_TOKENS or p_canon in AMBIGUOUS_CANONICAL_SET:
                    if fp_sub != "case_insensitive_acronym_collision":
                        record("ambiguous_skill", "polysemous_short_or_domain_term", {
                            "job_id": job_id, "span": (ps, pe), "mention": mention,
                            "canonical_skill": p_canon, "section": sec,
                        })

                record("false_positive", fp_sub, {
                    "job_id": job_id, "span": (ps, pe), "mention": mention,
                    "canonical_skill": p_canon, "section": sec, "snippet": snippet,
                })

            # Evaluate False Negatives + Category 8 (tokenization_error)
            peer_keys = {(s["character_start"], s["character_end"], s["canonical_skill"]) for s in peer_predictions_by_job.get(job_id, [])}
            for key, g in gt_by_key.items():
                if key in pred_by_key:
                    continue
                gs, ge, g_canon = key
                mention = g.get("skill_mention", text[gs:ge] if text else "")
                m_lower = mention.lower()
                snippet = text[max(0, gs - 25):min(len(text), ge + 15)] if text else ""

                is_country_surface = (
                    m_lower in COUNTRY_SURFACE_LOWER
                    or g_canon in {"Data Privacy Compliance", "Local Accounting Standards"}
                )
                is_tok_miss = key in peer_keys or (text and ge < len(text) and text[ge] in {".", "/", "+", "#"})

                if is_tok_miss:
                    record("tokenization_error", "sentence_final_or_symbol_tokenization_mismatch", {
                        "job_id": job_id, "span": (gs, ge), "mention": mention,
                        "canonical_skill": g_canon, "trailing_char": text[ge:ge + 1] if text else "",
                        "snippet": snippet,
                    })

                if is_country_surface:
                    fn_sub = "country_specific_regulatory_or_accounting_surface"
                    if p := pred_by_bound.get((gs, ge)):
                        record("incorrect_normalization", "unmapped_or_divergent_country_surface", {
                            "job_id": job_id, "span": (gs, ge), "mention": mention,
                            "predicted": p.get("canonical_skill"), "expected": g_canon,
                        })
                elif is_tok_miss:
                    fn_sub = "punctuation_boundary_or_tokenization_miss"
                else:
                    fn_sub = "unlisted_synonym_or_surface_variant"

                if mention.lower() in AMBIGUOUS_SHORT_TOKENS or g_canon in AMBIGUOUS_CANONICAL_SET:
                    record("ambiguous_skill", "missed_ambiguous_or_acronym_variant", {
                        "job_id": job_id, "span": (gs, ge), "mention": mention, "canonical_skill": g_canon,
                    })

                record("false_negative", fn_sub, {
                    "job_id": job_id, "span": (gs, ge), "mention": mention,
                    "canonical_skill": g_canon, "snippet": snippet,
                })

            # Check peer disagreements not already classified as FN
            for key in sorted(peer_keys ^ set(pred_by_key.keys())):
                if key in gt_by_key and key in pred_by_key:
                    # Predicted by active, missed by peer
                    s_idx, e_idx, c_name = key
                    record("tokenization_error", "peer_extractor_tokenization_divergence", {
                        "job_id": job_id, "span": (s_idx, e_idx), "canonical_skill": c_name,
                        "mention": text[s_idx:e_idx] if text else "",
                    })

        return {
            "category_counts": counts,
            "sub_category_counts": {k: dict(v) for k, v in sub_counts.items()},
            "examples": examples,
        }

    @staticmethod
    def summarize_modality_distribution(
        predictions_by_job: dict[str, list[dict[str, Any]]],
    ) -> dict[str, int]:
        """Counts extracted skill spans grouped by Week 3 modality label."""
        counts: Counter[str] = Counter()
        for spans in predictions_by_job.values():
            for sp in spans:
                counts[str(sp.get("modality", "unspecified"))] += 1
        return dict(sorted(counts.items()))

    @staticmethod
    def evaluate_negation_cases(
        cases: list[dict[str, Any]],
        extractor: Any,
    ) -> dict[str, Any]:
        """Evaluates modality & negation classification against labeled clause cases."""
        correct = 0
        total = 0
        details: list[dict[str, Any]] = []
        for case in cases:
            text = case["text"]
            expected = case["expected_skills"]  # dict canonical -> (expected_modality, expected_is_negated)
            res = extractor.analyze_sentences_and_negation(text)
            by_canon = {sp["canonical_skill"]: sp for sp in res["all_spans"]}
            for canon, (exp_mod, exp_neg) in expected.items():
                total += 1
                sp = by_canon.get(canon)
                matched = (
                    sp is not None
                    and sp.get("modality") == exp_mod
                    and bool(sp.get("is_negated")) is bool(exp_neg)
                )
                if matched:
                    correct += 1
                details.append({
                    "text": text,
                    "skill": canon,
                    "expected_modality": exp_mod,
                    "expected_is_negated": exp_neg,
                    "predicted_modality": None if sp is None else sp.get("modality"),
                    "predicted_is_negated": None if sp is None else sp.get("is_negated"),
                    "passed": matched,
                })
        return {
            "total_assertions": total,
            "passed_assertions": correct,
            "accuracy": (correct / total) if total > 0 else 1.0,
            "details": details,
        }
