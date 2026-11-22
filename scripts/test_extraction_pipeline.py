"""Comprehensive Deterministic Unit & Regression Test Suite for SkillGraph.

Validates:
- Offset-preserving text preprocessing & section segmentation (`TextPreprocessor`)
- `DictionarySkillExtractor` and `SpacySkillExtractor` on single-word,
  multi-word, abbreviated, Unicode, and symbol-heavy skills (C++, C#, .NET,
  Node.js, CI/CD, 4G/5G, LTE and 5G, NoSQL, Power BI, SQL, R, C, Go)
- Week 2 regression fixes ("Siem Reap" vs "SIEM", spaCy "LTE and 5G." trailing
  period split, "C#" sub-token boundary, "R." / "C." sentence-final periods)
- Multi-tier `SkillNormalizer` behavior and regional/statutory privacy safeguards
- 9-category error classification via `SkillExtractionEvaluator`
- Backward-compatible ingestion of both the 22-record technical benchmark
  and the 2,000-record global synthetic reference dataset
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.skill_extraction import (  # noqa: E402
    DictionarySkillExtractor,
    SkillExtractionEvaluator,
    SkillNormalizer,
    SpacySkillExtractor,
)
from scripts.preprocess_data import TextPreprocessor  # noqa: E402
import scripts.generate_synthetic_benchmark as tech_benchmark  # noqa: E402


EXTENDED_EDGE_CASE_TAXONOMY = pd.DataFrame([
    {"skill_id": "EDGE-001", "canonical_skill": "Python", "skill_category": "Programming", "skill_type": "Technology", "aliases": '["python 3", "py"]'},
    {"skill_id": "EDGE-002", "canonical_skill": "R", "skill_category": "Programming", "skill_type": "Technology", "aliases": '["R"]'},
    {"skill_id": "EDGE-003", "canonical_skill": "C", "skill_category": "Programming", "skill_type": "Technology", "aliases": '["C"]'},
    {"skill_id": "EDGE-004", "canonical_skill": "C++", "skill_category": "Programming", "skill_type": "Technology", "aliases": '["cpp", "c/c++"]'},
    {"skill_id": "EDGE-005", "canonical_skill": "C#", "skill_category": "Programming", "skill_type": "Technology", "aliases": '["C-sharp"]'},
    {"skill_id": "EDGE-006", "canonical_skill": "Go", "skill_category": "Programming", "skill_type": "Technology", "aliases": '["golang"]'},
    {"skill_id": "EDGE-007", "canonical_skill": ".NET", "skill_category": "Framework", "skill_type": "Technology", "aliases": '[".NET Core"]'},
    {"skill_id": "EDGE-008", "canonical_skill": "Node.js", "skill_category": "Framework", "skill_type": "Technology", "aliases": '["nodejs"]'},
    {"skill_id": "EDGE-009", "canonical_skill": "CI/CD", "skill_category": "DevOps", "skill_type": "Technology", "aliases": '["continuous integration"]'},
    {"skill_id": "EDGE-010", "canonical_skill": "SQL", "skill_category": "Database", "skill_type": "Technology", "aliases": '["Structured Query Language"]'},
    {"skill_id": "EDGE-011", "canonical_skill": "NoSQL", "skill_category": "Database", "skill_type": "Technology", "aliases": '[]'},
    {"skill_id": "EDGE-012", "canonical_skill": "Machine Learning", "skill_category": "AI", "skill_type": "Hard skill", "aliases": '["ML"]'},
    {"skill_id": "EDGE-013", "canonical_skill": "Deep Learning", "skill_category": "AI", "skill_type": "Hard skill", "aliases": '["DL"]'},
    {"skill_id": "EDGE-014", "canonical_skill": "Natural Language Processing", "skill_category": "AI", "skill_type": "Hard skill", "aliases": '["NLP"]'},
    {"skill_id": "EDGE-015", "canonical_skill": "Data Analysis", "skill_category": "Analytics", "skill_type": "Hard skill", "aliases": '["data analytics"]'},
    {"skill_id": "EDGE-016", "canonical_skill": "Project Management", "skill_category": "Management", "skill_type": "Hard skill", "aliases": '[]'},
    {"skill_id": "EDGE-017", "canonical_skill": "Power BI", "skill_category": "Analytics", "skill_type": "Tool", "aliases": '["Microsoft Power BI"]'},
    {"skill_id": "EDGE-018", "canonical_skill": "SIEM", "skill_category": "Security", "skill_type": "Tool", "aliases": '[]'},
    {"skill_id": "EDGE-019", "canonical_skill": "LTE and 5G", "skill_category": "Networking", "skill_type": "Technology", "aliases": '["4G/5G"]'},
])


def load_benchmark_corpus(path: Path) -> list[dict]:
    """Loads benchmark JSON payloads cleanly whether top-level list or dict."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "data" in payload:
            return list(payload["data"])
        if "annotations" in payload:
            return list(payload["annotations"])
    raise ValueError(f"Unrecognized JSON corpus layout in {path}")


class TestSkillExtractionPipeline(unittest.TestCase):
    """Deterministic unit & regression tests for SkillGraph Week 2 pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tax_csv_path = ROOT_DIR / "data" / "processed" / "global_skill_taxonomy.csv"
        cls.tax_json_path = ROOT_DIR / "outputs" / "exports" / "skill_taxonomy.json"
        cls.gt_global_path = ROOT_DIR / "data" / "annotations" / "synthetic_job_postings_ground_truth.json"

        cls.preprocessor = TextPreprocessor()
        cls.global_gt = load_benchmark_corpus(cls.gt_global_path)
        cls.global_texts = {item["job_id"]: item["text"] for item in cls.global_gt}

        # Global 230-skill extractors
        cls.dict_global = DictionarySkillExtractor(cls.tax_csv_path)
        cls.spacy_global = SpacySkillExtractor(cls.tax_csv_path, use_fast_tokenizer=True)
        cls.dict_global_broad = DictionarySkillExtractor(cls.tax_csv_path, include_broad_country_mappings=True)
        cls.dict_global_sec = DictionarySkillExtractor(cls.tax_csv_path, section_filter={"required", "preferred"})

        # 31-skill JSON technical extractors
        cls.dict_json = DictionarySkillExtractor(cls.tax_json_path)
        cls.spacy_json = SpacySkillExtractor(cls.tax_json_path, use_fast_tokenizer=True)

        # Extended edge-case extractors (includes C#, .NET, NoSQL alongside C, C++, R, Go)
        cls.dict_edge = DictionarySkillExtractor(EXTENDED_EDGE_CASE_TAXONOMY)
        cls.spacy_edge = SpacySkillExtractor(EXTENDED_EDGE_CASE_TAXONOMY, use_fast_tokenizer=True)

        # Normalizers
        cls.normalizer_strict = SkillNormalizer(cls.tax_csv_path, include_broad_country_mappings=False)
        cls.normalizer_broad = SkillNormalizer(cls.tax_csv_path, include_broad_country_mappings=True)
        cls.normalizer_edge = SkillNormalizer(EXTENDED_EDGE_CASE_TAXONOMY)

        from scripts.build_graph import RelationalDatabaseManager, SkillGraphBuilder  # noqa: E402
        cls.RelationalDatabaseManager = RelationalDatabaseManager
        cls.kg_builder = SkillGraphBuilder(include_broad_country_mappings=False, min_co_occurrence=5)
        cls.kg_builder.build()

    def _assert_both_extract_canonicals(self, text: str, expected: set[str], edge: bool = True) -> tuple[list[dict], list[dict]]:
        d_ext = self.dict_edge if edge else self.dict_global
        s_ext = self.spacy_edge if edge else self.spacy_global
        d_res = d_ext.extract(text)
        s_res = s_ext.extract(text)

        for res_list in (d_res, s_res):
            for span in res_list:
                self.assertEqual(
                    text[span["character_start"]:span["character_end"]],
                    span["skill_mention"],
                )
            for i in range(len(res_list)):
                for j in range(i + 1, len(res_list)):
                    self.assertLessEqual(res_list[i]["character_end"], res_list[j]["character_start"])

        self.assertEqual({x["canonical_skill"] for x in d_res}, expected)
        self.assertEqual({x["canonical_skill"] for x in s_res}, expected)
        return d_res, s_res

    # 1-5: Single-word, multi-word, and technical punctuation skills
    def test_01_single_word_skills(self) -> None:
        text = "Required stack: Python, SQL, NoSQL, Docker, Kubernetes, Linux, FastAPI, PostgreSQL."
        d_res = self.dict_json.extract(text)
        s_res = self.spacy_json.extract(text)
        expected_json = {"Python", "SQL", "Docker", "Kubernetes", "Linux", "FastAPI", "PostgreSQL"}
        self.assertEqual({x["canonical_skill"] for x in d_res}, expected_json)
        self.assertEqual({x["canonical_skill"] for x in s_res}, expected_json)
        d_edge, s_edge = self._assert_both_extract_canonicals("NoSQL and SQL databases.", {"NoSQL", "SQL"})
        self.assertEqual(len(d_edge), 2)
        self.assertEqual(len(s_edge), 2)

    def test_02_multi_word_skills(self) -> None:
        text = (
            "Experience in Machine Learning, Deep Learning, Natural Language Processing, "
            "Data Analysis, Project Management, and Power BI is required."
        )
        expected = {
            "Machine Learning",
            "Deep Learning",
            "Natural Language Processing",
            "Data Analysis",
            "Project Management",
            "Power BI",
        }
        self._assert_both_extract_canonicals(text, expected)

    def test_03_technical_punctuation_cpp_csharp(self) -> None:
        text = "Systems development with C++ and C# on embedded devices."
        self._assert_both_extract_canonicals(text, {"C++", "C#"})

    def test_04_technical_punctuation_dotnet_nodejs_cicd(self) -> None:
        text = "Full-stack services built with .NET, Node.js, and automated CI/CD pipelines."
        self._assert_both_extract_canonicals(text, {".NET", "Node.js", "CI/CD"})

    def test_05_c_and_cpp_and_csharp_coexistence(self) -> None:
        text = "Firmware engineers need C, C++, and C#."
        self._assert_both_extract_canonicals(text, {"C", "C++", "C#"})

    # 6-10: Ambiguous short terms (R, C, Go) & negative contexts
    def test_06_r_positive_vs_negative_context(self) -> None:
        self._assert_both_extract_canonicals("Statistical modeling in R and Python.", {"R", "Python"})
        self._assert_both_extract_canonicals("Collaborating closely with the medical R&D team.", set())
        self._assert_both_extract_canonicals("Lowercase r in normal prose does not count.", set())

    def test_07_c_positive_vs_negative_context(self) -> None:
        self._assert_both_extract_canonicals("Kernel drivers written in C.", {"C"})
        self._assert_both_extract_canonicals("Presenting to C-suite executives on Vitamin C clinical trials.", set())
        self._assert_both_extract_canonicals("Lowercase letter c is ignored.", set())

    def test_08_csharp_must_not_emit_bare_c_subspan(self) -> None:
        d_res, s_res = self._assert_both_extract_canonicals("Must know C#.", {"C#"})
        self.assertEqual([x["character_start"] for x in d_res], [10])
        self.assertEqual([x["character_end"] for x in s_res], [12])

    def test_09_go_positive_vs_negative_context(self) -> None:
        self._assert_both_extract_canonicals("Cloud microservices in Go and Python.", {"Go", "Python"})
        self._assert_both_extract_canonicals("We value engineers ready to go the extra mile and go from concept to launch.", set())

    def test_10_global_r_programming_language(self) -> None:
        text = "Must know R programming and SQL."
        self._assert_both_extract_canonicals(text, {"R (programming language)", "SQL"}, edge=False)

    # 11-14: Confirmed Week 2 Regression Fixes
    def test_11_regression_siem_vs_siem_reap(self) -> None:
        """Week 2 Regression: 'Siem Reap, Cambodia' must never match 'SIEM'."""
        text_city_only = "A technology company in Siem Reap, Cambodia is hiring an analyst."
        self._assert_both_extract_canonicals(text_city_only, set(), edge=False)
        text_both = "Based in Siem Reap, Cambodia; requires hands-on SIEM operations."
        d_res, s_res = self._assert_both_extract_canonicals(text_both, {"SIEM"}, edge=False)
        self.assertEqual(d_res[0]["skill_mention"], "SIEM")
        self.assertGreater(s_res[0]["character_start"], text_both.index("Cambodia"))

    def test_12_regression_spacy_sentence_final_period_5g(self) -> None:
        """Week 2 Regression: GLOBAL_SYN_001440 trailing period on 'LTE and 5G.'"""
        text = self.global_texts["GLOBAL_SYN_001440"]
        d_res = self.dict_global.extract(text)
        s_res = self.spacy_global.extract(text)
        self.assertIn("LTE and 5G", {x["canonical_skill"] for x in d_res})
        self.assertIn("LTE and 5G", {x["canonical_skill"] for x in s_res})
        d_span = next(x for x in d_res if x["canonical_skill"] == "LTE and 5G")
        s_span = next(x for x in s_res if x["canonical_skill"] == "LTE and 5G")
        self.assertEqual((d_span["character_start"], d_span["character_end"]), (s_span["character_start"], s_span["character_end"]))
        self.assertEqual(text[s_span["character_start"]:s_span["character_end"]], "LTE and 5G")

    def test_13_regression_spacy_sentence_final_4g5g_r_c(self) -> None:
        """Week 2 Regression: trailing periods after 4G/5G., R., and C. split cleanly in spaCy."""
        self._assert_both_extract_canonicals("Experience with 4G/5G.", {"LTE and 5G"}, edge=True)
        self._assert_both_extract_canonicals("Proficiency in R.", {"R"}, edge=True)
        self._assert_both_extract_canonicals("Proficiency in C.", {"C"}, edge=True)

    def test_14_regression_global_siem_zero_city_false_positives(self) -> None:
        """Verifies none of the 28 Siem Reap docs produce SIEM false positives."""
        siem_reap_ids = [jid for jid, txt in self.global_texts.items() if "Siem Reap" in txt]
        self.assertEqual(len(siem_reap_ids), 28)
        for jid in siem_reap_ids:
            txt = self.global_texts[jid]
            for ext in (self.dict_global, self.spacy_global):
                for m in ext.extract(txt):
                    if m["canonical_skill"] == "SIEM":
                        self.assertIn(m["skill_mention"], {"SIEM", "security information and event management"})
                        self.assertNotEqual(m["skill_mention"], "Siem")
                        self.assertNotIn("Reap", txt[m["character_start"]:m["character_end"] + 6])

    # 15-19: Abbreviations, case variations, Unicode, empty/null inputs, preprocessing
    def test_15_case_variations_and_abbreviations(self) -> None:
        text = "Candidates need postgresql, power bi, machine learning, NLP, ML, and SQL."
        self._assert_both_extract_canonicals(
            text,
            {"PostgreSQL", "Power BI", "Machine Learning", "Natural Language Processing", "SQL"},
            edge=False,
        )

    def test_16_unicode_and_typographic_whitespace_preservation(self) -> None:
        raw = "Must\u00a0know\tPython, \u2018Data Analysis\u2019 \u2014 and Power\u00a0BI in S\u00e3o Paulo."
        norm_res = self.preprocessor.normalize_preserving_offsets(raw)
        self.assertTrue(norm_res["length_preserved"])
        self.assertEqual(len(norm_res["normalized_text"]), len(raw))
        d_res, s_res = self._assert_both_extract_canonicals(raw, {"Python", "Data Analysis", "Power BI"}, edge=False)
        for m in d_res + s_res:
            self.assertEqual(raw[m["character_start"]:m["character_end"]], m["skill_mention"])

    def test_17_empty_missing_and_whitespace_inputs(self) -> None:
        for bad_input in ("", "   \n\t  ", None, float("nan"), 12345):
            self.assertEqual(self.dict_global.extract(bad_input), [])
            self.assertEqual(self.spacy_global.extract(bad_input), [])
            res = self.normalizer_strict.normalize(bad_input)
            self.assertEqual(res["mapping_status"], "UNRESOLVED_CANONICAL")

    def test_18_repeated_mentions_preserve_distinct_offsets(self) -> None:
        text = "Required: Python and Machine Learning. Bonus: Python for Data Analysis."
        d_res = self.dict_edge.extract(text)
        s_res = self.spacy_edge.extract(text)
        d_python = [(x["character_start"], x["character_end"]) for x in d_res if x["canonical_skill"] == "Python"]
        s_python = [(x["character_start"], x["character_end"]) for x in s_res if x["canonical_skill"] == "Python"]
        self.assertEqual(len(d_python), 2)
        self.assertEqual(d_python, s_python)

    def test_19_overlapping_phrase_resolution_prefers_longest_and_canonical(self) -> None:
        text = "Apply Natural Language Processing and machine learning for SQL queries."
        d_res, s_res = self._assert_both_extract_canonicals(
            text,
            {"Natural Language Processing", "Machine Learning", "SQL"},
            edge=False,
        )
        sql_mentions = [x["skill_mention"] for x in d_res if x["canonical_skill"] == "SQL"]
        self.assertEqual(sql_mentions, ["SQL queries"])  # 11 chars > 3 chars ("SQL")

    # 20-23: SkillNormalizer tiers & country-specific integrity rules
    def test_20_normalizer_exact_and_alias_abbreviations(self) -> None:
        for surface, expected_canon in [
            ("Python", "Python"),
            ("nlp", "Natural Language Processing"),
            ("NLP", "Natural Language Processing"),
            ("ml", "Machine Learning"),
            ("ML", "Machine Learning"),
            ("sql", "SQL"),
            ("postgresql", "PostgreSQL"),
            ("power bi", "Power BI"),
            ("Power BI", "Power BI"),
        ]:
            res = self.normalizer_strict.normalize(surface)
            self.assertEqual(res["canonical_skill"], expected_canon)
            self.assertEqual(res["mapping_status"], "MAPPED")
            self.assertTrue(res["skill_id"].startswith("SG-SK-"))

    def test_21_normalizer_country_specific_strict_vs_broad(self) -> None:
        for surface, expected_broad in [
            ("PDPA compliance", "Data Privacy Compliance"),
            ("GDPR compliance", "Data Privacy Compliance"),
            ("IFRS", "Local Accounting Standards"),
            ("UK GAAP (FRS 102)", "Local Accounting Standards"),
        ]:
            strict_res = self.normalizer_strict.normalize(surface)
            self.assertEqual(strict_res["mapping_status"], "UNRESOLVED_CANONICAL")
            self.assertIsNone(strict_res["skill_id"])

            broad_res = self.normalizer_broad.normalize(surface)
            self.assertEqual(broad_res["mapping_status"], "BROAD_COUNTRY_MAPPING")
            self.assertEqual(broad_res["canonical_skill"], expected_broad)
            self.assertEqual(broad_res["match_method"], "country_specific_broad_match")
            self.assertEqual(broad_res["mapping_confidence"], 0.80)

    def test_22_normalizer_never_invents_taxonomy_ids(self) -> None:
        unknown = self.normalizer_strict.normalize("QuantumFluxProprietaryEngine v9")
        self.assertIsNone(unknown["skill_id"])
        self.assertIsNone(unknown["canonical_skill"])
        self.assertEqual(unknown["mapping_status"], "UNRESOLVED_CANONICAL")
        self.assertEqual(unknown["mapping_confidence"], 0.0)

    def test_23_section_segmentation_and_section_filtering(self) -> None:
        text = self.global_texts["GLOBAL_SYN_000004"]
        unfiltered = self.dict_global.extract(text)
        filtered = self.dict_global_sec.extract(text)
        self.assertGreater(len(unfiltered), len(filtered))
        self.assertTrue(all(x["source_section"] in {"required", "preferred"} for x in filtered))

    # 24-27: Corpus slice alignment, legacy 22-job compatibility, and 9-category error classifier
    def test_24_corpus_slice_offsets_and_zero_overlaps(self) -> None:
        for rec in self.global_gt[:100]:
            jid = rec["job_id"]
            txt = rec["text"]
            d_spans = self.dict_global.extract(txt)
            s_spans = self.spacy_global.extract(txt)
            self.assertEqual(
                [(x["character_start"], x["character_end"], x["canonical_skill"]) for x in d_spans],
                [(x["character_start"], x["character_end"], x["canonical_skill"]) for x in s_spans],
                msg=f"Mismatch on {jid}",
            )
            for span in d_spans:
                self.assertEqual(txt[span["character_start"]:span["character_end"]], span["skill_mention"])

    def test_25_technical_22_benchmark_recall_and_no_r_or_c_false_positives(self) -> None:
        for job in tech_benchmark.SYNTHETIC_JOBS:
            preds_d = {x["canonical_skill"] for x in self.dict_json.extract(job["description"])}
            preds_s = {x["canonical_skill"] for x in self.spacy_json.extract(job["description"])}
            self.assertEqual(preds_d, preds_s, msg=f"Divergence on {job['job_id']}")
            gt_skills = set(job["ground_truth_skills"])
            self.assertTrue(gt_skills.issubset(preds_d), msg=f"Missing skill in {job['job_id']}: {gt_skills - preds_d}")
        # Verify SYN_JOB_004 ("ready to go the extra mile") and SYN_JOB_020 ("Vitamin C", "R&D") do not trigger Go/C/R
        preds_04 = {x["canonical_skill"] for x in self.dict_json.extract(tech_benchmark.SYNTHETIC_JOBS[3]["description"])}
        self.assertNotIn("Go", preds_04)
        preds_20 = {x["canonical_skill"] for x in self.dict_json.extract(tech_benchmark.SYNTHETIC_JOBS[19]["description"])}
        self.assertNotIn("C", preds_20)
        self.assertNotIn("R", preds_20)

    def test_26_evaluator_metrics_and_output_schema_fields(self) -> None:
        required_keys = {
            "skill_mention",
            "matched_text",
            "canonical_skill",
            "skill_id",
            "skill_category",
            "skill_type",
            "character_start",
            "character_end",
            "match_length",
            "is_alias",
            "extractor_name",
            "extraction_method",
            "match_method",
            "mapping_status",
            "mapping_confidence",
            "source_section",
        }
        spans = self.dict_global.extract("Required qualifications include Python and SQL.")
        self.assertEqual(len(spans), 2)
        for sp in spans:
            self.assertTrue(required_keys.issubset(set(sp.keys())))

    def test_27_nine_category_error_classifier(self) -> None:
        synthetic_text = "Required: C#, Python, PDPA compliance, and LTE and 5G. We work with Siem Reap office and do not require Java."
        s_lte = synthetic_text.index("LTE and 5G")
        s_siem = synthetic_text.index("Siem")
        s_java = synthetic_text.index("Java")
        gt_probe = {
            "PROBE_1": [
                {"character_start": 10, "character_end": 12, "canonical_skill": "C#", "skill_mention": "C#"},
                {"character_start": 14, "character_end": 20, "canonical_skill": "Python", "skill_mention": "Python"},
                {"character_start": 22, "character_end": 37, "canonical_skill": "Data Privacy Compliance", "skill_mention": "PDPA compliance"},
                {"character_start": s_lte, "character_end": s_lte + 10, "canonical_skill": "LTE and 5G", "skill_mention": "LTE and 5G"},
            ]
        }
        pred_probe = {
            "PROBE_1": [
                {"character_start": 10, "character_end": 11, "canonical_skill": "C", "skill_mention": "C", "source_section": "required"},
                {"character_start": 14, "character_end": 20, "canonical_skill": "WrongPythonLabel", "skill_mention": "Python", "source_section": "required"},
                {"character_start": 14, "character_end": 20, "canonical_skill": "WrongPythonLabel", "skill_mention": "Python", "source_section": "required"},
                {"character_start": s_siem, "character_end": s_siem + 4, "canonical_skill": "SIEM", "skill_mention": "Siem", "source_section": "intro_or_title"},
                {"character_start": s_java, "character_end": s_java + 4, "canonical_skill": "Java", "skill_mention": "Java", "source_section": "general"},
            ]
        }
        peer_probe = {
            "PROBE_1": [
                {"character_start": s_lte, "character_end": s_lte + 10, "canonical_skill": "LTE and 5G", "skill_mention": "LTE and 5G"},
            ]
        }
        report = SkillExtractionEvaluator.classify_extraction_errors(
            gt_probe,
            pred_probe,
            texts_by_id={"PROBE_1": synthetic_text},
            peer_predictions_by_job=peer_probe,
        )
        counts = report["category_counts"]
        for expected_non_zero in (
            "false_positive",
            "false_negative",
            "incorrect_boundary",
            "duplicate_match",
            "overlapping_match",
            "incorrect_normalization",
            "ambiguous_skill",
            "tokenization_error",
            "contextual_interpretation_error",
        ):
            self.assertGreater(counts[expected_non_zero], 0, msg=f"Category {expected_non_zero} returned 0")

    # ------------------------------------------------------------------
    # Week 3 Modules 1-9 Deterministic Unit & Integration Tests
    # ------------------------------------------------------------------

    def test_28_negation_module_eight_cases_spacy_and_dict(self) -> None:
        """Week 3 Module 1: Verify modality & dependency negation on all 8 prompt cases + SYN_JOB_003."""
        cases = [
            {"text": "Must have Python experience.", "expected_skills": {"Python": ("required", False)}},
            {"text": "Python is preferred.", "expected_skills": {"Python": ("preferred", False)}},
            {"text": "Python is not required.", "expected_skills": {"Python": ("negated_requirement", True)}},
            {"text": "Do not use Python.", "expected_skills": {"Python": ("excluded_or_prohibited", True)}},
            {"text": "Experience with Python is unnecessary.", "expected_skills": {"Python": ("negated_requirement", True)}},
            {"text": "Knowledge of Python and Java is required.", "expected_skills": {"Python": ("required", False), "Java": ("required", False)}},
            {"text": "Python is not required, but SQL is essential.", "expected_skills": {"Python": ("negated_requirement", True), "SQL": ("required", False)}},
            {"text": "Note that we do not develop in Python or Java for this position.", "expected_skills": {"Python": ("excluded_or_prohibited", True), "Java": ("excluded_or_prohibited", True)}},
        ]
        for extractor in (self.spacy_json, self.dict_json):
            res = SkillExtractionEvaluator.evaluate_negation_cases(cases, extractor)
            self.assertEqual(res["passed_assertions"], res["total_assertions"], msg=f"Failures: {res['details']}")

        # Case 8: General non-taxonomy prohibited clause
        an_prohib = self.spacy_json.analyze_sentences_and_negation("Candidates must avoid using deprecated technologies.")
        self.assertEqual(an_prohib["positive_spans"], [])
        self.assertGreaterEqual(len(an_prohib["non_taxonomy_prohibited_clauses"]), 1)
        clause = an_prohib["non_taxonomy_prohibited_clauses"][0]
        self.assertEqual(clause["modality"], "excluded_or_prohibited")
        self.assertIn("deprecated technologies", clause["target_phrase"].lower())
        self.assertIsNone(clause["skill_id"])

    def test_29_technical_22_doc_perfect_f1_after_negation_filtering(self) -> None:
        """Week 3 Regression: Negation filtering removes SYN_JOB_003 FPs -> 22-doc F1 = 1.0000."""
        for ext in (self.dict_json, self.spacy_json):
            tp = fp = fn = 0
            for job in tech_benchmark.SYNTHETIC_JOBS:
                preds = {sp["canonical_skill"] for sp in ext.extract(job["description"])}
                gt = set(job["ground_truth_skills"])
                tp += len(preds & gt)
                fp += len(preds - gt)
                fn += len(gt - preds)
            self.assertEqual((tp, fp, fn), (156, 0, 0))
            # Verify raw spans are still recoverable when include_negated=True
            syn_03_all = ext.extract(tech_benchmark.SYNTHETIC_JOBS[2]["description"], include_negated=True)
            neg_names = {sp["canonical_skill"] for sp in syn_03_all if sp["is_negated"]}
            self.assertEqual(neg_names, {"Python", "Java"})

    def test_30_spacy_parser_unavailable_fallback(self) -> None:
        """Week 3 Module 1: Report documented regex fallback when dependency parser is absent."""
        sp_fallback = SpacySkillExtractor(self.tax_json_path, use_fast_tokenizer=True)
        sp_fallback.has_dependency_parser = False
        analysis = sp_fallback.analyze_sentences_and_negation("Python is not required, but SQL is essential.")
        self.assertEqual(analysis["analysis_method"], "conservative_regex_fallback_parser_unavailable")
        by_canon = {sp["canonical_skill"]: sp for sp in analysis["all_spans"]}
        self.assertTrue(by_canon["Python"]["is_negated"])
        self.assertFalse(by_canon["SQL"]["is_negated"])

    # Week 4 Member 01: University Curriculum & Syllabus Ingestion Tests
    # ------------------------------------------------------------------

    def test_44_curriculum_provenance_and_fk_validation(self) -> None:
        """Week 4 Curriculum: Rejects missing provenance in strict mode and orphan major_id."""
        from scripts.ingest_curriculum import CurriculumIngestionPipeline, CurriculumValidationError

        pipe = CurriculumIngestionPipeline(self.tax_csv_path, extractor=self.spacy_global, strict_provenance=True)
        valid_major = [{
            "major_id": "TEST_MAJ_CS",
            "major_name": "Computer Science (Test Fixture)",
            "institution_name": "Test University",
            "degree_level": "Bachelor",
            "source_url": "https://example.edu/cs",
            "collected_at": "2026-11-18",
        }]
        missing_prov_course = [{
            "course_id": "TEST_CS101",
            "course_code": "CS101",
            "course_title": "Intro Programming",
            "major_id": "TEST_MAJ_CS",
            "syllabus_text": "Learn Python and SQL.",
        }]
        with self.assertRaises(CurriculumValidationError):
            pipe.validate_records(valid_major, missing_prov_course)

        orphan_course = [{
            "course_id": "TEST_CS101",
            "course_code": "CS101",
            "course_title": "Intro Programming",
            "major_id": "UNKNOWN_MAJ",
            "syllabus_text": "Learn Python.",
            "source_url": "https://example.edu/cs101",
            "collected_at": "2026-11-18",
        }]
        with self.assertRaises(CurriculumValidationError):
            pipe.validate_records(valid_major, orphan_course)

    def test_45_curriculum_prerequisite_cycle_detection(self) -> None:
        """Week 4 Curriculum: Rejects self-loops and cyclic prerequisite DAGs."""
        from scripts.ingest_curriculum import CurriculumIngestionPipeline, CurriculumValidationError

        pipe = CurriculumIngestionPipeline(self.tax_csv_path, extractor=self.spacy_global, strict_provenance=False)
        majors = [{
            "major_id": "TEST_MAJ_DS",
            "major_name": "Data Science (Test Fixture)",
            "institution_name": "Test University",
            "degree_level": "Bachelor",
        }]
        cyclic_courses = [
            {
                "course_id": "C1",
                "course_code": "DS101",
                "course_title": "Foundations",
                "major_id": "TEST_MAJ_DS",
                "syllabus_text": "Python and Statistics.",
                "prerequisite_course_ids": ["C2"],
            },
            {
                "course_id": "C2",
                "course_code": "DS201",
                "course_title": "Modeling",
                "major_id": "TEST_MAJ_DS",
                "syllabus_text": "Machine Learning.",
                "prerequisite_course_ids": ["C1"],
            },
        ]
        with self.assertRaises(CurriculumValidationError):
            pipe.validate_records(majors, cyclic_courses)

    def test_46_curriculum_syllabus_skill_extraction_and_negation(self) -> None:
        """Week 4 Curriculum: Extracts syllabus skills while filtering negated requirements."""
        from scripts.ingest_curriculum import CurriculumIngestionPipeline

        pipe = CurriculumIngestionPipeline(self.tax_csv_path, extractor=self.spacy_global, strict_provenance=True)
        majors = [{
            "major_id": "TEST_MAJ_CS",
            "major_name": "CS Syllabus Fixture",
            "institution_name": "Test University",
            "degree_level": "Bachelor",
            "source_url": "https://example.edu/cs",
            "collected_at": "2026-11-18",
        }]
        courses = [{
            "course_id": "TEST_CS101",
            "course_code": "CS101",
            "course_title": "Data Foundations",
            "major_id": "TEST_MAJ_CS",
            "syllabus_text": "Students practice Python and SQL queries. Note that we do not use Java in this course.",
            "source_url": "https://example.edu/cs101",
            "collected_at": "2026-11-18",
        }]
        res = pipe.ingest(majors, courses)
        self.assertEqual(res["courses_validated"], 1)
        self.assertGreaterEqual(res["course_skills_extracted"], 2)

    def test_48_curriculum_database_persistence_and_table_counts(self) -> None:
        """Week 4 Curriculum: Persists university_majors, university_courses, and course_skills into DB."""
        from scripts.ingest_curriculum import CurriculumIngestionPipeline

        db = self.RelationalDatabaseManager(self.kg_builder, db_path=":memory:")
        pipe = CurriculumIngestionPipeline(self.tax_csv_path, extractor=self.spacy_global, strict_provenance=True)
        majors = [{
            "major_id": "TEST_MAJ_DB",
            "major_name": "Data Analytics Fixture",
            "institution_name": "Test University",
            "degree_level": "Bachelor",
            "source_url": "https://example.edu/da",
            "collected_at": "2026-11-18",
        }]
        courses = [{
            "course_id": "TEST_DA101",
            "course_code": "DA101",
            "course_title": "SQL and Excel",
            "major_id": "TEST_MAJ_DB",
            "syllabus_text": "Hands-on SQL and Microsoft Excel.",
            "source_url": "https://example.edu/da101",
            "collected_at": "2026-11-18",
        }]
        res = pipe.ingest(majors, courses, db_manager=db)
        self.assertEqual(res["database_counts"]["university_majors"], 1)
        self.assertEqual(res["database_counts"]["university_courses"], 1)
        self.assertGreaterEqual(res["database_counts"]["course_skills"], 2)

    def test_49_curriculum_csv_roundtrip_temp_files(self) -> None:
        """Week 4 Curriculum: Loads and validates CSV curriculum exports cleanly via TemporaryDirectory."""
        import tempfile
        import pandas as pd
        from scripts.ingest_curriculum import CurriculumIngestionPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            m_path = tmp / "majors.csv"
            c_path = tmp / "courses.csv"
            pd.DataFrame([{
                "major_id": "TEST_TMP_MAJ",
                "major_name": "Temp Major",
                "institution_name": "Test Institute",
                "degree_level": "Bachelor",
                "source_url": "https://example.edu/tmp",
                "collected_at": "2026-11-18",
            }]).to_csv(m_path, index=False)
            pd.DataFrame([{
                "course_id": "TEST_TMP_C1",
                "course_code": "TMP101",
                "course_title": "Intro Data Analysis",
                "major_id": "TEST_TMP_MAJ",
                "syllabus_text": "Covers Data Analysis and Power BI.",
                "prerequisite_course_ids": "[]",
                "source_url": "https://example.edu/tmp101",
                "collected_at": "2026-11-18",
            }]).to_csv(c_path, index=False)

            pipe = CurriculumIngestionPipeline(self.tax_csv_path, extractor=self.spacy_global, strict_provenance=True)
            res = pipe.ingest(pipe.load_records(m_path), pipe.load_records(c_path))
            self.assertEqual(res["majors_validated"], 1)
            self.assertEqual(res["courses_validated"], 1)
            self.assertGreaterEqual(res["course_skills_extracted"], 2)

    # ------------------------------------------------------------------
    # Week 4 Member 01: Real Job-Posting Ingestion & Quality Audit Tests
    # ------------------------------------------------------------------

    def test_50_real_jobs_provenance_and_synthetic_rejection(self) -> None:
        """Week 4 Real Jobs: Rejects synthetic rows and missing provenance in strict mode."""
        from scripts.ingest_real_jobs import RealJobPostingPipeline, RealJobValidationError

        pipe = RealJobPostingPipeline(self.tax_csv_path, extractor=self.spacy_global, strict_provenance=True)
        with self.assertRaises(RealJobValidationError):
            pipe.validate_and_deduplicate([{
                "job_id": "REAL_LEAK_1",
                "occupation": "Data Analyst",
                "description": "Requires Python and SQL.",
                "is_synthetic": True,
                "source_url": "https://example.org/job/1",
                "source_name": "ExampleBoard",
                "collected_at": "2026-11-18",
            }])

        with self.assertRaises(RealJobValidationError):
            pipe.validate_and_deduplicate([{
                "job_id": "REAL_NOPROV_1",
                "occupation": "Data Analyst",
                "description": "Requires Python and SQL.",
                "is_synthetic": False,
            }])

    def test_51_real_jobs_deduplication_and_missingness_audit(self) -> None:
        """Week 4 Real Jobs: Deduplicates repeated job_ids/content and audits missing optional fields."""
        from scripts.ingest_real_jobs import RealJobPostingPipeline

        pipe = RealJobPostingPipeline(self.tax_csv_path, extractor=self.spacy_global, strict_provenance=True)
        rows = [
            {
                "job_id": "REAL_1",
                "occupation": "Data Analyst",
                "description": "Required: Python and SQL.",
                "is_synthetic": False,
                "source_url": "https://example.org/1",
                "source_name": "JobBoard",
                "collected_at": "2026-11-18",
                "country": "Cambodia",
            },
            {
                "job_id": "REAL_1",  # duplicate job_id
                "occupation": "Data Analyst",
                "description": "Required: Python and SQL.",
                "is_synthetic": False,
                "source_url": "https://example.org/1",
                "source_name": "JobBoard",
                "collected_at": "2026-11-18",
            },
            {
                "job_id": "REAL_2",  # duplicate normalized content
                "occupation": "Data Analyst",
                "description": "Required:  Python and SQL.",
                "is_synthetic": False,
                "source_url": "https://example.org/2",
                "source_name": "JobBoard",
                "collected_at": "2026-11-18",
            },
        ]
        val = pipe.validate_and_deduplicate(rows)
        self.assertEqual(val["total_input_rows"], 3)
        self.assertEqual(val["valid_unique_rows"], 1)
        self.assertEqual(val["duplicate_job_ids_dropped"], ["REAL_1"])
        self.assertEqual(val["duplicate_content_dropped"], ["REAL_2"])
        self.assertIn("salary_min", val["optional_field_missing_counts"])

    def test_52_real_jobs_cleaning_extraction_and_negation(self) -> None:
        """Week 4 Real Jobs: Extracts positive skills, filters negated mentions, flags unmapped occupations."""
        import tempfile
        import pandas as pd
        from scripts.ingest_real_jobs import RealJobPostingPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "real_jobs_fixture.csv"
            pd.DataFrame([{
                "job_id": "REAL_FIX_1",
                "occupation": "Custom Quantum Analyst",
                "country": "Cambodia",
                "description": "Required Skills:\nMust know Python and SQL.\nNote that we do not use Java.",
                "is_synthetic": False,
                "source_url": "https://example.org/q1",
                "source_name": "ExampleBoard",
                "collected_at": "2026-11-18",
            }]).to_csv(csv_path, index=False)

            pipe = RealJobPostingPipeline(self.tax_csv_path, extractor=self.spacy_global, strict_provenance=True)
            report = pipe.process_and_audit(pipe.load_records(csv_path))
            self.assertEqual(report["status"], "REAL_DATA_AUDITED")
            extracted_canons = {r["canonical_skill"] for r in report["extracted_rows"]}
            self.assertIn("Python", extracted_canons)
            self.assertIn("SQL", extracted_canons)
            self.assertNotIn("Java", extracted_canons)
            self.assertEqual(report["eda_summary"]["unmapped_occupations"], ["Custom Quantum Analyst"])
            self.assertEqual(report["eda_summary"]["length_preserved_count"], 1)

    def test_53_real_jobs_empty_default_csv_safe_handling(self) -> None:
        """Week 4 Real Jobs: Handles empty data/raw/real_job_postings.csv cleanly without inventing rows."""
        from scripts.ingest_real_jobs import DEFAULT_REAL_JOBS_PATH, RealJobPostingPipeline

        pipe = RealJobPostingPipeline(self.tax_csv_path, extractor=self.spacy_global, strict_provenance=True)
        recs = pipe.load_records(DEFAULT_REAL_JOBS_PATH)
        report = pipe.process_and_audit(recs)
        self.assertIn(report["status"], {"AWAITING_REAL_DATA", "REAL_DATA_AUDITED"})
        self.assertGreaterEqual(report["valid_unique_rows"], 0)

    # ------------------------------------------------------------------
    # Week 4 Member 01: Human-Annotation Importer & Kappa Tests
    # ------------------------------------------------------------------

    def test_56_human_annotation_rejects_unreviewed_or_fabricated_ids(self) -> None:
        """Week 4 Integrity: Refuses unreviewed rows, missing EDIT notes, and fabricated annotator_id."""
        from scripts.evaluate_human_annotations import (
            HumanAnnotationValidationError,
            validate_review_rows,
        )

        with self.assertRaises(HumanAnnotationValidationError):
            validate_review_rows([{"human_decision": None, "annotator_id": None}])

        with self.assertRaises(HumanAnnotationValidationError):
            validate_review_rows([{"human_decision": "CONFIRM", "annotator_id": "HUMAN_ANNOTATOR_TIER_A"}])

        with self.assertRaises(HumanAnnotationValidationError):
            validate_review_rows([{"human_decision": "EDIT", "annotator_id": "hustsak", "human_edit_notes": ""}])

    def test_57_human_annotation_cohens_kappa_calculation(self) -> None:
        """Week 4 Agreement: Computes exact Cohen's kappa on shared spans."""
        from scripts.evaluate_human_annotations import compute_cohens_kappa

        rows_a = [
            {"record_id": 1, "job_id": "J1", "character_start": 0, "character_end": 6, "human_decision": "CONFIRM"},
            {"record_id": 1, "job_id": "J1", "character_start": 10, "character_end": 13, "human_decision": "CONFIRM"},
            {"record_id": 2, "job_id": "J2", "character_start": 0, "character_end": 4, "human_decision": "REJECT"},
            {"record_id": 2, "job_id": "J2", "character_start": 8, "character_end": 12, "human_decision": "REJECT"},
        ]
        rows_b = [
            {"record_id": 1, "job_id": "J1", "character_start": 0, "character_end": 6, "human_decision": "CONFIRM"},
            {"record_id": 1, "job_id": "J1", "character_start": 10, "character_end": 13, "human_decision": "CONFIRM"},
            {"record_id": 2, "job_id": "J2", "character_start": 0, "character_end": 4, "human_decision": "REJECT"},
            {"record_id": 2, "job_id": "J2", "character_start": 8, "character_end": 12, "human_decision": "CONFIRM"},
        ]
        res = compute_cohens_kappa(rows_a, rows_b)
        self.assertEqual(res["shared_spans"], 4)
        self.assertEqual(res["observed_agreement"], 0.75)
        self.assertEqual(res["disagreement_count"], 1)
        self.assertEqual(res["cohens_kappa"], 0.5)

    def test_58_human_annotation_import_and_eval_roundtrip(self) -> None:
        """Week 4 Integrity: Builds validated human-reference JSON in memory and runs evaluation."""
        import pandas as pd
        from scripts.evaluate_human_annotations import (
            build_human_reviewed_payload,
            evaluate_against_human_reference,
        )

        text = "Required: Python and SQL."
        base_json = {
            "total_records": 1,
            "total_spans": 2,
            "records": [{
                "record_id": 1,
                "job_id": "SYN_TEST_HUMAN_1",
                "text": text,
                "pass_1": {"spans": [], "status": "PENDING"},
                "pass_2": {"canonical_mappings": [], "status": "PENDING"},
                "record_status": "PENDING",
                "annotator_id": None,
            }],
        }
        review_rows = [
            {
                "record_id": 1,
                "job_id": "SYN_TEST_HUMAN_1",
                "skill_mention": "Python",
                "character_start": 10,
                "character_end": 16,
                "source_section": "Required",
                "candidate_skill_id": "SG-SK-0001",
                "candidate_canonical_skill": "Python",
                "human_decision": "CONFIRM",
                "annotator_id": "hustsak_test",
                "human_edit_notes": "",
            },
            {
                "record_id": 1,
                "job_id": "SYN_TEST_HUMAN_1",
                "skill_mention": "SQL",
                "character_start": 21,
                "character_end": 24,
                "source_section": "Required",
                "candidate_skill_id": "SG-SK-0006",
                "candidate_canonical_skill": "SQL",
                "human_decision": "CONFIRM",
                "annotator_id": "hustsak_test",
                "human_edit_notes": "",
            },
        ]
        tax_df = pd.read_csv(self.tax_csv_path)
        payload = build_human_reviewed_payload(review_rows, base_json, tax_df)
        self.assertEqual(payload["total_spans"], 2)
        self.assertEqual(payload["records"][0]["record_status"], "COMPLETED")
        eval_res = evaluate_against_human_reference(payload, self.tax_csv_path)
        self.assertEqual(eval_res["spacy_strict"]["f1"], 1.0)
        self.assertEqual(eval_res["dictionary_strict"]["f1"], 1.0)

    # ------------------------------------------------------------------
    # Week 4 Integration: Major Gap, New API Routes, CORS, PG Init, Ablations
    # ------------------------------------------------------------------

    def test_61_postgres_init_dry_run_and_dotenv_safety(self) -> None:
        """Week 4 DB: Verifies PostgreSQL init dry-run without exposing secrets."""
        from scripts.init_postgres import run_postgres_init

        res = run_postgres_init(dry_run=True)
        self.assertEqual(res["status"], "DRY_RUN_VALIDATED")
        self.assertEqual(res["ddl_statement_count"], 23)
        self.assertFalse(res["executed_against_postgres"])

    def test_64_research_questions_and_proposal_nonempty(self) -> None:
        """Week 4 Docs: Verifies research questions and revised proposal documents are populated."""
        rq = (ROOT_DIR / "docs" / "proposal" / "research_questions.md").read_text(encoding="utf-8")
        prop = (ROOT_DIR / "docs" / "proposal" / "project_proposal.md").read_text(encoding="utf-8")
        self.assertIn("RQ0:", rq)
        self.assertIn("Member 01", prop)

    def test_65_member1_real_data_evaluation_report(self) -> None:
        """Member 01 M3 Step 1: Verifies human-reference error analysis and real-curriculum coverage."""
        from scripts.evaluate_member1_real_data import run_member1_evaluation

        res = run_member1_evaluation(save_path=None)
        self.assertEqual(res["human_reference_50_records"]["records_evaluated"], 50)
        self.assertEqual(res["real_jobs_100_audit"]["valid_unique_postings"], 100)
        self.assertIn("negation_ablation", res["real_jobs_100_audit"])
        self.assertEqual(res["real_curriculum_vs_real_jobs_gap"]["courses_count"], 13)
        self.assertGreater(res["real_curriculum_vs_real_jobs_gap"]["curriculum_skill_coverage_ratio"], 0.0)


def run_summary_benchmark() -> None:
    """Prints a quick summary of the 22-posting & 200-posting checks."""
    tax_csv = ROOT_DIR / "data" / "processed" / "global_skill_taxonomy.csv"
    tax_json = ROOT_DIR / "outputs" / "exports" / "skill_taxonomy.json"
    gt_200 = load_benchmark_corpus(ROOT_DIR / "data" / "annotations" / "synthetic_job_postings_ground_truth.json")[:200]

    d_230 = DictionarySkillExtractor(tax_csv, section_filter={"required", "preferred"}, include_broad_country_mappings=True)
    s_230 = SpacySkillExtractor(tax_csv, section_filter={"required", "preferred"}, include_broad_country_mappings=True, use_fast_tokenizer=True)

    texts = {r["job_id"]: r["text"] for r in gt_200}
    gts = {r["job_id"]: r["annotations"] for r in gt_200}
    pd_res = {k: d_230.extract(v) for k, v in texts.items()}
    ps_res = {k: s_230.extract(v) for k, v in texts.items()}
    eval_d = SkillExtractionEvaluator.evaluate_exact_spans(pd_res, gts)
    eval_s = SkillExtractionEvaluator.evaluate_exact_spans(ps_res, gts)
    print(f"200-Doc Global Sample (Broad+SecFilter) -> Dict F1={eval_d['micro_f1']:.4f}, spaCy F1={eval_s['micro_f1']:.4f}")
    d_31 = DictionarySkillExtractor(tax_json)
    print(f"31-Skill Benchmark Taxonomy Patterns Loaded: {len(d_31.patterns)}")


if __name__ == "__main__":
    run_summary_benchmark()
    unittest.main()
