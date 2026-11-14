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

        # Week 3 Knowledge Graph, Gap Engine, Relational DB, and Backend API
        from scripts.build_graph import (  # noqa: E402
            PriorityScoringModel,
            PriorityWeights,
            RelationalDatabaseManager,
            SkillGapAnalysisEngine,
            SkillGraphBuilder,
        )
        from backend.app.main import SkillGraphTestClient, create_app  # noqa: E402

        cls.PriorityScoringModel = PriorityScoringModel
        cls.PriorityWeights = PriorityWeights
        cls.RelationalDatabaseManager = RelationalDatabaseManager
        cls.kg_builder = SkillGraphBuilder(include_broad_country_mappings=False, min_co_occurrence=5)
        cls.kg_builder.build()
        cls.gap_engine = SkillGapAnalysisEngine(cls.kg_builder)
        cls.api_app, cls.db_layer, cls.api_service = create_app(builder=cls.kg_builder)
        cls.api_client = SkillGraphTestClient(cls.api_app)

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

    def test_31_complete_student_profile_zero_gap(self) -> None:
        """Week 3 Module 2: Student demonstrating all target-occupation skills has gap_score=0.0."""
        occ_info = self.db_layer.get_occupation_skills("SG-OC-002")
        full_skills = [{"skill_id": s["skill_id"], "proficiency_level": "advanced"} for s in occ_info["all_skills"]]
        res = self.gap_engine.analyze({
            "student_id": "SYN_COMPLETE_DA",
            "target_occupation": "Data Analyst",
            "skills": full_skills,
        })
        self.assertEqual(res["coverage_metrics"]["required_skill_coverage"], 1.0)
        self.assertEqual(res["coverage_metrics"]["overall_weighted_coverage"], 1.0)
        self.assertEqual(res["coverage_metrics"]["skill_gap_score"], 0.0)
        self.assertEqual(res["priority_recommendations"], [])

    def test_32_empty_student_profile_full_gap(self) -> None:
        """Week 3 Module 2: Empty student profile yields 0.0 coverage and honest unrecorded labels."""
        res = self.gap_engine.analyze({
            "student_id": "SYN_EMPTY_DS",
            "target_occupation": "Data Scientist",
            "skills": [],
        })
        self.assertEqual(res["coverage_metrics"]["required_skill_coverage"], 0.0)
        self.assertEqual(res["coverage_metrics"]["overall_weighted_coverage"], 0.0)
        self.assertEqual(res["coverage_metrics"]["skill_gap_score"], 1.0)
        self.assertTrue(all(r["profile_status"] == "not_recorded_in_supplied_profile" for r in res["priority_recommendations"]))
        self.assertTrue(all("not recorded in the supplied" in r["explanation"]["student_profile_status"] for r in res["priority_recommendations"]))

    def test_33_unknown_occupation_raises_and_returns_404(self) -> None:
        """Week 3 Module 2/7: Unknown occupation raises ValueError in engine and HTTP 404 in API."""
        with self.assertRaises(ValueError):
            self.gap_engine.analyze({"student_id": "SYN_ERR", "target_occupation": "Astral Wizard"})
        resp = self.api_client.post("/skill-gap/analyze", json={"student_id": "SYN_ERR", "target_occupation": "Astral Wizard", "skills": ["SQL"]})
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["error_code"], "RESOURCE_NOT_FOUND")

    def test_34_unknown_skill_handling_lenient_vs_strict(self) -> None:
        """Week 3 Module 2/7: Unknown skills are logged cleanly in lenient mode and rejected in strict mode."""
        res_lenient = self.gap_engine.analyze({
            "student_id": "SYN_UNK_SKILL",
            "target_occupation": "Data Analyst",
            "skills": ["Python", "NonExistentHyperSkill999"],
        }, strict_validation=False)
        self.assertEqual(len(res_lenient["unrecognized_skills"]), 1)
        self.assertEqual(res_lenient["unrecognized_skills"][0]["input_skill"], "NonExistentHyperSkill999")

        with self.assertRaises(ValueError):
            self.gap_engine.analyze({
                "student_id": "SYN_UNK_SKILL",
                "target_occupation": "Data Analyst",
                "skills": ["NonExistentHyperSkill999"],
            }, strict_validation=True)
        resp_strict = self.api_client.post("/skill-gap/analyze", json={
            "student_id": "SYN_UNK_SKILL",
            "target_occupation": "Data Analyst",
            "skills": ["NonExistentHyperSkill999"],
            "strict_validation": True,
        })
        self.assertEqual(resp_strict.status_code, 400)
        self.assertEqual(resp_strict.json()["error_code"], "INVALID_INPUT")

    def test_35_duplicate_student_skills_retains_highest_proficiency(self) -> None:
        """Week 3 Module 2: Duplicate skills deduplicate cleanly and retain highest proficiency."""
        res = self.gap_engine.analyze({
            "student_id": "SYN_DUP",
            "target_occupation": "Data Analyst",
            "skills": [
                {"skill": "SQL", "proficiency_level": "beginner"},
                {"skill_id": "SG-SK-0006", "proficiency_level": "advanced"},
                {"skill": "postgres", "proficiency_level": "intermediate"},
            ],
        })
        self.assertGreaterEqual(len(res["duplicate_skill_notes"]), 1)
        possessed_ids = {s["skill_id"] for s in res["possessed_skills"]}
        self.assertIn("SG-SK-0006", possessed_ids)  # SQL retained at advanced (gap=0.0)
        self.assertNotIn("SG-SK-0006", {s["skill_id"] for s in res["partially_demonstrated_skills"]})

    def test_36_missing_and_partial_proficiency_handling(self) -> None:
        """Week 3 Module 2: Handles omitted proficiency conservatively and surfaces partial gaps."""
        res = self.gap_engine.analyze({
            "student_id": "SYN_PROF",
            "target_occupation": "Data Analyst",
            "skills": [
                "Python",  # proficiency omitted -> demonstrated
                {"skill": "SQL", "proficiency_score": 0.35},  # 0.35 / 0.70 = 0.50 demonstrated -> gap 0.50
            ],
        })
        self.assertIn("SG-SK-0001", {s["skill_id"] for s in res["possessed_skills"]})
        partial_ids = {s["skill_id"]: s for s in res["partially_demonstrated_skills"]}
        self.assertIn("SG-SK-0006", partial_ids)
        self.assertEqual(partial_ids["SG-SK-0006"]["gap_factor"], 0.5)
        self.assertEqual(partial_ids["SG-SK-0006"]["profile_status"], "partially_demonstrated_in_supplied_profile")

    def test_37_required_vs_preferred_distinction_and_missing_mappings(self) -> None:
        """Week 3 Module 2/4: Preserves required vs preferred distinction and unmapped local laws."""
        norm_unresolved = self.normalizer_strict.normalize("LGPD compliance")
        self.assertEqual(norm_unresolved["mapping_status"], "UNRESOLVED_CANONICAL")
        res = self.gap_engine.analyze({
            "student_id": "SYN_REQ_PREF",
            "target_occupation": "Data Scientist",
            "skills": ["Python"],
        })
        req_ids = {s["skill_id"] for s in res["missing_required_skills"]}
        pref_ids = {s["skill_id"] for s in res["missing_preferred_skills"]}
        self.assertIn("SG-SK-0027", req_ids)  # Machine Learning is required in 17/17 DS postings
        self.assertIn("SG-SK-0029", pref_ids)  # Model Deployment is preferred-only in 9/17 DS postings
        self.assertTrue(req_ids.isdisjoint(pref_ids))

    def test_38_multi_occupation_opportunity_and_co_occurrence_semantics(self) -> None:
        """Week 3 Module 4/5: Multi-occupation counts & CO_OCCURS_WITH non-prerequisite tags are exact."""
        opp_sql = self.gap_engine.opp_analyzer.analyze_skill_opportunities("SG-SK-0006", exclude_occupation_id="SG-OC-002")
        self.assertEqual(opp_sql["required_occupation_count"], 6)
        self.assertEqual(opp_sql["preferred_only_occupation_count"], 4)
        self.assertEqual(opp_sql["total_occupation_count"], 10)
        self.assertEqual(opp_sql["co_occurrence_disclaimer"], "CO_OCCURS_WITH denotes symmetric statistical association only, never a pedagogical prerequisite.")
        for co in opp_sql["commonly_co_occurring_skills"]:
            self.assertEqual(co["relationship_semantics"], "statistical_co_occurrence_not_prerequisite")
        self.assertIn("Data Analyst", opp_sql["unlock_explanation"])

    def test_39_unavailable_prerequisite_and_no_fabricated_courses(self) -> None:
        """Week 3 Module 5: Reports missing prerequisites & courses honestly without fabricating."""
        res = self.gap_engine.analyze({"student_id": "SYN_LP", "target_occupation": "Data Scientist", "skills": []})
        lp = res["learning_path"]
        self.assertFalse(lp["prerequisite_graph_available"])
        self.assertEqual(lp["prerequisite_graph_status"], "UNAVAILABLE_IN_LOCAL_TAXONOMY")
        self.assertFalse(lp["learning_resources_available"])
        self.assertEqual(lp["learning_resources_status"], "UNAVAILABLE_NO_VERIFIED_COURSE_CATALOG")
        for rec in res["priority_recommendations"]:
            self.assertEqual(rec["prerequisite_info"]["status"], "UNAVAILABLE_IN_LOCAL_TAXONOMY")
            self.assertEqual(rec["scoring"]["components"]["D_status"], "UNAVAILABLE_NO_VERIFIED_PREREQUISITE_GRAPH")
            self.assertIsNone(rec["scoring"]["components"]["D_dependency_importance"])
            self.assertEqual(rec["scoring"]["components"]["C_status"], "UNAVAILABLE_NO_EMPIRICAL_COST_DATA")
            self.assertIsNone(rec["scoring"]["components"]["C_learning_cost"])

    def test_40_priority_score_bounds_no_double_counting_and_sensitivity(self) -> None:
        """Week 3 Module 3: Validates [0, 1] bounds, external O(s) non-overlap, and weight sensitivity."""
        res = self.gap_engine.analyze({
            "student_id": "SYN_SENS",
            "target_occupation": "Data Analyst",
            "skills": ["Communication"],
        }, run_sensitivity=True)
        # Machine Learning only required in DS & ML Eng; check Data Scientist single-target exclusion
        opp_excl_ds = self.gap_engine.opp_analyzer.analyze_skill_opportunities("SG-SK-0027", exclude_occupation_id="SG-OC-003")
        opp_incl_all = self.gap_engine.opp_analyzer.analyze_skill_opportunities("SG-SK-0027", exclude_occupation_id=None)
        self.assertLess(opp_excl_ds["raw_external_opportunity"], opp_incl_all["raw_external_opportunity"])

        for rec in res["priority_recommendations"]:
            self.assertGreaterEqual(rec["priority_score"], 0.0)
            self.assertLessEqual(rec["priority_score"], 1.0)
            c = rec["scoring"]["components"]
            self.assertTrue(0.0 <= c["G_gap_importance"] <= 1.0)
            self.assertTrue(0.0 <= c["O_external_opportunity"] <= 1.0)

        sens = res["weight_sensitivity"]
        self.assertIsNotNone(sens)
        self.assertIn("target_only", sens["top_k_by_preset"])
        self.assertIn("career_mobility_first", sens["top_k_by_preset"])
        self.assertEqual(sens["top_k_jaccard_vs_baseline"]["baseline_balanced_heuristic"], 1.0)

    def test_41_database_constraints_foreign_keys_unique_check(self) -> None:
        """Week 3 Module 6: Enforces PK, FK, UNIQUE, CHECK constraints & schema separation."""
        import sqlite3
        db = self.RelationalDatabaseManager(self.kg_builder, db_path=":memory:")
        counts = db.list_tables_and_counts()
        self.assertEqual(len(counts), 17)
        self.assertEqual(counts["skills"], 230)
        self.assertEqual(counts["occupations"], 53)
        self.assertEqual(counts["job_postings"], 2000)
        self.assertEqual(counts["university_courses"], 0)  # No fabricated courses

        # FK violation: unknown skill_id
        with self.assertRaises(sqlite3.IntegrityError):
            db.conn.execute("INSERT INTO students VALUES ('TMP_S', NULL, 'SG-OC-002', 1, '2026-01-01')")
            db.conn.execute("INSERT INTO student_skills VALUES ('TMP_S', 'SG-SK-9999', 'beginner', 0.5, 'test')")

        # CHECK constraint violation: proficiency_score > 1.0
        with self.assertRaises(sqlite3.IntegrityError):
            db.conn.execute("INSERT INTO student_skills VALUES ('TMP_S', 'SG-SK-0001', 'expert', 1.5, 'test')")

        # UNIQUE constraint violation: duplicate industry_name
        with self.assertRaises(sqlite3.IntegrityError):
            db.conn.execute("INSERT INTO industries VALUES ('IND_DUP', 'Information Technology', 1)")

    def test_42_api_endpoints_valid_and_invalid_requests(self) -> None:
        """Week 3 Module 7: Tests all REST endpoints, Pydantic validation, HTTP codes, and OpenAPI schema."""
        r_health = self.api_client.get("/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertEqual(r_health.json()["graph_nodes"], 2336)

        r_occ = self.api_client.get("/occupations?industry=Information%20Technology")
        self.assertEqual(r_occ.status_code, 200)
        self.assertGreaterEqual(len(r_occ.json()), 5)

        r_sk = self.api_client.get("/skills?search=Python")
        self.assertEqual(r_sk.status_code, 200)
        self.assertTrue(any(s["skill_id"] == "SG-SK-0001" for s in r_sk.json()))

        r_occ_sk = self.api_client.get("/occupations/SG-OC-002/skills")
        self.assertEqual(r_occ_sk.status_code, 200)
        self.assertGreater(r_occ_sk.json()["total_skills"], 10)

        # Invalid Pydantic payloads -> 422
        r_bad_empty = self.api_client.post("/skill-gap/analyze", json={"student_id": "", "target_occupation": "Data Analyst"})
        self.assertEqual(r_bad_empty.status_code, 422)
        r_bad_prof = self.api_client.post("/skill-gap/analyze", json={
            "student_id": "SYN_BAD_PROF",
            "target_occupation": "Data Analyst",
            "skills": [{"skill": "SQL", "proficiency_score": 2.5}],
        })
        self.assertEqual(r_bad_prof.status_code, 422)

        # Full end-to-end analyze -> get by analysis_id -> recommendations -> explanations
        r_post = self.api_client.post("/skill-gap/analyze", json={
            "student_id": "SYN_TEST_USER_42",
            "target_occupation": "Data Scientist",
            "skills": ["Python", {"skill": "SQL", "proficiency_level": "beginner"}],
            "top_k": 5,
        })
        self.assertEqual(r_post.status_code, 200)
        aid = r_post.json()["analysis_id"]
        self.assertEqual(self.api_client.get(f"/skill-gap/{aid}").status_code, 200)
        self.assertEqual(self.api_client.get("/recommendations/SYN_TEST_USER_42").status_code, 200)
        r_exp = self.api_client.get("/recommendations/SYN_TEST_USER_42/explanations")
        self.assertEqual(r_exp.status_code, 200)
        self.assertEqual(r_exp.json()["explanations_count"], 5)

        # Unknown analysis/student IDs -> 404
        self.assertEqual(self.api_client.get("/skill-gap/GAP-DOESNOTEXIST").status_code, 404)
        self.assertEqual(self.api_client.get("/recommendations/UNKNOWN_STUDENT_404").status_code, 404)

    def test_43_explanation_structural_and_numerical_consistency(self) -> None:
        """Week 3 Module 8: Verifies 7-part structured explanations match numerical calculations."""
        res = self.gap_engine.analyze({
            "student_id": "SYN_EXPLAIN_CHECK",
            "target_occupation": "Data Analyst",
            "skills": ["Python"],
        })
        required_exp_keys = {
            "skill", "skill_id", "why_identified", "occupation_requirement_role",
            "student_profile_status", "importance_assessment", "multi_occupation_relevance",
            "priority_calculation", "supporting_evidence",
        }
        for rec in res["priority_recommendations"]:
            exp = rec["explanation"]
            self.assertTrue(required_exp_keys.issubset(set(exp.keys())))
            self.assertEqual(exp["skill"], rec["canonical_skill"])
            self.assertEqual(exp["skill_id"], rec["skill_id"])
            self.assertEqual(exp["priority_calculation"]["priority_score"], rec["priority_score"])
            self.assertEqual(exp["supporting_evidence"]["required_postings"], rec["required_count"])
            self.assertEqual(exp["supporting_evidence"]["preferred_postings"], rec["preferred_count"])


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
