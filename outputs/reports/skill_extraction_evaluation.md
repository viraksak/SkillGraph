# SkillGraph — Week 1 Comprehensive Research & Evaluation Report

**Platform:** SkillGraph Core Research Engine  
**Milestone:** Week 1 Finalization — Dataset Audit, EDA, Annotation Design, & Extraction Baselines  
**Date:** November 2026  
**Status:** Completed & Empirically Verified  

---

## 1. Executive Summary & Research Objectives

The primary objective of Week 1 was to establish the empirical, taxonomic, and algorithmic foundation for the SkillGraph platform:
1. **Dataset Audit & Quality Assurance:** Validate the integrity, deduplication, and schema consistency of the collected global job-description corpus.
2. **Exploratory Data Analysis (EDA):** Investigate geographic, sectoral, occupational, and lexical distributions across the corpus to quantify characteristics and synthetic constraints.
3. **Annotation Architecture Review:** Evaluate annotation guidelines and inspect the 300-record human annotation template to assess occupational coverage and span conventions.
4. **Deterministic NLP Extraction Baselines:** Formulate, implement, and benchmark two independent extraction architectures:
   - **Method A:** Boundary-Aware Regular Expressions (`DictionarySkillExtractor`).
   - **Method B:** Token-Aware spaCy `PhraseMatcher` (`SpacySkillExtractor`).
5. **Exact Span-Level Evaluation & Error Analysis:** Measure strict span-level Precision, Recall, and F1, categorize structural failure modes, and isolate synthetic artifacts from real-world labor market phenomena.

---

## 2. Dataset Characteristics & Provenance Audit

| Dimension | Quantitative Metric | Verification Finding |
| :--- | :--- | :--- |
| **Total Records** | **2,000** job postings | Ingested from `data/processed/global_jobs_cleaned.csv` and `data/raw/synthetic_job_postings.*`. |
| **Real Records** | **0** records | Preserved separate empty placeholder files in `data/raw/real_job_postings.*`. |
| **Geographic Scope** | **20 countries** | Exactly 100 records per country across Asia (8), Europe (5), Americas (3), Africa (1), Oceania (2), Middle East (1). |
| **Industry Scope** | **15 industries** | Uniformly distributed between 129 (Construction) and 137 (Finance) postings; IT represents 6.65% (133 postings). |
| **Occupation Scope** | **53 occupations** | All 53 mapped to candidate ISCO-08 codes in `data/processed/global_occupation_mapping.csv`. |
| **Taxonomy Entities** | **230 canonical skills** | Spans 18 categories and 8 skill types in `data/processed/global_skill_taxonomy.csv`. |
| **Annotated Mentions** | **21,714 skill spans** | Verified in `data/processed/global_skill_mentions.csv` (averaging 10.86 mentions per posting). |
| **Generation Seed** | Seed 42 | Programmatic, template-driven generation using sentence frames and controlled vocabularies. |

---

## 3. Data Quality & Integrity Findings

An audit across all 40 columns confirmed:
- **Missing Values:** Exactly 9 fields have missing values (`company_name`, `salary_min`, `salary_max`, `salary_currency`, `source_url`, `posted_at`, `collected_at`, `license`, `attribution`), each showing **100.0% nulls (2,000 / 2,000)**. This is by design: real-world employer fields were intentionally omitted rather than inventing fake data.
- **Operational Completeness:** All operational fields (`description`, `required_skills`, `occupation`, `industry`, `country`, `seniority_level`) have **0% missing values**.
- **Duplicates:** Exactly **0 duplicate job IDs** and **0 duplicate text descriptions**. At the 0.80 TF-IDF cosine similarity threshold, all 2,000 records form distinct singleton clusters (maximum observed 5-gram Jaccard similarity is 0.34, well below the 0.55 threshold).

---

## 4. Key Exploratory Data Analysis (EDA) Findings

The complete analysis in `research/notebooks/03_eda.ipynb` produced 12 high-resolution visualizations in `outputs/visualizations/`:
1. **Geographic & Sectoral Balance:** Perfect uniformity across countries (100 each) and near-uniformity across industries (129–137 each) confirms synthetic balancing. Real-world labor-market skew is absent.
2. **Seniority Breakdown:** Mid-level (558, 27.9%) and Junior (518, 25.9%) roles dominate, followed by Senior (394, 19.7%), Entry-level (309, 15.5%), and Executive (221, 11.1%).
3. **Skill Composition:**
   - `English Language` is the single most prevalent competency (1,881 distinct postings, 94.1%).
   - Transversal core: `Communication` (633), `Microsoft Excel` (511), `Time Management` (460), `Team Leadership` (445), `Problem Solving` (431), `Data Analysis` (373).
   - Technical vs. Transversal: 56.3% of mentions (12,236) are technical (hard skills, technologies, tools), while 43.7% (9,478) are transversal (soft skills, languages, management).
4. **Text Characteristics:**
   - Character Length: Mean 1,130.8 chars (std: 92.8, min: 847, max: 1,419).
   - Word Count: Mean 159.2 words (std: 12.2, min: 119, max: 199).
   - Lexical Regularity: Active vocabulary is restricted to **1,676 unique word types** across 271,090 tokens (Type-Token Ratio = **0.0062**), reflecting template sentence frames.

---

## 5. Extraction Methodologies & Algorithmic Independence

Two modular extractors were developed and verified in `models/skill_extraction/`:

```
                       Input Job Description Text
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
[DictionarySkillExtractor]                             [SpacySkillExtractor]
• Compiles regex patterns with                        • Builds spaCy PhraseMatcher
  negative lookaround assertions:                       with dual-attribute vocabulary:
  (?<![A-Za-z0-9])phrase(?![A-Za-z0-9])                 - attr="LOWER" for general phrases
• Operates directly on Python string                    - attr="ORTH" for sensitive acronyms
  character buffer                                    • Operates on linguistic Doc tokens
          │                                                   │
          └─────────────────────────┬─────────────────────────┘
                                    ▼
                     [Greedy Longest-Match Overlap Resolver]
                     • Sorts candidate spans by descending length
                     • Eliminates overlapping sub-token spans
                     • Preserves exact character offsets: text[s:e] == mention
```

### Investigation: Why Were Benchmark Metrics Identical on the 100 Sample?
On the 100-sample benchmark in `04_skill_extraction.ipynb`, both extractors achieved identical scores (Micro Precision: 0.9243, Micro Recall: 0.9918, Micro F1: 0.9569). Our verification audit established:
1. **Algorithmic Independence:** Both extractors independently execute distinct underlying pattern-matching mechanisms.
2. **Shared Overlap Resolution:** Both extractors utilize the same deterministic greedy longest-match overlap resolver (`_resolve_overlaps`).
3. **Synthetic Regularity:** Across the first 100 benchmark records, skill phrases were generated from clean vocabulary matching exact taxonomy entries. Character boundaries and token boundaries aligned identically.
4. **Empirical Divergence Across the Full Corpus:** When audited across all 2,000 records, the two algorithms **diverged on record `GLOBAL_SYN_001440`**:
   - In `GLOBAL_SYN_001440`, the sentence ends with `"...RF optimisation, and LTE and 5G."`.
   - **Dictionary Regex:** Matched `'LTE and 5G'` at `[991:1001]` because `.` is non-alphanumeric, satisfying the lookahead boundary.
   - **spaCy PhraseMatcher:** Failed to match because spaCy's tokenizer attached the sentence-final period to the token (`['5G.']`), which failed to match the pattern doc (`['LTE', 'and', '5', 'G']`).
   - This proves the two models are functionally independent, with a 0.05% tokenization edge divergence on real punctuation boundaries.

---

## 6. Exact Span-Level Benchmark Evaluation

Evaluated against 100 benchmark postings (1,096 reference skill spans) using strict exact span matching (`start`, `end`, `canonical_skill`):

| Extraction Method | Micro Precision | Micro Recall | Micro F1-Score | Macro Precision | Macro Recall | Macro F1-Score | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dictionary Regex** | **0.9243** | **0.9918** | **0.9569** | **0.9393** | **0.9910** | **0.9615** | 1,087 | 89 | 9 |
| **spaCy PhraseMatcher** | **0.9243** | **0.9918** | **0.9569** | **0.9393** | **0.9910** | **0.9615** | 1,087 | 89 | 9 |

> **Evaluation Integrity Statement:**  
> These metrics reflect performance against **synthetic generator reference labels** (`human_verified = false`). They validate algorithmic implementation and span offset precision, but **must not be cited as empirical human ground-truth performance**.

---

## 7. Structural Error Analysis & Root Causes

Auditing the 89 False Positives (FP) and 9 False Negatives (FN) revealed three primary failure modes:

1. **Job Title & Department Name Bleed (Primary FP Driver):**  
   Dictionary matching scans the entire text indiscriminately. Words like `"Procurement"` or `"Customer Support"` appearing in company introductory text (e.g. *"Senior Procurement Officer in the Procurement department"*) are extracted as skill requirements, violating annotation guidelines.  
   *Mitigation:* Section-aware extraction restricting search to requirements blocks.
2. **Missing Regional Surface Variations (Primary FN Driver):**  
   In `GLOBAL_SYN_000020`, the phrase `"PDP Law compliance"` (Indonesia's Personal Data Protection Law) was labeled by the generator as `Data Privacy Compliance`, but the specific alias was absent from the local taxonomy.  
   *Mitigation:* Semantic vector similarity (Milestone 5).
3. **Syntactic Negation Blindness:**  
   Dictionary methods cannot evaluate governing verb negation (e.g. *"we do not develop in Python"*).  
   *Mitigation:* Dependency parsing (`dep == 'neg'`) in future ML/NER pipelines.

---

## 8. Annotation Preparation & Occupational Coverage Findings

An audit of `data/annotations/human_annotation_template.json` (300 records) revealed:
- **Sampling Skew:** Stratifying by country-industry pair ($20 \times 15 = 300$) created severe occupational imbalance:
  - Over-sampled: `Supply Chain Manager` (11), `Farm Manager` (10), `Procurement Officer` (9).
  - Severely under-sampled: `Data Scientist` (1), `Data Analyst` (2), `BI Analyst` (2).
- **Researcher Workflow:**
  - Blinded annotation protocol: label directly from raw text without generator hints.
  - Strict noun-phrase span boundaries (`text[start:end] == mention`).
  - Separation of human gold standard from generator references.

---

## 9. Week 1 Deliverables Summary

| Deliverable | Location | Status |
| :--- | :--- | :---: |
| Data Quality Report | `data/processed/data_quality_report.md` | Verified (28/28 checks passed) |
| Exploratory Data Analysis | `research/notebooks/03_eda.ipynb` | Executed (11 cells, 0 errors) |
| EDA Visualizations | `outputs/visualizations/eda_01_*.png` to `eda_12_*.png` | Generated (12 publication charts) |
| Dictionary Extractor | `models/skill_extraction/dictionary_extractor.py` | Implemented & Tested |
| spaCy PhraseMatcher | `models/skill_extraction/spacy_extractor.py` | Implemented & Tested |
| Evaluation Suite | `models/skill_extraction/evaluator.py` | Implemented & Tested |
| Baseline Extraction Notebook | `research/notebooks/04_skill_extraction.ipynb` | Executed (6 cells, 0 errors) |
| Final Research Report | `outputs/reports/skill_extraction_evaluation.md` | Documented & Verified |

---

## 10. Week 1 Boundaries & Week 2 Resolution Status

The following Week 1 handoff items have been updated to reflect Week 2 completion and remaining Week 3/4 scope:
1. **Skill Normalization (`Week 2 Completed`):** Built `SkillNormalizer` in [dictionary_extractor.py](file:///D:/Downloads/SkillGraph/models/skill_extraction/dictionary_extractor.py) with strict contextual disambiguation and opt-in regional compliance mapping across all 230 canonical skills (`SG-SK-0001`..`SG-SK-0230`), audited in [05_skill_normalization.ipynb](file:///D:/Downloads/SkillGraph/research/notebooks/05_skill_normalization.ipynb). International ESCO/O\*NET URI crosswalks and embedding-based alias matching remain queued for Week 3+.
2. **Knowledge Graph Preparation (`Week 2 Completed`):** Built relational schema and pure-Python adjacency/graph builder (`SkillGraphBuilder`) in [build_graph.py](file:///D:/Downloads/SkillGraph/scripts/build_graph.py), documented in [schema.md](file:///D:/Downloads/SkillGraph/docs/database/schema.md) and verified in [06_knowledge_graph.ipynb](file:///D:/Downloads/SkillGraph/research/notebooks/06_knowledge_graph.ipynb) (`2,336` active nodes, `31,503` active edges, `0` anomalies). PostgreSQL DDL deployment and NetworkX/Neo4j runtime ingestion remain queued for Week 3.
3. **Recommendation, Independent Human Study & Platform (`Week 3/4 Queued`):** Independent human review of `data/annotations/human_annotation_sample_50.json` (`50` records, `530` candidate spans), dependency-based negation filtering (`dep == 'neg'`), FastAPI backend, personalized learning-path algorithms, and frontend delivery.

---

# SkillGraph — Week 2 Technical Research & Engineering Report (Modules 1–9)

**Milestone:** Week 2 Autonomous Execution — Offset-Preserving Preprocessing, Extractor Remediation, Multi-Tier Normalization, 9-Category Error Audit, Scalability Benchmarking, & Knowledge Graph Readiness  
**Primary Benchmark Artifact:** `outputs/reports/skill_extraction_benchmark_report.json` (`EXP-02-WEEK2-EXTRACTION-NORMALIZATION-BENCHMARK`)  
**Ground-Truth Integrity Status:** `synthetic_job_postings_ground_truth.json` (`2,000` docs, `21,714` spans, `human_verified = false`) and `generate_synthetic_benchmark.py` (`22` docs, `156` skill targets, `human_verified = false`) are **synthetic generator reference benchmarks**. `data/annotations/human_annotation_sample_50.json` (`50` records, `530` candidate spans) remains **`PENDING` independent human confirmation** and is excluded from all P/R/F1 scoring.

---

## 11. Pipeline Audit & Three Remediated Algorithmic Defects

Re-auditing the Week 1 extractors across the full 2,000-posting synthetic corpus uncovered three concrete defects that lowered precision, recall, and standalone normalization correctness:

| Week 1 Defect | Affected Scope | Root Cause | Week 2 Remediation | Verified Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Title-case `'Siem'` false positives** | 28 FPs across 2,000 docs (`"Siem Reap, Cambodia"`) | `len <= 3` threshold failed to enforce strict uppercase matching on short acronyms (`SIEM`) | Enforced strict uppercase `ORTH` token rules and `AMBIGUOUS_SHORT_TOKENS` handling in both extractors | Eliminated all 28 `'Siem'` FPs (`1,639 -> 1,611` Strict Full-Text FPs) |
| **Sentence-final period tokenization miss** | 1 FN in `GLOBAL_SYN_001440` (`"LTE and 5G."`) | `en_core_web_sm` attached `.` to `5G.`, breaking `PhraseMatcher` token sequence `['LTE', 'and', '5', 'G']` | Added `suffix_search` rule `r"(?<=[A-Za-z0-9])\."` in `SpacySkillExtractor` while keeping `.js`/`.net` intact | Restored exact recall parity (`TP: 21,562 -> 21,563`, `FN: 152 -> 151`, `0` Dict vs. spaCy divergences) |
| **Standalone `SkillNormalizer` context regex & throughput bottleneck** | Standalone phrase probes & 2,000-doc runtime | Standalone normalizer lacked `is_valid_skill_context` regex boundaries; full `nlp(text)` pipeline ran slow on batch extraction | Fixed standalone regex context filtering in `SkillNormalizer`; added `use_fast_tokenizer=True` (`self.nlp.make_doc`) to `SpacySkillExtractor` | Probes `'Vitamin C'`/`'R&D'`/`'Go'` (idiomatic) resolve cleanly; `0.5728s` vs `30.1297s` on 2,000 docs (~52.6x speedup, `1.0` Jaccard parity) |

---

## 12. Module 2 — Offset-Preserving Text Preprocessing

Implemented `TextPreprocessor` in `scripts/preprocess_data.py` (`OFFSET_PRESERVING_TRANSLATION`, `normalize_preserving_offsets`, `normalize_compact`, `recover_original_span`, `segment_job_sections` using `SECTION_CUES`, `get_section_for_span`, and `verify_offset_alignment`) and executed validation in `research/notebooks/02_data_cleaning.ipynb`:

- **1:1 Character-Length Preservation:** `2,000 / 2,000` postings in `data/processed/global_jobs_cleaned.csv`, `50 / 50` records in `data/annotations/human_annotation_sample_50.json`, and `22 / 22` postings in `scripts/generate_synthetic_benchmark.py` maintain exact 1:1 length parity under `normalize_preserving_offsets`.
- **Span Offset Integrity:** `21,714 / 21,714` synthetic generator spans and `530 / 530` unconfirmed candidate spans align character-for-character across raw and normalized text.
- **Deterministic Section Segmentation:** Every posting (`2,000 / 2,000`) segmented cleanly into `intro_or_title`, `responsibilities`, `required`, and `preferred`. Ground-truth reference spans reside exclusively in qualification blocks (`required`: `15,852`, `preferred`: `5,862`).

---

## 13. Modules 3 & 4 — Multi-Tier Skill Normalization & Extractor Parity

`SkillNormalizer` in `models/skill_extraction/dictionary_extractor.py` maps surface forms against all `230` canonical skills (`18` categories, `8` skill types) in `data/processed/global_skill_taxonomy.csv` using `normalize_phrase_casing_rule`, `is_valid_skill_context`, and opt-in `COUNTRY_SPECIFIC_BROAD_SURFACE_MAP` (`SG-SK-0048 Data Privacy Compliance`, `SG-SK-0058 Local Accounting Standards`):

- **Edge-Case Disambiguation Probes (`research/notebooks/05_skill_normalization.ipynb`):**
  - `'Go'` (technical context) -> `MAPPED`
  - `'Go'` (idiomatic movement context) -> `UNRESOLVED_CANONICAL`
  - `'Vitamin C'`, `'R&D'` -> `UNRESOLVED_CANONICAL`
- **Full 2,000-Doc Mention Normalization (`21,714` spans):**
  - **Strict Mode (`include_broad_country_mappings=False`):** `exact_match = 7,707`, `alias_match = 13,856`, `unresolved_canonical = 151`.
  - **Broad Opt-In Mode (`include_broad_country_mappings=True`):** All `151` regional law and accounting mentions resolve as `country_specific_broad_match` (`0` unresolved).
- **Unconfirmed 50-Record Candidate Pool (`530` spans, `PENDING` human review):** `525 MAPPED`, `5 UNRESOLVED_CANONICAL` (preserving the 4 unresolved country-specific legal/regulatory mappings plus 1 broad phrase).
- **Cross-Architecture Agreement:** Across all 2,000 documents, `DictionarySkillExtractor`, `SpacySkillExtractor (use_fast_tokenizer=True)`, and `SpacySkillExtractor (use_fast_tokenizer=False)` achieved **100% span-level parity** (`23,174` shared strict spans, `23,300` shared broad spans, `0` divergences, Jaccard = `1.0000`).

---

## 14. Module 5 — Week 2 Four-Mode Evaluation & 22-Doc Edge Suite

### 14.1 2,000-Doc Synthetic Reference Benchmark (All Modes Show Exact Dict/spaCy Parity)

| Evaluation Mode | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | TP | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict Full-Text** (`Dict` / `spaCy Fast` / `spaCy Full`) | 0.930482 | 0.993046 | 0.960747 | 0.941030 | 0.993024 | 0.963771 | 21,563 | 1,611 | 151 |
| **Broad Country Full-Text** (`Dict` / `spaCy`) | 0.931931 | 1.000000 | 0.964767 | 0.942333 | 1.000000 | 0.967927 | 21,714 | 1,586 | 0 |
| **Strict Qualification Section-Filtered (`required` + `preferred`)** | 0.998842 | 0.993046 | 0.995936 | 0.998908 | 0.993024 | 0.995817 | 21,563 | 25 | 151 |
| **Broad Country + Qualification Section-Filtered** | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 21,714 | 0 | 0 |

In Strict Full-Text mode, `relaxed_f1` and `boundary_only_f1` both equal `0.960747` (`0.964767` in Broad Full-Text, `0.995936` in Strict Section-Filtered, `1.000000` in Broad + Section-Filtered).

> **Circularity Warning:** The `1.000000` F1 score under Broad + Section-Filtered mode validates deterministic alignment between the regex/spaCy extractors, section segmenter, and synthetic generator templates—it **must not** be interpreted as expected real-world accuracy on noisy external postings.

### 14.2 22-Doc Technical Edge-Case Suite (`generate_synthetic_benchmark.py`, 31 skills, 156 targets)

| Extraction Method | Micro Precision | Micro Recall | Micro F1 | TP | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DictionarySkillExtractor` (Week 2)** | 0.987342 | 1.000000 | 0.993631 | 156 | 2 | 0 |
| **`SpacySkillExtractor` Fast (Week 2)** | 0.987342 | 1.000000 | 0.993631 | 156 | 2 | 0 |

Improvements vs. Week 1 removed the regex C/symbol boundary FP (`FP: 3 -> 2`). The remaining 2 FPs occur identically in `SYN_JOB_003` (`"Python"` and `"Java"` in `"do not develop in Python or Java"`), reflecting intentional dictionary negation sensitivity queued for Milestone 3 dependency parsing (`dep == 'neg'`).

---

## 15. Nine-Category Error Analysis Across 2,000 Documents

`SkillExtractionEvaluator.classify_extraction_errors` in `models/skill_extraction/evaluator.py` decomposes every error across 9 diagnostic categories (note: non-exclusive structural tags such as `contextual_interpretation_error` and `incorrect_boundary` intentionally co-tag root causes alongside primary `FP`/`FN` counts):

| Error Category | Strict Full-Text | Strict Section-Filtered | Broad + Section-Filtered | Root-Cause Sub-Categories & Observed Corpus Examples |
| :--- | :---: | :---: | :---: | :--- |
| `false_positive` | **1,611** | **25** | **0** | `title_or_department_bleed` (728), `responsibility_duty_bleed` (858), `unannotated_lexical_match` (25). E.g. `GLOBAL_SYN_000004` `[93:104]` `'Procurement'` (`intro_or_title`), `[284:295]` `'procurement'` (`responsibilities`); `GLOBAL_SYN_000010` `[76:92]` `'Customer Support'`; `GLOBAL_SYN_000012` `[328:348]` `'infection prevention'`. |
| `false_negative` | **151** | **151** | **0** | `country_specific_regulatory_or_accounting_surface` (151). E.g. `GLOBAL_SYN_000020` `[1016:1034]` `'PDP Law compliance'`, `GLOBAL_SYN_000035` `[1000:1015]` `'LGPD compliance'`, `GLOBAL_SYN_000045` `[836:842]` `'K-IFRS'`, `GLOBAL_SYN_000049` `[955:978]` `'Brazilian CPC standards'`. |
| `incorrect_boundary` | **25** | **25** | **0** | `partial_span_boundary_mismatch` (25): matching embedded nationality/sub-phrases inside unmapped local standards. E.g. `GLOBAL_SYN_000178` pred `[826:836]` `'Vietnamese'` vs. gt `[826:863]` `'Vietnamese Accounting Standards (VAS)'`; `GLOBAL_SYN_000950` pred `[948:952]` `'Thai'` + `[953:972]` `'Financial Reporting'` vs. gt `[948:989]` `'Thai Financial Reporting Standards (TFRS)'`. |
| `duplicate_match` | **0** | **0** | **0** | Eliminated by deterministic `_resolve_overlaps`. |
| `overlapping_match` | **0** | **0** | **0** | Eliminated by descending-length overlap resolution. |
| `incorrect_normalization` | **0** | **0** | **0** | Zero mis-mapped canonical IDs among matched spans. |
| `ambiguous_skill` | **59** | **52** | **0** | `missed_ambiguous_or_acronym_variant` (52, e.g. `GLOBAL_SYN_000086` `'IFRS'`, `GLOBAL_SYN_000137` `'CIFRS'`, `GLOBAL_SYN_000142` `'IFRS and ASPE'`), `polysemous_short_or_domain_term` (7 in non-qualification sections, e.g. `GLOBAL_SYN_000069` `[500:503]` `'SQL'` in `responsibilities`). |
| `tokenization_error` | **26** | **26** | **0** | `sentence_final_or_symbol_tokenization_mismatch` (26): trailing period next to unmapped country standards (`GLOBAL_SYN_000035` `'LGPD compliance.'`, `GLOBAL_SYN_000181` `'UAE PDPL compliance.'`, `GLOBAL_SYN_000565` `'CIFRS.'`). |
| `contextual_interpretation_error` | **1,586** | **0** | **0** | `title_or_department_non_qualification_context` (728), `duty_clause_non_qualification_context` (858) — eliminated 100% by qualification section filtering. |

---

## 16. Module 6 — Automated Regression Testing (`scripts/test_extraction_pipeline.py`)

`scripts/test_extraction_pipeline.py` implements `TestSkillExtractionPipeline` (27 deterministic tests via Python stdlib `unittest`), covering offset-preserving normalization, compact offset recovery, section cue classification, `'Siem Reap'` suppression, `'LTE and 5G.'` punctuation attachment, `'C'`, `'C++'`, `'R'`, `'Go'`, `'Vitamin C'`, `'R&D'`, strict vs. broad normalization modes, 9-category error classification, and knowledge graph referential integrity (`pytest` is not installed in `.venv`).

---

## 17. Module 7 — Scalability & Memory Benchmarks

Uninstrumented wall-clock timings and isolated 200-doc `tracemalloc` steady-state memory measurements (`outputs/reports/skill_extraction_benchmark_report.json`):

| Corpus | Extractor Configuration | Init (s) | Extract (s) | Throughput (docs/s) | Throughput (chars/s) | Mean ms/doc | Steady-State Peak Traced Mem (KB / 200-doc batch) | Process RSS Delta (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2,000 docs** (`2,261,639` chars, `23,174` spans) | `DictionarySkillExtractor` | 0.0984 | 16.4355 | 121.69 | 137,607 | 8.218 | 107.1 | 16.11 |
| **2,000 docs** (`2,261,639` chars, `23,174` spans) | `SpacySkillExtractor` (`use_fast_tokenizer=True`) | 0.4113 | 0.5728 | 3,491.79 | 3,948,581 | 0.286 | 174.1 | 60.44 |
| **2,000 docs** (`2,261,639` chars, `23,174` spans) | `SpacySkillExtractor` (`use_fast_tokenizer=False`) | 0.3301 | 30.1297 | 66.38 | 75,063 | 15.065 | 4,179.9 | 57.92 |
| **22 tech docs** (`6,203` chars, `173` spans) | `DictionarySkillExtractor` | 0.0126 | 0.0093 | 2,362.14 | 666,015 | 0.423 | 17.5 | 0.21 |
| **22 tech docs** (`6,203` chars, `173` spans) | `SpacySkillExtractor` (`use_fast_tokenizer=True`) | 0.2785 | 0.0082 | 2,696.24 | 760,218 | 0.371 | 32.1 | 37.58 |
| **22 tech docs** (`6,203` chars, `173` spans) | `SpacySkillExtractor` (`use_fast_tokenizer=False`) | 0.2939 | 0.1565 | 140.53 | 39,623 | 7.116 | 1,433.6 | 40.29 |

*Note:* In `research/notebooks/04_skill_extraction.ipynb` Section 11 (200-doc benchmark where `tracemalloc` wraps initialization + batch execution together), total peak traced allocations measured `1,559.1 KB` (Dictionary regex, `97.6 docs/s`), `49,288.9 KB` (spaCy Fast including model/matcher init, `689.4 docs/s`), and `49,163.2 KB` (spaCy Full, `30.9 docs/s`). Recommendation: use `SpacySkillExtractor(use_fast_tokenizer=True)` in production ingestion (`3,491.79 docs/s`, `174.1 KB` steady-state peak batch overhead, `1.0000` Jaccard agreement).

---

## 18. Module 8 — Knowledge Graph Construction & Schema Readiness

Implemented `SkillGraphBuilder` in [build_graph.py](file:///D:/Downloads/SkillGraph/scripts/build_graph.py), documented schema & pipeline contracts in [schema.md](file:///D:/Downloads/SkillGraph/docs/database/schema.md) and [data_pipeline.md](file:///D:/Downloads/SkillGraph/docs/architecture/data_pipeline.md), and validated analytical explainability queries in [06_knowledge_graph.ipynb](file:///D:/Downloads/SkillGraph/research/notebooks/06_knowledge_graph.ipynb):

- **Verified Active Graph Scale:**
  - **2,336 Active Nodes (6 types):** `Skill` (`230`), `SkillCategory` (`18`), `Occupation` (`53`), `Industry` (`15`), `Country` (`20`), `JobPosting` (`2,000`). *(Note: `User` / `CurriculumCourse` / `LearningResource` schema nodes are defined in PostgreSQL DDL for Week 3+ without synthetic learner fabrication.)*
  - **31,503 Active Edges (8 types):** `REQUIRES_SKILL` (`15,819`), `PREFERS_SKILL` (`5,769`), `BELONGS_TO_CATEGORY` (`230`), `IN_OCCUPATION` (`2,000`), `IN_INDUSTRY` (`53`), `LOCATED_IN` (`2,000`), `OCCUPATION_DEMANDS_SKILL` (`1,645`), `CO_OCCURS_WITH` (`3,987`).
- **Referential Integrity & Dependencies:** `validate_graph()` returned `True` (`anomalies = 0`). Because `networkx` is not installed in `.venv` (`networkx_installed: False`), `SkillGraphBuilder` defaults to pure-Python adjacency dicts (`to_adjacency_dict()`) and normalized relational tables (`to_relational_tables()`) while exposing `to_networkx()` when `networkx` is installed.

---

## 19. Module 9 — Environment Limitations & Next-Milestone Roadmap

- **Optional Packages Absent in `.venv`:** Neither `pytest` nor `networkx` is installed in the offline `.venv`; tests use Python stdlib `unittest` (`scripts/test_extraction_pipeline.py`) and graph algorithms operate natively on deterministic adjacency dictionaries and DataFrames. Similarly, `fastapi`, `uvicorn`, `psycopg2`, `psycopg`, `asyncpg`, and `sqlalchemy` are absent from `.venv` and documented cleanly in [backend/requirements.txt](file:///D:/Downloads/SkillGraph/backend/requirements.txt).
- **Independent Human Evaluation Protocol (`Awaiting human validation`):** Complete manual review of [human_annotation_sample_50.json](file:///D:/Downloads/SkillGraph/data/annotations/human_annotation_sample_50.json) (`50` records, `530` candidate spans, currently `PENDING`), collect real-world job postings into `data/raw/`, and ingest real university course syllabi into `university_majors`, `university_courses`, and `course_skills`.

---

## 20. Week 3 Module 1 — Dependency-Based Negation & Modality Evaluation (`Implemented`, `Tested`, `Experimentally observed`)

All results below were generated by executing [07_recommendation_evaluation.ipynb](file:///D:/Downloads/SkillGraph/research/notebooks/07_recommendation_evaluation.ipynb) and recorded in the `"week3_extensions"` section of [skill_extraction_benchmark_report.json](file:///D:/Downloads/SkillGraph/outputs/reports/skill_extraction_benchmark_report.json).

### 20.1 Eight Negation & Modality Diagnostic Cases + `SYN_JOB_003` (`11/11` Assertions)
- **Evaluators:** `SpacySkillExtractor` (`spacy_dependency_parser` over `en_core_web_sm` dependency arcs) and `DictionarySkillExtractor` (`clause_scoped_regex`).
- **Measured Accuracy:** Both extractors passed **`11/11` assertions (`1.0000` accuracy)** across `"Must have Python experience"` (`required`), `"Python is preferred"` (`preferred`), `"Python is not required"` (`negated_requirement`), `"Do not use Python"` (`excluded_or_prohibited`), `"Experience with Python is unnecessary"` (`negated_requirement`), `"Knowledge of Python and Java is required"` (`required` for both), `"Python is not required, but SQL is essential"` (`Python=negated_requirement`, `SQL=required`), and `"Note that we do not develop in Python or Java for this position."` (`SYN_JOB_003`, `Python=excluded_or_prohibited`, `Java=excluded_or_prohibited`).
- **Non-Taxonomy Prohibited Clause:** `"Candidates must avoid using deprecated technologies."` captured `target_phrase="deprecated technologies"` under `excluded_or_prohibited` with `skill_id=None` (no hallucinated taxonomy ID).
- **Parser-Unavailable Fallback:** Disabling `has_dependency_parser` triggered `conservative_regex_fallback_parser_unavailable` and correctly isolated `"Python"` (negated) from `"SQL"` (required).

### 20.2 22-Document Technical Benchmark Before vs. After Negation Filtering

| Configuration | Mode | TP | FP | FN | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Week 2 Baseline (No negation exclusion / `include_negated=True`) | Raw spans | 156 | 2 | 0 | 0.987342 | 1.000000 | 0.993631 |
| **Week 3 `DictionarySkillExtractor`** | `include_negated=False` | **156** | **0** | **0** | **1.000000** | **1.000000** | **1.000000** |
| **Week 3 `SpacySkillExtractor`** | `include_negated=False` | **156** | **0** | **0** | **1.000000** | **1.000000** | **1.000000** |

### 20.3 2,000-Document Synthetic Corpus Modality Distribution
- **`SpacySkillExtractor` strict exact-span extraction (`include_negated=False`, `1.1647s`):** `TP = 21,563`, `FP = 1,611`, `FN = 151`, **`Precision = 0.930482`, `Recall = 0.993046`, `F1 = 0.960747`** (Week 2 strict benchmark preserved cleanly).
- **Corpus Modality Breakdown (`include_negated=True`):** `required: 15,819`, `preferred: 5,769`, `mentioned: 858`, `contextually_irrelevant: 728` (zero real negated skill clauses exist in the synthetic generator's 2,000 templates beyond `"no prior experience"` boilerplate).

---

## 21. Week 3 Modules 2–5 — Skill Gap Analysis, Worked Calculation & Weight Sensitivity (`Implemented`, `Tested`, `Experimentally observed`)

### 21.1 Synthetic Student Profile Benchmarks
| Profile | Target Occupation | Required Coverage | Overall Weighted Coverage | Skill Gap Score | Possessed | Partial | Missing Req | Missing Pref | Unrecognized |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `empty_ds` | Data Scientist | 0.0000 | 0.0000 | 1.0000 | 0 | 0 | 22 | 7 | 0 |
| `partial_da` | Data Analyst | 0.1190 | 0.1806 | 0.8194 | 2 | 1 | 18 | 5 | 0 |
| `edge_bi` | BI Analyst | 0.0208 | 0.0836 | 0.9164 | 1 | 1 | 23 | 4 | 1 |
| `complete_da` | Data Analyst | 1.0000 | 1.0000 | 0.0000 | 26 | 0 | 0 | 0 | 0 |

### 21.2 Worked Mathematical Calculation (`SYNTHETIC_STUDENT_001 -> Data Analyst`, `alpha=0.70, beta=0.30, gamma_eff=0.0, delta_eff=0.0`)
| Rank | Skill | Req Type | Gap Factor | Req Share | Pref Share | `G(s)` | Raw Ext Opp | Max Norm | `O(s)` | `P(s)` | Tier |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | English Language | required | 1.0 | 0.9375 | 0.0 | 0.9375 | 48.9921 | 48.9921 | 1.0000 | **0.9563** | Critical / High |
| 2 | Data Analysis | required | 1.0 | 1.0000 | 0.0 | 1.0000 | 7.8907 | 48.9921 | 0.1611 | **0.7483** | Critical / High |
| 3 | Tableau | required | 1.0 | 0.6250 | 0.0 | 0.6250 | 0.9118 | 48.9921 | 0.0186 | **0.4431** | Medium |
| 4 | Statistics | required | 1.0 | 0.5625 | 0.0 | 0.5625 | 5.2964 | 48.9921 | 0.1081 | **0.4262** | Medium |
| 5 | Data Visualization | required | 1.0 | 0.5625 | 0.0 | 0.5625 | 2.2826 | 48.9921 | 0.0466 | **0.4077** | Medium |
| 6 | SQL (`beginner=0.35`) | required | 0.5 | 1.0000 | 0.0 | 0.5000 | 4.7960 | 48.9921 | 0.0979 | **0.3794** | Medium |

### 21.3 Weight Sensitivity Across `BASELINE_WEIGHT_PRESETS` (Top-5 vs Baseline Heuristic)
- **`target_only` `(alpha=1.0, beta=0.0)`:** Top-5 = `[Data Analysis, English Language, Tableau, Statistics, Data Visualization]` (`Jaccard = 1.0000`).
- **`baseline_balanced_heuristic` `(alpha=0.7, beta=0.3)`:** Top-5 = `[English Language, Data Analysis, Tableau, Statistics, Data Visualization]` (`Jaccard = 1.0000`).
- **`equal_gap_opportunity` `(alpha=0.5, beta=0.5)`:** Top-5 = `[English Language, Data Analysis, Statistics, Tableau, Data Visualization]` (`Jaccard = 1.0000`, `Statistics` rises above `Tableau`).
- **`career_mobility_first` `(alpha=0.3, beta=0.7)`:** Top-5 = `[English Language, Data Analysis, Attention to Detail, Statistics, Microsoft Excel]` (`Jaccard = 0.4286`, general cross-industry skills displace role-specific BI tools).

---

## 22. Week 3 Module 4 & 6 — Multi-Occupation Opportunity & Learning Path (`Implemented`, `Tested`)

| Skill | Skill ID | Required Occs | Preferred-Only Occs | Total Occs | Raw Ext Opp | Normalized `O(s)` | Top Co-Occurring Bridges (Non-Prerequisite) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Python | `SG-SK-0001` | 5 | 8 | 13 | 5.7866 | 0.1159 | English Language, Statistics, Data Analysis |
| SQL | `SG-SK-0006` | 6 | 4 | 10 | 5.7960 | 0.1161 | English Language, Data Analysis, Python |
| Statistics | `SG-SK-0024` | 8 | 1 | 9 | 5.8589 | 0.1173 | English Language, Data Analysis, Report Writing |
| Power BI | `SG-SK-0022` | 3 | 4 | 7 | 3.2736 | 0.0656 | English Language, Data Analysis, Microsoft Excel |
| Machine Learning | `SG-SK-0027` | 2 | 0 | 2 | 2.0000 | 0.0401 | Python, English Language, Apache Spark |

- **Learning Path Status (`SYNTHETIC_STUDENT_001 -> Data Analyst`):** Strategy = `transparent_priority_and_category_heuristic` (`Stage 1: 2`, `Stage 2: 20`, `Stage 3: 2`). `prerequisite_graph_available = False` (`UNAVAILABLE_IN_LOCAL_TAXONOMY`), `learning_resources_available = False` (`UNAVAILABLE_NO_VERIFIED_COURSE_CATALOG`). No courses, hours, or fake prerequisite edges are fabricated.

---

## 23. Week 3 Modules 7–9 — Relational DB, FastAPI/ASGI Layer & Verification Summary (`Implemented`, `Tested`)

- **Relational DB (`RelationalDatabaseManager`, `23` PostgreSQL DDL statements -> `17` tables):** Reference tables (`18` categories, `230` skills, `325` aliases, `15` industries, `53` occupations, `20` countries, `2,000` postings, `21,588` mentions, `1,645` occupation demands, `3,987` co-occurrences) plus `university_majors` (`0`), `university_courses` (`0`), `course_skills` (`0`) verified in SQLite (`PRAGMA foreign_keys = ON`). Host PostgreSQL 17 service active on `127.0.0.1:5432`, awaiting `.env` credentials and Python DB driver installation (`live_pg_execution_ready = False`).
- **REST API (`backend/app/main.py`):** OpenAPI `3.1.0` specification exposes `8` paths and `13` schemas (`200` on `/health`, `/occupations`, `/skills`, `/occupations/{id}/skills`, `POST /skill-gap/analyze`, `/skill-gap/{id}`, `/recommendations/{id}`, `/recommendations/{id}/explanations`; `400`/`404`/`422` enforced cleanly).
- **Automated Regression Suite (`scripts/test_extraction_pipeline.py`):** Extended from `27` (Week 2) to `43` deterministic unit and integration tests covering all Week 3 modules.

