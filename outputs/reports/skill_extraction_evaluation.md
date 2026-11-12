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

## 10. Boundaries & Remaining Scope (Week 2 Handoff)

The following items are strictly out-of-scope for Week 1 and remain queued for subsequent milestones:
1. **Week 2 (Skill Normalization):** Mapping local canonical IDs (`SG-SK-####`) to international standards (ESCO URIs and O*NET Concept IDs) and embedding-based alias matching.
2. **Week 2/3 (Knowledge Graph):** Constructing the Neo4j/NetworkX graph linking occupations, skills, industries, and countries.
3. **Week 3/4 (Recommendation & Platform):** Personalized learning path algorithms, FastAPI backend, Next.js frontend, and academic paper writing.
