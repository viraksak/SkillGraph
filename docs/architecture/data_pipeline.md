# SkillGraph — Week 2 End-to-End NLP & Knowledge Graph Data Pipeline

**Project:** SkillGraph: An Explainable Knowledge Graph-Based Framework for Career-Oriented Skill Gap Analysis and Learning Path Recommendation  
**Status:** Week 2 Implemented & Validated (`scripts/preprocess_data.py`, `models/skill_extraction/`, `scripts/build_graph.py`)

---

## 1. Pipeline Architecture Overview

```
[Raw / Cleaned Job Descriptions]
   │  data/processed/global_jobs_cleaned.csv (2,000 postings)
   │  scripts/generate_synthetic_benchmark.py (22 technical edge-case postings)
   ▼
[Stage 1: Offset-Preserving Preprocessing & Section Segmentation] (scripts/preprocess_data.py)
   │  • TextPreprocessor.normalize_preserving_offsets(): strict 1:1 length invariant
   │  • Protects C++, C#, .NET, Node.js, CI/CD, 4G/5G, SFRS(I)
   │  • TextPreprocessor.segment_job_sections(): intro_or_title | responsibilities | required | preferred
   ▼
[Stage 2: Dual-Track Skill Extraction] (models/skill_extraction/)
   │  ├── Track A: DictionarySkillExtractor (boundary-aware regex + symbol isolation)
   │  └── Track B: SpacySkillExtractor (en_core_web_sm PhraseMatcher + custom period-suffix rule)
   │  • Fixed Week 1 defects: all-uppercase acronym enforcement (SIEM vs Siem Reap),
   │    sentence-final period split (5G., 4G/5G., R., C.), C# sub-token guard, context filters
   ▼
[Stage 3: Multi-Tier Taxonomic Normalization] (SkillNormalizer)
   │  • Tier 1: exact_match (conf 1.00) | Tier 2: alias_match (conf 0.95)
   │  • Tier 3: context_verified_short_token (conf 0.90)
   │  • Tier 4: country_specific_broad_match (conf 0.80, opt-in) vs UNRESOLVED_CANONICAL (conf 0.00)
   ▼
[Stage 4: 9-Category Error Auditing & Agreement Verification] (SkillExtractionEvaluator)
   │  • Strict Exact Span, Relaxed Overlap, Boundary-Only, Extractor Jaccard Agreement
   │  • Automated 9-category error taxonomy classifier + unittest regression suite (27 tests)
   ▼
[Stage 5: Knowledge Graph Preparation & PostgreSQL Relational Export] (scripts/build_graph.py)
   │  • Nodes (2,336 active): Skill (230), SkillCategory (18), Occupation (53), Industry (15),
   │    Country (20), JobPosting (2,000)
   │  • Edges (31,503 active): REQUIRES_SKILL, PREFERS_SKILL, BELONGS_TO_CATEGORY,
   │    IN_OCCUPATION, IN_INDUSTRY, LOCATED_IN, OCCUPATION_DEMANDS_SKILL, CO_OCCURS_WITH
   │  • Extension schemas ready: UniversityMajor, UniversityCourse, Student
```

---

## 2. Stage-by-Stage Technical Specifications

### Stage 1 — Offset-Preserving Preprocessing (`scripts/preprocess_data.py`)
- **Length-Preservation Guarantee:** Non-standard single-character whitespace (`\t`, `\r`, `\u00a0`, `\u200b`) and typographic quotes/dashes (`\u2018`, `\u2019`, `\u201c`, `\u201d`, `\u2013`, `\u2014`) are translated 1-to-1 so `len(normalized_text) == len(original_text)` and `original_text[start:end] == skill_mention` remains strictly invariant.
- **Reversible Compaction:** `normalize_compact()` returns `(compact_text, index_map)` for callers requiring collapsed whitespace alongside exact offset recovery via `recover_original_span()`.
- **Section Segmentation:** `segment_job_sections()` identifies `intro_or_title`, `responsibilities`, `required`, and `preferred` blocks across all 2,000 postings using structural cues (`2,000 / 2,000` postings segmented cleanly; `21,714 / 21,714` synthetic reference spans localized into `required` (15,852) and `preferred` (5,862)).

### Stage 2 — Dual-Track Skill Extraction (`models/skill_extraction/`)
- **All-Uppercase Acronym Enforcement:** Replaced Week 1 `len(phrase) <= 3` rule with full `isupper()` and ambiguous short-token detection (`normalize_phrase_casing_rule()`), preventing Title-case geographic tokens (`"Siem Reap"`) from matching `"SIEM"` (`SG-SK-0044`).
- **Symbol & Context Protection (`is_valid_skill_context`):** Isolates single-letter tokens (`C`, `R`) against neighboring `+`, `#`, `&`, `/`, `-` and filename `.` while allowing sentence-final `"R."` and `"C."`, and filters negative idioms (`"Vitamin C"`, `"C-suite"`, `"R&D"`, `"ready to go"`).
- **Custom spaCy Tokenizer Rule (`SpacySkillExtractor._configure_tokenizer`):** Appends trailing period rule `(?<=[A-Za-z0-9])\.` to `nlp.Defaults.suffixes` so sentence-final `"LTE and 5G."` (`GLOBAL_SYN_001440`), `"4G/5G."`, `"R."`, and `"C."` tokenize identically to `"LTE and 5G"`, while preserving `".NET"` and `"Node.js"`.
- **Fast Tokenizer Mode (`use_fast_tokenizer=True`):** Calls `self.nlp.make_doc(text)` instead of executing unused downstream neural pipes (`tagger`, `parser`, `ner`), achieving **100% span identity** (`Jaccard = 1.0000`) with full pipeline mode at **~26x higher throughput**.

### Stage 2b — Week 3 Dependency-Based Negation & Modality Detection (`Implemented` & `Tested`)
- Classifies every extracted mention into `{required, preferred, mentioned, negated_requirement, excluded_or_prohibited, contextually_irrelevant}` using spaCy dependency-parse governor traversal (`neg` children, coordination conjunct propagation, complement clause governors, `avoid`/`without`/`excluded` cues) in `SpacySkillExtractor`, with conservative clause-scoped regex fallback (`DictionarySkillExtractor` or when `has_dependency_parser=False`).
- Excludes negated and prohibited skills by default (`include_negated=False`), eliminating false-positive extractions on negative constraints (e.g., `SYN_JOB_003` `"Note that we do not develop in Python or Java for this position."`, moving the 22-doc technical benchmark from `F1=0.993631` to `1.000000` with `0` FP / `0` FN) while retaining raw mentions when `include_negated=True` (`analyze_sentences_and_negation()`).

### Stage 3 — Multi-Tier Skill Normalization (`SkillNormalizer`)
- Operates strictly against verified entries loaded from `data/processed/global_skill_taxonomy.csv` (230 canonical skills) or `outputs/exports/skill_taxonomy.json` (31 skills).
- Never fabricates taxonomy IDs for unseen tokens; holds regional statutory privacy laws and local accounting frameworks `UNRESOLVED_CANONICAL` by default unless `include_broad_country_mappings=True` explicitly activates `BROAD_COUNTRY_MAPPING` (confidence `0.80`).

### Stage 4 — Knowledge Graph Preparation (`scripts/build_graph.py`)
- Outputs validated relational tables (`to_relational_tables()`) and pure-Python adjacency exports (`to_adjacency_dict()`, with optional `to_networkx()` when `networkx` is installed).
- Enforces strict semantic distinction between symmetric market co-occurrence (`CO_OCCURS_WITH`, measured by raw co-occurrence count, Jaccard similarity, and PMI) and pedagogical prerequisite relationships (`PREREQUISITE_OF`), which must come from curriculum ontologies rather than job posting text co-occurrence.

### Stage 5 — Week 3 Skill Gap, Priority Scoring, Multi-Occupation Opportunity & Learning Path Pipeline (`Implemented` & `Tested`)
- **`SkillGapAnalysisEngine`:** Evaluates a student skill profile against a resolved target occupation (`OCCUPATION_DEMANDS_SKILL` edges), separating `possessed_skills`, `partially_demonstrated_skills`, `missing_required_skills`, and `missing_preferred_skills`, and distinguishing absent profile records (`not_recorded_in_supplied_profile`) from verified inability.
- **`PriorityScoringModel` & `MultiOccupationAnalyzer`:** Computes bounded `P(s) = clamp((alpha*G(s) + beta*O(s) + gamma_eff*D(s) - delta_eff*C(s)) / (alpha + beta + gamma_eff), 0, 1)` (`alpha=0.70, beta=0.30` baseline heuristic), automatically zeroing `gamma_eff` / `delta_eff` when verified prerequisite graphs or empirical learning costs are absent (`UNAVAILABLE_NO_VERIFIED_PREREQUISITE_GRAPH`, `UNAVAILABLE_NO_EMPIRICAL_COST_DATA`), and runs weight sensitivity across 4 presets.
- **`LearningPathRecommender`:** Sequences skills using `transparent_priority_and_category_heuristic` (`Stage 1`–`Stage 3`) while reporting `UNAVAILABLE_IN_LOCAL_TAXONOMY` and `UNAVAILABLE_NO_VERIFIED_COURSE_CATALOG` rather than synthesizing courses, hours, or fake prerequisites.
- **`RelationalDatabaseManager` & REST API:** Synchronizes graph entities (`2,336` nodes / `31,503` edges) and analysis runs into the `17`-table schema (`sqlite3` mirror with `PRAGMA foreign_keys = ON` + PostgreSQL 17 DDL ready) served via [backend/app/main.py](file:///D:/Downloads/SkillGraph/backend/app/main.py).

---

## 3. Data Governance & Evaluation Disclaimers
- **Synthetic Development Data Only (`Experimentally observed`):** All `2,000` postings in `global_jobs_cleaned.csv`, all `22` postings in `generate_synthetic_benchmark.py`, and all demo student profiles are synthetic. Evaluation scores against generator-produced labels measure deterministic extraction correctness and circular taxonomy coverage—**not** real-world generalization.
- **Unconfirmed Human Candidate Pool (`Awaiting human validation`):** The 50-record sample (`data/annotations/human_annotation_sample_50.json`, 530 spans) remains `PENDING` independent human confirmation and is strictly excluded from benchmark scoring.
- **Absent Institutional Curricula (`Awaiting data`):** `university_majors`, `university_courses`, and `course_skills` remain empty (`0` rows) pending verified university course catalog ingestion.

