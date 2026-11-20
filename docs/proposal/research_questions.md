# SkillGraph — Research Questions, Hypotheses & Evaluation Framework

**Project:** *SkillGraph: An Explainable Knowledge Graph-Based Framework for Career-Oriented Skill Gap Analysis and Learning Path Recommendation*

---

## 1. Primary Research Question

> **RQ0:** How can a knowledge graph integrate university curricula and labor-market skill requirements to identify student skill gaps and prioritize skills that maximize career opportunities?

---

## 2. Sub-Research Questions & Testable Hypotheses

### RQ1 — Accurate & Context-Aware Skill Extraction
> How accurately can an NLP pipeline extract and normalize canonical labor-market and syllabus skills while distinguishing required, preferred, and negated/prohibited mentions?

- **Hypothesis ($H_1$):** Adding dependency-based negation and clause modality classification (`required`, `preferred`, `negated_requirement`, `excluded_or_prohibited`) reduces false positives caused by negative constraints (e.g., *"we do not develop in Python or Java"*) compared to surface keyword matching alone.
- **Evaluation Metrics:** Strict span-level Precision, Recall, $F_1$, boundary-only $F_1$, 9-category error counts, and human inter-annotator agreement (Cohen's $\kappa$) on [human_annotation_sample_50.json](file:///D:/Downloads/SkillGraph/data/annotations/human_annotation_sample_50.json) (`PENDING` manual review) and real postings ([real_job_postings.csv](file:///D:/Downloads/SkillGraph/data/raw/real_job_postings.csv)).

### RQ2 — Curriculum-to-Market Alignment via Knowledge Graph Modeling
> How can an explainable bipartite/multi-relational Knowledge Graph link university courses (`university_courses` -> `course_skills`) to target occupations (`occupation_skill_demand`) without conflating statistical market co-occurrence with pedagogical prerequisites?

- **Hypothesis ($H_2$):** Separating symmetric `CO_OCCURS_WITH` edges (Jaccard, PMI) from explicit syllabus-derived `PREREQUISITE_OF` edges prevents invalid prerequisite claims while enabling transparent major-vs-occupation coverage audits (`analyze_major_vs_occupation`).
- **Evaluation Metrics:** Referential integrity (`0` anomalies), DAG acyclicity, major-to-occupation coverage ratios, and explicit provenance coverage (`source_url`, `collected_at`).

### RQ3 — Multi-Occupation Priority Scoring & Domain Calibration
> How does combining target-occupation gap importance $G(s)$, external cross-occupation career opportunity $O(s)$, and verified prerequisite readiness $D(s)$ affect skill recommendation rankings across weight presets and domain-vs-language filters?

- **Hypothesis ($H_3$):** Weighting external career portability $O(s)$ shifts recommendations toward transferable cross-role skills (quantified by Top-$K$ Jaccard shifts across `BASELINE_WEIGHT_PRESETS`), while `exclude_skill_types=["Language"]` prevents ubiquitous spoken-language requirements from crowding out core technical/domain gaps.
- **Evaluation Metrics:** Weight-sensitivity Top-$K$ Jaccard similarity, rank-shift ablations with/without $\gamma D(s)$ and `exclude_skill_types`.

### RQ4 — Explainability & Evidence Transparency
> Can every skill recommendation be accompanied by deterministic, auditable evidence linking posting counts, external occupation reach, mathematical score components, and explicit unavailability notices when prerequisite or course data is absent?

- **Hypothesis ($H_4$):** A 7-part structured explanation schema (`StructuredExplanationModel`) guarantees full numerical traceability between `priority_score` and underlying graph counts while preventing hallucinated competence or fabricated course recommendations.
- **Evaluation Metrics:** Structural & numerical consistency checks (`100%` match between explanation components and priority formula) and endpoint contract validation (`200`/`400`/`404`/`422`).
