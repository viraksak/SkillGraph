# SkillGraph — Revised Execution Proposal (Weeks 4–10)
**Project:** *SkillGraph: An Explainable Knowledge Graph-Based Framework for Career-Oriented Skill Gap Analysis and Learning Path Recommendation*  
**Team:** 2 Members — **Member 01 (`hustsak` / Data Science & NLP Lead)** & **Member 02 (Recommendation & Product Lead)**  
**Repository:** `https://github.com/viraksak/SkillGraph.git`

---

## 1. Why Change the Approach?

Weeks 1–3 already built and verified the core technical engine (`49/49` passing unit & integration tests in [test_extraction_pipeline.py](file:///D:/Downloads/SkillGraph/scripts/test_extraction_pipeline.py)):
- **NLP Extraction & Negation:** `DictionarySkillExtractor`, `SpacySkillExtractor`, `SkillNormalizer`, and clause-level negation filtering (`230` taxonomy skills).
- **Knowledge Graph & Gap Engine:** `2,336` nodes, `31,503` edges, `SkillGapAnalysisEngine`, `PriorityScoringModel`, `MultiOccupationAnalyzer`, and `LearningPathRecommender` in [build_graph.py](file:///D:/Downloads/SkillGraph/scripts/build_graph.py).
- **Curriculum Ingestion Pipeline:** `CurriculumIngestionPipeline` in [ingest_curriculum.py](file:///D:/Downloads/SkillGraph/scripts/ingest_curriculum.py) (provenance validation, prerequisite DAG cycle checks, negation-aware syllabus skill extraction, verified `PREREQUISITE_OF` derivation).
- **Backend & SQL Schema:** `17`-table PostgreSQL DDL + SQLite execution mirror + OpenAPI `3.1` REST service in [backend/app/main.py](file:///D:/Downloads/SkillGraph/backend/app/main.py).

**What is blocking completion is scope size on data collection & annotation — not missing code.** Trying to manually annotate all `530` spans alone in one sitting or scrape thousands of jobs stalls progress.

### The New "Bounded Pilot + Split Ownership" Approach
1. **Replace unbounded data scraping with a Scoped Real-World Pilot:**
   - **1–2 target university majors** (`10–15` real courses with public syllabi & explicit prerequisites).
   - **50–100 real regional job postings** across `3–5` target IT/Data occupations (e.g., Data Analyst, Data Scientist, Data Engineer, Software Developer, ML Engineer) with exact `source_url` and `collected_at`.
2. **Replace overwhelming single-pass annotation with Flagged-First + Shared Agreement Review:**
   - Review the `13` flagged rows + language policy first, then confirm/edit remaining rows in `10`-record daily batches (`5` batches total) using [candidates_review_sheet_all50.csv](file:///D:/Downloads/SkillGraph/data/candidates_review_sheet_all50.csv), plus a **10-record shared overlap** between Member 01 and Member 02 for inter-annotator agreement (Cohen's kappa).
3. **Strict Parallel Lanes via Feature Branches:**
   - Member 01 works exclusively on data, NLP validation, syllabus extraction, and paper evaluation.
   - Member 02 works exclusively on PostgreSQL live cutover, FastAPI/Uvicorn deployment, weight/domain tuning, and the Next.js frontend.

---

## 2. Exact Task Split (Who Builds What)

### Member 01 — Data Science & NLP Lead (`hustsak`)
| Task | Deliverable / File | Target |
| :--- | :--- | :--- |
| **1. Complete Human Reference Sample** | [candidates_review_sheet_all50.csv](file:///D:/Downloads/SkillGraph/data/candidates_review_sheet_all50.csv) -> [human_annotation_sample_50.json](file:///D:/Downloads/SkillGraph/data/annotations/human_annotation_sample_50.json) | `50` records reviewed in 5 batches (`10`/day) |
| **2. Collect Real Job Postings Pilot** | [real_job_postings.csv](file:///D:/Downloads/SkillGraph/data/raw/real_job_postings.csv) | `50–100` real postings with `source_url` + `collected_at` |
| **3. Collect Real University Syllabi** | Curriculum CSVs (`majors.csv`, `courses.csv`) | `1–2` majors, `10–15` real courses + explicit prerequisites |
| **4. Run Real-Data NLP & Curriculum Pipelines** | [preprocess_data.py](file:///D:/Downloads/SkillGraph/scripts/preprocess_data.py) + [ingest_curriculum.py](file:///D:/Downloads/SkillGraph/scripts/ingest_curriculum.py) | Extract `course_skills` & verified `PREREQUISITE_OF` DAG |
| **5. Human-Reference & Real-Data Evaluation** | [skill_extraction_evaluation.md](file:///D:/Downloads/SkillGraph/outputs/reports/skill_extraction_evaluation.md) | Precision, Recall, F1, error analysis, Cohen's kappa |
| **6. Paper Sections (Research Half)** | Research Paper Draft | Methodology, Data Provenance, Taxonomy, NLP Experiments |

### Member 02 — Recommendation & Product Lead (Teammate)
| Task | Deliverable / File | Target |
| :--- | :--- | :--- |
| **1. Co-Annotate Agreement Subset** | `10` shared records (`~100` spans) | Independent agreement check with Member 01 |
| **2. Live PostgreSQL 17 Setup** | Local `.env` + [docker-compose.yml](file:///D:/Downloads/SkillGraph/docker-compose.yml) + [schema.md](file:///D:/Downloads/SkillGraph/docs/database/schema.md) | Execute `get_postgresql_ddl()` (`17` tables) on PG 17 |
| **3. Live FastAPI + Uvicorn Cutover** | [backend/app/main.py](file:///D:/Downloads/SkillGraph/backend/app/main.py) + [backend/requirements.txt](file:///D:/Downloads/SkillGraph/backend/requirements.txt) | Install `fastapi`, `uvicorn`, `psycopg[binary]`; verify Swagger UI |
| **4. Domain-Skill Filtering / Calibration** | [build_graph.py](file:///D:/Downloads/SkillGraph/scripts/build_graph.py) (`PriorityScoringModel`) | Optional language/soft-skill filter so technical roles rank accurately |
| **5. Frontend Web Application** | Next.js + TypeScript + Tailwind + D3.js | Student skill input, gap dashboard, graph view, 7-part explanations |
| **6. Paper Sections (System Half)** | Research Paper Draft + Demo | Architecture, Graph/Priority Model, API/UI Evaluation |

---

## 3. Week-by-Week Finish Plan (Weeks 4–10)

| Week | Member 01 (`hustsak`) | Member 02 (Teammate) | Joint Done Gate |
| :--- | :--- | :--- | :--- |
| **Week 4** | Finish batches 1–3 (`30` records) of human review; collect `10–15` real course syllabi. | Stand up PostgreSQL 17 + `.env` + `uvicorn`; test all `8` API endpoints live. | PG 17 tables created; API live on `localhost:8000`. |
| **Week 5** | Finish batches 4–5 (`20` records); collect `50–100` real job postings. | Co-annotate `10` shared records; scaffold Next.js frontend + `/occupations` & `/skills` pages. | All `50` human records reviewed; real data in `data/raw/`. |
| **Week 6** | Run `ingest_curriculum.py` & `preprocess_data.py` on real data; compute human-reference F1 & kappa. | Connect frontend skill-gap form to `POST /skill-gap/analyze` & `/explanations`. | Real `course_skills` & verified `D(s)` prerequisites loaded. |
| **Week 7** | Error analysis on real postings & human sample; validate taxonomy coverage gaps. | Add D3.js skill-gap & career-opportunity visualization; add domain-skill filter toggle. | End-to-end UI works on real curriculum + real jobs. |
| **Week 8** | Run statistical & ablation experiments (with/without negation, with/without `gamma*D(s)`). | Full product & API integration testing; write deployment/testing report. | Feature freeze; full system test passes. |
| **Week 9** | Write paper sections: Methodology, Data, NLP Evaluation, Limitations. | Write paper sections: Architecture, Priority Model, System Evaluation. | Complete research paper draft reviewed by both. |
| **Week 10** | Rehearse live demo & finalize slides (NLP + evaluation part). | Rehearse live demo & finalize slides (UI + explanation walkthrough). | Final submission, presentation & live demo ready. |

---

## 4. Non-Negotiable Rules to Finish Cleanly

1. **Always branch off `main`:** Member 01 uses `feature/new-approach` / `feature/real-eval`; Member 02 uses `feature/postgres-fastapi-ui`. Never push unreviewed work directly to `main`.
2. **Never mix synthetic and real data:** Keep synthetic `2,000` postings clearly labeled `synthetic`; evaluate real postings and human labels separately.
3. **Never fake prerequisites or courses:** Only feed explicit syllabus prerequisites into `CurriculumIngestionPipeline` so `gamma * D(s)` remains scientifically defensible.
4. **Friday 15-Minute Check-In:** Each member shows runnable command/UI output proving what finished that week.
