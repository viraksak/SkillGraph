# SkillGraph Repository Structure

```
SkillGraph/
│
├── README.md                                  # Project overview and high-level architecture
├── LICENSE                                    # Open source licensing terms
├── CONTRIBUTING.md                            # Contribution guidelines and research standards
├── .gitignore                                 # Git ignore rules (.venv, caches, checkpoints)
├── .env.example                               # Template for local environment variables
├── docker-compose.yml                         # Container orchestration configuration
├── PROJECT_STRUCTURE.md                       # Comprehensive directory and file tree documentation
│
├── .github/                                   # GitHub community and automation configurations
│   ├── PULL_REQUEST_TEMPLATE.md               # Standard pull request checklist
│   ├── ISSUE_TEMPLATE/                        # Issue templates for community & research tasks
│   │   ├── feature_request.md                 # Template for proposing new capabilities
│   │   ├── bug_report.md                      # Template for bug tracking
│   │   ├── research_task.md                   # Template for research hypotheses & experiments
│   │   └── ds_learning.md                     # Template for data science milestones
│   └── workflows/                             # CI/CD pipelines
│       └── tests.yml                          # Continuous integration test runner
│
├── docs/                                      # Engineering and academic documentation
│   ├── proposal/                              # Research proposal and foundational questions
│   │   ├── project_proposal.md                # Problem statement, methodology, roadmap
│   │   └── research_questions.md              # Formal research hypotheses
│   ├── architecture/                          # Architecture blueprints
│   │   ├── system_architecture.md             # End-to-end multi-tier architecture
│   │   └── data_pipeline.md                   # Data ingestion, transformation, and graph pipelines
│   ├── database/                              # Database designs
│   │   └── schema.md                          # Relational, vector, and graph schema specs
│   ├── api/                                   # Interface specifications
│   │   └── api_documentation.md               # REST and GraphQL API endpoints
│   └── diagrams/                              # Architecture and ER diagrams
│
├── data/                                      # Data governance & storage tier
│   ├── README.md                              # Data catalog, governance & synthetic data disclaimer
│   ├── raw/                                   # Ingested immutable source datasets
│   │   ├── synthetic_job_postings.csv         # Benchmark synthetic job postings (CSV)
│   │   └── synthetic_job_postings.json        # Benchmark synthetic job postings (JSON)
│   ├── annotations/                           # Expert-curated ground truth annotations
│   │   └── synthetic_job_postings_ground_truth.json # Ground truth technical skills per sample
│   ├── processed/                             # Preprocessed text & extraction comparisons
│   │   └── extracted_skills_comparison.csv    # Multi-model extracted skills dataset
│   └── external/                              # External standard taxonomies (ESCO, O*NET)
│
├── research/                                  # Research, experiments, and scholarly papers
│   ├── literature_review/                     # Related literature notes & citations
│   ├── experiments/                           # Experiment tracking configs and metrics
│   ├── evaluation/                            # Benchmark evaluation scripts & results
│   ├── paper/                                 # Research paper drafts (LaTeX / Markdown)
│   └── notebooks/                             # Jupyter research notebooks
│       ├── 01_skill_extraction_baseline.ipynb # Week 1 Lesson 1: Baseline skill extraction (Executed)
│       ├── 02_data_cleaning.ipynb             # Text normalization & corpus cleaning
│       ├── 03_eda.ipynb                       # Exploratory data analysis on skills
│       ├── 04_skill_extraction.ipynb          # Advanced entity extraction & NER
│       ├── 05_skill_normalization.ipynb       # Taxonomic mapping & entity resolution
│       ├── 06_knowledge_graph.ipynb           # Skill ontology & graph construction
│       └── 07_recommendation_evaluation.ipynb # Personalized learning path recommendation
│
├── models/                                    # Model weights and extraction engines
│   ├── skill_extraction/                      # NER and pattern matching models
│   ├── skill_matching/                        # Semantic similarity and embedding matchers
│   ├── knowledge_graph/                       # Graph embeddings (Node2Vec, TransE, GNN)
│   └── recommendation/                        # Graph-based path recommendation algorithms
│
├── backend/                                   # FastAPI backend service
│   ├── requirements.txt                       # Python dependencies for backend & ML
│   ├── app/                                   # Application source package
│   │   ├── main.py                            # FastAPI entry point
│   │   ├── api/                               # Route handlers / controllers
│   │   ├── database/                          # DB session, engine, and migrations
│   │   ├── models/                            # SQLAlchemy ORM models
│   │   ├── schemas/                           # Pydantic request/response schemas
│   │   └── services/                          # Business logic & extraction orchestration
│   └── tests/                                 # Pytest unit and integration test suite
│
├── frontend/                                  # Next.js / React user interface
│   ├── package.json                           # Node.js dependencies and scripts
│   ├── app/                                   # Next.js App Router pages
│   ├── components/                            # Reusable UI component library
│   ├── features/                              # Feature-specific components (e.g. GraphViewer)
│   ├── hooks/                                 # Custom React hooks
│   └── lib/                                   # Utility functions and API clients
│
├── scripts/                                   # Automation and pipeline utility scripts
│   ├── download_data.py                       # Data collection utility
│   ├── preprocess_data.py                     # Data cleaning runner
│   ├── generate_synthetic_benchmark.py        # Lesson 1 synthetic data & annotation generator
│   ├── build_baseline_notebook.py             # Reproducible notebook generator
│   ├── test_extraction_pipeline.py            # Extraction pipeline unit test
│   └── build_graph.py                         # Graph building pipeline script
│
└── outputs/                                   # Research outputs, reports, and visualizations
    ├── reports/                               # Quantitative metrics and evaluation summaries
    │   ├── skill_extraction_benchmark_report.json # Structured JSON evaluation metrics
    │   └── skill_extraction_evaluation.md     # Markdown benchmark comparison report
    ├── visualizations/                        # Generated figures and charts (300 DPI)
    │   ├── skill_frequency_distribution.png   # Corpus-wide skill frequency distribution
    │   └── model_comparison_metrics.png       # Regex vs. spaCy benchmark comparison chart
    └── exports/                               # Serialized data artifacts and taxonomies
        └── skill_taxonomy.json                # Canonical skills taxonomy with aliases
```

