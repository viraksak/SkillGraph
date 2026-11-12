"""Generate synthetic benchmark job postings and ground-truth annotations for Lesson 1.

DISCLAIMER:
All generated job postings are strictly synthetic datasets created for NLP
skill extraction benchmarking and do NOT represent actual hiring market demand.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_ANNOTATIONS = ROOT_DIR / "data" / "annotations"
OUTPUTS_EXPORTS = ROOT_DIR / "outputs" / "exports"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_ANNOTATIONS.mkdir(parents=True, exist_ok=True)
OUTPUTS_EXPORTS.mkdir(parents=True, exist_ok=True)

# Canonical Taxonomy with aliases and categories
SKILL_TAXONOMY = {
    "Python": {
        "category": "Programming Language",
        "aliases": ["python", "python3", "python 3", "py"],
    },
    "R": {
        "category": "Programming Language",
        "aliases": ["r"],
    },
    "C": {
        "category": "Programming Language",
        "aliases": ["c"],
    },
    "C++": {
        "category": "Programming Language",
        "aliases": ["c++", "cpp", "c/c++"],
    },
    "Go": {
        "category": "Programming Language",
        "aliases": ["go", "golang"],
    },
    "Java": {
        "category": "Programming Language",
        "aliases": ["java"],
    },
    "JavaScript": {
        "category": "Programming Language",
        "aliases": ["javascript", "js"],
    },
    "TypeScript": {
        "category": "Programming Language",
        "aliases": ["typescript", "ts"],
    },
    "SQL": {
        "category": "Query Language",
        "aliases": ["sql"],
    },
    "Machine Learning": {
        "category": "AI / Data Science",
        "aliases": ["machine learning", "ml"],
    },
    "Deep Learning": {
        "category": "AI / Data Science",
        "aliases": ["deep learning", "dl"],
    },
    "Natural Language Processing": {
        "category": "AI / Data Science",
        "aliases": ["natural language processing", "nlp"],
    },
    "Computer Vision": {
        "category": "AI / Data Science",
        "aliases": ["computer vision", "cv"],
    },
    "PyTorch": {
        "category": "Machine Learning Framework",
        "aliases": ["pytorch", "torch"],
    },
    "TensorFlow": {
        "category": "Machine Learning Framework",
        "aliases": ["tensorflow", "tf"],
    },
    "Scikit-Learn": {
        "category": "Machine Learning Framework",
        "aliases": ["scikit-learn", "scikit learn", "sklearn"],
    },
    "PostgreSQL": {
        "category": "Database",
        "aliases": ["postgresql", "postgres", "psql"],
    },
    "MongoDB": {
        "category": "Database",
        "aliases": ["mongodb", "mongo"],
    },
    "Redis": {
        "category": "Database",
        "aliases": ["redis"],
    },
    "Apache Spark": {
        "category": "Data Engineering",
        "aliases": ["apache spark", "spark", "pyspark"],
    },
    "Apache Kafka": {
        "category": "Data Engineering",
        "aliases": ["apache kafka", "kafka"],
    },
    "Docker": {
        "category": "DevOps & Cloud",
        "aliases": ["docker"],
    },
    "Kubernetes": {
        "category": "DevOps & Cloud",
        "aliases": ["kubernetes", "k8s"],
    },
    "Amazon Web Services": {
        "category": "DevOps & Cloud",
        "aliases": ["amazon web services", "aws"],
    },
    "Google Cloud Platform": {
        "category": "DevOps & Cloud",
        "aliases": ["google cloud platform", "gcp", "google cloud"],
    },
    "CI/CD": {
        "category": "DevOps & Cloud",
        "aliases": ["ci/cd", "ci cd", "continuous integration"],
    },
    "React": {
        "category": "Frontend Framework",
        "aliases": ["react", "reactjs", "react.js"],
    },
    "Node.js": {
        "category": "Backend Framework",
        "aliases": ["node.js", "nodejs", "node"],
    },
    "FastAPI": {
        "category": "Backend Framework",
        "aliases": ["fastapi"],
    },
    "Linux": {
        "category": "Operating Systems",
        "aliases": ["linux"],
    },
    "Git": {
        "category": "Developer Tools",
        "aliases": ["git"],
    },
}

SYNTHETIC_JOBS = [
    {
        "job_id": "SYN_JOB_001",
        "title": "Senior Data Scientist",
        "department": "Analytics & AI",
        "seniority": "Senior",
        "description": "We are seeking a Senior Data Scientist to lead our statistical modeling initiatives. You will build predictive algorithms using Python and R, deploying models with Scikit-Learn and PyTorch. Candidates must be fluent in SQL and PostgreSQL for querying large data warehouses. Experience working with the R&D team and reporting to C-suite executives is valued, but core coding skills in Python and R are required. Familiarity with Git version control is essential.",
        "ground_truth_skills": ["Python", "R", "Scikit-Learn", "PyTorch", "SQL", "PostgreSQL", "Git"],
        "negative_context_notes": "Contains non-skill occurrences: 'R&D', 'C-suite'."
    },
    {
        "job_id": "SYN_JOB_002",
        "title": "Machine Learning Engineer",
        "department": "Applied Machine Learning",
        "seniority": "Mid-Level",
        "description": "Looking for a passionate Machine Learning Engineer. Responsibilities include building Deep Learning pipelines using TensorFlow and PyTorch. You will package models into Docker containers and deploy on Kubernetes (k8s) clusters on Amazon Web Services (AWS). Strong experience with Python3, CI/CD automation, and Git is mandatory.",
        "ground_truth_skills": ["Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Docker", "Kubernetes", "Amazon Web Services", "Python", "CI/CD", "Git"],
        "negative_context_notes": "Uses aliases: 'k8s', 'AWS', 'Python3'."
    },
    {
        "job_id": "SYN_JOB_003",
        "title": "Systems Software Engineer",
        "department": "Core Infrastructure",
        "seniority": "Senior",
        "description": "Our infrastructure team is hiring a Systems Software Engineer. The role requires expert knowledge of C and C++ for low-latency kernel development on Linux. You will write high-performance network protocols and debug multithreaded systems. Must have solid command of Git and Docker. Note that we do not develop in Python or Java for this position.",
        "ground_truth_skills": ["C", "C++", "Linux", "Git", "Docker"],
        "negative_context_notes": "Includes single-letter C and symbol-heavy C++. Also tests negative constraint filtering if applicable, but C and C++ are the target technical competencies."
    },
    {
        "job_id": "SYN_JOB_004",
        "title": "Big Data Engineer",
        "department": "Data Platform",
        "seniority": "Senior",
        "description": "Join our data platform team building streaming architectures. Essential requirements include Apache Spark (pyspark), Apache Kafka, and SQL. All pipelines are orchestrated on Google Cloud Platform (GCP) and containerized with Docker. Hands-on experience with Python and PostgreSQL is expected. We value candidates ready to go the extra mile.",
        "ground_truth_skills": ["Apache Spark", "Apache Kafka", "SQL", "Google Cloud Platform", "Docker", "Python", "PostgreSQL"],
        "negative_context_notes": "Contains non-skill phrase 'go the extra mile' (must not match Go programming language). Uses alias 'GCP', 'pyspark'."
    },
    {
        "job_id": "SYN_JOB_005",
        "title": "NLP Research Scientist",
        "department": "Research Lab",
        "seniority": "Mid/Senior",
        "description": "Seeking an NLP Research Scientist with a track record in Natural Language Processing and Deep Learning. Must have hands-on experience training large language models with PyTorch on Linux clusters. Solid programming in Python, experience with Scikit-Learn, and experiment tracking with Git are required. Active collaboration with university R&D partners.",
        "ground_truth_skills": ["Natural Language Processing", "Deep Learning", "PyTorch", "Linux", "Python", "Scikit-Learn", "Git"],
        "negative_context_notes": "Contains non-skill mention 'R&D'."
    },
    {
        "job_id": "SYN_JOB_006",
        "title": "Cloud Backend Developer",
        "department": "Cloud Engineering",
        "seniority": "Mid-Level",
        "description": "We are developing cloud microservices using Go (Golang) and FastAPI with Python. The backend relies on PostgreSQL, Redis caching, and Docker. Deployment is handled via CI/CD pipelines on Amazon Web Services. You will manage schema migrations and write unit tests in Git repos.",
        "ground_truth_skills": ["Go", "FastAPI", "Python", "PostgreSQL", "Redis", "Docker", "CI/CD", "Amazon Web Services", "Git"],
        "negative_context_notes": "Validates single-word Go / Golang."
    },
    {
        "job_id": "SYN_JOB_007",
        "title": "Full Stack Engineer (TypeScript & React)",
        "department": "Product Engineering",
        "seniority": "Mid-Level",
        "description": "Seeking a Full Stack Engineer proficient in TypeScript, React (ReactJS), and Node.js. On the server side, you will interact with MongoDB and Redis. Candidates must possess strong knowledge of Git, Docker, and continuous integration (CI/CD).",
        "ground_truth_skills": ["TypeScript", "React", "Node.js", "MongoDB", "Redis", "Git", "Docker", "CI/CD"],
        "negative_context_notes": "Aliases: 'ReactJS', 'Node.js'."
    },
    {
        "job_id": "SYN_JOB_008",
        "title": "Computer Vision Engineer",
        "department": "Robotics & Perception",
        "seniority": "Senior",
        "description": "Develop edge vision pipelines using Computer Vision and Deep Learning algorithms. Core languages are C++ and Python. Must have production experience optimizing models in PyTorch or TensorFlow for embedded Linux devices. Git and Docker skills are expected. Collaboration with hardware engineers in our R&D facility.",
        "ground_truth_skills": ["Computer Vision", "Deep Learning", "C++", "Python", "PyTorch", "TensorFlow", "Linux", "Git", "Docker"],
        "negative_context_notes": "Contains 'R&D', C++."
    },
    {
        "job_id": "SYN_JOB_009",
        "title": "DevOps & Site Reliability Engineer",
        "department": "Infrastructure",
        "seniority": "Senior",
        "description": "Looking for an SRE to manage our multi-cloud infrastructure across Amazon Web Services (AWS) and Google Cloud Platform. Must have deep expertise with Kubernetes, Docker, Linux, and Terraform. Implement automated CI/CD workflows and maintain cluster security using Git operations and Go scripts.",
        "ground_truth_skills": ["Amazon Web Services", "Google Cloud Platform", "Kubernetes", "Docker", "Linux", "CI/CD", "Git", "Go"],
        "negative_context_notes": "Multi-cloud tech stack with 'AWS', 'Go'."
    },
    {
        "job_id": "SYN_JOB_010",
        "title": "Junior Data Analyst",
        "department": "Business Intelligence",
        "seniority": "Junior",
        "description": "Entry-level Data Analyst position. Required technical skills include SQL, PostgreSQL, and Python. Experience using R for exploratory data analysis and statistical tests is a huge plus. Must be comfortable managing reports and presenting findings to managerial staff.",
        "ground_truth_skills": ["SQL", "PostgreSQL", "Python", "R"],
        "negative_context_notes": "Includes single-letter 'R' in valid programming context."
    },
    {
        "job_id": "SYN_JOB_011",
        "title": "Java Enterprise Backend Architect",
        "department": "Enterprise Services",
        "seniority": "Principal",
        "description": "Lead enterprise application architecture using Java and Spring Boot. Database design requires PostgreSQL and Redis. Experience with Docker, Kubernetes, CI/CD, and Git repository management is required. Must be able to go from concept to production seamlessly.",
        "ground_truth_skills": ["Java", "PostgreSQL", "Redis", "Docker", "Kubernetes", "CI/CD", "Git"],
        "negative_context_notes": "Contains 'go from concept' (negative for Go)."
    },
    {
        "job_id": "SYN_JOB_012",
        "title": "MLOps Platform Engineer",
        "department": "AI Infrastructure",
        "seniority": "Mid-Level",
        "description": "Build automated Machine Learning platforms. Essential skills: Python, Docker, Kubernetes (k8s), and CI/CD pipelines. You will deploy PyTorch and Scikit-Learn models on Amazon Web Services (AWS) infrastructure. Familiarity with Apache Kafka event streaming is desirable.",
        "ground_truth_skills": ["Machine Learning", "Python", "Docker", "Kubernetes", "CI/CD", "PyTorch", "Scikit-Learn", "Amazon Web Services", "Apache Kafka"],
        "negative_context_notes": "Combines ML and DevOps skills."
    },
    {
        "job_id": "SYN_JOB_013",
        "title": "Frontend Software Engineer",
        "department": "Web Platforms",
        "seniority": "Junior/Mid",
        "description": "We are seeking a Frontend Engineer skilled in JavaScript, TypeScript, and React.js. You will consume REST APIs provided by Node.js backends. Version control using Git is mandatory. Attention to detail and UI performance is valued.",
        "ground_truth_skills": ["JavaScript", "TypeScript", "React", "Node.js", "Git"],
        "negative_context_notes": "Aliases: 'React.js'."
    },
    {
        "job_id": "SYN_JOB_014",
        "title": "Data Warehouse Architect",
        "department": "Data Governance",
        "seniority": "Lead",
        "description": "Architect enterprise data warehousing solutions. Requirements: advanced SQL, PostgreSQL, Apache Spark for ETL processing, and Python scripting. Cloud experience in Google Cloud Platform (GCP) or AWS is highly desirable. Proficient in Git.",
        "ground_truth_skills": ["SQL", "PostgreSQL", "Apache Spark", "Python", "Google Cloud Platform", "Amazon Web Services", "Git"],
        "negative_context_notes": "Contains dual cloud mentions 'GCP' and 'AWS'."
    },
    {
        "job_id": "SYN_JOB_015",
        "title": "Embedded Firmware Developer",
        "department": "IoT Hardware",
        "seniority": "Mid-Level",
        "description": "Responsible for developing micro-controller firmware using C and C++ on RTOS and embedded Linux. Must be skilled in Git, debugging hardware protocols, and optimizing power usage. Candidates must have a BS in Computer Science or Electrical Engineering.",
        "ground_truth_skills": ["C", "C++", "Linux", "Git"],
        "negative_context_notes": "Single letter C alongside C++."
    },
    {
        "job_id": "SYN_JOB_016",
        "title": "AI Research Scientist (Deep Learning & Generative Models)",
        "department": "Frontier AI",
        "seniority": "Staff Scientist",
        "description": "Conduct cutting-edge research in Deep Learning and Natural Language Processing. Requirements include extensive Python development, mastery of PyTorch and TensorFlow, and model distributed training on Linux clusters with Docker. Track record of papers and open-source contributions on Git.",
        "ground_truth_skills": ["Deep Learning", "Natural Language Processing", "Python", "PyTorch", "TensorFlow", "Linux", "Docker", "Git"],
        "negative_context_notes": "Advanced multi-term AI skills."
    },
    {
        "job_id": "SYN_JOB_017",
        "title": "Cloud Security & Infrastructure Engineer",
        "department": "Information Security",
        "seniority": "Senior",
        "description": "Secure our cloud environments across Amazon Web Services and Google Cloud Platform. Hands-on experience with Kubernetes, Docker, and Linux system auditing. Scripting in Python or Go, and managing infrastructure via Git CI/CD workflows.",
        "ground_truth_skills": ["Amazon Web Services", "Google Cloud Platform", "Kubernetes", "Docker", "Linux", "Python", "Go", "Git", "CI/CD"],
        "negative_context_notes": "Scripting in Go."
    },
    {
        "job_id": "SYN_JOB_018",
        "title": "Quantitative Financial Analyst",
        "department": "Quantitative Research",
        "seniority": "Mid-Level",
        "description": "Quantitative trading group seeking an analyst. Strong statistical modeling skills in R and Python required. Must write complex SQL queries against PostgreSQL financial databases. Experience with C++ for high frequency execution is an asset. Regular interaction with senior traders.",
        "ground_truth_skills": ["R", "Python", "SQL", "PostgreSQL", "C++"],
        "negative_context_notes": "Includes R and C++."
    },
    {
        "job_id": "SYN_JOB_019",
        "title": "Backend Microservices Developer (Go & Kubernetes)",
        "department": "Core Platform",
        "seniority": "Senior",
        "description": "Develop resilient, low-latency microservices written in Go. You will deploy workloads onto Kubernetes using Docker and CI/CD pipelines. Persistence layers utilize PostgreSQL and Redis. Collaboration via Git.",
        "ground_truth_skills": ["Go", "Kubernetes", "Docker", "CI/CD", "PostgreSQL", "Redis", "Git"],
        "negative_context_notes": "Go programming language."
    },
    {
        "job_id": "SYN_JOB_020",
        "title": "Healthcare AI Analyst",
        "department": "Clinical Data Lab",
        "seniority": "Mid-Level",
        "description": "Analyze clinical trials and electronic health records. Qualifications: Python, Scikit-Learn, and SQL. Must have familiarity with Natural Language Processing techniques for clinical texts. Team collaborates closely with the medical R&D division and Vitamin C clinical study group. Git experience required.",
        "ground_truth_skills": ["Python", "Scikit-Learn", "SQL", "Natural Language Processing", "Git"],
        "negative_context_notes": "Contains 'medical R&D' (false positive risk for R) and 'Vitamin C' (false positive risk for C)."
    },
    {
        "job_id": "SYN_JOB_021",
        "title": "Modern Web & Mobile App Developer",
        "department": "Digital Experience",
        "seniority": "Mid-Level",
        "description": "Looking for a modern web developer proficient in JavaScript, TypeScript, and React. Backend integrations use Node.js and MongoDB. Automated testing and builds run through Git and CI/CD.",
        "ground_truth_skills": ["JavaScript", "TypeScript", "React", "Node.js", "MongoDB", "Git", "CI/CD"],
        "negative_context_notes": "Standard web stack."
    },
    {
        "job_id": "SYN_JOB_022",
        "title": "Real-Time Streaming Platform Engineer",
        "department": "Event Systems",
        "seniority": "Senior",
        "description": "Build high-throughput streaming systems using Apache Kafka and Apache Spark. We develop in Java and Python, running distributed workers on Kubernetes and Docker. Backend databases include PostgreSQL and Redis on Amazon Web Services (AWS).",
        "ground_truth_skills": ["Apache Kafka", "Apache Spark", "Java", "Python", "Kubernetes", "Docker", "PostgreSQL", "Redis", "Amazon Web Services"],
        "negative_context_notes": "Complex streaming and storage architecture."
    }
]


def main():
    print("Generating synthetic benchmark datasets...")

    # 1. Export Raw Job Postings (CSV & JSON)
    df_jobs = pd.DataFrame(SYNTHETIC_JOBS)
    
    # Save CSV with explicit metadata flag
    df_raw = df_jobs[["job_id", "title", "department", "seniority", "description"]].copy()
    df_raw["is_synthetic"] = True
    df_raw["dataset_version"] = "1.0.0-synthetic-benchmark"
    df_raw.to_csv(DATA_RAW / "synthetic_job_postings.csv", index=False)
    print(f"Saved raw CSV to: {DATA_RAW / 'synthetic_job_postings.csv'}")

    with open(DATA_RAW / "synthetic_job_postings.json", "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "dataset_name": "SkillGraph Synthetic Job Postings Benchmark",
                "version": "1.0.0",
                "disclaimer": "This dataset is strictly synthetic and generated solely for NLP entity extraction benchmarks. It does not represent actual market demand.",
                "total_records": len(SYNTHETIC_JOBS)
            },
            "data": SYNTHETIC_JOBS
        }, f, indent=2)
    print(f"Saved raw JSON to: {DATA_RAW / 'synthetic_job_postings.json'}")

    # 2. Export Ground Truth Annotations
    annotations_data = []
    for item in SYNTHETIC_JOBS:
        annotations_data.append({
            "job_id": item["job_id"],
            "title": item["title"],
            "ground_truth_skills": sorted(item["ground_truth_skills"]),
            "negative_context_notes": item["negative_context_notes"]
        })

    with open(DATA_ANNOTATIONS / "synthetic_job_postings_ground_truth.json", "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "version": "1.0.0",
                "annotation_type": "Expert Curated Technical Skills Ground Truth",
                "total_annotated": len(annotations_data)
            },
            "annotations": annotations_data
        }, f, indent=2)
    print(f"Saved annotations to: {DATA_ANNOTATIONS / 'synthetic_job_postings_ground_truth.json'}")

    # 3. Export Skill Taxonomy
    with open(OUTPUTS_EXPORTS / "skill_taxonomy.json", "w", encoding="utf-8") as f:
        json.dump(SKILL_TAXONOMY, f, indent=2)
    print(f"Saved taxonomy to: {OUTPUTS_EXPORTS / 'skill_taxonomy.json'}")


if __name__ == "__main__":
    main()

