# SkillGraph Data Directory

This directory contains datasets and annotations used for SkillGraph research experiments and pipeline validation.

## Structure

- `raw/`: Raw datasets. For Week 1 Lesson 1, contains `synthetic_job_postings.csv` and `.json`.
- `annotations/`: Expert/manually curated ground truth annotations. Contains `synthetic_job_postings_ground_truth.json`.
- `processed/`: Cleaned and transformed datasets with extracted and normalized entity outputs.
- `external/`: External standard taxonomies (e.g., ESCO, O*NET, Lightcast) when integrated in subsequent milestones.

## Synthetic Data Disclaimer

All sample job postings in `raw/synthetic_job_postings.*` are **strictly synthetic** benchmark samples constructed for natural language processing benchmarking, boundary validation (e.g., C, C++, R), and algorithmic comparison. They do **not** claim to represent real-world hiring market statistics or industry skill demand.

