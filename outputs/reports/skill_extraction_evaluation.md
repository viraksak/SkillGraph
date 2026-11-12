# SkillGraph Research Experiment 01: Baseline Evaluation Report

- **Experiment:** Technical Skill Extraction from Job Descriptions
- **Milestone:** Week 1, Lesson 1
- **Evaluated Samples:** 22 Synthetic Job Postings

## Performance Metrics

| Metric | Regex Baseline | spaCy PhraseMatcher |
| :--- | :--- | :--- |
| **Micro Precision** | 0.9811 | 0.9873 |
| **Micro Recall** | 1.0000 | 1.0000 |
| **Micro F1** | 0.9905 | 0.9936 |
| **Macro Precision** | 0.9820 | 0.9870 |
| **Macro Recall** | 1.0000 | 1.0000 |
| **Macro F1** | 0.9898 | 0.9924 |
| **False Positives (Total)** | 3 | 2 |
| **False Negatives (Total)** | 0 | 0 |

## Summary of Findings
1. **spaCy PhraseMatcher** outperformed regex on token boundary accuracy, avoiding alias leakage (e.g. 'js' inside 'Node.js').
2. Neither dictionary approach can handle **grammatical negation** (e.g., 'we do not develop in Python').
3. Next steps for Lesson 2: integrate contextual embeddings and grammatical dependency filtering.
