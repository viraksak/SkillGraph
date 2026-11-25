
import json
import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.skill_extraction.annotation_loader import load_annotation_dataset
from models.skill_extraction.dictionary_extractor import DictionarySkillExtractor
from models.skill_extraction.evaluator import SkillExtractionEvaluator


ANNOTATION_PATH = Path(
    "data/annotations/human_annotation_sample_50.json"
)

TAXONOMY_PATH = Path(
    "data/processed/global_skill_taxonomy.csv"
)

REPORT_DIR = Path("outputs/reports")



def main():
    # 1. Load human annotations
    annotation_data = load_annotation_dataset(str(ANNOTATION_PATH))

    ground_truth_by_job = annotation_data["ground_truth_by_job"]

    # 2. Load original job descriptions
    with ANNOTATION_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    records_by_job = {
        record["job_id"]: record
        for record in dataset["records"]
    }

    # 3. Initialize dictionary extractor
    extractor = DictionarySkillExtractor(str(TAXONOMY_PATH))

    # 4. Extract skills from every annotated job
    predictions_by_job = {}

    for job_id in ground_truth_by_job:
        record = records_by_job.get(job_id)

        if record is None:
            raise ValueError(f"Missing job description: {job_id}")

        predictions_by_job[job_id] = extractor.extract(record["text"])

    # 5. Evaluate predictions against human annotations
    results = SkillExtractionEvaluator.evaluate_exact_spans(
        predictions_by_job=predictions_by_job,
        ground_truth_by_job=ground_truth_by_job,
    )

    # 6. Create report directory
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 7. Save per-document metrics
    results["per_doc_df"].to_csv(
        REPORT_DIR / "per_document_evaluation.csv",
        index=False,
    )

    # 8. Save false-positive and false-negative examples
    import pandas as pd

    pd.DataFrame(results["fp_examples"]).to_csv(
        REPORT_DIR / "false_positives.csv",
        index=False,
    )

    pd.DataFrame(results["fn_examples"]).to_csv(
        REPORT_DIR / "false_negatives.csv",
        index=False,
    )

    # 9. Save overall metrics
    report = {
        key: value
        for key, value in results.items()
        if key != "per_doc_df"
    }

    report["dataset_name"] = annotation_data["dataset_name"]
    report["evaluation_type"] = annotation_data["evaluation_type"]
    report["provenance_note"] = annotation_data["provenance_note"]
    report["evaluation_limitation_statement"] = (
        annotation_data["evaluation_limitation_statement"]
    )

    with (REPORT_DIR / "evaluation_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    # 10. Display results
    print("\n========== SKILL EXTRACTION EVALUATION ==========")

    print("Documents evaluated:", len(ground_truth_by_job))
    print("Total TP:", results["total_tp"])
    print("Total FP:", results["total_fp"])
    print("Total FN:", results["total_fn"])

    print("\nMICRO METRICS")
    print(f"Precision: {results['micro_precision']:.4f}")
    print(f"Recall:    {results['micro_recall']:.4f}")
    print(f"F1:        {results['micro_f1']:.4f}")

    print("\nMACRO METRICS")
    print(f"Precision: {results['macro_precision']:.4f}")
    print(f"Recall:    {results['macro_recall']:.4f}")
    print(f"F1:        {results['macro_f1']:.4f}")

    print("\nREPORTS SAVED IN:", REPORT_DIR.resolve())


if __name__ == "__main__":
    main()
