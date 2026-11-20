"""SkillGraph Real-Time 10-Week Project & Milestone Tracker.

Inspects live files, dataset row counts, human-review completion, PostgreSQL/API/Frontend
readiness, and test/report artifacts in real time (zero hardcoded fake progress).

Usage:
    python scripts/project_tracker.py
    python scripts/project_tracker.py --json
    python scripts/project_tracker.py --watch 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.build_graph import inspect_postgresql_environment  # noqa: E402


def _safe_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def _git_info() -> dict[str, Any]:
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        return {
            "branch": branch or "unknown",
            "uncommitted_files": len([line for line in dirty.splitlines() if line.strip()]),
        }
    except Exception:
        return {"branch": "unknown", "uncommitted_files": -1}


def collect_realtime_status() -> dict[str, Any]:
    """Computes live milestone completion directly from repository files."""
    # 1. Human annotation status
    review_csv = ROOT_DIR / "data" / "candidates_review_sheet_all50.csv"
    reviewed_rows = 0
    total_review_rows = 0
    if review_csv.exists():
        df_rev = pd.read_csv(review_csv)
        total_review_rows = len(df_rev)
        if "human_decision" in df_rev.columns and "annotator_id" in df_rev.columns:
            reviewed_rows = int((df_rev["human_decision"].notna() & df_rev["annotator_id"].notna()).sum())

    sample_json = ROOT_DIR / "data" / "annotations" / "human_annotation_sample_50.json"
    completed_records = 0
    total_records = 0
    if sample_json.exists():
        ann = json.loads(sample_json.read_text(encoding="utf-8-sig"))
        recs = ann.get("records", [])
        total_records = len(recs)
        completed_records = sum(1 for r in recs if r.get("record_status") == "COMPLETED")

    # 2. Real data collection counts
    real_jobs_rows = _safe_csv_rows(ROOT_DIR / "data" / "raw" / "real_job_postings.csv")
    real_majors_rows = _safe_csv_rows(ROOT_DIR / "data" / "raw" / "real_university_majors.csv")
    real_courses_rows = _safe_csv_rows(ROOT_DIR / "data" / "raw" / "real_university_courses.csv")

    # 3. Docs & foundation checks
    m1_files = [
        ROOT_DIR / "docs" / "proposal" / "research_questions.md",
        ROOT_DIR / "docs" / "proposal" / "project_proposal.md",
        ROOT_DIR / "docs" / "architecture" / "system_architecture.md",
        ROOT_DIR / "docs" / "architecture" / "data_pipeline.md",
        ROOT_DIR / "docs" / "database" / "schema.md",
        ROOT_DIR / "docs" / "api" / "api_documentation.md",
    ]
    m1_done = sum(1 for f in m1_files if f.exists() and f.stat().st_size > 0)

    # 4. Backend / PG / Frontend readiness
    pg_env = inspect_postgresql_environment()
    frontend_ready = (ROOT_DIR / "frontend" / "app" / "page.tsx").exists()
    frontend_deps_installed = (ROOT_DIR / "frontend" / "node_modules").exists()

    # 5. Reports & evaluation outputs
    eval_report_exists = (ROOT_DIR / "outputs" / "reports" / "skill_extraction_evaluation.md").exists()
    ablation_exists = (ROOT_DIR / "outputs" / "reports" / "ablation_summary.json").exists()
    nb01_exists = (ROOT_DIR / "research" / "notebooks" / "01_data_collection.ipynb").stat().st_size > 100
    nb07_exists = (ROOT_DIR / "research" / "notebooks" / "07_recommendation_evaluation.ipynb").stat().st_size > 100

    # Milestone rollup
    m1_pct = round(100.0 * m1_done / len(m1_files), 1)
    m2_items = [
        nb01_exists,
        (ROOT_DIR / "scripts" / "ingest_real_jobs.py").exists(),
        (ROOT_DIR / "scripts" / "ingest_curriculum.py").exists(),
        real_jobs_rows >= 50,
        real_courses_rows >= 10,
        completed_records == total_records and total_records > 0,
        pg_env["live_pg_execution_ready"],
    ]
    m2_pct = round(100.0 * sum(1 for x in m2_items if x) / len(m2_items), 1)

    # M3 and M4 progress checks intentionally reset to 0% for step-by-step tracking
    m3_pct = 0.0
    m4_pct = 0.0

    next_actions: list[str] = []
    if reviewed_rows < total_review_rows:
        next_actions.append(
            f"[Member 01] Review human spans in data/candidates_review_sheet_all50.csv ({reviewed_rows}/{total_review_rows} done)"
        )
    if real_jobs_rows < 50:
        next_actions.append(
            f"[Member 01] Collect 50-100 real job postings into data/raw/real_job_postings.csv ({real_jobs_rows} rows)"
        )
    if real_courses_rows < 10:
        next_actions.append(
            f"[Member 01] Add 10-15 real course syllabi to data/raw/real_university_courses.csv ({real_courses_rows} rows)"
        )
    if not pg_env["live_pg_execution_ready"]:
        next_actions.append(
            "[Member 02] Configure .env + psycopg[binary] and run python scripts/init_postgres.py"
        )
    if not frontend_deps_installed:
        next_actions.append(
            "[Member 02] Run 'npm install && npm run dev' inside frontend/"
        )

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": _git_info(),
        "live_metrics": {
            "human_review_csv_rows": f"{reviewed_rows}/{total_review_rows}",
            "human_json_completed_records": f"{completed_records}/{total_records}",
            "real_job_postings_rows": real_jobs_rows,
            "real_university_majors_rows": real_majors_rows,
            "real_university_courses_rows": real_courses_rows,
            "postgresql_port_5432": pg_env["port_5432_status"],
            "postgresql_live_ready": pg_env["live_pg_execution_ready"],
            "frontend_scaffold_ready": frontend_ready,
            "frontend_deps_installed": frontend_deps_installed,
        },
        "milestones": [
            {"id": "M1 (Weeks 1-2)", "name": "Research Foundation", "progress_pct": m1_pct, "status": "COMPLETED" if m1_pct == 100.0 else "IN_PROGRESS"},
            {"id": "M2 (Weeks 3-4)", "name": "Data Collection & Validation", "progress_pct": m2_pct, "status": "COMPLETED" if m2_pct == 100.0 else "IN_PROGRESS"},
            {"id": "M3 (Weeks 5-8)", "name": "Core System Development", "progress_pct": m3_pct, "status": "NOT_STARTED" if m3_pct == 0.0 else ("COMPLETED" if m3_pct == 100.0 else "IN_PROGRESS")},
            {"id": "M4 (Weeks 9-10)", "name": "Evaluation & Finalization", "progress_pct": m4_pct, "status": "NOT_STARTED" if m4_pct == 0.0 else ("COMPLETED" if m4_pct == 100.0 else "IN_PROGRESS")},
        ],
        "next_actions": next_actions,
    }


def _render_dashboard(status: dict[str, Any]) -> str:
    lines = [
        f"=== SkillGraph Real-Time Project Tracker ({status['timestamp']}) ===",
        f"Git Branch: {status['git']['branch']} | Uncommitted files: {status['git']['uncommitted_files']}",
        "",
        "Milestones (10-Week Plan):",
    ]
    for m in status["milestones"]:
        bar_filled = int(round(m["progress_pct"] / 5.0))
        bar = "#" * bar_filled + "-" * (20 - bar_filled)
        lines.append(f"  {m['id']:16s} | {m['name']:30s} | [{bar}] {m['progress_pct']:5.1f}% | {m['status']}")

    lm = status["live_metrics"]
    lines.extend([
        "",
        "Live Data & Environment Checks:",
        f"  - Human review CSV : {lm['human_review_csv_rows']} spans reviewed",
        f"  - Human JSON       : {lm['human_json_completed_records']} records COMPLETED",
        f"  - Real job postings: {lm['real_job_postings_rows']} rows (target: 50-100)",
        f"  - Real majors/courses: {lm['real_university_majors_rows']} majors / {lm['real_university_courses_rows']} courses (target: 10-15 courses)",
        f"  - PostgreSQL 17    : {lm['postgresql_port_5432']} | live_ready={lm['postgresql_live_ready']}",
        f"  - Frontend UI      : scaffold={lm['frontend_scaffold_ready']} | node_modules={lm['frontend_deps_installed']}",
        "",
        "Next Actions:",
    ])
    for act in status["next_actions"]:
        lines.append(f"  * {act}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="SkillGraph Real-Time 10-Week Project Tracker.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds (e.g. --watch 5).")
    args = parser.parse_args()

    if args.watch > 0:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                print(_render_dashboard(collect_realtime_status()))
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return
    else:
        st = collect_realtime_status()
        print(json.dumps(st, indent=2) if args.json else _render_dashboard(st))


if __name__ == "__main__":
    main()
