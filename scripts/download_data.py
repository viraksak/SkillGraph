"""SkillGraph Week 4 / Member 01 Module: Public Real Job-Posting Collector.

Fetches 100 genuine, publicly available English job postings aligned with SkillGraph's
IT / Data / Analytics / Software / Cloud / Cybersecurity / Business scope from open
unauthenticated JSON APIs (Remotive categories and Arbeitnow pages), strips HTML cleanly,
maps canonical occupation titles where applicable while preserving raw ``job_title``,
records full provenance (``source_url``, ``source_name``, ``posted_at``, ``collected_at``),
and writes ``data/raw/real_job_postings.csv`` and ``.json``.

Research Integrity Guardrail:
    Every record carries ``is_synthetic=False``, ``data_type='real'``, and an exact
    verifiable ``source_url``. Zero synthetic rows are ever mixed into this file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from typing import Any

import httpx
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RAW_CSV_PATH = ROOT_DIR / "data" / "raw" / "real_job_postings.csv"
RAW_JSON_PATH = ROOT_DIR / "data" / "raw" / "real_job_postings.json"
OCC_MAPPING_PATH = ROOT_DIR / "data" / "processed" / "global_occupation_mapping.csv"

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
REMOTIVE_CATEGORIES = ("data", "software-dev", "devops", "product", "qa", "marketing")
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"

ENGLISH_MARKERS = (" and ", " the ", " with ", " experience ", " team ", " work ", " you ", " will ")
NON_ENGLISH_TITLE_CUES = ("(m/w/d)", "(f/m/d)", "(w/m/d)", "(m/f/x)", "kfz-", " für ", " und ", " mit ", "(h/f)")
DOMAIN_TITLE_KEYWORDS = (
    "data", "analyst", "analytics", "scientist", "machine learning", "ai ", "ml ",
    "software", "developer", "engineer", "devops", "cloud", "backend", "frontend",
    "full-stack", "full stack", "security", "cyber", "penetration", "network",
    "database", "sql", "python", "qa ", "testing", "product manager", "marketing",
    "bi ", "business intelligence", "architect", "sre", "platform",
)

# Maps real job titles cleanly to closest SkillGraph occupation names when exact substring matches
TITLE_TO_CANONICAL_OCC: tuple[tuple[str, str], ...] = (
    ("data scientist", "Data Scientist"),
    ("data engineer", "Data Engineer"),
    ("data analyst", "Data Analyst"),
    ("business intelligence", "Business Intelligence Analyst"),
    ("machine learning", "Machine Learning Engineer"),
    ("ai engineer", "Machine Learning Engineer"),
    ("cybersecurity", "Cybersecurity Analyst"),
    ("security operations", "Cybersecurity Analyst"),
    ("penetration tester", "Cybersecurity Analyst"),
    ("network engineer", "Network Engineer"),
    ("product manager", "Project Manager"),
    ("project manager", "Project Manager"),
    ("digital marketing", "Digital Marketing Specialist"),
    ("marketing", "Marketing Specialist"),
    ("software", "Software Developer"),
    ("developer", "Software Developer"),
    ("full-stack", "Software Developer"),
    ("back-end", "Software Developer"),
    ("backend", "Software Developer"),
    ("frontend", "Software Developer"),
    ("devops", "Software Developer"),
    ("testing engineer", "Quality Control Inspector"),
    ("qa ", "Quality Control Inspector"),
)

TAG_RE = re.compile(r"<[^>]+>")
BLOCK_TAG_RE = re.compile(r"</?(p|div|li|br|h[1-6]|ul|ol)[^>]*>", re.IGNORECASE)


def _strip_html(raw_html: str) -> str:
    text = BLOCK_TAG_RE.sub("\n", raw_html or "")
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    lines = [" ".join(ln.split()) for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _is_relevant_english_tech_job(title: str, desc: str) -> bool:
    t_low = title.lower()
    if any(cue in t_low for cue in NON_ENGLISH_TITLE_CUES):
        return False
    if not any(kw in t_low for kw in DOMAIN_TITLE_KEYWORDS):
        return False
    d_low = f" {desc.lower()} "
    if sum(1 for tok in ENGLISH_MARKERS if tok in d_low) < 4:
        return False
    if any(de in d_low for de in (" und ", " für ", " mit ", " wir ", " deine ")):
        return False
    return len(desc) >= 200


def _load_schema_columns() -> list[str]:
    if RAW_CSV_PATH.exists():
        first_line = RAW_CSV_PATH.read_text(encoding="utf-8-sig").splitlines()[0]
        return [c.strip() for c in first_line.split(",") if c.strip()]
    return [
        "job_id", "country", "country_code", "region", "industry", "occupation",
        "job_title", "company_name", "location", "description", "responsibilities",
        "required_skills", "preferred_skills", "education_level", "experience_years_min",
        "experience_years_max", "employment_type", "language_requirements", "salary_min",
        "salary_max", "salary_currency", "original_language", "source_name", "source_url",
        "posted_at", "collected_at", "generated_at", "data_type", "is_synthetic", "verified",
        "seniority_level", "occupation_id", "isco08_code_candidate", "source_dataset",
        "license", "attribution", "generator_seed",
    ]


def collect_public_real_jobs(target_count: int = 100, timeout_s: float = 15.0) -> list[dict[str, Any]]:
    """Collects 100 real public English IT/Data/Business postings aligned with SkillGraph."""
    cols = _load_schema_columns()
    collected_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    occ_df = pd.read_csv(OCC_MAPPING_PATH) if OCC_MAPPING_PATH.exists() else pd.DataFrame()
    occ_lookup = {
        str(r["occupation"]).strip().lower(): r
        for r in occ_df.to_dict(orient="records")
    } if not occ_df.empty else {}

    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()

    def _resolve_canonical_occ(raw_title: str) -> tuple[str, dict[str, Any] | None]:
        t_low = raw_title.lower()
        if t_low in occ_lookup:
            rec = occ_lookup[t_low]
            return str(rec["occupation"]), rec
        for substr, canon_occ in TITLE_TO_CANONICAL_OCC:
            if substr in t_low and canon_occ.lower() in occ_lookup:
                return canon_occ, occ_lookup[canon_occ.lower()]
        return raw_title, None

    def _add_record(
        job_id: str,
        title: str,
        company: str,
        location: str,
        industry: str,
        emp_type: str,
        desc: str,
        source_name: str,
        source_url: str,
        posted_at: str,
        attribution: str,
    ) -> None:
        if len(rows) >= target_count or job_id in seen_ids:
            return
        if not title or not desc or not source_url or not _is_relevant_english_tech_job(title, desc):
            return
        h = hashlib.sha256(f"{title.lower()}||{' '.join(desc.lower().split())}".encode("utf-8")).hexdigest()[:16]
        if h in seen_hashes:
            return
        seen_ids.add(job_id)
        seen_hashes.add(h)

        canon_occ, occ_rec = _resolve_canonical_occ(title)
        base = {c: "" for c in cols}
        base.update({
            "job_id": job_id,
            "country": location or "Remote",
            "industry": str(occ_rec.get("industry")) if occ_rec else (industry or "Information Technology"),
            "occupation": canon_occ,
            "job_title": title,
            "company_name": company,
            "location": location or "Remote",
            "description": desc,
            "employment_type": emp_type or "Full-time",
            "original_language": "en",
            "source_name": source_name,
            "source_url": source_url,
            "posted_at": posted_at,
            "collected_at": collected_at,
            "data_type": "real",
            "is_synthetic": False,
            "verified": True,
            "occupation_id": str(occ_rec["occupation_id"]) if occ_rec else "",
            "isco08_code_candidate": str(occ_rec.get("isco08_code_candidate", "")) if occ_rec else "",
            "source_dataset": source_name,
            "attribution": attribution,
        })
        rows.append(base)

    # 1. Remotive Public API across relevant tech/data categories
    for cat in REMOTIVE_CATEGORIES:
        if len(rows) >= target_count:
            break
        try:
            r_rem = httpx.get(REMOTIVE_URL, params={"category": cat, "limit": 50}, timeout=timeout_s)
            if r_rem.status_code == 200:
                for item in r_rem.json().get("jobs", []):
                    _add_record(
                        job_id=f"REAL_REM_{item.get('id')}",
                        title=str(item.get("title") or "").strip(),
                        company=str(item.get("company_name") or "").strip(),
                        location=str(item.get("candidate_required_location") or "Remote").strip(),
                        industry=str(item.get("category") or "Information Technology").strip(),
                        emp_type=str(item.get("job_type") or "full_time").strip(),
                        desc=_strip_html(str(item.get("description") or "")),
                        source_name="Remotive Public API",
                        source_url=str(item.get("url") or "").strip(),
                        posted_at=str(item.get("publication_date") or "").strip(),
                        attribution="Remotive public remote-jobs API (remotive.com)",
                    )
        except Exception as exc:
            print(f"Warning: Remotive category {cat} skipped ({exc})")

    # 2. Arbeitnow Public API across pages 1..5
    for page in range(1, 6):
        if len(rows) >= target_count:
            break
        try:
            r_arb = httpx.get(ARBEITNOW_URL, params={"page": page}, timeout=timeout_s)
            if r_arb.status_code == 200:
                for item in r_arb.json().get("data", []):
                    slug = str(item.get("slug") or "")
                    short_id = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:10]
                    tags = item.get("tags") or []
                    _add_record(
                        job_id=f"REAL_ARB_{short_id}",
                        title=str(item.get("title") or "").strip(),
                        company=str(item.get("company_name") or "").strip(),
                        location=str(item.get("location") or "Remote").strip(),
                        industry=", ".join(str(t) for t in tags[:3]) if tags else "Information Technology",
                        emp_type=", ".join(item.get("job_types") or ["Full-time"]),
                        desc=_strip_html(str(item.get("description") or "")),
                        source_name="Arbeitnow Public API",
                        source_url=str(item.get("url") or "").strip(),
                        posted_at=str(item.get("created_at") or "").strip(),
                        attribution="Arbeitnow public job-board API (arbeitnow.com)",
                    )
                    if len(rows) >= target_count:
                        break
        except Exception as exc:
            print(f"Warning: Arbeitnow page {page} skipped ({exc})")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect 100 real domain-aligned job postings into data/raw/real_job_postings.*")
    parser.add_argument("--target-count", type=int, default=100, help="Target number of real postings (default: 100).")
    args = parser.parse_args()

    rows = collect_public_real_jobs(target_count=args.target_count)
    if not rows:
        print("No real postings collected; existing data/raw/real_job_postings.* left unchanged.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(RAW_CSV_PATH, index=False, encoding="utf-8")
    RAW_JSON_PATH.write_bytes((json.dumps(rows, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"Saved {len(rows)} domain-aligned real job postings to {RAW_CSV_PATH} and {RAW_JSON_PATH}")


if __name__ == "__main__":
    main()
