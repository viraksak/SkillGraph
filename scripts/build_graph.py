"""Week 2 Knowledge Graph Preparation & Validation Pipeline (`scripts/build_graph.py`).

Constructs an explainable, provenance-backed Knowledge Graph representation from
SkillGraph's processed job corpus (`global_jobs_cleaned.csv`), canonical skill
taxonomy (`global_skill_taxonomy.csv`), and occupation catalog
(`global_occupation_mapping.csv`) using the Week 2 NLP extraction &
normalization pipeline.

Design Safeguards:
- Distinguishes extracted `REQUIRES_SKILL` (`required` section) from
  `PREFERS_SKILL` (`preferred` section) with full span & confidence metadata.
- Explicitly labels `CO_OCCURS_WITH` as symmetric statistical co-occurrence,
  never as a pedagogical prerequisite (`relationship_semantics =
  "statistical_co_occurrence_not_prerequisite"`).
- Defines curriculum & student schemas (`UniversityCourse`, `UniversityMajor`,
  `Student`) cleanly as typed structural definitions without fabricating data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
import sqlite3
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.skill_extraction import SkillNormalizer, SpacySkillExtractor  # noqa: E402
from models.skill_extraction.dictionary_extractor import parse_aliases_field  # noqa: E402

CURRICULUM_EXTENSION_SCHEMAS: dict[str, dict[str, Any]] = {
    "UniversityMajor": {
        "primary_key": "major_id",
        "columns": ["major_id", "major_name", "institution_name", "degree_level", "target_occupation_ids"],
        "status": "schema_ready_awaiting_curriculum_ingestion",
    },
    "UniversityCourse": {
        "primary_key": "course_id",
        "columns": ["course_id", "course_code", "course_title", "major_id", "syllabus_text", "taught_skill_ids"],
        "status": "schema_ready_awaiting_curriculum_ingestion",
    },
    "Student": {
        "primary_key": "student_id",
        "columns": ["student_id", "major_id", "target_occupation_id", "mastered_skill_ids"],
        "status": "schema_ready_awaiting_student_profiles",
    },
}


def _slug_id(prefix: str, label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(label).strip().upper())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return f"{prefix}_{cleaned.strip('_')}"


class SkillGraphBuilder:
    """Builds relational tables and NetworkX-ready graph structures from SkillGraph data."""

    def __init__(
        self,
        jobs_path: str | Path | None = None,
        taxonomy_path: str | Path | None = None,
        occupations_path: str | Path | None = None,
        include_broad_country_mappings: bool = False,
        min_co_occurrence: int = 5,
    ) -> None:
        self.jobs_path = Path(jobs_path or (ROOT_DIR / "data" / "processed" / "global_jobs_cleaned.csv"))
        self.taxonomy_path = Path(taxonomy_path or (ROOT_DIR / "data" / "processed" / "global_skill_taxonomy.csv"))
        self.occupations_path = Path(occupations_path or (ROOT_DIR / "data" / "processed" / "global_occupation_mapping.csv"))
        self.include_broad_country_mappings = include_broad_country_mappings
        self.min_co_occurrence = min_co_occurrence

        self.df_jobs = pd.read_csv(self.jobs_path)
        self.df_taxonomy = pd.read_csv(self.taxonomy_path)
        self.df_occupations = pd.read_csv(self.occupations_path)

        self.extractor = SpacySkillExtractor(
            self.df_taxonomy,
            include_broad_country_mappings=self.include_broad_country_mappings,
            section_filter={"required", "preferred"},
            use_fast_tokenizer=True,
        )

        self.nodes: dict[str, pd.DataFrame] = {}
        self.edges: dict[str, pd.DataFrame] = {}
        self._built = False

    def build(self) -> dict[str, Any]:
        """Constructs all entity node tables and relationship edge tables."""
        # 1. Build entity node tables
        df_skills = self.df_taxonomy[[
            "skill_id",
            "canonical_skill",
            "skill_category",
            "skill_type",
            "taxonomy_source",
            "esco_uri",
            "onet_element_id",
            "mapping_status",
        ]].copy()
        df_skills["category_id"] = df_skills["skill_category"].apply(lambda c: _slug_id("CAT", c))

        categories = sorted(self.df_taxonomy["skill_category"].dropna().unique().tolist())
        df_categories = pd.DataFrame([
            {
                "category_id": _slug_id("CAT", cat),
                "category_name": cat,
                "skill_count": int((self.df_taxonomy["skill_category"] == cat).sum()),
            }
            for cat in categories
        ])

        industries = sorted(self.df_occupations["industry"].dropna().unique().tolist())
        df_industries = pd.DataFrame([
            {
                "industry_id": _slug_id("IND", ind),
                "industry_name": ind,
                "occupation_count": int((self.df_occupations["industry"] == ind).sum()),
            }
            for ind in industries
        ])

        df_occs = self.df_occupations[[
            "occupation_id",
            "occupation",
            "industry",
            "department_label",
            "isco08_code_candidate",
            "isco08_label_candidate",
            "mapping_confidence",
            "mapping_status",
            "esco_occupation_uri",
            "onet_soc_code",
        ]].copy()
        df_occs["industry_id"] = df_occs["industry"].apply(lambda i: _slug_id("IND", i))

        df_countries = (
            self.df_jobs[["country_code", "country", "region"]]
            .drop_duplicates()
            .sort_values("country_code")
            .reset_index(drop=True)
        )

        df_job_nodes = self.df_jobs[[
            "job_id",
            "job_title",
            "occupation_id",
            "occupation",
            "industry",
            "country_code",
            "seniority_level",
            "education_level",
            "experience_years_min",
            "experience_years_max",
            "employment_type",
            "is_synthetic",
            "split",
        ]].copy()

        self.nodes = {
            "Skill": df_skills,
            "SkillCategory": df_categories,
            "Occupation": df_occs,
            "Industry": df_industries,
            "Country": df_countries,
            "JobPosting": df_job_nodes,
        }

        # 2. Structural taxonomy & posting metadata edges
        df_belongs = df_skills[["skill_id", "category_id", "skill_category"]].copy()
        df_in_occ = df_job_nodes[["job_id", "occupation_id"]].copy()
        df_in_ind = df_occs[["occupation_id", "industry_id", "industry"]].copy()
        df_loc_in = df_job_nodes[["job_id", "country_code"]].copy()

        # 3. Extract & normalize job skill edges (REQUIRES_SKILL / PREFERS_SKILL)
        requires_rows: list[dict[str, Any]] = []
        prefers_rows: list[dict[str, Any]] = []

        skill_doc_freq: Counter[str] = Counter()
        co_occur_counts: Counter[tuple[str, str]] = Counter()
        occ_job_counts: Counter[str] = Counter(df_job_nodes["occupation_id"].tolist())
        occ_skill_stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {"required_postings": set(), "preferred_postings": set(), "all_postings": set()}
        )

        for row in self.df_jobs.itertuples(index=False):
            job_id = str(row.job_id)
            occ_id = str(row.occupation_id)
            spans = self.extractor.extract(str(row.description))

            doc_skill_ids: set[str] = set()
            for sp in spans:
                sid = sp["skill_id"]
                doc_skill_ids.add(sid)
                edge_rec = {
                    "job_id": job_id,
                    "skill_id": sid,
                    "canonical_skill": sp["canonical_skill"],
                    "skill_mention": sp["skill_mention"],
                    "character_start": sp["character_start"],
                    "character_end": sp["character_end"],
                    "source_section": sp["source_section"],
                    "extractor_name": sp["extractor_name"],
                    "match_method": sp["match_method"],
                    "mapping_status": sp["mapping_status"],
                    "mapping_confidence": sp["mapping_confidence"],
                }
                if sp["source_section"] == "required":
                    requires_rows.append(edge_rec)
                    occ_skill_stats[(occ_id, sid)]["required_postings"].add(job_id)
                elif sp["source_section"] == "preferred":
                    prefers_rows.append(edge_rec)
                    occ_skill_stats[(occ_id, sid)]["preferred_postings"].add(job_id)
                occ_skill_stats[(occ_id, sid)]["all_postings"].add(job_id)

            for sid in doc_skill_ids:
                skill_doc_freq[sid] += 1
            for s1, s2 in combinations(sorted(doc_skill_ids), 2):
                co_occur_counts[(s1, s2)] += 1

        # 4. Aggregate Occupation -> Skill demand edges
        occ_skill_rows: list[dict[str, Any]] = []
        skill_name_lookup = dict(zip(df_skills["skill_id"], df_skills["canonical_skill"]))
        occ_title_lookup = dict(zip(df_occs["occupation_id"], df_occs["occupation"]))

        for (occ_id, sid), stat in occ_skill_stats.items():
            total_occ_jobs = max(1, occ_job_counts[occ_id])
            job_cnt = len(stat["all_postings"])
            occ_skill_rows.append({
                "occupation_id": occ_id,
                "occupation": occ_title_lookup.get(occ_id, ""),
                "skill_id": sid,
                "canonical_skill": skill_name_lookup.get(sid, ""),
                "job_count": job_cnt,
                "required_count": len(stat["required_postings"]),
                "preferred_count": len(stat["preferred_postings"]),
                "occupation_total_postings": total_occ_jobs,
                "demand_share": round(job_cnt / total_occ_jobs, 4),
            })

        df_occ_skill = pd.DataFrame(occ_skill_rows).sort_values(
            ["occupation_id", "demand_share", "job_count"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

        # 5. Build symmetric CO_OCCURS_WITH statistical association edges (NOT prerequisites)
        n_docs = len(self.df_jobs)
        co_occur_rows: list[dict[str, Any]] = []
        for (s1, s2), cnt in co_occur_counts.items():
            if cnt < self.min_co_occurrence:
                continue
            f1 = skill_doc_freq[s1]
            f2 = skill_doc_freq[s2]
            union = f1 + f2 - cnt
            jaccard = cnt / union if union > 0 else 0.0
            p_xy = cnt / n_docs
            p_x = f1 / n_docs
            p_y = f2 / n_docs
            pmi = math.log2(p_xy / (p_x * p_y)) if (p_xy > 0 and p_x > 0 and p_y > 0) else 0.0
            co_occur_rows.append({
                "source_skill_id": s1,
                "source_skill": skill_name_lookup.get(s1, ""),
                "target_skill_id": s2,
                "target_skill": skill_name_lookup.get(s2, ""),
                "co_occurrence_count": cnt,
                "jaccard_similarity": round(jaccard, 4),
                "pmi_score": round(pmi, 4),
                "is_symmetric": True,
                "relationship_semantics": "statistical_co_occurrence_not_prerequisite",
            })

        df_co_occur = pd.DataFrame(co_occur_rows).sort_values(
            ["co_occurrence_count", "pmi_score"],
            ascending=[False, False],
        ).reset_index(drop=True)

        self.edges = {
            "REQUIRES_SKILL": pd.DataFrame(requires_rows),
            "PREFERS_SKILL": pd.DataFrame(prefers_rows),
            "BELONGS_TO_CATEGORY": df_belongs,
            "IN_OCCUPATION": df_in_occ,
            "IN_INDUSTRY": df_in_ind,
            "LOCATED_IN": df_loc_in,
            "OCCUPATION_DEMANDS_SKILL": df_occ_skill,
            "CO_OCCURS_WITH": df_co_occur,
        }
        self._built = True
        return self.summary()

    def validate_graph(self) -> dict[str, Any]:
        """Verifies referential integrity, zero orphan edges, and edge semantics."""
        if not self._built:
            self.build()

        valid_jobs = set(self.nodes["JobPosting"]["job_id"])
        valid_skills = set(self.nodes["Skill"]["skill_id"])
        valid_cats = set(self.nodes["SkillCategory"]["category_id"])
        valid_occs = set(self.nodes["Occupation"]["occupation_id"])
        valid_inds = set(self.nodes["Industry"]["industry_id"])
        valid_countries = set(self.nodes["Country"]["country_code"])

        orphan_req = sum(
            ~(self.edges["REQUIRES_SKILL"]["job_id"].isin(valid_jobs) & self.edges["REQUIRES_SKILL"]["skill_id"].isin(valid_skills))
        )
        orphan_pref = sum(
            ~(self.edges["PREFERS_SKILL"]["job_id"].isin(valid_jobs) & self.edges["PREFERS_SKILL"]["skill_id"].isin(valid_skills))
        )
        orphan_cat = sum(
            ~(self.edges["BELONGS_TO_CATEGORY"]["skill_id"].isin(valid_skills) & self.edges["BELONGS_TO_CATEGORY"]["category_id"].isin(valid_cats))
        )
        orphan_occ = sum(
            ~(self.edges["IN_OCCUPATION"]["job_id"].isin(valid_jobs) & self.edges["IN_OCCUPATION"]["occupation_id"].isin(valid_occs))
        )
        orphan_ind = sum(
            ~(self.edges["IN_INDUSTRY"]["occupation_id"].isin(valid_occs) & self.edges["IN_INDUSTRY"]["industry_id"].isin(valid_inds))
        )
        orphan_ctry = sum(
            ~(self.edges["LOCATED_IN"]["job_id"].isin(valid_jobs) & self.edges["LOCATED_IN"]["country_code"].isin(valid_countries))
        )
        self_loops = int((self.edges["CO_OCCURS_WITH"]["source_skill_id"] == self.edges["CO_OCCURS_WITH"]["target_skill_id"]).sum())

        total_anomalies = int(orphan_req + orphan_pref + orphan_cat + orphan_occ + orphan_ind + orphan_ctry + self_loops)
        return {
            "referential_integrity_passed": total_anomalies == 0,
            "total_anomalies": total_anomalies,
            "orphan_edges": {
                "REQUIRES_SKILL": int(orphan_req),
                "PREFERS_SKILL": int(orphan_pref),
                "BELONGS_TO_CATEGORY": int(orphan_cat),
                "IN_OCCUPATION": int(orphan_occ),
                "IN_INDUSTRY": int(orphan_ind),
                "LOCATED_IN": int(orphan_ctry),
            },
            "co_occurrence_self_loops": self_loops,
        }

    def explain_skill_gap(
        self,
        known_skills: set[str] | list[str],
        target_occupation: str,
        top_k: int = 8,
    ) -> pd.DataFrame:
        """Explains missing target-occupation skills ranked by demand share & co-occurrence."""
        if not self._built:
            self.build()

        known_lower = {s.strip().lower() for s in known_skills}
        df_demand = self.edges["OCCUPATION_DEMANDS_SKILL"]
        df_target = df_demand[df_demand["occupation"].str.lower() == target_occupation.strip().lower()].copy()
        if df_target.empty:
            raise ValueError(f"Unknown target occupation: {target_occupation!r}")

        df_co = self.edges["CO_OCCURS_WITH"]
        known_ids = set(
            self.nodes["Skill"][self.nodes["Skill"]["canonical_skill"].str.lower().isin(known_lower)]["skill_id"]
        )

        gap_rows: list[dict[str, Any]] = []
        for row in df_target.itertuples(index=False):
            if row.canonical_skill.lower() in known_lower:
                continue
            bridges = df_co[
                ((df_co["source_skill_id"] == row.skill_id) & (df_co["target_skill_id"].isin(known_ids)))
                | ((df_co["target_skill_id"] == row.skill_id) & (df_co["source_skill_id"].isin(known_ids)))
            ]
            bridge_names = sorted({
                b.target_skill if b.source_skill_id == row.skill_id else b.source_skill
                for b in bridges.itertuples(index=False)
            })
            gap_rows.append({
                "missing_skill_id": row.skill_id,
                "missing_skill": row.canonical_skill,
                "demand_share": row.demand_share,
                "required_postings": row.required_count,
                "preferred_postings": row.preferred_count,
                "co_occurring_known_skills": bridge_names[:5],
                "explanation": (
                    f"Demanded in {row.job_count}/{row.occupation_total_postings} "
                    f"({row.demand_share * 100:.1f}%) postings for {target_occupation}; "
                    f"statistically co-occurs with {len(bridge_names)} known skill(s)."
                ),
            })

        return pd.DataFrame(gap_rows).head(top_k)

    def to_relational_tables(self) -> dict[str, pd.DataFrame]:
        """Returns dictionary of PostgreSQL-ready node & edge DataFrames."""
        if not self._built:
            self.build()
        return {**{f"node_{k}": v.copy() for k, v in self.nodes.items()}, **{f"edge_{k}": v.copy() for k, v in self.edges.items()}}

    def to_adjacency_dict(self) -> dict[str, Any]:
        """Exports NetworkX-compatible node-link / adjacency payload without requiring networkx."""
        if not self._built:
            self.build()
        return {
            "directed_core": False,
            "node_types": {k: len(v) for k, v in self.nodes.items()},
            "edge_types": {k: len(v) for k, v in self.edges.items()},
            "extension_schemas": CURRICULUM_EXTENSION_SCHEMAS,
            "co_occurrence_edges": self.edges["CO_OCCURS_WITH"].to_dict("records"),
        }

    def to_networkx(self) -> Any:
        """Builds an `nx.MultiDiGraph` if `networkx` is installed in the environment."""
        if importlib.util.find_spec("networkx") is None:
            raise ImportError(
                "Optional package 'networkx' is not installed in this Python environment. "
                "Use `to_relational_tables()` or `to_adjacency_dict()` instead, or install networkx."
            )
        import networkx as nx  # type: ignore

        if not self._built:
            self.build()

        g = nx.MultiDiGraph()
        for ntype, df_n in self.nodes.items():
            pk = {
                "Skill": "skill_id",
                "SkillCategory": "category_id",
                "Occupation": "occupation_id",
                "Industry": "industry_id",
                "Country": "country_code",
                "JobPosting": "job_id",
            }[ntype]
            for rec in df_n.to_dict("records"):
                g.add_node(f"{ntype}:{rec[pk]}", node_type=ntype, **rec)

        for rec in self.edges["REQUIRES_SKILL"].to_dict("records"):
            g.add_edge(f"JobPosting:{rec['job_id']}", f"Skill:{rec['skill_id']}", rel_type="REQUIRES_SKILL", **rec)
        for rec in self.edges["PREFERS_SKILL"].to_dict("records"):
            g.add_edge(f"JobPosting:{rec['job_id']}", f"Skill:{rec['skill_id']}", rel_type="PREFERS_SKILL", **rec)
        return g

    def summary(self) -> dict[str, Any]:
        """Returns node and edge counts across the constructed Knowledge Graph."""
        if not self._built:
            self.build()
        node_counts = {k: len(v) for k, v in self.nodes.items()}
        edge_counts = {k: len(v) for k, v in self.edges.items()}
        return {
            "total_active_nodes": sum(node_counts.values()),
            "total_active_edges": sum(edge_counts.values()),
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "networkx_installed": importlib.util.find_spec("networkx") is not None,
        }


PROFICIENCY_GAP_MAP: dict[str, tuple[float, float]] = {
    # level -> (demonstrated_score, remaining_gap_weight)
    "none": (0.0, 1.0),
    "unrecorded": (0.0, 1.0),
    "basic": (0.50, 0.50),
    "beginner": (0.50, 0.50),
    "novice": (0.50, 0.50),
    "developing": (0.65, 0.35),
    "intermediate": (1.00, 0.00),
    "proficient": (1.00, 0.00),
    "advanced": (1.00, 0.00),
    "expert": (1.00, 0.00),
}
PROFICIENCY_TARGET_THRESHOLD = 0.70


def inspect_postgresql_environment() -> dict[str, Any]:
    """Checks local PostgreSQL binaries, service status, and driver availability safely."""
    psql_path = Path(r"C:\Program Files\PostgreSQL\17\bin\psql.exe")
    pg_isready_path = Path(r"C:\Program Files\PostgreSQL\17\bin\pg_isready.exe")

    driver_installed = {
        pkg: importlib.util.find_spec(pkg) is not None
        for pkg in ("psycopg2", "psycopg", "asyncpg", "sqlalchemy")
    }
    env_configured = {
        key: bool(os.environ.get(key))
        for key in ("SKILLGRAPH_DB_HOST", "SKILLGRAPH_DB_PORT", "SKILLGRAPH_DB_NAME", "SKILLGRAPH_DB_USER", "SKILLGRAPH_DB_PASSWORD")
    }
    pg_ready_status = "unknown"
    if pg_isready_path.exists():
        try:
            proc = subprocess.run(
                [str(pg_isready_path), "-h", "127.0.0.1", "-p", "5432"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            pg_ready_status = proc.stdout.strip() or proc.stderr.strip()
        except Exception as exc:
            pg_ready_status = f"error: {exc}"

    return {
        "psql_exists": psql_path.exists(),
        "pg_isready_exists": pg_isready_path.exists(),
        "port_5432_status": pg_ready_status,
        "python_pg_drivers": driver_installed,
        "env_vars_configured": env_configured,
        "live_pg_execution_ready": any(driver_installed.values()) and all(env_configured.values()),
        "offline_sql_fallback": "sqlite3_with_foreign_keys_enabled",
    }


POSTGRESQL_DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS skill_categories (
        category_id VARCHAR(64) PRIMARY KEY,
        category_name VARCHAR(128) NOT NULL UNIQUE,
        skill_count INTEGER NOT NULL DEFAULT 0 CHECK (skill_count >= 0)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS skills (
        skill_id VARCHAR(32) PRIMARY KEY,
        canonical_skill VARCHAR(256) NOT NULL UNIQUE,
        category_id VARCHAR(64) NOT NULL REFERENCES skill_categories(category_id),
        skill_type VARCHAR(64) NOT NULL,
        taxonomy_source VARCHAR(64) NOT NULL,
        esco_uri TEXT,
        onet_element_id VARCHAR(64),
        mapping_status VARCHAR(64) NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS skill_aliases (
        alias_id INTEGER PRIMARY KEY,
        skill_id VARCHAR(32) NOT NULL REFERENCES skills(skill_id) ON DELETE CASCADE,
        alias_text VARCHAR(256) NOT NULL,
        case_sensitive INTEGER NOT NULL DEFAULT 0 CHECK (case_sensitive IN (0, 1)),
        UNIQUE (skill_id, alias_text)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS industries (
        industry_id VARCHAR(64) PRIMARY KEY,
        industry_name VARCHAR(128) NOT NULL UNIQUE,
        occupation_count INTEGER NOT NULL DEFAULT 0 CHECK (occupation_count >= 0)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS occupations (
        occupation_id VARCHAR(32) PRIMARY KEY,
        occupation VARCHAR(256) NOT NULL UNIQUE,
        industry_id VARCHAR(64) NOT NULL REFERENCES industries(industry_id),
        department_label VARCHAR(128),
        isco08_code_candidate VARCHAR(32),
        isco08_label_candidate VARCHAR(256),
        mapping_confidence REAL CHECK (mapping_confidence IS NULL OR (mapping_confidence >= 0.0 AND mapping_confidence <= 1.0)),
        mapping_status VARCHAR(64),
        esco_occupation_uri TEXT,
        onet_soc_code VARCHAR(32)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS countries (
        country_code VARCHAR(8) PRIMARY KEY,
        country VARCHAR(128) NOT NULL UNIQUE,
        region VARCHAR(64) NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS job_postings (
        job_id VARCHAR(64) PRIMARY KEY,
        occupation_id VARCHAR(32) NOT NULL REFERENCES occupations(occupation_id),
        country_code VARCHAR(8) NOT NULL REFERENCES countries(country_code),
        job_title VARCHAR(256) NOT NULL,
        seniority_level VARCHAR(64) NOT NULL,
        education_level VARCHAR(64) NOT NULL,
        experience_years_min INTEGER CHECK (experience_years_min IS NULL OR experience_years_min >= 0),
        experience_years_max INTEGER CHECK (experience_years_max IS NULL OR experience_years_max >= experience_years_min),
        employment_type VARCHAR(64) NOT NULL,
        description TEXT NOT NULL,
        is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic IN (0, 1)),
        split VARCHAR(32) NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS job_skill_mentions (
        mention_id INTEGER PRIMARY KEY,
        job_id VARCHAR(64) NOT NULL REFERENCES job_postings(job_id) ON DELETE CASCADE,
        skill_id VARCHAR(32) REFERENCES skills(skill_id),
        skill_mention VARCHAR(256) NOT NULL,
        character_start INTEGER NOT NULL CHECK (character_start >= 0),
        character_end INTEGER NOT NULL CHECK (character_end > character_start),
        source_section VARCHAR(32) NOT NULL,
        extractor_name VARCHAR(64) NOT NULL,
        match_method VARCHAR(64) NOT NULL,
        mapping_status VARCHAR(64) NOT NULL,
        mapping_confidence REAL NOT NULL CHECK (mapping_confidence >= 0.0 AND mapping_confidence <= 1.0)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_jsm_job_id ON job_skill_mentions(job_id);",
    "CREATE INDEX IF NOT EXISTS idx_jsm_skill_id ON job_skill_mentions(skill_id);",
    "CREATE INDEX IF NOT EXISTS idx_jsm_section ON job_skill_mentions(source_section);",
    """
    CREATE TABLE IF NOT EXISTS occupation_skill_demand (
        occupation_id VARCHAR(32) NOT NULL REFERENCES occupations(occupation_id),
        skill_id VARCHAR(32) NOT NULL REFERENCES skills(skill_id),
        job_count INTEGER NOT NULL CHECK (job_count >= 0),
        required_count INTEGER NOT NULL CHECK (required_count >= 0),
        preferred_count INTEGER NOT NULL CHECK (preferred_count >= 0),
        occupation_total_postings INTEGER NOT NULL CHECK (occupation_total_postings > 0),
        demand_share REAL NOT NULL CHECK (demand_share >= 0.0 AND demand_share <= 1.0),
        PRIMARY KEY (occupation_id, skill_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_osd_skill_id ON occupation_skill_demand(skill_id);",
    """
    CREATE TABLE IF NOT EXISTS skill_co_occurrences (
        source_skill_id VARCHAR(32) NOT NULL REFERENCES skills(skill_id),
        target_skill_id VARCHAR(32) NOT NULL REFERENCES skills(skill_id),
        co_occurrence_count INTEGER NOT NULL CHECK (co_occurrence_count >= 1),
        jaccard_similarity REAL NOT NULL CHECK (jaccard_similarity >= 0.0 AND jaccard_similarity <= 1.0),
        pmi_score REAL NOT NULL,
        relationship_semantics VARCHAR(64) NOT NULL DEFAULT 'statistical_co_occurrence_not_prerequisite',
        CHECK (source_skill_id < target_skill_id),
        PRIMARY KEY (source_skill_id, target_skill_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS university_majors (
        major_id VARCHAR(64) PRIMARY KEY,
        major_name VARCHAR(256) NOT NULL UNIQUE,
        institution_name VARCHAR(256) NOT NULL,
        degree_level VARCHAR(64) NOT NULL,
        data_provenance VARCHAR(64) NOT NULL DEFAULT 'awaiting_verified_curriculum_ingestion'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS university_courses (
        course_id VARCHAR(64) PRIMARY KEY,
        course_code VARCHAR(32) NOT NULL,
        course_title VARCHAR(256) NOT NULL,
        major_id VARCHAR(64) REFERENCES university_majors(major_id) ON DELETE CASCADE,
        syllabus_text TEXT,
        data_provenance VARCHAR(64) NOT NULL DEFAULT 'awaiting_verified_curriculum_ingestion',
        UNIQUE (major_id, course_code)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS course_skills (
        course_id VARCHAR(64) NOT NULL REFERENCES university_courses(course_id) ON DELETE CASCADE,
        skill_id VARCHAR(32) NOT NULL REFERENCES skills(skill_id),
        extraction_method VARCHAR(64) NOT NULL,
        target_proficiency_level VARCHAR(32),
        PRIMARY KEY (course_id, skill_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS students (
        student_id VARCHAR(64) PRIMARY KEY,
        major_id VARCHAR(64) REFERENCES university_majors(major_id),
        target_occupation_id VARCHAR(32) REFERENCES occupations(occupation_id),
        is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic IN (0, 1)),
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS student_skills (
        student_id VARCHAR(64) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
        skill_id VARCHAR(32) NOT NULL REFERENCES skills(skill_id),
        proficiency_level VARCHAR(32) NOT NULL,
        proficiency_score REAL CHECK (proficiency_score IS NULL OR (proficiency_score >= 0.0 AND proficiency_score <= 1.0)),
        evidence_source VARCHAR(128),
        PRIMARY KEY (student_id, skill_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS skill_gap_analyses (
        analysis_id VARCHAR(64) PRIMARY KEY,
        student_id VARCHAR(64) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
        target_occupation_id VARCHAR(32) NOT NULL REFERENCES occupations(occupation_id),
        required_coverage REAL NOT NULL CHECK (required_coverage >= 0.0 AND required_coverage <= 1.0),
        overall_coverage REAL NOT NULL CHECK (overall_coverage >= 0.0 AND overall_coverage <= 1.0),
        skill_gap_score REAL NOT NULL CHECK (skill_gap_score >= 0.0 AND skill_gap_score <= 1.0),
        weights_json TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic IN (0, 1)),
        created_at TEXT NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_sga_student_id ON skill_gap_analyses(student_id);",
    """
    CREATE TABLE IF NOT EXISTS recommendation_explanations (
        explanation_id INTEGER PRIMARY KEY,
        analysis_id VARCHAR(64) NOT NULL REFERENCES skill_gap_analyses(analysis_id) ON DELETE CASCADE,
        student_id VARCHAR(64) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
        skill_id VARCHAR(32) NOT NULL REFERENCES skills(skill_id),
        rank_order INTEGER NOT NULL CHECK (rank_order >= 1),
        requirement_type VARCHAR(32) NOT NULL,
        profile_status VARCHAR(64) NOT NULL,
        priority_score REAL NOT NULL CHECK (priority_score >= 0.0 AND priority_score <= 1.0),
        learning_stage VARCHAR(128) NOT NULL,
        explanation_json TEXT NOT NULL,
        UNIQUE (analysis_id, skill_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_re_student_id ON recommendation_explanations(student_id);",
)


def get_postgresql_ddl() -> str:
    """Returns non-destructive PostgreSQL DDL for all 17 SkillGraph tables and indexes."""
    # Translate SQLite integer-key convenience syntax cleanly to PostgreSQL BIGSERIAL/BOOLEAN/NUMERIC
    pg_blocks: list[str] = []
    for stmt in POSTGRESQL_DDL_STATEMENTS:
        s = stmt.strip()
        s = s.replace("alias_id INTEGER PRIMARY KEY", "alias_id BIGSERIAL PRIMARY KEY")
        s = s.replace("mention_id INTEGER PRIMARY KEY", "mention_id BIGSERIAL PRIMARY KEY")
        s = s.replace("explanation_id INTEGER PRIMARY KEY", "explanation_id BIGSERIAL PRIMARY KEY")
        s = s.replace("case_sensitive INTEGER NOT NULL DEFAULT 0 CHECK (case_sensitive IN (0, 1))", "case_sensitive BOOLEAN NOT NULL DEFAULT FALSE")
        s = s.replace("is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic IN (0, 1))", "is_synthetic BOOLEAN NOT NULL DEFAULT TRUE")
        s = s.replace("created_at TEXT NOT NULL", "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        s = s.replace("weights_json TEXT NOT NULL", "weights_json JSONB NOT NULL")
        s = s.replace("payload_json TEXT NOT NULL", "payload_json JSONB NOT NULL")
        s = s.replace("explanation_json TEXT NOT NULL", "explanation_json JSONB NOT NULL")
        s = s.replace("REAL", "NUMERIC(8,4)")
        pg_blocks.append(s)
    return "\n\n".join(pg_blocks) + "\n"


class RelationalDatabaseManager:
    """Relational SQL schema manager and offline SQLite execution mirror for PostgreSQL."""

    def __init__(self, builder: SkillGraphBuilder | None = None, db_path: str = ":memory:") -> None:
        self.builder = builder or SkillGraphBuilder(include_broad_country_mappings=False, min_co_occurrence=5)
        if not self.builder._built:
            self.builder.build()
        self.db_path = db_path
        # check_same_thread=False allows safe reuse inside ASGI/test threads on in-memory DB
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._initialize_schema_and_seed()

    def _initialize_schema_and_seed(self) -> None:
        cur = self.conn.cursor()
        for stmt in POSTGRESQL_DDL_STATEMENTS:
            cur.execute(stmt)

        # Seed reference & labor-market tables cleanly from SkillGraphBuilder (idempotent)
        for r in self.builder.nodes["SkillCategory"].itertuples(index=False):
            cur.execute(
                "INSERT OR IGNORE INTO skill_categories (category_id, category_name, skill_count) VALUES (?, ?, ?)",
                (r.category_id, r.category_name, int(r.skill_count)),
            )

        for r in self.builder.nodes["Skill"].itertuples(index=False):
            cur.execute(
                """INSERT OR IGNORE INTO skills
                (skill_id, canonical_skill, category_id, skill_type, taxonomy_source, esco_uri, onet_element_id, mapping_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r.skill_id, r.canonical_skill, r.category_id, r.skill_type, r.taxonomy_source,
                    None if pd.isna(r.esco_uri) else str(r.esco_uri),
                    None if pd.isna(r.onet_element_id) else str(r.onet_element_id),
                    r.mapping_status,
                ),
            )

        for row in self.builder.df_taxonomy.itertuples(index=False):
            for alias in parse_aliases_field(getattr(row, "aliases", None)):
                cur.execute(
                    "INSERT OR IGNORE INTO skill_aliases (skill_id, alias_text, case_sensitive) VALUES (?, ?, ?)",
                    (row.skill_id, alias, 1 if alias.isupper() else 0),
                )

        for r in self.builder.nodes["Industry"].itertuples(index=False):
            cur.execute(
                "INSERT OR IGNORE INTO industries (industry_id, industry_name, occupation_count) VALUES (?, ?, ?)",
                (r.industry_id, r.industry_name, int(r.occupation_count)),
            )

        for r in self.builder.nodes["Occupation"].itertuples(index=False):
            cur.execute(
                """INSERT OR IGNORE INTO occupations
                (occupation_id, occupation, industry_id, department_label, isco08_code_candidate,
                 isco08_label_candidate, mapping_confidence, mapping_status, esco_occupation_uri, onet_soc_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r.occupation_id, r.occupation, r.industry_id, r.department_label,
                    None if pd.isna(r.isco08_code_candidate) else str(r.isco08_code_candidate),
                    None if pd.isna(r.isco08_label_candidate) else str(r.isco08_label_candidate),
                    None if pd.isna(r.mapping_confidence) else float(r.mapping_confidence),
                    r.mapping_status,
                    None if pd.isna(r.esco_occupation_uri) else str(r.esco_occupation_uri),
                    None if pd.isna(r.onet_soc_code) else str(r.onet_soc_code),
                ),
            )

        for r in self.builder.nodes["Country"].itertuples(index=False):
            cur.execute(
                "INSERT OR IGNORE INTO countries (country_code, country, region) VALUES (?, ?, ?)",
                (r.country_code, r.country, r.region),
            )

        desc_lookup = dict(zip(self.builder.df_jobs["job_id"], self.builder.df_jobs["description"]))
        job_rows = [
            (
                r.job_id, r.occupation_id, r.country_code, r.job_title, r.seniority_level,
                r.education_level,
                None if pd.isna(r.experience_years_min) else int(r.experience_years_min),
                None if pd.isna(r.experience_years_max) else int(r.experience_years_max),
                r.employment_type, str(desc_lookup.get(r.job_id, "")),
                1 if bool(r.is_synthetic) else 0, r.split,
            )
            for r in self.builder.nodes["JobPosting"].itertuples(index=False)
        ]
        cur.executemany(
            """INSERT OR IGNORE INTO job_postings
            (job_id, occupation_id, country_code, job_title, seniority_level, education_level,
             experience_years_min, experience_years_max, employment_type, description, is_synthetic, split)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            job_rows,
        )

        # Insert job skill mentions if empty
        existing_mentions = cur.execute("SELECT COUNT(*) FROM job_skill_mentions").fetchone()[0]
        if existing_mentions == 0:
            mention_df = pd.concat(
                [self.builder.edges["REQUIRES_SKILL"], self.builder.edges["PREFERS_SKILL"]],
                ignore_index=True,
            )
            mention_rows = [
                (
                    r.job_id, r.skill_id, r.skill_mention, int(r.character_start), int(r.character_end),
                    r.source_section, r.extractor_name, r.match_method, r.mapping_status, float(r.mapping_confidence),
                )
                for r in mention_df.itertuples(index=False)
            ]
            cur.executemany(
                """INSERT INTO job_skill_mentions
                (job_id, skill_id, skill_mention, character_start, character_end,
                 source_section, extractor_name, match_method, mapping_status, mapping_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                mention_rows,
            )

        demand_rows = [
            (
                r.occupation_id, r.skill_id, int(r.job_count), int(r.required_count),
                int(r.preferred_count), int(r.occupation_total_postings), float(r.demand_share),
            )
            for r in self.builder.edges["OCCUPATION_DEMANDS_SKILL"].itertuples(index=False)
        ]
        cur.executemany(
            """INSERT OR IGNORE INTO occupation_skill_demand
            (occupation_id, skill_id, job_count, required_count, preferred_count, occupation_total_postings, demand_share)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            demand_rows,
        )

        co_rows = [
            (
                r.source_skill_id, r.target_skill_id, int(r.co_occurrence_count),
                float(r.jaccard_similarity), float(r.pmi_score), r.relationship_semantics,
            )
            for r in self.builder.edges["CO_OCCURS_WITH"].itertuples(index=False)
        ]
        cur.executemany(
            """INSERT OR IGNORE INTO skill_co_occurrences
            (source_skill_id, target_skill_id, co_occurrence_count, jaccard_similarity, pmi_score, relationship_semantics)
            VALUES (?, ?, ?, ?, ?, ?)""",
            co_rows,
        )
        self.conn.commit()

    def list_tables_and_counts(self) -> dict[str, int]:
        """Returns row counts for all 17 normalized schema tables."""
        tables = [
            "skill_categories", "skills", "skill_aliases", "industries", "occupations",
            "countries", "job_postings", "job_skill_mentions", "occupation_skill_demand",
            "skill_co_occurrences", "university_majors", "university_courses",
            "course_skills", "students", "student_skills",
            "skill_gap_analyses", "recommendation_explanations",
        ]
        counts: dict[str, int] = {}
        for t in tables:
            counts[t] = int(self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
        return counts

    def fetch_occupations(self, industry: str | None = None, search: str | None = None) -> list[dict[str, Any]]:
        """Queries normalized occupations joined with industries."""
        sql = """
            SELECT o.occupation_id, o.occupation, o.industry_id, i.industry_name AS industry,
                   o.department_label, o.isco08_code_candidate, o.isco08_label_candidate,
                   (SELECT COUNT(*) FROM job_postings jp WHERE jp.occupation_id = o.occupation_id) AS posting_count
            FROM occupations o
            JOIN industries i ON i.industry_id = o.industry_id
            WHERE 1=1
        """
        params: list[Any] = []
        if industry:
            sql += " AND LOWER(i.industry_name) = LOWER(?)"
            params.append(industry.strip())
        if search:
            sql += " AND LOWER(o.occupation) LIKE ?"
            params.append(f"%{search.strip().lower()}%")
        sql += " ORDER BY o.occupation_id ASC"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def fetch_industries(self) -> list[dict[str, Any]]:
        """Lists normalized industries with occupation counts."""
        sql = """
            SELECT i.industry_id, i.industry_name,
                   (SELECT COUNT(*) FROM occupations o WHERE o.industry_id = i.industry_id) AS occupation_count
            FROM industries i
            ORDER BY i.industry_name ASC
        """
        return [dict(r) for r in self.conn.execute(sql).fetchall()]

    def fetch_majors(self) -> list[dict[str, Any]]:
        """Lists ingested university majors with course counts."""
        sql = """
            SELECT m.major_id, m.major_name, m.institution_name, m.degree_level, m.data_provenance,
                   (SELECT COUNT(*) FROM university_courses c WHERE c.major_id = m.major_id) AS course_count
            FROM university_majors m
            ORDER BY m.major_id ASC
        """
        return [dict(r) for r in self.conn.execute(sql).fetchall()]

    def fetch_courses(self, major_id: str | None = None) -> list[dict[str, Any]]:
        """Lists ingested university courses with extracted skill counts."""
        sql = """
            SELECT c.course_id, c.course_code, c.course_title, c.major_id, c.data_provenance,
                   (SELECT COUNT(*) FROM course_skills cs WHERE cs.course_id = c.course_id) AS skill_count
            FROM university_courses c
            WHERE 1=1
        """
        params: list[Any] = []
        if major_id:
            sql += " AND c.major_id = ?"
            params.append(major_id.strip())
        sql += " ORDER BY c.course_id ASC"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def fetch_skills(
        self,
        category: str | None = None,
        skill_type: str | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        """Queries canonical skills joined with categories."""
        sql = """
            SELECT s.skill_id, s.canonical_skill, sc.category_name AS skill_category,
                   s.skill_type, s.taxonomy_source, s.mapping_status
            FROM skills s
            JOIN skill_categories sc ON sc.category_id = s.category_id
            WHERE 1=1
        """
        params: list[Any] = []
        if category:
            sql += " AND LOWER(sc.category_name) = LOWER(?)"
            params.append(category.strip())
        if skill_type:
            sql += " AND LOWER(s.skill_type) = LOWER(?)"
            params.append(skill_type.strip())
        if search:
            sql += " AND LOWER(s.canonical_skill) LIKE ?"
            params.append(f"%{search.strip().lower()}%")
        sql += " ORDER BY s.skill_id ASC"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def fetch_occupation_skills(self, occupation_id: str) -> dict[str, Any] | None:
        """Fetches occupation metadata plus required and preferred skill breakdown."""
        occ = self.conn.execute(
            """SELECT o.occupation_id, o.occupation, i.industry_name AS industry,
                      o.isco08_code_candidate, o.isco08_label_candidate
               FROM occupations o
               JOIN industries i ON i.industry_id = o.industry_id
               WHERE LOWER(o.occupation_id) = LOWER(?) OR LOWER(o.occupation) = LOWER(?)""",
            (occupation_id.strip(), occupation_id.strip()),
        ).fetchone()
        if occ is None:
            return None

        skills = self.conn.execute(
            """SELECT osd.skill_id, s.canonical_skill, sc.category_name AS skill_category,
                      s.skill_type, osd.job_count, osd.required_count, osd.preferred_count,
                      osd.occupation_total_postings, osd.demand_share
               FROM occupation_skill_demand osd
               JOIN skills s ON s.skill_id = osd.skill_id
               JOIN skill_categories sc ON sc.category_id = s.category_id
               WHERE osd.occupation_id = ?
               ORDER BY osd.required_count DESC, osd.demand_share DESC, s.skill_id ASC""",
            (occ["occupation_id"],),
        ).fetchall()
        rows = [dict(r) for r in skills]
        return {
            **dict(occ),
            "total_skills": len(rows),
            "required_skills": [r for r in rows if r["required_count"] > 0],
            "preferred_only_skills": [r for r in rows if r["required_count"] == 0 and r["preferred_count"] > 0],
            "all_skills": rows,
        }

    def save_analysis_result(
        self,
        analysis: dict[str, Any],
        engine: Any,
        profile: dict[str, Any],
    ) -> None:
        """Persists student profile (separately from job requirements) and gap results."""
        cur = self.conn.cursor()
        student_id = analysis["student_id"]
        occ_id = analysis["target_occupation"]["occupation_id"]
        is_syn = 1 if analysis["is_synthetic_profile"] else 0

        cur.execute(
            """INSERT INTO students (student_id, major_id, target_occupation_id, is_synthetic, created_at)
               VALUES (?, NULL, ?, ?, ?)
               ON CONFLICT(student_id) DO UPDATE SET target_occupation_id = excluded.target_occupation_id""",
            (student_id, occ_id, is_syn, analysis["created_at"]),
        )
        known_map, _, _ = engine._normalize_student_profile(profile, strict_validation=False)
        for sid, entry in known_map.items():
            cur.execute(
                """INSERT INTO student_skills (student_id, skill_id, proficiency_level, proficiency_score, evidence_source)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(student_id, skill_id) DO UPDATE SET
                       proficiency_level = excluded.proficiency_level,
                       proficiency_score = excluded.proficiency_score,
                       evidence_source = excluded.evidence_source""",
                (student_id, sid, entry["proficiency_level"], entry["proficiency_score"], entry["evidence_source"]),
            )

        cov = analysis["coverage_metrics"]
        weights_json = json.dumps(
            analysis["priority_recommendations"][0]["scoring"]["weights_applied"]
            if analysis["priority_recommendations"]
            else asdict(BASELINE_WEIGHT_PRESETS["baseline_balanced_heuristic"])
        )
        cur.execute(
            """INSERT INTO skill_gap_analyses
               (analysis_id, student_id, target_occupation_id, required_coverage,
                overall_coverage, skill_gap_score, weights_json, payload_json, is_synthetic, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis["analysis_id"],
                student_id,
                occ_id,
                cov["required_skill_coverage"],
                cov["overall_weighted_coverage"],
                cov["skill_gap_score"],
                weights_json,
                json.dumps(analysis),
                is_syn,
                analysis["created_at"],
            ),
        )

        for rec in analysis["priority_recommendations"]:
            cur.execute(
                """INSERT INTO recommendation_explanations
                   (analysis_id, student_id, skill_id, rank_order, requirement_type,
                    profile_status, priority_score, learning_stage, explanation_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis["analysis_id"],
                    student_id,
                    rec["skill_id"],
                    rec["rank"],
                    rec["requirement_type"],
                    rec["profile_status"],
                    rec["priority_score"],
                    rec["learning_stage"],
                    json.dumps(rec["explanation"]),
                ),
            )
        self.conn.commit()

    def fetch_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        """Retrieves a persisted analysis payload by `analysis_id`."""
        row = self.conn.execute(
            "SELECT payload_json FROM skill_gap_analyses WHERE analysis_id = ?",
            (analysis_id.strip(),),
        ).fetchone()
        return None if row is None else json.loads(row["payload_json"])

    def fetch_latest_student_recommendations(self, student_id: str) -> dict[str, Any] | None:
        """Retrieves latest recommendations and explanations for a `student_id`."""
        row = self.conn.execute(
            """SELECT payload_json FROM skill_gap_analyses
               WHERE student_id = ?
               ORDER BY created_at DESC, rowid DESC LIMIT 1""",
            (student_id.strip(),),
        ).fetchone()
        return None if row is None else json.loads(row["payload_json"])


def main() -> None:
    """Builds and validates the SkillGraph Knowledge Graph and Week 3 Gap Engine."""
    builder = SkillGraphBuilder(include_broad_country_mappings=False, min_co_occurrence=5)
    summary = builder.build()
    validation = builder.validate_graph()

    print("=== SkillGraph Week 2 Knowledge Graph Build & Validation ===")
    print(f"Total active nodes: {summary['total_active_nodes']} -> {summary['node_counts']}")
    print(f"Total active edges: {summary['total_active_edges']} -> {summary['edge_counts']}")
    print(f"Referential integrity passed: {validation['referential_integrity_passed']} (anomalies={validation['total_anomalies']})")
    print(f"NetworkX installed in .venv: {summary['networkx_installed']}")

    df_top_co = builder.edges["CO_OCCURS_WITH"].head(5)
    print("\nTop 5 Symmetric Skill Co-Occurrences (statistical_co_occurrence_not_prerequisite):")
    for row in df_top_co.itertuples(index=False):
        print(f"  {row.source_skill} <-> {row.target_skill}: count={row.co_occurrence_count}, Jaccard={row.jaccard_similarity}, PMI={row.pmi_score}")

    gap = builder.explain_skill_gap(known_skills=["Python", "SQL", "Communication"], target_occupation="Data Scientist", top_k=5)
    print("\nSample Explainable Skill Gap (Student={Python, SQL, Communication} -> Data Scientist):")
    for row in gap.itertuples(index=False):
        print(f"  - {row.missing_skill}: {row.explanation}")

    db = RelationalDatabaseManager(builder)
    pg_env = inspect_postgresql_environment()
    print(f"SQL Schema Tables Initialized: {len(db.list_tables_and_counts())} | PostgreSQL port_5432={pg_env['port_5432_status']} | pg_drivers_ready={pg_env['live_pg_execution_ready']}")


if __name__ == "__main__":
    main()

