"""SkillGraph Week 4 / Member 02 Helper: PostgreSQL 17 Initialization & Audit Script.

Reads connection settings strictly from environment variables (or optional local `.env`
file without storing secrets in source code), verifies driver availability, and runs
``get_postgresql_ddl()`` idempotently against PostgreSQL 17 when ready.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.build_graph import (  # noqa: E402
    POSTGRESQL_DDL_STATEMENTS,
    get_postgresql_ddl,
    inspect_postgresql_environment,
)


def load_local_dotenv(dotenv_path: Path | None = None) -> list[str]:
    """Loads key=value pairs from local `.env` into os.environ without overwriting existing vars."""
    target = dotenv_path or (ROOT_DIR / ".env")
    loaded_keys: list[str] = []
    if not target.exists():
        return loaded_keys
    for raw_line in target.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
            loaded_keys.append(key)
    return loaded_keys


def run_postgres_init(dry_run: bool = False) -> dict[str, Any]:
    """Audits PostgreSQL readiness and applies DDL if drivers and credentials are configured."""
    loaded_env = load_local_dotenv()
    env_audit = inspect_postgresql_environment()
    ddl_sql = get_postgresql_ddl()
    ddl_statements = [s.strip() for s in ddl_sql.split(";\n\n") if s.strip()]

    result: dict[str, Any] = {
        "dotenv_loaded_keys": loaded_env,
        "ddl_statement_count": len(POSTGRESQL_DDL_STATEMENTS),
        "postgresql_env": env_audit,
        "executed_against_postgres": False,
        "status": "DRY_RUN_VALIDATED" if dry_run else "AWAITING_DRIVER_OR_CREDENTIALS",
    }

    if dry_run or not env_audit["live_pg_execution_ready"]:
        return result

    host = os.environ.get("SKILLGRAPH_DB_HOST", "127.0.0.1")
    port = int(os.environ.get("SKILLGRAPH_DB_PORT", "5432"))
    dbname = os.environ.get("SKILLGRAPH_DB_NAME", "skillgraph")
    user = os.environ.get("SKILLGRAPH_DB_USER", "postgres")
    password = os.environ.get("SKILLGRAPH_DB_PASSWORD", "")

    driver = None
    for pkg in ("psycopg", "psycopg2"):
        if env_audit["python_pg_drivers"].get(pkg):
            driver = importlib.import_module(pkg)
            break

    if driver is None:
        return result

    conn = driver.connect(host=host, port=port, dbname=dbname, user=user, password=password)
    try:
        with conn:
            with conn.cursor() as cur:
                for stmt in ddl_statements:
                    cur.execute(stmt)
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"
                )
                table_count = int(cur.fetchone()[0])
        result["executed_against_postgres"] = True
        result["public_tables_count"] = table_count
        result["status"] = "POSTGRESQL_DDL_APPLIED"
    finally:
        conn.close()

    return result


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    report = run_postgres_init(dry_run=dry_run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
