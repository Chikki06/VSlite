#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import os
import sqlite3
import sys
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upsert inference image metadata into viewer DB")
    parser.add_argument("--db-path", required=True, help="Path to sqlite DB file")
    parser.add_argument("--manifest", required=True, help="CSV with name,resolution,image_type,project")
    parser.add_argument("--table", default=None, help="Optional explicit table name")
    parser.add_argument(
        "--helper-path",
        default="/home/labuser/SocketInference/viewer/misc",
        help="Path containing available_images_db_helper.py",
    )
    return parser.parse_args()


def read_manifest(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"name", "resolution", "image_type", "project"}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        for row in reader:
            if not row.get("name"):
                continue
            rows.append({
                "name": row["name"].strip(),
                "resolution": row["resolution"].strip(),
                "image_type": row["image_type"].strip(),
                "project": row["project"].strip(),
            })
    return rows


def try_helper_upsert(rows: List[Dict[str, str]], helper_path: str) -> Optional[Tuple[int, int]]:
    if not os.path.isdir(helper_path):
        return None

    if helper_path not in sys.path:
        sys.path.insert(0, helper_path)

    try:
        from available_images_db_helper import add_available_image  # type: ignore
    except Exception:
        return None

    success = 0
    failures = 0
    for row in rows:
        try:
            add_available_image(
                row["name"],
                row["resolution"],
                row["image_type"],
                project=row["project"],
            )
            success += 1
        except Exception as exc:
            print(f"helper upsert failed for {row['name']}: {exc}", file=sys.stderr)
            failures += 1
    return success, failures


def pick_table(conn: sqlite3.Connection, explicit: Optional[str]) -> str:
    cur = conn.cursor()
    tables = {
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    if explicit:
        if explicit not in tables:
            raise ValueError(f"Requested table not found: {explicit}")
        return explicit

    for candidate in ("tissue_images", "available_images", "images"):
        if candidate in tables:
            return candidate

    raise ValueError("Could not auto-detect target table. Use --table.")


def get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    pragma = cur.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in pragma]


def first_present(columns: List[str], candidates: List[str]) -> Optional[str]:
    colset = set(columns)
    for candidate in candidates:
        if candidate in colset:
            return candidate
    return None


def sqlite_fallback_upsert(db_path: str, rows: List[Dict[str, str]], table: Optional[str]) -> Tuple[int, int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    selected_table = pick_table(conn, table)
    columns = get_columns(conn, selected_table)

    name_col = first_present(columns, ["name", "image_name", "filename"])
    resolution_col = first_present(columns, ["resolution", "image_resolution"])
    image_type_col = first_present(columns, ["image_type", "type", "modality"])
    project_col = first_present(columns, ["project", "project_name"])
    updated_at_col = first_present(columns, ["updated_at", "updated_on", "modified_at"])
    created_at_col = first_present(columns, ["created_at", "created_on"])

    required = [name_col, resolution_col, image_type_col]
    if any(col is None for col in required):
        raise ValueError(
            "Could not map required columns (name/resolution/image_type) for fallback sqlite upsert"
        )

    success = 0
    failures = 0

    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for row in rows:
        try:
            payload: Dict[str, str] = {
                name_col: row["name"],  # type: ignore[index]
                resolution_col: row["resolution"],  # type: ignore[index]
                image_type_col: row["image_type"],  # type: ignore[index]
            }
            if project_col is not None:
                payload[project_col] = row["project"]
            if updated_at_col is not None:
                payload[updated_at_col] = now

            where_cols = [name_col, image_type_col]
            if project_col is not None:
                where_cols.append(project_col)

            where_sql = " AND ".join([f"{col} = ?" for col in where_cols])
            where_vals = [payload[col] for col in where_cols]

            found = cur.execute(
                f"SELECT rowid FROM {selected_table} WHERE {where_sql} LIMIT 1",
                where_vals,
            ).fetchone()

            if found is not None:
                update_cols = [resolution_col]
                if updated_at_col is not None:
                    update_cols.append(updated_at_col)
                update_sql = ", ".join([f"{col} = ?" for col in update_cols])
                update_vals = [payload[col] for col in update_cols] + where_vals
                cur.execute(
                    f"UPDATE {selected_table} SET {update_sql} WHERE {where_sql}",
                    update_vals,
                )
            else:
                insert_payload = dict(payload)
                if created_at_col is not None:
                    insert_payload[created_at_col] = now
                cols_sql = ", ".join(insert_payload.keys())
                placeholders = ", ".join(["?" for _ in insert_payload])
                cur.execute(
                    f"INSERT INTO {selected_table} ({cols_sql}) VALUES ({placeholders})",
                    list(insert_payload.values()),
                )

            success += 1
        except Exception as exc:
            print(f"sqlite fallback failed for {row['name']}: {exc}", file=sys.stderr)
            failures += 1

    conn.commit()
    conn.close()
    return success, failures


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.db_path):
        print(f"DB path does not exist: {args.db_path}", file=sys.stderr)
        return 2

    rows = read_manifest(args.manifest)
    if not rows:
        print("Manifest is empty; nothing to update")
        return 0

    helper_result = try_helper_upsert(rows, args.helper_path)
    if helper_result is not None:
        success, failures = helper_result
        print(f"DB update via helper complete: success={success} failures={failures}")
        return 0 if failures == 0 else 3

    success, failures = sqlite_fallback_upsert(args.db_path, rows, args.table)
    print(f"DB update via sqlite fallback complete: success={success} failures={failures}")
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
