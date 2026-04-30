"""BIRD schema helpers - DevTablesIndex and column meaning extraction."""

import csv
import json
import os
import re
from typing import Dict, List, Optional


class DevTablesIndex:
    def __init__(self, tables_json_path: str):
        self.by_db: Dict[str, dict] = {}
        if not tables_json_path:
            return
        with open(tables_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            self.by_db[rec["db_id"]] = rec

    def get(self, db_id: str) -> Optional[dict]:
        return self.by_db.get(db_id)


def schema_from_dev_tables(
    rec: dict, col_meanings: Dict[str, str] | None = None
) -> str:
    col_meanings = col_meanings or {}

    table_names = rec["table_names_original"]
    col_names = rec["column_names_original"]
    col_types = rec.get("column_types", [])
    pks = rec.get("primary_keys", [])
    fks = rec.get("foreign_keys", [])

    pk_set = set()
    for pk in pks:
        if isinstance(pk, list):
            pk_set.update(pk)
        else:
            pk_set.add(pk)

    cols_by_table: Dict[int, List[int]] = {i: [] for i in range(len(table_names))}
    for idx, (tid, _) in enumerate(col_names):
        if tid != -1:
            cols_by_table[tid].append(idx)

    lines = []
    for tid, tname in enumerate(table_names):
        lines.append(f"TABLE {tname}")
        for cidx in cols_by_table[tid]:
            _, cname = col_names[cidx]
            ctype = col_types[cidx] if cidx < len(col_types) else ""
            pk_tag = " PRIMARY KEY" if cidx in pk_set else ""

            key = f"{tname}.{cname}"
            meaning = col_meanings.get(key, "")
            meaning_tag = f" -- {meaning}" if meaning else ""

            lines.append(f"  - {cname} ({ctype}){pk_tag}{meaning_tag}")
        lines.append("")

    if fks:
        lines.append("FOREIGN KEYS")
        for a, b in fks:
            ta, ca = col_names[a]
            tb, cb = col_names[b]
            if ta != -1 and tb != -1:
                lines.append(f"  - {table_names[ta]}.{ca} -> {table_names[tb]}.{cb}")

    return "\n".join(lines).strip()


def column_meanings_from_dev_tables(rec: dict) -> dict:
    table_names = rec["table_names_original"]
    col_orig = rec["column_names_original"]
    col_human = rec.get("column_names", col_orig)

    meanings = {}
    for idx, (tid, orig_name) in enumerate(col_orig):
        if tid == -1:
            continue
        human_name = col_human[idx][1] if idx < len(col_human) else orig_name
        key = f"{table_names[tid]}.{orig_name}"
        if human_name and human_name != orig_name:
            meanings[key] = human_name
    return meanings


def _clean(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_column_meanings_from_db_desc(
    db_root: str, db_id: str, max_chars_per_col: int = 300
) -> dict:
    desc_dir = os.path.join(db_root, db_id, "database_description")
    if not os.path.isdir(desc_dir):
        return {}

    meanings = {}
    for fname in sorted(os.listdir(desc_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        table = os.path.splitext(fname)[0]
        path = os.path.join(desc_dir, fname)

        try:
            with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    orig = _clean(row.get("original_column_name", ""))
                    if not orig:
                        continue
                    col_desc = _clean(row.get("column_description", ""))
                    val_desc = _clean(row.get("value_description", ""))
                    data_fmt = _clean(row.get("data_format", ""))

                    parts = []
                    if col_desc and col_desc.lower() != orig.lower():
                        parts.append(col_desc)
                    else:
                        parts.append(orig)
                    if data_fmt:
                        parts.append(f"type={data_fmt}")
                    if val_desc:
                        parts.append(val_desc)

                    meaning = " | ".join([p for p in parts if p])[:max_chars_per_col]

                    key = f"{table}.{orig}"
                    meanings[key] = meaning
        except Exception:
            continue

    return meanings
