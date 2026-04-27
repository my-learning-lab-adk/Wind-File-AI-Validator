"""
file_scanner.py
Scans a wind-data TSV file for:
  - metadata (latitude, longitude, elevation)
  - NaN / blank cells
  - 9999 sentinel cells
  - column name list (raw + normalized)
No LLM involved — pure pandas. Exact, full-coverage sweep.
"""

import logging
import os
import re
import unicodedata
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ---------- helpers ----------

def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )

def normalize_column(name: str) -> str:
    """Mirrors the app's normalize_dataframe_labels() behaviour."""
    if name is None:
        return ""
    s = str(name)
    s = _strip_accents(s)
    s = s.replace("\xa0", "_").replace("'", "").replace("\u2019", "")
    s = s.lower()
    s = re.sub(r"[ /:?()\-]", "_", s)
    s = re.sub(r"\[", "[", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _detect_header_row(path: str, encoding: str = "ISO-8859-1", max_scan: int = 60) -> int:
    """Find the line index (0-based) that starts with Date/Time or Timestamp."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        with open(path, "r", encoding=encoding, errors="replace") as f:
            for i, line in enumerate(f):
                if i > max_scan:
                    break
                low = line.strip().lower()
                if low.startswith("date/time") or low.startswith("timestamp") or low.startswith("date\ttime"):
                    return i
    except OSError as exc:
        raise ValueError(f"Cannot read file: {exc}") from exc
    raise ValueError("Could not find a Date/Time or Timestamp header row in the first 60 lines.")


def _read_metadata(path: str, header_row: int, encoding: str = "ISO-8859-1") -> dict:
    """Pull latitude / longitude / elevation from the lines above the header."""
    meta = {"latitude": None, "longitude": None, "elevation": None, "raw_header_lines": []}
    with open(path, "r", encoding=encoding, errors="replace") as f:
        for i, line in enumerate(f):
            if i >= header_row:
                break
            line = line.rstrip("\r\n")
            meta["raw_header_lines"].append(line)
            low = line.lower()
            if "latitude" in low:
                m = re.search(r"[-+]?\d+(?:\.\d+)?", line)
                if m:
                    meta["latitude"] = float(m.group())
            elif "longitude" in low:
                m = re.search(r"[-+]?\d+(?:\.\d+)?", line)
                if m:
                    meta["longitude"] = float(m.group())
            elif "elevation" in low:
                m = re.search(r"[-+]?\d+(?:\.\d+)?\s*\w*", line)
                if m:
                    meta["elevation"] = m.group().strip()
    return meta


# ---------- main scan ----------

def scan_file(path: str, encoding: str = "ISO-8859-1") -> dict:
    """
    Run a full sweep of the TSV file.

    Returns a dict with:
      metadata          -> lat/lon/elev + raw header lines
      columns           -> [{original, normalized}, ...]
      na_cells          -> [{column, rows:[...], count}, ...]   (NaN or blank)
      sentinel_9999     -> [{column, rows:[...], count}, ...]
      all_missing_cols  -> columns where every value is NA/blank/9999
      shape             -> (rows, cols)
      timestamp_column  -> name of timestamp col (best guess)
      sample_rows       -> first 5 rows as list of dicts
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError("The uploaded file is empty.")

    header_row = _detect_header_row(path, encoding=encoding)
    metadata = _read_metadata(path, header_row, encoding=encoding)

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            skiprows=header_row,
            encoding=encoding,
            engine="python",
            on_bad_lines="skip",
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse the file as TSV: {exc}") from exc

    # drop fully-empty junk columns like "Unnamed: 15" that are entirely NaN
    df = df.dropna(axis=1, how="all")

    if df.empty:
        raise ValueError("The file contains no data rows after parsing.")
    if df.shape[1] == 0:
        raise ValueError("The file contains no valid columns after removing empty ones.")

    columns_info = [
        {"original": str(c), "normalized": normalize_column(c)} for c in df.columns
    ]

    # --- timestamp column (detected early so other checks can skip it) ---
    ts_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl.startswith("date/time") or cl.startswith("timestamp") or cl == "date_time":
            ts_col = str(c)
            break

    # --- NaN / blank cells ---
    na_cells = []
    for col in df.columns:
        col_series = df[col]
        if col_series.dtype == object:
            mask = col_series.isna() | col_series.astype(str).str.strip().eq("")
        else:
            mask = col_series.isna()
        rows = df.index[mask].tolist()
        if rows:
            na_cells.append(
                {"column": str(col), "rows": rows[:200], "count": int(len(rows))}
            )

    # --- 9999 sentinel cells ---
    sentinel_cells = []
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        mask = numeric == 9999
        rows = df.index[mask].tolist()
        if rows:
            sentinel_cells.append(
                {"column": str(col), "rows": rows[:200], "count": int(len(rows))}
            )

    # --- special character cells (exclude timestamp column, allow digits, letters, whitespace, dot) ---
    special_char_cells = []
    special_char_pattern = re.compile(r'[^\w\s.]', re.UNICODE)
    for col in df.columns:
        if ts_col and str(col) == ts_col:
            continue
        col_series = df[col].dropna().astype(str)
        mask = col_series.apply(lambda v: bool(special_char_pattern.search(v)) if v.strip() != "" else False)
        rows = col_series.index[mask].tolist()
        if rows:
            special_char_cells.append(
                {"column": str(col), "rows": rows[:200], "count": int(len(rows))}
            )

    # --- duplicate rows (based on timestamp column, keep first) ---
    duplicate_rows = []
    if ts_col:
        dupes = df[ts_col].duplicated(keep="first")
        duplicate_rows = df.index[dupes].tolist()

    # --- columns where everything is missing ---
    all_missing = []
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        is_na = df[col].isna() | (df[col].astype(str).str.strip() == "")
        is_9999 = numeric == 9999
        if (is_na | is_9999).all():
            all_missing.append(str(col))

    # --- build grouped validation messages ---
    grouped_messages = {}

    if na_cells:
        msg = "Invalid cell due to the presence of nan"
        grouped_messages[msg] = [
            {"row": row, "column": item["column"]}
            for item in na_cells for row in item["rows"]
        ]

    if sentinel_cells:
        msg = "Invalid cell due to the presence of 9999"
        grouped_messages[msg] = [
            {"row": row, "column": item["column"]}
            for item in sentinel_cells for row in item["rows"]
        ]

    if special_char_cells:
        msg = "Invalid cell due to the presence of special character"
        grouped_messages[msg] = [
            {"row": row, "column": item["column"]}
            for item in special_char_cells for row in item["rows"]
        ]

    if duplicate_rows:
        grouped_messages["Duplicate row"] = [{"row": row} for row in duplicate_rows[:200]]

    return {
        "metadata": metadata,
        "columns": columns_info,
        "na_cells": na_cells,
        "sentinel_9999": sentinel_cells,
        "special_char_cells": special_char_cells,
        "duplicate_rows": duplicate_rows[:200],
        "duplicate_row_count": len(duplicate_rows),
        "all_missing_cols": all_missing,
        "shape": (int(df.shape[0]), int(df.shape[1])),
        "timestamp_column": ts_col,
        "sample_rows": df.head(5).astype(str).to_dict(orient="records"),
        "validation_messages": grouped_messages,
    }


def build_report_text(scan: dict, max_rows_shown: int = 10) -> str:
    """
    Full diagnostic report of the scanned file, structured for LLM consumption.
    Covers: metadata, shape, all column names, per-column NaN/9999 counts,
    affected row indices, and a top-level summary.
    """
    lines = []
    m          = scan["metadata"]
    total_rows = scan["shape"][0]
    total_cols = scan["shape"][1]

    na_map           = {i["column"]: i for i in scan["na_cells"]}
    sentinel_map     = {i["column"]: i for i in scan["sentinel_9999"]}
    special_char_map = {i["column"]: i for i in scan.get("special_char_cells", [])}
    total_nan        = sum(i["count"] for i in scan["na_cells"])
    total_9999       = sum(i["count"] for i in scan["sentinel_9999"])
    total_special    = sum(i["count"] for i in scan.get("special_char_cells", []))
    cols_with_nan        = len(scan["na_cells"])
    cols_with_9999       = len(scan["sentinel_9999"])
    cols_with_special    = len(scan.get("special_char_cells", []))
    duplicate_row_count  = scan.get("duplicate_row_count", 0)
    fully_missing        = len(scan["all_missing_cols"])
    clean_cols           = total_cols - len(
        {i["column"] for i in scan["na_cells"]} |
        {i["column"] for i in scan["sentinel_9999"]} |
        {i["column"] for i in scan.get("special_char_cells", [])}
    )

    # ── 1. SUMMARY ────────────────────────────────────────────────────────────
    lines.append("=== SUMMARY ===")
    lines.append(f"Total rows             : {total_rows}")
    lines.append(f"Total columns          : {total_cols}")
    lines.append(f"Timestamp column       : {scan['timestamp_column']}")
    lines.append(f"Columns with NaN/blank : {cols_with_nan}  (total NaN cells: {total_nan})")
    lines.append(f"Columns with 9999      : {cols_with_9999}  (total 9999 cells: {total_9999})")
    lines.append(f"Columns with special ch: {cols_with_special}  (total special char cells: {total_special})")
    lines.append(f"Duplicate rows         : {duplicate_row_count}")
    lines.append(f"Fully-missing columns  : {fully_missing}")
    lines.append(f"Clean columns          : {clean_cols}  (no NaN, no 9999, no special chars)")

    # ── 2. METADATA ───────────────────────────────────────────────────────────
    lines.append("\n=== METADATA ===")
    lines.append(f"Latitude : {m['latitude']}")
    lines.append(f"Longitude: {m['longitude']}")
    lines.append(f"Elevation: {m['elevation']}")

    # ── 3. COLUMN NAMES ───────────────────────────────────────────────────────
    lines.append(f"\n=== COLUMNS ({total_cols} total) ===")
    lines.append("  #   Original name               Normalized name")
    lines.append("  " + "-" * 60)
    for i, c in enumerate(scan["columns"], 1):
        lines.append(f"  {i:<4}{c['original']:<32}{c['normalized']}")

    # ── 4. FULLY-MISSING COLUMNS ──────────────────────────────────────────────
    lines.append("\n=== FULLY-MISSING COLUMNS (all rows are NaN or 9999) ===")
    if scan["all_missing_cols"]:
        for col in scan["all_missing_cols"]:
            lines.append(f"  • {col}")
    else:
        lines.append("  (none)")

    # ── 5. NaN / BLANK CELLS — per column ────────────────────────────────────
    lines.append(f"\n=== NaN / BLANK CELLS ({cols_with_nan} columns affected, {total_nan} total cells) ===")
    if not scan["na_cells"]:
        lines.append("  (none — file has no missing or blank values)")
    for item in scan["na_cells"]:
        shown = item["rows"][:max_rows_shown]
        extra = (
            f" … (+{item['count'] - max_rows_shown} more rows)"
            if item["count"] > max_rows_shown else ""
        )
        pct = item["count"] / total_rows * 100 if total_rows else 0
        lines.append(
            f"  {item['column']}: {item['count']} missing ({pct:.1f}% of rows)"
            f"  — row indices: {shown}{extra}"
        )

    # ── 6. 9999 SENTINEL CELLS — per column ───────────────────────────────────
    lines.append(f"\n=== 9999 SENTINEL CELLS ({cols_with_9999} columns affected, {total_9999} total cells) ===")
    if not scan["sentinel_9999"]:
        lines.append("  (none — no 9999 sentinel values found)")
    for item in scan["sentinel_9999"]:
        shown = item["rows"][:max_rows_shown]
        extra = (
            f" … (+{item['count'] - max_rows_shown} more rows)"
            if item["count"] > max_rows_shown else ""
        )
        pct = item["count"] / total_rows * 100 if total_rows else 0
        lines.append(
            f"  {item['column']}: {item['count']} sentinel values ({pct:.1f}% of rows)"
            f"  — row indices: {shown}{extra}"
        )

    # ── 7. SPECIAL CHARACTER CELLS — per column ─────────────────────────────
    lines.append(f"\n=== SPECIAL CHARACTER CELLS ({cols_with_special} columns affected, {total_special} total cells) ===")
    special_char_items = scan.get("special_char_cells", [])
    if not special_char_items:
        lines.append("  (none — no special character values found)")
    for item in special_char_items:
        shown = item["rows"][:max_rows_shown]
        extra = (
            f" … (+{item['count'] - max_rows_shown} more rows)"
            if item["count"] > max_rows_shown else ""
        )
        pct = item["count"] / total_rows * 100 if total_rows else 0
        lines.append(
            f"  {item['column']}: {item['count']} cells with special chars ({pct:.1f}% of rows)"
            f"  — row indices: {shown}{extra}"
        )

    # ── 8. DUPLICATE ROWS ─────────────────────────────────────────────────────
    lines.append(f"\n=== DUPLICATE ROWS ({duplicate_row_count} rows) ===")
    dup_rows = scan.get("duplicate_rows", [])
    if not dup_rows:
        lines.append("  (none — no duplicate timestamps found)")
    else:
        shown = dup_rows[:max_rows_shown]
        extra = (
            f" … (+{duplicate_row_count - max_rows_shown} more rows)"
            if duplicate_row_count > max_rows_shown else ""
        )
        lines.append(f"  Duplicate row indices: {shown}{extra}")

    # ── 9. PER-COLUMN STATUS TABLE ────────────────────────────────────────────
    lines.append(f"\n=== PER-COLUMN STATUS TABLE ===")
    lines.append("  Column name                      NaN count   9999 count   Special chars   Status")
    lines.append("  " + "-" * 85)
    for c in scan["columns"]:
        col      = c["original"]
        n_nan    = na_map[col]["count"]          if col in na_map          else 0
        n_9999   = sentinel_map[col]["count"]    if col in sentinel_map    else 0
        n_special = special_char_map[col]["count"] if col in special_char_map else 0
        if col in scan["all_missing_cols"]:
            status = "FULLY MISSING"
        else:
            issues = []
            if n_nan > 0:
                issues.append("NaN")
            if n_9999 > 0:
                issues.append("9999")
            if n_special > 0:
                issues.append("special chars")
            status = "has " + " + ".join(issues) if issues else "clean"
        lines.append(f"  {col:<36}{n_nan:<12}{n_9999:<16}{n_special:<16}{status}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python file_scanner.py <path_to_file>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    try:
        scan = scan_file(path)
        print(build_report_text(scan))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
