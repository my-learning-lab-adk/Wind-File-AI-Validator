"""
sample_4.py  –  Wind Data File Validator
Run:  streamlit run sample_4.py

Requirements:
  pip install streamlit pandas langchain langchain-community
              sentence-transformers faiss-cpu requests

Needs:
  - ./rules_index/   (FAISS index built from validation PDFs)
  - Ollama running:  ollama pull gemma3:12b

Flow:
  1. User drag-drops a .txt / .tsv wind file.
  2. File is scanned immediately (pure Python — zero LLM).
  3. LLM auto-checks every column name against rules_index and
     generates a short per-column reason for each violation.
  4. Colored table shows the first 100 rows:
       • Column header  → green (ok) | red (rule violation) + hover reason
       • Row index cell → green (clean row) | red (has NaN/9999) + hover summary
       • Data cell      → green (valid)  | red (NaN or 9999) + hover reason
  5. Chat section below for deeper analysis.
"""

import html as _html_mod
import json
import os
import tempfile

import pandas as pd
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from file_scanner import _detect_header_row, build_report_text, scan_file
from tool_registry import build_system_prompt


# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://192.168.0.80:11434/"
MODEL_NAME      = "gemma3:12b"
INDEX_DIR       = "embeddings/rules_index_1"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
PREVIEW_ROWS    = 100


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading LLM…")
def load_llm():
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=MODEL_NAME,
        temperature=0,
        # client_kwargs={"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}},
    )


@st.cache_resource(show_spinner="Loading rule index…")
def load_vectordb():
    from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_resource(show_spinner="Loading rule index…")
def load_vectordb():

    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectordb = FAISS.load_local(
        "rules_index",   # folder name
        emb,
        allow_dangerous_deserialization=True
    )

    return vectordb


# ── Helpers ───────────────────────────────────────────────────────────────────
def _esc(text: str) -> str:
    """HTML-escape for safe embedding in attribute values and text nodes."""
    return _html_mod.escape(str(text).replace("\n", " ").replace("\r", ""), quote=True)


# ── LLM column check (auto-runs on upload) ────────────────────────────────────
def run_column_check(scan: dict, progress_bar=None) -> tuple:
    """
    Calls the LLM in batches of 15 columns to identify which column names
    violate rules and why.

    Args:
        scan        : scan dict from scan_file()
        progress_bar: optional st.progress element — updated after each batch

    Returns:
        bad_cols  : set[str]        — names of flagged columns
        col_reasons: dict[str, str] — {col_name: short_reason}
    """
    BATCH_SIZE = 15

    vectordb = load_vectordb()
    col_list = [c["original"] for c in scan["columns"]]
    total_batches = max(1, (len(col_list) + BATCH_SIZE - 1) // BATCH_SIZE)

    # Fetch rule context once — reused for every batch
    rule_context = "(rule index not available — generic naming check only)"
    if vectordb is not None:
        hits = vectordb.similarity_search(
        "column validation rules sensor format unit naming", k=8
        )
        rule_context = "\n\n---\n\n".join(
        f"[{h.metadata.get('rule_id', '?')}]\n{h.page_content}"
        for h in hits
        )

    bad_cols: set[str]       = set()
    reasons:  dict[str, str] = {}

    for batch_num, i in enumerate(range(0, len(col_list), BATCH_SIZE), start=1):
        batch     = col_list[i : i + BATCH_SIZE]
        batch_end = min(i + BATCH_SIZE, len(col_list))

        prompt = f"""You are a strict wind-data validation expert.

Your task is to validate each column name STRICTLY based on the provided rules.

COLUMN NAMES:
{json.dumps(batch, ensure_ascii=False)}

VALIDATION RULES:
{rule_context}

INSTRUCTIONS:

1. Check EACH column independently.
2. A column is INVALID if it violates ANY rule.
3. You MUST strictly follow the rules — do NOT assume or guess.
4. Use rule definitions, not just examples.
5. If multiple issues exist, explain the most obvious issue based on the column name.
Focus on what is clearly wrong, not internal rule names.
6. Be strict — even small deviations should be marked INVALID.

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "columns": [
    {{
      "name": "<exact column name>",
      "valid": true or false,
      "reason": "Explain in simple, human-readable language what is wrong with the column name. Mention the exact issue (e.g., invalid character, wrong format, invalid unit), but do NOT mention rule IDs."
    }}
  ]
}}

RULES FOR OUTPUT:
- Do NOT add extra text
- Do NOT skip any column
- Each column must appear exactly once
- Keep reason short but specific
"""

        resp = load_llm().invoke([
            SystemMessage(content="Return only valid JSON. No markdown, no explanation."),
            HumanMessage(content=prompt),
        ]).content

        try:
            cleaned = resp.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            data      = json.loads(cleaned.strip())
            batch_set = set(batch)
            for entry in data.get("columns", []):
                name   = str(entry.get("name",   ""))
                valid  = entry.get("valid", True)
                reason = str(entry.get("reason", ""))
                if name in batch_set:
                    reasons[name] = reason
                    if not valid:
                        bad_cols.add(name)
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass

        if progress_bar is not None:
            pct = int((batch_num / total_batches) * 100)
            progress_bar.progress(
                pct,
                text=f"Checking columns {i + 1}–{batch_end} of {len(col_list)} ({pct}%)",
            )

    return bad_cols, reasons


# ── LLM call helper (for the ReAct agent) ────────────────────────────────────
def call_llm(messages: list) -> str:
    lc = []
    for m in messages:
        if m["role"] == "system":
            lc.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            lc.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc.append(AIMessage(content=m["content"]))
    return load_llm().invoke(lc).content


# ── HTML table with hover tooltips ────────────────────────────────────────────

_TABLE_CSS = """
<style>
/* ── table shell ── */
.wv-wrap { overflow-x: auto; margin: 0; }
.wv {
    border-collapse: collapse;
    font-size: 12px;
    font-family: 'Courier New', monospace;
    width: 100%;
}
.wv th, .wv td {
    border: 1px solid #2e2e2e;
    padding: 3px 8px;
    white-space: nowrap;
    position: relative;   /* needed for tooltip positioning */
}

/* ── cell colours ── */
.c-ok  { background: #1a3a1a; color: #4ade80; }
.c-bad { background: #3a1a1a; color: #f87171; }

/* ── header colours ── */
.h-ok  { background: #14532d; color: #bbf7d0; font-weight: 700; }
.h-bad { background: #7f1d1d; color: #fecaca; font-weight: 700; }

/* ── row-index cell ── */
.r-ok  { background: #0f2e18; color: #6ee7b7; font-weight: 700; text-align: right; min-width: 40px; }
.r-bad { background: #450a0a; color: #fca5a5; font-weight: 700; text-align: right; min-width: 40px; }
.idx   { background: #1e1e1e; color: #9ca3af; font-weight: 700; text-align: center; }

/* ── tooltip via data-tip + ::after ── */
[data-tip] { cursor: default; }
[data-tip]::after {
    content      : attr(data-tip);
    display      : none;
    position     : absolute;
    z-index      : 9999;
    top          : calc(100% + 4px);
    left         : 0;
    background   : #0f172a;
    color        : #e2e8f0;
    border       : 1px solid #475569;
    border-radius: 6px;
    padding      : 7px 11px;
    font-size    : 11px;
    font-weight  : 400;
    font-family  : sans-serif;
    white-space  : normal;
    width        : max-content;
    max-width    : 300px;
    line-height  : 1.5;
    box-shadow   : 0 6px 16px rgba(0,0,0,.7);
    pointer-events: none;
}
[data-tip]:hover::after { display: block; }
</style>
"""


def render_colored_table(
    df: pd.DataFrame,
    scan: dict,
    bad_cols: set,
    col_reasons: dict,
) -> str:
    """
    Build an HTML preview table with colour-coded cells and hover tooltips.

    Colour logic:
      Column header  → green (ok) | red (naming rule violation) + tooltip with LLM reason
      Row index cell → green (all cells valid) | red (any NaN/9999 in this row) + tooltip
      Data cell      → green (valid) | red (NaN or 9999) + tooltip explaining the issue
    """
    # Fast lookup: column → set of row indices that are bad
    na_rows: dict[str, set] = {
        item["column"]: set(item["rows"]) for item in scan["na_cells"]
    }
    s9_rows: dict[str, set] = {
        item["column"]: set(item["rows"]) for item in scan["sentinel_9999"]
    }

    preview = df.head(PREVIEW_ROWS)
    parts   = [_TABLE_CSS, '<div class="wv-wrap"><table class="wv">']

    # ── Header row ────────────────────────────────────────────────────────────
    parts.append("<thead><tr>")
    parts.append('<th class="idx">#</th>')

    for col in df.columns:
        cs     = str(col)
        reason = col_reasons.get(cs, "")
        if cs in bad_cols:
            tip = reason or "Column name violates naming rules."
            parts.append(
                f'<th class="h-bad" data-tip="{_esc(tip)}">{_esc(cs)}</th>'
            )
        else:
            tip = reason or "Column name is valid."
            parts.append(
                f'<th class="h-ok" data-tip="{_esc(tip)}">{_esc(cs)}</th>'
            )

    parts.append("</tr></thead><tbody>")

    # ── Data rows ─────────────────────────────────────────────────────────────
    for idx, row in preview.iterrows():
        # Classify every cell in this row first (needed for row-index colouring)
        cell_issues: list[tuple[str, str]] = []   # [(col_name, issue_type), ...]

        for col in df.columns:
            cs  = str(col)
            val = row[col]

            is_na  = (cs in na_rows  and idx in na_rows[cs]) or pd.isna(val)
            is_999 = cs in s9_rows and idx in s9_rows[cs]
            if not is_999:
                try:
                    if float(val) == 9999:
                        is_999 = True
                except (ValueError, TypeError):
                    pass

            if is_999:
                cell_issues.append((cs, "sentinel 9999"))
            elif is_na:
                cell_issues.append((cs, "NaN / blank"))

        # Row-index cell
        if cell_issues:
            shown   = cell_issues[:5]
            extra   = len(cell_issues) - 5
            summary = ", ".join(f"{c} ({t})" for c, t in shown)
            if extra > 0:
                summary += f" … +{extra} more column(s)"
            tip = f"Row {idx} has bad cells: {summary}"
            parts.append(f'<tr><td class="r-bad" data-tip="{_esc(tip)}">{idx}</td>')
        else:
            parts.append(f'<tr><td class="r-ok">{idx}</td>')

        # Data cells
        for col in df.columns:
            cs  = str(col)
            val = row[col]

            is_na  = (cs in na_rows  and idx in na_rows[cs]) or pd.isna(val)
            is_999 = cs in s9_rows and idx in s9_rows[cs]
            if not is_999:
                try:
                    if float(val) == 9999:
                        is_999 = True
                except (ValueError, TypeError):
                    pass

            display = "" if pd.isna(val) else _esc(str(val))

            if is_999:
                tip = (
                    f"Sentinel value 9999 in \u2018{cs}\u2019 \u2014 "
                    "indicates an invalid or unavailable measurement. "
                    "This is a placeholder used when the sensor had no valid reading."
                )
                parts.append(f'<td class="c-bad" data-tip="{_esc(tip)}">{display}</td>')
            elif is_na:
                tip = (
                    f"Missing value (NaN) in \u2018{cs}\u2019 \u2014 "
                    "no data was recorded for this sensor at this timestamp. "
                    "The cell is empty or blank in the source file."
                )
                parts.append(f'<td class="c-bad" data-tip="{_esc(tip)}">{display}</td>')
            else:
                parts.append(f'<td class="c-ok">{display}</td>')

        parts.append("</tr>")

    parts.append("</tbody></table></div>")

    if len(df) > PREVIEW_ROWS:
        parts.append(
            f'<p style="color:#6b7280;font-size:11px;margin-top:6px">'
            f"Showing first {PREVIEW_ROWS:,} of {len(df):,} rows"
            f"</p>"
        )

    return "".join(parts)


# ── Agent tools ───────────────────────────────────────────────────────────────
def _no_file() -> str:
    return "No file uploaded yet. Ask the user to upload a file first."


@tool
def get_file_summary(query: str) -> str:
    """Returns total rows, columns, NaN counts, 9999 counts, and clean column count."""
    scan = st.session_state.get("scan")
    if not scan:
        return _no_file()
    rows, cols = scan["shape"]
    na_cols    = len(scan["na_cells"])
    s9_cols    = len(scan["sentinel_9999"])
    total_nan  = sum(i["count"] for i in scan["na_cells"])
    total_999  = sum(i["count"] for i in scan["sentinel_9999"])
    fully      = len(scan["all_missing_cols"])
    clean      = cols - len(
        {i["column"] for i in scan["na_cells"]} |
        {i["column"] for i in scan["sentinel_9999"]}
    )
    return (
        f"Total rows             : {rows:,}\n"
        f"Total columns          : {cols}\n"
        f"Timestamp column       : {scan['timestamp_column']}\n"
        f"Columns with NaN/blank : {na_cols}  (total NaN cells: {total_nan:,})\n"
        f"Columns with 9999      : {s9_cols}  (total 9999 cells: {total_999:,})\n"
        f"Fully-missing columns  : {fully}\n"
        f"Clean columns          : {clean}  (no NaN, no 9999)"
    )


@tool
def get_file_metadata(query: str) -> str:
    """Returns the file's latitude, longitude, and elevation."""
    scan = st.session_state.get("scan")
    if not scan:
        return _no_file()
    m = scan["metadata"]
    return (
        f"Latitude : {m['latitude']}\n"
        f"Longitude: {m['longitude']}\n"
        f"Elevation: {m['elevation']}"
    )


@tool
def get_column_list(query: str) -> str:
    """Returns column names. Pass a keyword to filter, or empty string for all."""
    scan = st.session_state.get("scan")
    if not scan:
        return _no_file()
    cols    = [c["original"] for c in scan["columns"]]
    keyword = query.strip().upper()
    if keyword:
        cols = [c for c in cols if keyword in c.upper()]
        if not cols:
            return f"No columns found containing '{query}'."
        header = f"Columns containing '{query}': {len(cols)}\n"
    else:
        header = f"Total columns: {len(cols)}\n"
    return header + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(cols))


@tool
def get_fully_missing_columns(query: str) -> str:
    """Returns columns where every value is NaN or 9999 (no valid data at all)."""
    scan = st.session_state.get("scan")
    if not scan:
        return _no_file()
    missing = scan["all_missing_cols"]
    if not missing:
        return "No fully-missing columns — all columns have at least some valid data."
    return "Fully-missing columns:\n" + "\n".join(f"  • {c}" for c in missing)


@tool
def get_nan_details(query: str) -> str:
    """Returns per-column NaN/blank counts."""
    scan = st.session_state.get("scan")
    if not scan:
        return _no_file()
    if not scan["na_cells"]:
        return "No NaN or blank values found in the file."
    lines = [f"Columns with NaN/blank values ({len(scan['na_cells'])} columns):"]
    for item in scan["na_cells"]:
        lines.append(f"  {item['column']} : {item['count']:,}")
    return "\n".join(lines)


@tool
def get_9999_details(query: str) -> str:
    """Returns per-column 9999 sentinel value counts."""
    scan = st.session_state.get("scan")
    if not scan:
        return _no_file()
    if not scan["sentinel_9999"]:
        return "No 9999 sentinel values found in the file."
    lines = [f"Columns with 9999 sentinel values ({len(scan['sentinel_9999'])} columns):"]
    for item in scan["sentinel_9999"]:
        lines.append(f"  {item['column']} : {item['count']:,}")
    return "\n".join(lines)


@tool
def get_column_status(query: str) -> str:
    """Returns a combined table of every column with its NaN count and 9999 count."""
    scan = st.session_state.get("scan")
    if not scan:
        return _no_file()
    na_map = {i["column"]: i["count"] for i in scan["na_cells"]}
    s9_map = {i["column"]: i["count"] for i in scan["sentinel_9999"]}
    lines  = ["Column name                              NaN count   9999 count"]
    lines.append("-" * 65)
    for c in scan["columns"]:
        col = c["original"]
        lines.append(f"  {col:<40}{na_map.get(col, 0):<12}{s9_map.get(col, 0)}")
    return "\n".join(lines)


@tool
def retrieve_validation_rules(query: str) -> str:
    """Searches the validation-rule PDF index and returns the most relevant excerpts."""
    vectordb = load_vectordb()
    if vectordb is None:
        return "Rule index not loaded — place the unzipped rules_index/ folder next to this app."
    hits   = vectordb.similarity_search(query, k=4)
    chunks = [
    f"[{h.metadata.get('rule_id', '?')}]\n{h.page_content}"
    for h in hits
    ]
    return "\n\n---\n\n".join(chunks)


@tool
def check_column_names(query: str) -> str:
    """Returns flagged column names and LLM-generated reasons from the auto validation."""
    scan = st.session_state.get("scan")
    if scan is None:
        return "No file uploaded yet."
    bad     = st.session_state.get("bad_columns", set())
    reasons = st.session_state.get("col_reasons", {})
    all_cols = [c["original"] for c in scan["columns"]]
    return json.dumps(
        {
            "flagged_columns": [
                {"name": c, "reason": reasons.get(c, "")} for c in sorted(bad)
            ],
            "ok_columns": [c for c in all_cols if c not in bad],
        },
        ensure_ascii=False,
        indent=2,
    )

def run_metadata_check(header_lines, vectordb):
    import json

    if vectordb is None:
        return [{
            "line": "Metadata validation skipped",
            "valid": False,
            "reason": "Rules file not loaded"
        }]

    # 🔍 Retrieve metadata-related rules
    hits = vectordb.similarity_search(
        "metadata validation rules latitude longitude elevation format coordinate", k=6
    )

    rule_context = "\n\n---\n\n".join(
        f"[{h.metadata.get('rule_id', '?')}]\n{h.page_content}"
        for h in hits
    )

    # 🧠 Build prompt
    prompt = f"""You are a strict data validation expert.

Validate the following metadata lines:

{chr(10).join(header_lines)}

VALIDATION RULES:
{rule_context}

INSTRUCTIONS:
- Check each metadata line independently
- Detect format issues, invalid characters, missing units, or invalid ranges
- Be strict but explain clearly in simple human language
- Do NOT mention rule IDs

OUTPUT FORMAT (STRICT JSON):
{{
  "metadata": [
    {{
      "line": "<original line>",
      "valid": true or false,
      "reason": "<simple explanation>"
    }}
  ]
}}
"""

    try:
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=MODEL_NAME,
            temperature=0
        )

        resp = llm.invoke(prompt)
        text = resp.content.strip()

        # Extract JSON safely
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return []

        data = json.loads(text[start:end])
        return data.get("metadata", [])

    except Exception as e:
        return [{"line": "Metadata validation failed", "valid": False, "reason": str(e)}]

@tool
def get_color_preview_summary(query: str) -> str:
    """Returns a text summary of the colored preview table (red headers, red cell counts)."""
    df   = st.session_state.get("df")
    scan = st.session_state.get("scan")
    if df is None or scan is None:
        return "No file preview available — please upload a file first."
    bad    = st.session_state.get("bad_columns", set())
    na_map = {i["column"]: i["count"] for i in scan["na_cells"]}
    s9_map = {i["column"]: i["count"] for i in scan["sentinel_9999"]}
    lines  = [
        f"Red headers  ({len(bad)}): {', '.join(sorted(bad)) or '(none)'}",
        f"Green headers ({len(df.columns) - len(bad)}): all others",
        "",
        "Red-cell counts per column (NaN + 9999):",
    ]
    any_bad = False
    for col in df.columns:
        cs = str(col)
        t  = na_map.get(cs, 0) + s9_map.get(cs, 0)
        if t:
            lines.append(
                f"  {cs}: {t} total "
                f"({na_map.get(cs, 0)} NaN/blank, {s9_map.get(cs, 0)} x 9999)"
            )
            any_bad = True
    if not any_bad:
        lines.append("  (none — all cells look clean)")
    lines.append(f"\nTotal rows: {len(df):,}  |  Columns: {len(df.columns)}")
    lines.append(f"Preview shows first {min(PREVIEW_ROWS, len(df))} rows.")
    return "\n".join(lines)


# ── ReAct agent ───────────────────────────────────────────────────────────────
ALL_TOOLS = [
    get_file_summary,
    get_file_metadata,
    get_column_list,
    get_fully_missing_columns,
    get_nan_details,
    get_9999_details,
    get_column_status,
    retrieve_validation_rules,
    check_column_names,
    get_color_preview_summary,
]

REACT_SYSTEM = build_system_prompt([t.name for t in ALL_TOOLS])


def run_agent(user_input: str, thought_container=None) -> str:
    tool_map = {t.name: t for t in ALL_TOOLS}
    messages = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user",   "content": user_input},
    ]

    for _ in range(8):
        response = call_llm(messages)
        messages.append({"role": "assistant", "content": response})

        if thought_container is not None:
            for line in response.splitlines():
                if line.startswith(("Thought:", "Action:", "Action Input:")):
                    thought_container.markdown(f"💭 **{line}**")

        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()

        action = action_input = None
        for line in response.splitlines():
            if line.startswith("Action:"):
                action = line[len("Action:"):].strip()
            elif line.startswith("Action Input:"):
                action_input = line[len("Action Input:"):].strip()

        if action and action in tool_map:
            observation = tool_map[action].invoke(action_input or "")
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            # LLM tried an unknown action or got confused — nudge it for a clean answer
            messages.append({
                "role": "user",
                "content": (
                    "You tried an unrecognised action. "
                    "Based on everything gathered so far, give your final answer now.\n"
                    "Final Answer:"
                ),
            })
            final = call_llm(messages)
            if "Final Answer:" in final:
                return final.split("Final Answer:")[-1].strip()
            return final.strip()

    return "Agent reached max iterations without a final answer."


# ═══════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Wind File Validator", layout="wide")
st.title("🌬️ Wind Data File Validator")

# ── Session state initialisation ──────────────────────────────────────────────
_DEFAULTS: list[tuple] = [
    ("metadata_results", []),
    ("metadata_check_done", False),
    ("messages",           []),
    ("scan",               None),
    ("scan_text",          ""),
    ("file_name",          None),
    ("file_size",          -1),
    ("df",                 None),
    ("bad_columns",        set()),
    ("col_reasons",        {}),
    ("column_check_done",  False),
]
for key, default in _DEFAULTS:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar — file upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Upload File")
    uploaded = st.file_uploader(
        "Drag & drop a TSV / .txt wind file here",
        type=["txt", "tsv"],
        help="Tab-separated wind-data files with a metadata header are supported.",
    )

    new_file = uploaded and (
        uploaded.name != st.session_state.file_name
        or uploaded.size != st.session_state.file_size
    )

    if new_file:
        # Reset all state for the new file
        st.session_state.bad_columns      = set()
        st.session_state.col_reasons      = {}
        st.session_state.column_check_done = False
        st.session_state.scan             = None
        st.session_state.scan_text        = ""
        st.session_state.df               = None
        st.session_state.messages         = []
        st.session_state.metadata_results = []
        st.session_state.metadata_check_done = False

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # ── Step 1: pure-Python scan (no LLM) ─────────────────────────────────
        with st.spinner("Scanning file…"):
            try:
                scan = scan_file(tmp_path)
                st.session_state.scan      = scan
                st.session_state.scan_text = build_report_text(scan)
                st.session_state.file_name = uploaded.name
                st.session_state.file_size = uploaded.size

                header_row = _detect_header_row(tmp_path, encoding="ISO-8859-1")
                st.session_state.df = pd.read_csv(
                    tmp_path,
                    sep="\t",
                    skiprows=header_row,
                    encoding="ISO-8859-1",
                    engine="python",
                    on_bad_lines="skip",
                ).dropna(axis=1, how="all")

                st.success(f"✅ Scanned: {uploaded.name}")
            except Exception as e:
                st.error(f"Scan failed: {e}")

        # ── Step 2: LLM column check — auto-triggered immediately ─────────────
        if st.session_state.scan is not None:
            with st.spinner("Checking column names against rules…"):
                progress_bar = st.progress(0, text="Starting column name check…")
                try:
                    bad, reasons = run_column_check(
                        st.session_state.scan, progress_bar=progress_bar
                    )
                    progress_bar.empty()
                    st.session_state.bad_columns      = bad
                    st.session_state.col_reasons      = reasons
                    st.session_state.column_check_done = True
                    if bad:
                        st.warning(f"⚠️ {len(bad)} column(s) flagged")
                    else:
                        st.success("✅ All column names look valid")
                except Exception as e:
                    progress_bar.empty()
                    st.warning(f"Column check failed: {e}")
                    st.session_state.column_check_done = True
        # ── Step 3: Metadata validation ─────────────────────────────
        if st.session_state.scan is not None and not st.session_state.metadata_check_done:
            header_lines = st.session_state.scan["metadata"].get("raw_header_lines", [])

            if header_lines:
                with st.spinner("Validating metadata…"):
                    try:
                        vectordb = load_vectordb()
                        metadata_results = run_metadata_check(header_lines, vectordb)

                        st.session_state.metadata_results = metadata_results
                        st.session_state.metadata_check_done = True

                    except Exception as e:
                        st.warning(f"Metadata check failed: {e}")
                        st.session_state.metadata_check_done = True
    # Sidebar metrics
    if st.session_state.scan:
        s = st.session_state.scan
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Rows",      f"{s['shape'][0]:,}")
        c2.metric("Columns",   s["shape"][1])
        c1.metric("NaN cols",  len(s["na_cells"]))
        c2.metric("9999 cols", len(s["sentinel_9999"]))
        with st.expander("Full scan report"):
            st.code(st.session_state.scan_text, language="text")
        st.divider()
        st.header("2. Conversation")
        if st.button("🗑️  Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── Nothing uploaded yet ───────────────────────────────────────────────────────
if st.session_state.scan is None:
    st.info("👆 Drag and drop a wind-data .txt file in the sidebar to begin.")
    st.stop()

# Rule index warning


scan = st.session_state.scan
meta = scan["metadata"]

# ── Metadata panel ────────────────────────────────────────────────────────────
st.subheader("📋 File Metadata")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Latitude",  meta["latitude"]  if meta["latitude"]  is not None else "—")
mc2.metric("Longitude", meta["longitude"] if meta["longitude"] is not None else "—")
mc3.metric("Elevation", meta["elevation"] if meta["elevation"] is not None else "—")
mc4.metric("File",      st.session_state.file_name or "—")

if meta.get("raw_header_lines"):
    with st.expander("Raw header lines from file"):
        for line in meta["raw_header_lines"]:
            st.text(line)
# 🚨 Show metadata validation results
if st.session_state.get("metadata_results"):
    bad_meta = [m for m in st.session_state.metadata_results if not m["valid"]]

    if bad_meta:
        for item in bad_meta:
            st.error(f"{item['line']} — {item['reason']}")
    elif st.session_state.get("metadata_check_done"):
        st.success("Metadata looks valid.")

st.divider()

# ── Colored preview table ──────────────────────────────────────────────────────
st.subheader(f"📊 Data Preview — first {PREVIEW_ROWS} rows")

bad_cols = st.session_state.get("bad_columns", set())
reasons  = st.session_state.get("col_reasons",  {})

# Flagged columns summary banner
if bad_cols:
    flagged_md = "  |  ".join(
        f"`{c}` — {reasons.get(c, 'rule violation')}" for c in sorted(bad_cols)
    )
    st.error(f"**{len(bad_cols)} column(s) flagged:** {flagged_md}")
elif st.session_state.column_check_done:
    st.success("All column names comply with naming rules.")

# Legend
leg = st.columns(5)
_badge = lambda bg, fg, txt: (
    f'<span style="background:{bg};color:{fg};padding:2px 9px;'
    f'border-radius:4px;font-size:11px;font-weight:600">{txt}</span>'
)
leg[0].markdown(_badge("#14532d", "#bbf7d0", "■ Header OK"),           unsafe_allow_html=True)
leg[1].markdown(_badge("#7f1d1d", "#fecaca", "■ Header Violation"),    unsafe_allow_html=True)
leg[2].markdown(_badge("#1a3a1a", "#4ade80", "■ Cell Valid"),           unsafe_allow_html=True)
leg[3].markdown(_badge("#3a1a1a", "#f87171", "■ Cell NaN / 9999"),     unsafe_allow_html=True)
leg[4].markdown(
    '<span style="font-size:11px;color:#9ca3af">Hover red cells for details</span>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.df is not None:
    st.markdown(
        render_colored_table(st.session_state.df, scan, bad_cols, reasons),
        unsafe_allow_html=True,
    )
else:
    st.warning("Data preview could not be loaded.")

st.divider()

# ── Chat section ───────────────────────────────────────────────────────────────
st.subheader("💬 Ask the Validator")
st.caption(
    "Ask about sensors, rules, missing values, column names, "
    "or anything else about this file."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask anything about this file…")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    history_lines = [
        f"USER: {m['content']}"
        for m in st.session_state.messages[:-1]
        if m["role"] == "user"
    ]
    history_prefix = (
        "CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"
        if history_lines else ""
    )
    full_input = history_prefix + f"USER: {user_query}"

    with st.chat_message("assistant"):
        thought_box = st.expander("💭 Model reasoning", expanded=False)
        with st.spinner("Thinking…"):
            try:
                answer = run_agent(full_input, thought_container=thought_box)
            except Exception as e:
                answer = f"[Agent error: {e}]"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})