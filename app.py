"""
backend.py  –  FastAPI backend for the Wind Data File Validator

Endpoints:
  POST /api/analyse   – Upload a .txt file → scan + LLM column check → returns analysis
  POST /api/chat      – Ask questions about an already-analysed file (ReAct agent)
  GET  /api/health    – Health check

Run:
  uvicorn backend:app --reload --port 8000

Requirements (same as sample_5.py):
  pip install fastapi uvicorn[standard] pandas langchain langchain-community
              sentence-transformers faiss-cpu langchain-ollama python-multipart
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from file_scanner import _detect_header_row, build_report_text, scan_file
from tool_registry import build_system_prompt

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://192.168.0.80:11434"
MODEL_NAME      = "gemma3:12b"
INDEX_DIR       = "embeddings/rules_index_1"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Wind File Validator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your React dev server in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Module-level cached resources (loaded once on first use) ───────────────────
_llm: ChatOllama | None = None
_vectordb = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=MODEL_NAME,
            temperature=0,
        )
    return _llm


def get_vectordb():
    global _vectordb
    if _vectordb is None and os.path.isdir(INDEX_DIR):
        emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _vectordb = FAISS.load_local(
            INDEX_DIR, emb, allow_dangerous_deserialization=True
        )
    return _vectordb


# ── In-memory session store ────────────────────────────────────────────────────
# session_id -> {scan, df, bad_cols, col_reasons, file_name}
sessions: dict[str, dict] = {}


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _call_llm(messages: list[dict]) -> str:
    lc = []
    for m in messages:
        if m["role"] == "system":
            lc.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            lc.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc.append(AIMessage(content=m["content"]))
    return get_llm().invoke(lc).content


def run_column_check(scan: dict) -> tuple[set[str], dict[str, str]]:
    """Calls the LLM to validate column names against rules."""
    vectordb  = get_vectordb()
    col_list  = [c["original"] for c in scan["columns"]]

    rule_context = "(rule index not available — generic naming check only)"
    if vectordb is not None:
        hits = vectordb.similarity_search(
            "column naming conventions sensor names format units prefix suffix", k=6
        )
        rule_context = "\n\n---\n\n".join(
            f"[{h.metadata.get('source', '?')} p.{h.metadata.get('page', '?')}]\n{h.page_content}"
            for h in hits
        )

    prompt = f"""You are a wind-data validation expert.

Below are the column names from a wind-data file.
Evaluate EVERY column name against the naming rules provided and decide
whether each one is VALID or INVALID.

COLUMN NAMES (evaluate each one):
{json.dumps(col_list, ensure_ascii=False)}

NAMING RULES (from validation documents):
{rule_context}

For each column name judge it strictly against the rules above.
Return ONLY a JSON object — no markdown, no extra text:
{{
  "columns": [
    {{
      "name": "<exact column name as given>",
      "valid": true or false,
      "reason": "<one clear sentence explaining why it is valid or what rule it violates>"
    }},
    ...
  ]
}}

Every column in the input must appear exactly once in the output.
Use "valid": false when the column breaks any rule.
Use "valid": true only when the column fully complies."""

    resp = get_llm().invoke([
        SystemMessage(content="Return only valid JSON. No markdown, no explanation."),
        HumanMessage(content=prompt),
    ]).content

    bad_cols: set[str]       = set()
    reasons:  dict[str, str] = {}
    try:
        cleaned = resp.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
        data    = json.loads(cleaned.strip())
        col_set = set(col_list)
        for entry in data.get("columns", []):
            name   = str(entry.get("name",   ""))
            valid  = entry.get("valid", True)
            reason = str(entry.get("reason", ""))
            if name in col_set:
                reasons[name] = reason
                if not valid:
                    bad_cols.add(name)
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    return bad_cols, reasons


# ── Agent tools (session-aware, no Streamlit dependency) ──────────────────────

def _make_tools(session: dict):
    """Returns a dict of tool-name → callable, bound to the given session."""

    scan: dict        = session["scan"]
    bad_cols: set     = set(session.get("bad_cols", []))
    col_reasons: dict = session.get("col_reasons", {})

    def get_file_summary(_: str) -> str:
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

    def get_file_metadata(_: str) -> str:
        m = scan["metadata"]
        return (
            f"Latitude : {m['latitude']}\n"
            f"Longitude: {m['longitude']}\n"
            f"Elevation: {m['elevation']}"
        )

    def get_column_list(query: str) -> str:
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

    def get_fully_missing_columns(_: str) -> str:
        missing = scan["all_missing_cols"]
        if not missing:
            return "No fully-missing columns — all columns have at least some valid data."
        return "Fully-missing columns:\n" + "\n".join(f"  • {c}" for c in missing)

    def get_nan_details(_: str) -> str:
        if not scan["na_cells"]:
            return "No NaN or blank values found in the file."
        lines = [f"Columns with NaN/blank values ({len(scan['na_cells'])} columns):"]
        for item in scan["na_cells"]:
            lines.append(f"  {item['column']} : {item['count']:,}")
        return "\n".join(lines)

    def get_9999_details(_: str) -> str:
        if not scan["sentinel_9999"]:
            return "No 9999 sentinel values found in the file."
        lines = [f"Columns with 9999 sentinel values ({len(scan['sentinel_9999'])} columns):"]
        for item in scan["sentinel_9999"]:
            lines.append(f"  {item['column']} : {item['count']:,}")
        return "\n".join(lines)

    def get_column_status(_: str) -> str:
        na_map = {i["column"]: i["count"] for i in scan["na_cells"]}
        s9_map = {i["column"]: i["count"] for i in scan["sentinel_9999"]}
        lines  = ["Column name                              NaN count   9999 count"]
        lines.append("-" * 65)
        for c in scan["columns"]:
            col = c["original"]
            lines.append(f"  {col:<40}{na_map.get(col, 0):<12}{s9_map.get(col, 0)}")
        return "\n".join(lines)

    def retrieve_validation_rules(query: str) -> str:
        vectordb = get_vectordb()
        if vectordb is None:
            return "Rule index not loaded — place the unzipped rules_index/ folder next to this app."
        hits   = vectordb.similarity_search(query, k=4)
        chunks = [
            f"[{h.metadata.get('source', '?')} p.{h.metadata.get('page', '?')}]\n{h.page_content}"
            for h in hits
        ]
        return "\n\n---\n\n".join(chunks)

    def check_column_names(_: str) -> str:
        return json.dumps(
            {
                "flagged_columns": [
                    {"name": c, "reason": col_reasons.get(c, "")} for c in sorted(bad_cols)
                ],
                "ok_columns": [
                    c["original"] for c in scan["columns"] if c["original"] not in bad_cols
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    return {
        "get_file_summary":          get_file_summary,
        "get_file_metadata":         get_file_metadata,
        "get_column_list":           get_column_list,
        "get_fully_missing_columns": get_fully_missing_columns,
        "get_nan_details":           get_nan_details,
        "get_9999_details":          get_9999_details,
        "get_column_status":         get_column_status,
        "retrieve_validation_rules": retrieve_validation_rules,
        "check_column_names":        check_column_names,
    }


# ── ReAct agent ────────────────────────────────────────────────────────────────

REACT_SYSTEM = build_system_prompt([
    "get_file_summary",
    "get_file_metadata",
    "get_column_list",
    "get_fully_missing_columns",
    "get_nan_details",
    "get_9999_details",
    "get_column_status",
    "retrieve_validation_rules",
    "check_column_names",
])


def run_agent(user_input: str, session: dict) -> str:
    tool_map = _make_tools(session)
    messages = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user",   "content": user_input},
    ]

    for _ in range(8):
        response = _call_llm(messages)
        messages.append({"role": "assistant", "content": response})

        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()

        action = action_input = None
        for line in response.splitlines():
            if line.startswith("Action:"):
                action = line[len("Action:"):].strip()
            elif line.startswith("Action Input:"):
                action_input = line[len("Action Input:"):].strip()

        if action and action in tool_map:
            observation = tool_map[action](action_input or "")
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            messages.append({
                "role": "user",
                "content": (
                    "You tried an unrecognised action. "
                    "Based on everything gathered so far, give your final answer now.\n"
                    "Final Answer:"
                ),
            })
            final = _call_llm(messages)
            if "Final Answer:" in final:
                return final.split("Final Answer:")[-1].strip()
            return final.strip()

    return "Agent reached max iterations without a final answer."


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message:    str
    history:    list[dict] = []


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.post("/api/analyse")
async def analyse_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "upload")[1] or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        scan      = scan_file(tmp_path)
        scan_text = build_report_text(scan)

        header_row = _detect_header_row(tmp_path, encoding="ISO-8859-1")
        df = pd.read_csv(
            tmp_path,
            sep="\t",
            skiprows=header_row,
            encoding="ISO-8859-1",
            engine="python",
            on_bad_lines="skip",
        ).dropna(axis=1, how="all")

        bad_cols, col_reasons = run_column_check(scan)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    # Store session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "scan":        scan,
        "df":          df,
        "bad_cols":    list(bad_cols),
        "col_reasons": col_reasons,
        "file_name":   file.filename,
    }

    return {
        "session_id":       session_id,
        "file_name":        file.filename,
        "analysis":         scan_text,
        "flagged_columns":  list(bad_cols),
        "col_reasons":      col_reasons,
        "metadata":         scan["metadata"],
        "shape": {
            "rows":    scan["shape"][0],
            "columns": scan["shape"][1],
        },
        "summary": {
            "nan_cols":          len(scan["na_cells"]),
            "sentinel_cols":     len(scan["sentinel_9999"]),
            "fully_missing_cols": len(scan["all_missing_cols"]),
        },
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload and analyse a file first.",
        )

    # Build history prefix if provided
    history_prefix = ""
    if req.history:
        lines = [
            f"USER: {m['content']}"
            for m in req.history
            if m.get("role") == "user"
        ]
        if lines:
            history_prefix = "CONVERSATION HISTORY:\n" + "\n".join(lines) + "\n\n"

    full_input = history_prefix + f"USER: {req.message}"

    try:
        answer = run_agent(full_input, session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return {"answer": answer}


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "index": os.path.isdir(INDEX_DIR)}
