"""U4 mental-model loop — the living Capability Ledger.

Closes the compile loop: after Claude Code builds the gaps, the creator reports back
what was implemented; that report is **merged into the engine's capability registry**
so the next compile knows those systems exist and the reasoning model stops asking to
re-implement them. This is the "spec sheet that knows status".

Pure logic over dicts (no Mongo/LLM here — the route persists + calls Bedrock). The
ledger is stored per engine; the base registry lives in ``compile_tools._REGISTRY``.

Ledger shape (one doc per engine, persisted by the route):
  {"engine", "extra_systems": [...], "extra_consumes": [...],
   "entries": [{"at", "systems", "consumes", "repo", "commit", "files", "summary"}]}
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def empty_ledger(engine: str) -> Dict[str, Any]:
    return {"engine": engine, "extra_systems": [], "extra_consumes": [], "entries": []}


def merge_registry(base: Dict[str, Any], ledger: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fold a ledger's accumulated capabilities into the base registry. Deterministic,
    order-preserving, deduped. Returns a new dict (base untouched)."""
    reg = {
        "engine": base.get("engine"),
        "contract_version": base.get("contract_version"),
        "systems": list(base.get("systems", [])),
        "contract_fields": list(base.get("contract_fields", [])),
        "consumes": list(base.get("consumes", [])),
    }
    if base.get("unknown_engine"):
        reg["unknown_engine"] = True
    if not ledger:
        reg["entry_count"] = 0
        return reg
    for s in ledger.get("extra_systems", []):
        if s not in reg["systems"]:
            reg["systems"].append(s)
    for c in ledger.get("extra_consumes", []):
        if c not in reg["consumes"]:
            reg["consumes"].append(c)
    reg["entry_count"] = len(ledger.get("entries", []))
    reg["last_entry"] = ledger["entries"][-1] if ledger.get("entries") else None
    return reg


def apply_report(ledger: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    """Append a build report to the ledger and merge its capabilities in. Returns the
    updated ledger (new dict). ``report`` = {systems?, consumes?, repo?, commit?,
    files?, summary?}. Empty/duplicate items are ignored."""
    led = {
        "engine": ledger.get("engine"),
        "extra_systems": list(ledger.get("extra_systems", [])),
        "extra_consumes": list(ledger.get("extra_consumes", [])),
        "entries": list(ledger.get("entries", [])),
    }
    systems = [s for s in (report.get("systems") or []) if s]
    consumes = [c for c in (report.get("consumes") or []) if c]
    for s in systems:
        if s not in led["extra_systems"]:
            led["extra_systems"].append(s)
    for c in consumes:
        if c not in led["extra_consumes"]:
            led["extra_consumes"].append(c)
    led["entries"].append({
        "at": datetime.now(timezone.utc).isoformat(),
        "systems": systems,
        "consumes": consumes,
        "repo": report.get("repo") or "",
        "commit": report.get("commit") or "",
        "files": [f for f in (report.get("files") or []) if f],
        "summary": (report.get("summary") or "").strip(),
    })
    return led


# ── deterministic extractor (no LLM): scan a Claude-Code done-note ─────────────
_COMMIT_RE = re.compile(r"\b([0-9a-f]{7,40})\b")
_REPO_RE = re.compile(r"(https?://github\.com/[\w.-]+/[\w.-]+|[\w.-]+/[\w.-]+\.git)")
_FILE_RE = re.compile(r"\b([\w./-]+\.(?:ts|tsx|js|jsx|py|cs|json))\b")


def extract_from_note(
    note: str, known_layers: List[str], known_systems: List[str]
) -> Dict[str, Any]:
    """Best-effort deterministic structuring of a freeform done-note: find which known
    layers/systems it mentions, plus repo/commit/file hints. The creator reviews and
    confirms before apply — this never writes state, it only suggests."""
    text = note or ""
    low = text.lower()
    consumes = [l for l in known_layers if l and l.lower() in low]
    systems = [s for s in known_systems if s and s.lower() in low]
    repo_m = _REPO_RE.search(text)
    # commit: prefer a hex token near the word "commit"
    commit = ""
    for m in _COMMIT_RE.finditer(text):
        tok = m.group(1)
        if len(tok) >= 7 and not tok.isdigit():
            commit = tok
            break
    files = sorted(set(_FILE_RE.findall(text)))[:40]
    return {
        "systems": systems,
        "consumes": consumes,
        "repo": repo_m.group(1) if repo_m else "",
        "commit": commit,
        "files": files,
        "summary": text.strip()[:500],
    }
