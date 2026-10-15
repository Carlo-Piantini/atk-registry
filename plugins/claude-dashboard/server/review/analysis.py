"""Analysis pass: `claude -p` turns the deterministic passes into a walkthrough.

Stage A (one call): group the reviewable files and order groups easy -> hard.
Stage B (one call per group, in parallel): split each file into logical steps,
each with what it does, the agent's reasoning, and an optional visual.

The model never sees a whole 75-file diff at once: stage A reads a manifest,
stage B reads one group's hunks. Coverage is enforced here, not trusted: every
file lands in exactly one group, every hunk in exactly one step.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from regen import invoke_claude

from .diff import ReviewDiff
from .provenance import Edit, HunkProvenance
from .structure import Structure

CALL_TIMEOUT = 300.0
PARALLEL = 4
HUNK_LINES_MAX = 160
NARRATION_CHARS = 700
BRIEF_CHARS = 2500

GROUP_SYSTEM = """You plan a code review. A reviewer must review a large \
change an AI coding agent made. You split the change into coherent groups of \
related files and order the groups from easiest to hardest to review, so the \
reviewer builds understanding before reaching the hard parts.

Rules:
- Every file in the manifest goes in exactly one group. Use the exact paths.
- A group is a unit a reviewer can hold in their head: one feature slice or \
layer (e.g. seed data, data access, API surface, a UI area). 3-12 files is \
typical; never one group per file, never one giant group.
- Tests, stories, and test page-objects go in the group with the code they \
cover, so they read as evidence for that code.
- Order: foundations and low-risk first (data, types, seed, config), then the \
logic that builds on them, then the UI that consumes it. Difficulty 1-5 \
reflects logic density and risk, not file count.
- summary: one short paragraph, plain words: what the group changes and why, \
from the reviewer's point of view.

Reply with only a JSON object in <json></json> tags:
{"groups": [{"id": "g1", "title": "...", "difficulty": 1, "summary": "...", \
"files": ["path", ...]}]}"""

STEP_SYSTEM = """You walk a reviewer through one group of a code change an \
AI coding agent made. For each file, split its hunks into logical steps: \
one step = one coherent change a reviewer approves or rejects as a unit \
(e.g. "add the resolver field", "wire the column into the table"). A step \
may span several hunks; trivial files may be one step.

For each step:
- title: imperative, under 10 words.
- what: 1-2 sentences, what the code does now.
- why: 1-2 sentences, why the agent made this change. Ground it in the \
provided agent context (brief, narration) when it says why; set why_source \
to "transcript". If the context only narrates mechanics, infer from the \
code and the brief and set why_source to "inferred". Never invent intent.
- risk: optional, one sentence, only for a real concern a reviewer should check.
- visual: help the reviewer see the behavior. kind is "none", "mermaid" \
(flowchart TD or sequenceDiagram; plain, no style/classDef lines) or \
"table" (a small HTML <table>). Use "none" when a visual adds nothing — \
most small steps.

Every hunk id appears in exactly one step. Reply with only JSON in \
<json></json> tags:
{"files": [{"path": "...", "steps": [{"title": "...", "hunks": ["h1"], \
"what": "...", "why": "...", "why_source": "transcript", "risk": null, \
"visual": {"kind": "none", "body": ""}}]}]}"""


@dataclass
class CallMetrics:
    stage: str
    seconds: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    model: str | None
    ok: bool
    error: str | None = None


@dataclass
class Analysis:
    groups: list[dict]
    skim: list[str]
    files: dict[str, list[dict]]  # path -> steps
    problems: list[str] = field(default_factory=list)
    calls: list[CallMetrics] = field(default_factory=list)


def _extract_json(body: str) -> dict:
    m = re.search(r"<json>(.*?)</json>", body, re.S)
    raw = m.group(1) if m else body
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n|\n```$", "", raw)
    try:
        return json.loads(raw)
    except ValueError:
        # Asked for HTML tables, the model sometimes writes <br/> for the
        # JSON's own line breaks. As whitespace that's harmless; inside a
        # string it becomes a newline, which strict=False accepts.
        return json.loads(re.sub(r"<br\s*/?>", "\n", raw), strict=False)


RETRIES = 1


def _call(stage: str, system: str, user: str, model: str) -> tuple[dict | None, CallMetrics]:
    """One structured call, retried once; the metric records every attempt's
    time and cost, and on failure what the CLI actually returned."""
    t0 = time.monotonic()
    cost = 0.0
    err = None
    for attempt in range(RETRIES + 1):
        r = None
        try:
            r = invoke_claude(system_prompt=system, user_message=user, model=model, timeout=CALL_TIMEOUT)
            cost += r.cost_usd or 0
            data = _extract_json(r.body)
            return data, CallMetrics(stage, round(time.monotonic() - t0, 1), r.input_tokens,
                                     r.output_tokens, round(cost, 4), r.model, True,
                                     f"retried after: {err}" if err else None)
        except Exception as e:  # noqa: BLE001 — report every failure as a metric
            detail = ""
            if r is not None:
                detail = f" | rc={r.returncode} body={r.body[:160]!r} stderr={r.stderr[-160:]!r}"
            err = f"attempt {attempt + 1}: {type(e).__name__}: {e}"[:200] + detail
    return None, CallMetrics(stage, round(time.monotonic() - t0, 1), None, None, round(cost, 4),
                             None, False, err)


def ticket_context(diff: ReviewDiff, prov: list[HunkProvenance], edits: dict[str, Edit]) -> str:
    """The distinct human prompts and subagent briefs behind the change."""
    seen: dict[str, int] = {}
    for p in prov:
        for u in p.edits[:1]:
            e = edits.get(u)
            if not e:
                continue
            for txt in (e.brief, e.user_prompt):
                if txt and len(txt) > 80:
                    seen[txt] = seen.get(txt, 0) + 1
    top = sorted(seen, key=lambda t: -seen[t])[:4]
    return "\n\n---\n\n".join(t[:BRIEF_CHARS] for t in top) or "(no agent context found)"


def _manifest(diff: ReviewDiff, st: Structure, paths: list[str]) -> str:
    rows = []
    by = {f.path: f for f in diff.files}
    for p in paths:
        f, fi = by[p], st.files[p]
        heads = sorted({h.header[:50] for h in f.hunks if h.header})[:4]
        imp = [i for i in fi.imports if i in paths]
        rows.append(f"- {p} | {fi.kind} | {f.status} +{f.additions}/-{f.deletions}"
                    + (f" | imports: {', '.join(imp)}" if imp else "")
                    + (f" | near: {'; '.join(heads)}" if heads else ""))
    return "\n".join(rows)


def group_files(diff: ReviewDiff, st: Structure, context: str, model: str) -> tuple[list[dict], list[str], CallMetrics]:
    reviewable = [f.path for f in diff.files if not st.files[f.path].skim]
    user = (f"<agent_context>\n{context}\n</agent_context>\n\n"
            f"<manifest count=\"{len(reviewable)}\">\n{_manifest(diff, st, reviewable)}\n</manifest>")
    data, m = _call("group", GROUP_SYSTEM, user, model)
    problems: list[str] = []
    groups = (data or {}).get("groups") or []
    placed: dict[str, str] = {}
    for g in groups:
        g["files"] = [p for p in g.get("files", []) if p in reviewable]
        for p in g["files"]:
            if p in placed:
                problems.append(f"file in two groups: {p}")
            placed[p] = g["id"]
    missing = [p for p in reviewable if p not in placed]
    if missing:
        problems.append(f"{len(missing)} files ungrouped by the model")
        groups.append({"id": "g-rest", "title": "Remaining files", "difficulty": 3,
                       "summary": "Files the grouping pass did not place.", "files": missing})
    return groups, problems, m


def _group_prompt(group: dict, diff: ReviewDiff, prov: dict, edits: dict[str, Edit], context: str) -> tuple[str, dict]:
    by = {f.path: f for f in diff.files}
    ids: dict[str, tuple[str, str]] = {}
    parts = [f"<agent_context>\n{context}\n</agent_context>",
             f"<group title=\"{group['title']}\">\n{group.get('summary', '')}\n</group>"]
    n = 0
    for path in group["files"]:
        f = by[path]
        blocks = []
        for h in f.hunks:
            n += 1
            hid = f"h{n}"
            ids[hid] = (path, h.hash)
            lines = h.lines[:HUNK_LINES_MAX]
            more = f"\n… {len(h.lines) - HUNK_LINES_MAX} more lines" if len(h.lines) > HUNK_LINES_MAX else ""
            p = prov.get((path, h.hash))
            note = ""
            if p and p.edits and p.status in ("attributed", "partial"):
                e = edits.get(p.edits[0])
                if e and e.reasoning:
                    note = f"\n<agent_narration via=\"{e.tool}\">{e.reasoning[-NARRATION_CHARS:]}</agent_narration>"
            blocks.append(f"<hunk id=\"{hid}\" at=\"-{h.old_start} +{h.new_start}\">{note}\n"
                          + "\n".join(lines) + more + "\n</hunk>")
        parts.append(f"<file path=\"{path}\" status=\"{f.status}\">\n" + "\n".join(blocks) + "\n</file>")
    return "\n\n".join(parts), ids


def analyze(diff: ReviewDiff, st: Structure, prov: list[HunkProvenance], edits: dict[str, Edit],
            model: str = "sonnet") -> Analysis:
    context = ticket_context(diff, prov, edits)
    groups, problems, gm = group_files(diff, st, context, model)
    groups.sort(key=lambda g: g.get("difficulty", 3))  # stable: keeps model order within a level
    skim = [f.path for f in diff.files if st.files[f.path].skim]
    pmap = {(p.path, p.hunk_hash): p for p in prov}
    result = Analysis(groups=groups, skim=skim, files={}, problems=problems, calls=[gm])

    def run(g):
        user, ids = _group_prompt(g, diff, pmap, edits, context)
        data, m = _call(f"steps:{g['id']}", STEP_SYSTEM, user, model)
        return g, ids, data, m

    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        for g, ids, data, m in pool.map(run, groups):
            result.calls.append(m)
            if data is None:
                problems.append(f"{g['id']}: step call failed ({m.error})")
                continue
            used: dict[str, int] = {}
            for fobj in data.get("files") or []:
                path = fobj.get("path")
                if path not in g["files"]:
                    continue
                steps = fobj.get("steps") or []
                for s in steps:
                    # A hunk claimed twice stays with its first step; the
                    # repeat is reported so the prompt can be tuned.
                    keep = []
                    for h in s.get("hunks", []):
                        if h in ids and ids[h][0] == path:
                            hh = ids[h][1]
                            used[hh] = used.get(hh, 0) + 1
                            if used[hh] == 1:
                                keep.append(hh)
                    s["hunks"] = keep
                result.files[path] = [s for s in steps if s["hunks"]]
            for hid, (path, hh) in ids.items():
                c = used.get(hh, 0)
                if c != 1:
                    problems.append(f"{g['id']}: hunk {hid} in {path.rsplit('/', 1)[-1]} covered {c}x")
    return result
