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
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from regen import invoke_claude

from .diff import ReviewDiff
from .provenance import Edit, HunkProvenance
from .structure import Structure

CALL_TIMEOUT = 300.0
PARALLEL = 6
# Step calls are packed to about this much diff text (~10k tokens) so no
# single call runs long; a big group becomes several parallel calls.
CHUNK_CHARS = 40_000
HUNK_LINES_MAX = 160
SPLIT_LINES = 60  # hunks longer than this are shown with line indexes and may be split
SPLIT_LINES_MAX = 450
NARRATION_CHARS = 700
BRIEF_CHARS = 2500

GROUP_SYSTEM = """You plan a code review. A reviewer must review a large \
change an AI coding agent made. You split the change into coherent groups of \
related files and order the groups from easiest to hardest to review, so the \
reviewer builds understanding before reaching the hard parts.

Rules:
- Every file in the manifest goes in exactly one group.
- A group is a unit a reviewer can hold in their head: one feature slice or \
layer (e.g. seed data, data access, API surface, a UI area). 3-12 files is \
typical; never one group per file, never one giant group.
- Tests, stories, and test page-objects go in the group with the code they \
cover, so they read as evidence for that code.
- Review order is computed afterwards from the imports: a group is reviewed \
after every group it builds on. So group along dependency lines: keep a \
service and the code it needs together or upstream, and never put a \
consumer (resolver, page) in the same group as an unrelated provider just \
to balance sizes.
- difficulty 1-5 reflects logic density and risk, not file count; it only \
orders groups that don't depend on each other.
- summary: one short paragraph, plain words: what the group changes and why, \
from the reviewer's point of view. Do not refer to other groups by number \
or position.

Files are numbered in the manifest; refer to them ONLY by number. Reply \
with only a JSON object in <json></json> tags:
{"groups": [{"id": "g1", "title": "...", "difficulty": 1, "summary": "...", \
"files": [1, 4, 5]}]}"""

STEP_SYSTEM = """You walk a reviewer through one group of a code change an \
AI coding agent made. For each file, split its hunks into logical steps: \
one step = one coherent change a reviewer approves or rejects as a unit \
(e.g. "add the resolver field", "wire the column into the table"). A step \
may span several hunks; trivial files may be one step.

Large hunks (a new file, a big rewrite) are shown with line indexes \
("L12|"). Split them by unit (a type, a function, a component, a group of \
related tests) using ranges: "h3:1-40" claims lines 1-40 of hunk h3. Aim \
for steps a reviewer reads in a minute or two (roughly 15-60 lines); never \
one step for a 150-line file.

For each step:
- title: imperative, under 10 words.
- what: 1-2 sentences describing what the code is meant to do. Describe \
intent and mechanism; NEVER vouch for correctness ("already handles X", \
"correctly filters Y") — the reviewer is the one who decides that.
- why: 1-2 sentences, why the agent made this change. Ground it in the \
provided agent context (brief, narration) when it says why; set why_source \
to "transcript". If the context only narrates mechanics, infer from the \
code and the brief and set why_source to "inferred". Never invent intent.
- risk: optional, one sentence naming a specific thing to verify in THIS \
code, or null. Look through these lenses and report the sharpest one that \
applies: lifecycle/status states the code may mishandle (archived, draft, \
deleted, failed, empty); cost that grows with data (per-row queries, loading \
whole populations to show a page); numbers shown vs numbers computed \
(rounding, units, deltas); requirements in the brief this code may not \
meet; sibling code, docs or call sites that may also need the change; \
error handling that differs from neighbouring methods. No generic advice.
- visual: help the reviewer see the behavior. kind is "none", "mermaid" \
(flowchart TD or sequenceDiagram; plain, no style/classDef lines) or \
"table" (a small HTML <table>). Use "none" when a visual adds nothing — \
most small steps.

Every hunk (or every line of a ranged hunk) belongs to exactly one step; \
never list the same hunk in two steps. Reply with only JSON in \
<json></json> tags:
{"files": [{"path": "...", "steps": [{"title": "...", "hunks": ["h1", "h3:1-40"], \
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
    """The first complete JSON object in the reply, wherever it starts:
    tolerates <json> tags, code fences, prose around it and trailing text."""
    m = re.search(r"<json>(.*?)</json>", body, re.S)
    raw = (m.group(1) if m else body).strip()
    dec = json.JSONDecoder(strict=False)
    last_err: Exception | None = None
    variants = (raw, re.sub(r"<br\s*/?>", "\n", raw), body)
    # Outermost object first in every variant, only then any later "{":
    # an inner object must never win over a repairable outer one.
    for first_only in (True, False):
        for text in variants:
            starts = [i for i, ch in enumerate(text) if ch == "{"]
            for i in starts[:1] if first_only else starts[1:]:
                try:
                    obj, _ = dec.raw_decode(text, i)
                except ValueError as e:
                    last_err = e
                    continue
                if isinstance(obj, dict):
                    return obj
    raise last_err or ValueError("no JSON object in reply")


RETRIES = 1
# Hidden thinking is most of each call's output tokens, and output tokens set
# its wall time. Overridable per stage for experiments.
# Measured in M0: grouping at low is ~2x faster with no visible quality loss.
# Steps keep the CLI default: lower effort was much faster but no better and
# possibly worse on review recall (n=8 findings; too few to tune further).
EFFORT = {"group": os.environ.get("CCD_REVIEW_EFFORT_GROUP") or "low",
          "steps": os.environ.get("CCD_REVIEW_EFFORT_STEPS") or None}
TASK_TAIL = {
    "group": "\n\nTask: group every numbered file above per the rules and reply with only the <json> object.",
    "steps": "\n\nTask: write the walkthrough steps for every hunk above per the rules and reply with only the <json> object.",
    "judge": "\n\nTask: give a verdict for every finding above and reply with only the <json> object.",
}


def _call(stage: str, system: str, user: str, model: str) -> tuple[dict | None, CallMetrics]:
    """One structured call, retried once; the metric records every attempt's
    time and cost, and on failure what the CLI actually returned."""
    t0 = time.monotonic()
    cost = 0.0
    err = None
    for attempt in range(RETRIES + 1):
        r = None
        try:
            kind = "group" if stage == "group" else "judge" if stage == "judge" else "steps"
            r = invoke_claude(system_prompt=system, user_message=user + TASK_TAIL[kind], model=model,
                              timeout=CALL_TIMEOUT,
                              effort=EFFORT.get(kind))
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
    """One numbered row per file; imports refer to other rows by number so
    the model never has to echo paths back."""
    rows = []
    by = {f.path: f for f in diff.files}
    num = {p: i for i, p in enumerate(paths, 1)}
    for p in paths:
        f, fi = by[p], st.files[p]
        heads = sorted({h.header[:50] for h in f.hunks if h.header})[:4]
        imp = [str(num[i]) for i in fi.imports if i in num]
        rows.append(f"{num[p]}. {p} | {fi.kind} | {f.status} +{f.additions}/-{f.deletions}"
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
    def path_of(x):
        if isinstance(x, int) or (isinstance(x, str) and x.strip().isdigit()):
            i = int(x)
            return reviewable[i - 1] if 1 <= i <= len(reviewable) else None
        return x if x in reviewable else None  # tolerate a path anyway

    for g in groups:
        g["files"] = [p for p in map(path_of, g.get("files", [])) if p]
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
            if len(h.lines) > SPLIT_LINES:
                cap = SPLIT_LINES_MAX
                lines = [f"L{i}|{ln}" for i, ln in enumerate(h.lines[:cap], 1)]
            else:
                cap = HUNK_LINES_MAX
                lines = h.lines[:cap]
            more = f"\n… {len(h.lines) - cap} more lines" if len(h.lines) > cap else ""
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


_REF_RE = re.compile(r"^(h\d+)(?::(\d+)-(\d+))?$")


def _ranges(lines: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for n in sorted(lines):
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def _ref(hash_: str, a: int, b: int, size: int) -> str:
    return hash_ if (a, b) == (1, size) else f"{hash_}:{a}-{b}"


def _claim(refs: list, ids: dict, path: str, sizes: dict, claimed: dict) -> tuple[list[str], int]:
    """Turn a step's "h3" / "h3:1-40" refs into hunk-hash refs, keeping only
    lines no earlier step claimed (first claim wins). Also returns how many
    requested lines were already claimed."""
    out: list[str] = []
    dup = 0
    for r in refs:
        m = _REF_RE.match(str(r).strip())
        if not m or m.group(1) not in ids or ids[m.group(1)][0] != path:
            continue
        hh = ids[m.group(1)][1]
        size = sizes[hh]
        a, b = (int(m.group(2)), int(m.group(3))) if m.group(2) else (1, size)
        a, b = max(1, a), min(size, b)
        taken = claimed.setdefault(hh, set())
        new = [n for n in range(a, b + 1) if n not in taken]
        dup += (b - a + 1) - len(new)
        taken.update(new)
        out += [_ref(hh, x, y, size) for x, y in _ranges(new)]
    return out, dup


def _fill_gaps(g: dict, ids: dict, sizes: dict, claimed: dict, files: dict, problems: list,
               blank: dict | None = None) -> None:
    """Attach lines no step claimed to the step holding the adjacent line of
    the same hunk (or a catch-all step). Gaps of blank lines are attached
    quietly; anything else is reported."""
    for hid, (path, hh) in ids.items():
        size = sizes[hh]
        missing = [n for n in range(1, size + 1) if n not in claimed.get(hh, set())]
        if not missing:
            continue
        steps = files.setdefault(path, [])

        def owner(line: int):
            for st in steps:
                for r in st["hunks"]:
                    h, _, rg = r.partition(":")
                    if h != hh:
                        continue
                    a, b = map(int, rg.split("-")) if rg else (1, size)
                    if a <= line <= b:
                        return st
            return None

        real = [n for n in missing if not (blank and n in blank.get(hh, set()))]
        if real:
            problems.append(f"{g['id']}: {hid} in {path.rsplit('/', 1)[-1]} had {len(real)} unclaimed lines")
        for a, b in _ranges(missing):
            host = owner(a - 1) or owner(b + 1)
            if host is None:
                host = {"title": "Remaining changes", "hunks": [], "what": "Lines the walkthrough did not assign to a step.",
                        "why": "", "why_source": "inferred", "risk": None, "visual": {"kind": "none", "body": ""}}
                steps.append(host)
            host["hunks"].append(_ref(hh, a, b, size))
            claimed.setdefault(hh, set()).update(range(a, b + 1))
        for st in steps:  # merge adjacent ranges of one hunk back together
            st["hunks"] = _merge_refs(st["hunks"], sizes)


def _merge_refs(refs: list[str], sizes: dict) -> list[str]:
    lines: dict[str, set[int]] = {}
    order: list[str] = []
    for r in refs:
        h, _, rg = r.partition(":")
        a, b = map(int, rg.split("-")) if rg else (1, sizes[h])
        if h not in lines:
            order.append(h)
        lines.setdefault(h, set()).update(range(a, b + 1))
    return [_ref(h, a, b, sizes[h]) for h in order for a, b in _ranges(lines[h])]


TEST_KINDS = ("test", "test-support")


def order_groups(groups: list[dict], st: Structure) -> list[dict]:
    """Review order: a group comes after every group it builds on (a file in
    it imports a file there, or it consumes generated API types or a query
    client and they hold API code), difficulty breaks ties, and test-only groups
    (e2e suites) go last. Cycles fall back to difficulty. Sets each group's
    `builds_on` to the ids it depends on."""
    where = {p: g["id"] for g in groups for p in g["files"]}
    deps: dict[str, set[str]] = {g["id"]: set() for g in groups}
    for a, b in st.edges:
        ga, gb = where.get(a), where.get(b)
        if ga and gb and ga != gb and st.files[a].kind not in TEST_KINDS:
            deps[ga].add(gb)
    # Clients reach the API through generated code (GraphQL/OpenAPI types),
    # which sits in the skim group, so no import edge links them. A group
    # importing generated code builds on every group holding API code.
    # API providers only: a frontend's src/api/queries.ts is a client, not the API.
    api_groups = {g["id"] for g in groups
                  if any(st.files[p].kind == "api" and not st.files[p].api_client for p in g["files"])}
    for a, b in st.edges:
        ga = where.get(a)
        if (ga and ga not in api_groups and st.files[a].kind not in TEST_KINDS
                and b in st.files and st.files[b].kind == "generated"):
            deps[ga] |= api_groups
    # Same for files that query the API through a client package (gql``).
    for p, fi in st.files.items():
        ga = where.get(p)
        if ga and ga not in api_groups and fi.api_client and fi.kind not in TEST_KINDS:
            deps[ga] |= api_groups
    by_id = {g["id"]: g for g in groups}
    pos = {g["id"]: i for i, g in enumerate(groups)}
    test_only = {g["id"] for g in groups
                 if g["files"] and all(st.files[p].kind in TEST_KINDS for p in g["files"])}

    def rank(gid: str) -> tuple:
        return (gid in test_only, by_id[gid].get("difficulty", 3), pos[gid])

    out: list[str] = []
    left = set(deps)
    while left:
        ready = [g for g in left if not (deps[g] & left)]
        if not ready or all(g in test_only for g in ready) and left - test_only:
            # A cycle (routers and pages import each other). Break it at the
            # group with the fewest unmet dependencies — never by pulling a
            # test-only group forward.
            pool = (left - test_only) or left
            nxt = min(pool, key=lambda g: (len(deps[g] & left),) + rank(g))
        else:
            nxt = min(ready, key=rank)
        out.append(nxt)
        left.discard(nxt)
    for gid in out:
        by_id[gid]["builds_on"] = sorted(deps[gid], key=out.index)
    return [by_id[g] for g in out]


def chunk_groups(groups: list[dict], diff: ReviewDiff, budget: int = CHUNK_CHARS) -> list[dict]:
    """Split each group's files into step-call units of at most `budget`
    characters of diff (a file larger than the budget gets a unit alone).
    Units share the group's title and summary; ids are "g3", "g3.2", ..."""
    size = {f.path: sum(len(ln) + 1 for h in f.hunks for ln in h.lines) for f in diff.files}
    units: list[dict] = []
    for g in groups:
        parts: list[list[str]] = [[]]
        used = 0
        for p in g["files"]:
            if parts[-1] and used + size.get(p, 0) > budget:
                parts.append([])
                used = 0
            parts[-1].append(p)
            used += size.get(p, 0)
        parts = [x for x in parts if x]  # an empty group needs no call
        for k, files in enumerate(parts, 1):
            units.append({**g, "id": g["id"] if len(parts) == 1 else f"{g['id']}.{k}", "files": files})
    return units


def analyze(diff: ReviewDiff, st: Structure, prov: list[HunkProvenance], edits: dict[str, Edit],
            model: str = "sonnet") -> Analysis:
    context = ticket_context(diff, prov, edits)
    groups, problems, gm = group_files(diff, st, context, model)
    groups = order_groups(groups, st)
    skim = [f.path for f in diff.files if st.files[f.path].skim]
    pmap = {(p.path, p.hunk_hash): p for p in prov}
    result = Analysis(groups=groups, skim=skim, files={}, problems=problems, calls=[gm])

    sizes = {h.hash: len(h.lines) for f in diff.files for h in f.hunks}
    blank = {h.hash: {i for i, ln in enumerate(h.lines, 1) if not ln[1:].strip()}
             for f in diff.files for h in f.hunks}

    def run(g):
        user, ids = _group_prompt(g, diff, pmap, edits, context)
        data, m = _call(f"steps:{g['id']}", STEP_SYSTEM, user, model)
        return g, ids, data, m

    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        for g, ids, data, m in pool.map(run, chunk_groups(groups, diff)):
            result.calls.append(m)
            if data is None:
                problems.append(f"{g['id']}: step call failed ({m.error})")
                continue
            claimed: dict[str, set[int]] = {}
            for fobj in data.get("files") or []:
                path = fobj.get("path")
                if path not in g["files"]:
                    continue
                steps = fobj.get("steps") or []
                for s in steps:
                    s["hunks"], dup = _claim(s.get("hunks", []), ids, path, sizes, claimed)
                    if dup:
                        problems.append(f"{g['id']}: {dup} lines of {path.rsplit('/', 1)[-1]} claimed twice; kept the first claim")
                result.files[path] = [s for s in steps if s["hunks"]]
            _fill_gaps(g, ids, sizes, claimed, result.files, problems, blank)
    return result
    return result
