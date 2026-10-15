"""Provenance pass: which transcript edit wrote each hunk, and why.

Every Edit/MultiEdit/Write tool call in a Claude Code transcript carries the
text it wrote, and sits after the agent's own words about what it was doing.
Matching a hunk's added lines against those written texts recovers the edit
that produced it; the surrounding assistant text is the agent's reasoning,
recorded at the time rather than reconstructed after the fact.

Shell edits (sed, python heredocs via the Bash tool) match the same way
against the command text. Hunks with no match (formatters, codegen, human
edits) are reported as unattributed — never guessed.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field

from .diff import ReviewDiff

EDIT_TOOLS = ("Edit", "MultiEdit", "Write")
# Reasoning context kept per edit: the last few assistant text blocks since
# the user's most recent message.
REASONING_BLOCKS = 3
REASONING_CHARS = 2000
PROMPT_CHARS = 3000
ATTRIBUTED = 0.6
PARTIAL = 0.2


@dataclass
class Edit:
    rel_hint: str  # absolute file_path as written by the tool
    tool: str
    written: list[str]  # new_string / content blocks
    replaced: list[str]  # old_string blocks (Edit/MultiEdit)
    session: str
    uuid: str
    timestamp: str
    branch: str | None
    sidechain: bool
    reasoning: str
    thinking: str
    user_prompt: str  # latest human-authored prompt in the main chat
    brief: str = ""  # a subagent's original task brief (its first message)


@dataclass
class HunkProvenance:
    path: str
    hunk_hash: str
    status: str  # attributed | partial | none | trivial
    coverage: float
    edits: list[str] = field(default_factory=list)  # edit uuids, best first
    branch_match: bool = False


def _norm(line: str) -> str:
    return " ".join(line.split())


def _significant(lines: list[str]) -> list[str]:
    """Lines worth matching: drop blanks and pure punctuation (`});`, `}`)."""
    out = []
    for ln in lines:
        n = _norm(ln)
        if len(n) >= 4 and any(ch.isalnum() for ch in n):
            out.append(n)
    return out


_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def _text_of(content) -> str:
    if isinstance(content, list):
        content = "\n".join(x.get("text", "") for x in content
                            if isinstance(x, dict) and x.get("type") == "text")
    if not isinstance(content, str):
        return ""
    return _REMINDER_RE.sub("", content)


def scan(transcripts: list[str], wanted_suffixes: set[str]) -> list[Edit]:
    """All edit tool calls touching a wanted path, with their reasoning."""
    edits: list[Edit] = []
    for path in transcripts:
        texts: list[str] = []
        thinking = ""
        prompt = ""
        brief = ""
        try:
            fh = open(path, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if '"assistant"' not in line and '"user"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                msg = d.get("message") or {}
                content = msg.get("content")
                if d.get("type") == "user":
                    if d.get("toolUseResult") or d.get("isMeta"):
                        continue
                    text = _text_of(content).strip()
                    if not text:
                        continue
                    if d.get("isSidechain") and not brief:
                        brief = text[:PROMPT_CHARS]  # the subagent's task
                    elif (d.get("origin") or {}).get("kind") == "human" or d.get("turnOrigin") == "human":
                        prompt = text[:PROMPT_CHARS]
                    # Any new instruction resets the reasoning window.
                    texts, thinking = [], ""
                    continue
                if not isinstance(content, list):
                    continue
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    t = blk.get("type")
                    if t == "text" and blk.get("text", "").strip():
                        texts = (texts + [blk["text"].strip()])[-REASONING_BLOCKS:]
                    elif t == "thinking" and blk.get("thinking", "").strip():
                        thinking = blk["thinking"].strip()
                    elif t == "tool_use" and blk.get("name") == "Bash":
                        # Agents also edit through the shell (sed, python
                        # heredocs). The command text holds what it wrote, so it
                        # matches like an edit; it is tied to every changed file
                        # whose basename it mentions, and lines must still match.
                        cmd = (blk.get("input") or {}).get("command") or ""
                        for s in wanted_suffixes:
                            if os.path.basename(s) in cmd:
                                edits.append(Edit(
                                    rel_hint="/" + s, tool="Bash", written=[cmd],
                                    replaced=[cmd], session=d.get("sessionId") or "",
                                    uuid=f"{blk.get('id') or d.get('uuid')}:{s}",
                                    timestamp=d.get("timestamp") or "",
                                    branch=d.get("gitBranch"), sidechain=bool(d.get("isSidechain")),
                                    reasoning="\n\n".join(texts)[-REASONING_CHARS:],
                                    thinking=thinking[-REASONING_CHARS:], user_prompt=prompt, brief=brief,
                                ))
                    elif t == "tool_use" and blk.get("name") in EDIT_TOOLS:
                        inp = blk.get("input") or {}
                        fp = inp.get("file_path") or ""
                        if not any(fp.endswith("/" + s) for s in wanted_suffixes):
                            continue
                        if blk["name"] == "Write":
                            written, replaced = [inp.get("content") or ""], []
                        elif blk["name"] == "MultiEdit":
                            ops = inp.get("edits") or []
                            written = [o.get("new_string") or "" for o in ops]
                            replaced = [o.get("old_string") or "" for o in ops]
                        else:
                            written = [inp.get("new_string") or ""]
                            replaced = [inp.get("old_string") or ""]
                        edits.append(Edit(
                            rel_hint=fp, tool=blk["name"], written=written,
                            replaced=replaced, session=d.get("sessionId") or "",
                            uuid=blk.get("id") or d.get("uuid") or "",
                            timestamp=d.get("timestamp") or "",
                            branch=d.get("gitBranch"), sidechain=bool(d.get("isSidechain")),
                            reasoning="\n\n".join(texts)[-REASONING_CHARS:],
                            thinking=thinking[-REASONING_CHARS:], user_prompt=prompt, brief=brief,
                        ))
    edits.sort(key=lambda e: e.timestamp)
    return edits


def transcripts_from_glob(pattern: str) -> list[str]:
    """Main transcripts plus subagent transcripts nested under each session."""
    base = os.path.expanduser(pattern)
    out = set(glob.glob(base))
    for p in list(out):
        out.update(glob.glob(os.path.join(p[: -len(".jsonl")], "**", "*.jsonl"), recursive=True))
    return sorted(out)


def attribute(diff: ReviewDiff, edits: list[Edit], branch: str | None) -> list[HunkProvenance]:
    by_path: dict[str, list[Edit]] = {}
    for f in diff.files:
        by_path[f.path] = [e for e in edits if e.rel_hint.endswith("/" + f.path)]

    out: list[HunkProvenance] = []
    for f in diff.files:
        cands = by_path[f.path]
        # Prefer edits made on this ticket's branch; fall back to any branch.
        on_branch = [e for e in cands if branch and e.branch == branch]
        for h in f.hunks:
            target = _significant(h.added) or _significant(h.removed)
            use_removed = not _significant(h.added)
            if not target:
                out.append(HunkProvenance(f.path, h.hash, "trivial", 1.0))
                continue
            best = None
            for pool, is_branch in ((on_branch, True), (cands, False)):
                scored = []
                covered: set[int] = set()
                for e in pool:
                    blob = e.replaced if use_removed else e.written
                    lines = {_norm(x) for b in blob for x in b.splitlines()}
                    hit = {i for i, ln in enumerate(target) if ln in lines}
                    if hit:
                        scored.append((len(hit), e.timestamp, e.uuid))
                        covered |= hit
                if scored:
                    cov = len(covered) / len(target)
                    ranked = [u for _, _, u in sorted(scored, reverse=True)]
                    cand = (cov, is_branch, ranked)
                    if best is None or cov > best[0] + 0.1:
                        best = cand
                if best and best[0] >= ATTRIBUTED:
                    break
            if best is None:
                out.append(HunkProvenance(f.path, h.hash, "none", 0.0))
                continue
            cov, is_branch, ranked = best
            status = "attributed" if cov >= ATTRIBUTED else "partial" if cov >= PARTIAL else "none"
            out.append(HunkProvenance(f.path, h.hash, status, round(cov, 3),
                                      ranked[:5], is_branch))
    return out
