"""Diff ingestion: a branch's changes vs its base as files and hunks.

The diff is taken against the merge-base (`base...head`), the same range a PR
shows. Every hunk gets a content hash so a later approval can be pinned to the
exact code it approved and invalidated when that code changes.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import asdict, dataclass, field

GIT_TIMEOUT = 60

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    header: str
    lines: list[str] = field(default_factory=list)  # raw, with +/-/space prefix
    hash: str = ""

    @property
    def added(self) -> list[str]:
        return [ln[1:] for ln in self.lines if ln.startswith("+")]

    @property
    def removed(self) -> list[str]:
        return [ln[1:] for ln in self.lines if ln.startswith("-")]


@dataclass
class FileDiff:
    path: str
    old_path: str | None
    status: str  # A, M, D, R
    binary: bool = False
    additions: int = 0
    deletions: int = 0
    hunks: list[Hunk] = field(default_factory=list)


@dataclass
class ReviewDiff:
    repo: str
    base: str
    head: str
    base_sha: str
    head_sha: str
    files: list[FileDiff]
    diff_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args],
        check=True, capture_output=True, text=True, errors="replace",
        timeout=GIT_TIMEOUT,
    ).stdout


def _hunk_hash(path: str, h: Hunk) -> str:
    body = "\n".join(ln for ln in h.lines if ln[:1] in "+-")
    return hashlib.sha256(f"{path}\0{body}".encode()).hexdigest()[:16]


def parse_patch(patch: str) -> list[FileDiff]:
    """Parse `git diff` output (no color, with -M) into FileDiffs."""
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    hunk: Hunk | None = None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(.*) b/(.*)$", line)
            a, b = (m.group(1), m.group(2)) if m else ("", "")
            cur = FileDiff(path=b, old_path=a if a != b else None, status="M")
            files.append(cur)
            hunk = None
            continue
        if cur is None:
            continue
        if hunk is None:
            if line.startswith("new file mode"):
                cur.status = "A"
            elif line.startswith("deleted file mode"):
                cur.status = "D"
            elif line.startswith("rename from "):
                cur.status, cur.old_path = "R", line[len("rename from "):]
            elif line.startswith("rename to "):
                cur.path = line[len("rename to "):]
            elif line.startswith("Binary files "):
                cur.binary = True
        m = _HUNK_RE.match(line)
        if m:
            hunk = Hunk(
                old_start=int(m.group(1)), old_len=int(m.group(2) or 1),
                new_start=int(m.group(3)), new_len=int(m.group(4) or 1),
                header=m.group(5).strip(),
            )
            cur.hunks.append(hunk)
            continue
        if hunk is not None and line[:1] in ("+", "-", " ") and not line.startswith(("+++ ", "--- ")):
            hunk.lines.append(line)
            if line.startswith("+"):
                cur.additions += 1
            elif line.startswith("-"):
                cur.deletions += 1
        elif hunk is not None and line.startswith("\\"):
            continue  # "\ No newline at end of file"
    for f in files:
        for h in f.hunks:
            h.hash = _hunk_hash(f.path, h)
    return files


def collect(repo: str, base: str, head: str) -> ReviewDiff:
    """The diff of `head` against its merge-base with `base`."""
    base_sha = git(repo, "merge-base", base, head).strip()
    head_sha = git(repo, "rev-parse", head).strip()
    patch = git(repo, "diff", "--no-color", "--no-ext-diff", "-M", "--unified=3",
                base_sha, head_sha)
    files = parse_patch(patch)
    diff_hash = hashlib.sha256(
        "\n".join(h.hash for f in files for h in f.hunks).encode()
    ).hexdigest()[:16]
    return ReviewDiff(repo=repo, base=base, head=head, base_sha=base_sha,
                      head_sha=head_sha, files=files, diff_hash=diff_hash)
