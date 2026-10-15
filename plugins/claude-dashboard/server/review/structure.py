"""Structure pass: cheap, deterministic signals about a diff's files.

Two outputs seed the analysis pass and drive the relationship graphs:
  - a kind tag per file (seed, migration, test, generated, ui, api, ...), and
    whether it belongs in the "skim" group (generated / bulky / translations);
  - an import graph restricted to the changed files (edge a -> b: a imports b).

Import resolution is deliberately heuristic (TS/JS only for now): relative
specifiers resolve exactly; aliased ones ("@/lib/x", "~/x") match by path
suffix against the changed set. Both are cheap and good enough to say which
changed files talk to each other.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field

from .diff import ReviewDiff, git

# Above this many added lines a non-source file is "bulky" and goes to skim.
BULKY_LINES = 1500

_SOURCE_EXT = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
_IMPORT_RE = re.compile(
    r"""(?:^|\s)(?:import|export)\s[^'"]*?from\s*['"]([^'"]+)['"]"""
    r"""|(?:^|\s)import\s*['"]([^'"]+)['"]"""
    r"""|(?:import|require)\(\s*['"]([^'"]+)['"]\s*\)""",
    re.M,
)


@dataclass
class FileInfo:
    path: str
    kind: str
    skim: bool
    skim_reason: str | None = None
    imports: list[str] = field(default_factory=list)  # changed files it imports


@dataclass
class Structure:
    files: dict[str, FileInfo]
    edges: list[tuple[str, str]]


def classify(path: str, additions: int) -> tuple[str, bool, str | None]:
    """(kind, skim, skim_reason) from the path alone, plus size."""
    p = path.lower()
    name = posixpath.basename(p)
    if name in ("package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lockb", "poetry.lock"):
        return "lockfile", True, "lockfile"
    if "/__generated__/" in p or ".gen." in name or name.endswith(".gql") or name.endswith(".graphql"):
        return "generated", True, "generated code/schema"
    if "/migrations/" in p:
        if name.endswith(".json"):
            return "migration", True, "migration snapshot"
        return "migration", False, None
    if name.endswith((".po", ".pot")) or "/locales/" in p or "/i18n/" in p:
        return "i18n", True, "translation catalog"
    if name.endswith(".snap") or "/__snapshots__/" in p:
        return "snapshot", True, "test snapshot"
    if "/page-objects/" in p:
        return "test-support", False, None
    if (re.search(r"\.(spec|test)\.[a-z]+$", name) or "/__tests__/" in p
            or p.startswith(("tests/", "test/", "e2e/")) or "/e2e/" in p or name.startswith("test_")):
        return "test", False, None
    if ".stories." in name:
        return "story", False, None
    if "/seed" in p or "/fixtures/" in p:
        return "seed", False, None
    if name.endswith((".md", ".mdx")):
        return "docs", False, None
    if name.endswith((".json", ".yaml", ".yml", ".toml")) or name.startswith((".", "tsconfig")) or "config" in name:
        kind = "config"
    elif name.endswith((".tsx", ".jsx", ".css", ".scss")) or "/components/" in p or "/pages/" in p or "/routes/" in p:
        kind = "ui"
    elif any(s in p for s in ("/api/", "/server/", "/modules/", "/resolvers/", "/repositories/", "/services/")):
        kind = "api"
    else:
        kind = "source"
    if additions > BULKY_LINES and not name.endswith(_SOURCE_EXT):
        return kind, True, f"bulky ({additions} lines)"
    return kind, False, None


def _strip_ext(path: str) -> str:
    for ext in _SOURCE_EXT:
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def _resolve(spec: str, importer: str, changed: dict[str, str]) -> str | None:
    """The changed file `spec` refers to, or None. `changed` maps
    extension-less path (and dir for index files) -> real path."""
    if spec.startswith("."):
        target = posixpath.normpath(posixpath.join(posixpath.dirname(importer), spec))
        return changed.get(_strip_ext(target)) or changed.get(target + "/index")
    # Aliased / package import: drop the alias head, match by path suffix.
    tail = re.sub(r"^(@/|~/|#/|@\w[\w-]*/)", "", spec)
    if "/" not in tail and tail == spec:
        return None  # bare package ("react"), never a changed file
    tail = _strip_ext(tail)
    hits = [real for key, real in changed.items()
            if key.endswith("/" + tail) or key.endswith("/" + tail + "/index")]
    return hits[0] if len(hits) == 1 else None


def analyze(diff: ReviewDiff) -> Structure:
    files: dict[str, FileInfo] = {}
    for f in diff.files:
        kind, skim, why = classify(f.path, f.additions)
        files[f.path] = FileInfo(path=f.path, kind=kind, skim=skim, skim_reason=why)

    changed: dict[str, str] = {}
    for f in diff.files:
        if f.path.endswith(_SOURCE_EXT):
            key = _strip_ext(f.path)
            changed[key] = f.path

    edges: list[tuple[str, str]] = []
    for f in diff.files:
        if f.status == "D" or f.binary or not f.path.endswith(_SOURCE_EXT):
            continue
        try:
            src = git(diff.repo, "show", f"{diff.head_sha}:{f.path}")
        except Exception:
            continue
        seen: set[str] = set()
        for m in _IMPORT_RE.finditer(src):
            spec = next(g for g in m.groups() if g)
            tgt = _resolve(spec, f.path, changed)
            if tgt and tgt != f.path and tgt not in seen:
                seen.add(tgt)
                edges.append((f.path, tgt))
        files[f.path].imports = sorted(seen)
    return Structure(files=files, edges=edges)
