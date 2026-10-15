"""Tests for the review package's deterministic passes: diff parsing, the
structure pass (kinds, skim, import graph) and transcript provenance.
Builds a throwaway git repo and transcript; no network, no `claude -p`.
Run: ../.venv/bin/python test_review.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from review.diff import collect, parse_patch
from review.provenance import attribute, scan
from review.structure import analyze, classify

_tmp = Path(tempfile.mkdtemp(prefix="ccd-review-test-"))
REPO = _tmp / "repo"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True,
                          capture_output=True, text=True).stdout


def _write(rel: str, text: str) -> None:
    p = REPO / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


REPO.mkdir()
_git("init", "-q", "-b", "main")
_git("config", "user.email", "t@example.com")
_git("config", "user.name", "t")
_write("src/api/score.ts", "export function score(a: number) {\n  return a;\n}\n")
_write("src/ui/old.tsx", "export const Old = () => null;\n")
_git("add", "-A")
_git("commit", "-qm", "base")
_git("switch", "-qc", "feature")
_write("src/api/score.ts",
       "export function score(a: number, weight: number) {\n  return a * weight;\n}\n")
_write("src/ui/ScoreCard.tsx",
       "import { score } from '../api/score';\nexport const ScoreCard = () => score(1, 2);\n")
_write("src/ui/__tests__/ScoreCard.test.tsx",
       "import { ScoreCard } from '../ScoreCard';\ntest('renders', () => ScoreCard());\n")
_write("src/__generated__/graphql.ts", "export type Q = { a: number };\n")
_git("mv", "src/ui/old.tsx", "src/ui/legacy.tsx")
_git("add", "-A")
_git("commit", "-qm", "feature")

DIFF = collect(str(REPO), "main", "feature")
BY = {f.path: f for f in DIFF.files}


# ─── diff ──────────────────────────────────────────────────────────────

def test_collect_lists_every_changed_file_with_status():
    assert BY["src/api/score.ts"].status == "M"
    assert BY["src/ui/ScoreCard.tsx"].status == "A"
    assert BY["src/ui/legacy.tsx"].status == "R"
    assert BY["src/ui/legacy.tsx"].old_path == "src/ui/old.tsx"


def test_hunks_carry_added_and_removed_lines():
    h = BY["src/api/score.ts"].hunks[0]
    assert "  return a * weight;" in h.added
    assert "  return a;" in h.removed
    assert BY["src/api/score.ts"].additions == 2 and BY["src/api/score.ts"].deletions == 2


def test_hunk_hash_changes_with_content_and_is_stable():
    again = collect(str(REPO), "main", "feature")
    assert again.diff_hash == DIFF.diff_hash
    patched = parse_patch(
        "diff --git a/x.ts b/x.ts\n--- a/x.ts\n+++ b/x.ts\n@@ -1 +1 @@\n-a\n+b\n")
    other = parse_patch(
        "diff --git a/x.ts b/x.ts\n--- a/x.ts\n+++ b/x.ts\n@@ -1 +1 @@\n-a\n+c\n")
    assert patched[0].hunks[0].hash != other[0].hunks[0].hash


def test_parse_patch_ignores_no_newline_marker():
    files = parse_patch("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n"
                        "\\ No newline at end of file\n+b\n")
    assert files[0].hunks[0].lines == ["-a", "+b"]


# ─── structure ─────────────────────────────────────────────────────────

def test_classify_kinds_and_skim():
    assert classify("apps/x/src/__generated__/graphql.ts", 10)[:2] == ("generated", True)
    assert classify("apps/x/locales/en-US/messages.po", 10)[:2] == ("i18n", True)
    assert classify("db/migrations/0001/snapshot.json", 10)[:2] == ("migration", True)
    assert classify("db/migrations/0001/migration.sql", 10)[:2] == ("migration", False)
    assert classify("src/ui/__tests__/A.test.tsx", 10)[0] == "test"
    assert classify("scripts/seed/data/x.ts", 10)[0] == "seed"
    assert classify("scripts/seed/x.spec.ts", 10)[0] == "test"
    assert classify("src/ui/A.stories.tsx", 10)[0] == "story"
    assert classify("data/huge.json", 5000)[1] is True
    assert classify("src/big.ts", 5000)[1] is False  # source is never skimmed by size


def test_import_graph_links_changed_files_only():
    st = analyze(DIFF)
    assert ("src/ui/ScoreCard.tsx", "src/api/score.ts") in st.edges
    assert ("src/ui/__tests__/ScoreCard.test.tsx", "src/ui/ScoreCard.tsx") in st.edges
    assert st.files["src/__generated__/graphql.ts"].skim


# ─── provenance ────────────────────────────────────────────────────────

def _transcript(events: list[dict]) -> str:
    p = _tmp / f"t{len(list(_tmp.glob('t*.jsonl')))}.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return str(p)


def _assistant(blocks: list[dict], branch: str = "feature", **kw) -> dict:
    return {"type": "assistant", "gitBranch": branch, "sessionId": "s1",
            "timestamp": kw.pop("ts", "2026-01-01T00:00:00Z"),
            "message": {"content": blocks}, **kw}


ABS = f"{REPO}/src/api/score.ts"
T_MAIN = _transcript([
    {"type": "user", "origin": {"kind": "human"}, "message": {"content": "Weight the score"}},
    _assistant([{"type": "text", "text": "Scores must honor the weight setting."},
                {"type": "tool_use", "id": "e1", "name": "Edit", "input": {
                    "file_path": ABS, "old_string": "  return a;",
                    "new_string": "export function score(a: number, weight: number) {\n  return a * weight;"}}]),
])
T_SHELL = _transcript([
    _assistant([{"type": "tool_use", "id": "b1", "name": "Bash", "input": {
        "command": "cat > src/ui/ScoreCard.tsx <<'EOF'\nexport const ScoreCard = () => score(1, 2);\nEOF"}}]),
])


def test_scan_captures_reasoning_and_prompt():
    edits = scan([T_MAIN], {"src/api/score.ts"})
    assert len(edits) == 1
    assert "honor the weight" in edits[0].reasoning
    assert edits[0].user_prompt == "Weight the score"


def test_attribute_matches_edit_and_shell_and_reports_gaps():
    edits = scan([T_MAIN, T_SHELL], {f.path for f in DIFF.files})
    prov = {p.path: p for p in attribute(DIFF, edits, "feature")}
    assert prov["src/api/score.ts"].status == "attributed"
    assert prov["src/api/score.ts"].edits[0] == "e1"
    assert prov["src/api/score.ts"].branch_match
    assert prov["src/ui/ScoreCard.tsx"].status in ("attributed", "partial")
    assert prov["src/ui/__tests__/ScoreCard.test.tsx"].status == "none"


def test_attribute_prefers_edits_on_the_ticket_branch():
    other = _transcript([_assistant([{"type": "tool_use", "id": "x1", "name": "Edit", "input": {
        "file_path": ABS, "old_string": "", "new_string": "  return a * weight;"}}],
        branch="other", ts="2026-02-01T00:00:00Z")])
    edits = scan([T_MAIN, other], {"src/api/score.ts"})
    prov = {p.path: p for p in attribute(DIFF, edits, "feature")}
    assert prov["src/api/score.ts"].edits[0] == "e1"


# ─── group ordering ────────────────────────────────────────────────────

def _st(kinds: dict[str, str], edges: list[tuple[str, str]]):
    from review.structure import FileInfo, Structure
    return Structure(files={p: FileInfo(p, k, False) for p, k in kinds.items()}, edges=edges)


def _order(groups, st):
    from review.analysis import order_groups
    return [g["id"] for g in order_groups([dict(g) for g in groups], st)]


def test_order_puts_providers_before_consumers_despite_difficulty():
    st = _st({"svc.ts": "api", "res.ts": "api"}, [("res.ts", "svc.ts")])
    groups = [{"id": "res", "difficulty": 1, "files": ["res.ts"]},
              {"id": "svc", "difficulty": 5, "files": ["svc.ts"]}]
    assert _order(groups, st) == ["svc", "res"]


def test_order_ui_on_generated_types_builds_on_api_groups():
    st = _st({"page.tsx": "ui", "graphql.ts": "generated", "res.ts": "api"},
             [("page.tsx", "graphql.ts")])
    groups = [{"id": "ui", "difficulty": 1, "files": ["page.tsx"]},
              {"id": "api", "difficulty": 4, "files": ["res.ts"]}]
    assert _order(groups, st) == ["api", "ui"]


def test_order_api_client_files_build_on_api_groups():
    st = _st({"queries.ts": "source", "res.ts": "api"}, [])
    st.files["queries.ts"].api_client = True
    groups = [{"id": "q", "difficulty": 1, "files": ["queries.ts"]},
              {"id": "api", "difficulty": 5, "files": ["res.ts"]}]
    assert _order(groups, st) == ["api", "q"]
    # A client living under an /api/ path is still a client, not a provider.
    st.files["queries.ts"].kind = "api"
    assert _order(groups, st) == ["api", "q"]


def test_order_test_only_groups_last_even_through_a_cycle():
    st = _st({"a.tsx": "ui", "b.tsx": "ui", "e2e.spec.ts": "test"},
             [("a.tsx", "b.tsx"), ("b.tsx", "a.tsx")])
    groups = [{"id": "e2e", "difficulty": 1, "files": ["e2e.spec.ts"]},
              {"id": "a", "difficulty": 3, "files": ["a.tsx"]},
              {"id": "b", "difficulty": 2, "files": ["b.tsx"]}]
    assert _order(groups, st) == ["b", "a", "e2e"]


def test_order_ignores_test_imports_and_records_builds_on():
    from review.analysis import order_groups
    st = _st({"x.ts": "api", "x.spec.ts": "test", "y.ts": "api"},
             [("x.spec.ts", "y.ts"), ("y.ts", "x.ts")])
    groups = [{"id": "gx", "difficulty": 3, "files": ["x.ts", "x.spec.ts"]},
              {"id": "gy", "difficulty": 1, "files": ["y.ts"]}]
    out = order_groups(groups, st)
    assert [g["id"] for g in out] == ["gx", "gy"]
    assert out[1]["builds_on"] == ["gx"] and out[0]["builds_on"] == []


# ─── analysis output parsing ───────────────────────────────────────────

def test_extract_json_tolerates_br_line_breaks():
    from review.analysis import _extract_json
    body = '<json><br/>{"files": [<br/>{"t": "<td>x<br/>y</td>"}]}</json>'
    assert _extract_json(body) == {"files": [{"t": "<td>x\ny</td>"}]}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


if __name__ == "__main__":
    from testutil import run_module_tests
    run_module_tests(globals())
