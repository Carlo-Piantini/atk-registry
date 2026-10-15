"""M0 bench: run the deterministic review passes over a corpus of real changes
and report the numbers M0 needs (size, skim share, graph connectivity,
provenance coverage).

Usage (from server/):
    python3 -m review.m0_bench <corpus.json> <out_dir>

corpus.json:
    {"repo": "/abs/repo", "transcripts_glob": "~/.claude/projects/*name*/*.jsonl",
     "tickets": [{"id": "...", "branch": "...", "base": "origin/main", "head": "<optional sha>"}]}

Per-ticket JSON (diff, structure, provenance, edits) is written to out_dir;
keep that directory out of version control — it holds the corpus's code.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict

from .diff import collect
from .provenance import attribute, scan, transcripts_from_glob
from .structure import analyze


def run(corpus_path: str, out_dir: str) -> list[dict]:
    corpus = json.load(open(corpus_path))
    os.makedirs(out_dir, exist_ok=True)
    diffs = {}
    for t in corpus["tickets"]:
        diffs[t["id"]] = collect(corpus["repo"], t["base"], t.get("head", t["branch"]))

    t0 = time.time()
    wanted = {f.path for d in diffs.values() for f in d.files}
    transcripts = transcripts_from_glob(corpus["transcripts_glob"])
    edits = scan(transcripts, wanted)
    scan_s = time.time() - t0

    edit_tool = {e.uuid: e.tool for e in edits}
    rows = []
    for t in corpus["tickets"]:
        d = diffs[t["id"]]
        s = analyze(d)
        prov = attribute(d, edits, t["branch"])
        skim_paths = {p for p, fi in s.files.items() if fi.skim}
        review_prov = [p for p in prov if p.path not in skim_paths and p.status != "trivial"]
        status = Counter(p.status for p in review_prov)
        added = {(f.path, h.hash): len(h.added) for f in d.files for h in f.hunks}
        lines_tot = sum(added[(p.path, p.hunk_hash)] for p in review_prov) or 1
        lines_attr = sum(added[(p.path, p.hunk_hash)] for p in review_prov if p.status == "attributed")
        used = {u for p in prov for u in p.edits[:1]}
        unattributed_files = Counter(p.path for p in review_prov if p.status == "none")
        src_files = [f for f in d.files if f.path not in skim_paths]
        connected = {x for e in s.edges if e[0] not in skim_paths and e[1] not in skim_paths for x in e}
        row = {
            "id": t["id"],
            "files": len(d.files),
            "hunks": sum(len(f.hunks) for f in d.files),
            "added": sum(f.additions for f in d.files),
            "removed": sum(f.deletions for f in d.files),
            "kinds": dict(Counter(fi.kind for fi in s.files.values())),
            "skim_files": len(skim_paths),
            "skim_added_share": round(sum(f.additions for f in d.files if f.path in skim_paths)
                                      / max(1, sum(f.additions for f in d.files)), 3),
            "graph_edges_excl_skim": sum(1 for a, b in s.edges if a not in skim_paths and b not in skim_paths),
            "graph_connected_share": round(len(connected) / max(1, len(src_files)), 3),
            "review_hunks": len(review_prov),
            "prov_hunks": dict(status),
            "prov_hunk_share": round(status["attributed"] / max(1, len(review_prov)), 3),
            "prov_line_share": round(lines_attr / lines_tot, 3),
            "prov_branch_match": round(sum(1 for p in review_prov if p.status == "attributed" and p.branch_match)
                                       / max(1, status["attributed"]), 3),
            "edits_used": len(used),
            "with_reasoning": len({e.uuid for e in edits if e.uuid in used and e.reasoning}),
            "via_shell": sum(1 for p in review_prov if p.status == "attributed" and p.edits and edit_tool.get(p.edits[0]) == "Bash"),
            "top_unattributed_files": unattributed_files.most_common(6),
        }
        rows.append(row)
        with open(os.path.join(out_dir, f"{t['id']}.json"), "w") as fh:
            json.dump({
                "summary": row,
                "diff": d.to_dict(),
                "structure": {"files": {p: asdict(fi) for p, fi in s.files.items()}, "edges": s.edges},
                "provenance": [asdict(p) for p in prov],
                "edits": {e.uuid: asdict(e) for e in edits if e.uuid in {u for p in prov for u in p.edits}},
            }, fh, indent=1)
    meta = {"transcripts": len(transcripts), "edits_scanned": len(edits), "scan_seconds": round(scan_s, 1)}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"meta": meta, "tickets": rows}, fh, indent=1)
    print(json.dumps(meta))
    return rows


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    for r in run(sys.argv[1], sys.argv[2]):
        print(json.dumps(r))
