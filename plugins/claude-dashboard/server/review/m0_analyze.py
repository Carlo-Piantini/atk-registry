"""M0: run the analysis pass on one corpus ticket and record cost/latency.

Usage (from server/, with the plugin venv for pydantic):
    ../.venv/bin/python -m review.m0_analyze <corpus.json> <out_dir> <ticket_id> [model]

Reads the corpus the same way m0_bench does; writes <ticket_id>.analysis.json
(or <CCD_REVIEW_RUN_NAME>.analysis.json, to keep experiment runs apart)
(groups, steps, problems, per-call metrics) to out_dir.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict

from .analysis import EFFORT, analyze
from .diff import collect
from .provenance import attribute, scan, transcripts_from_glob
from .structure import analyze as structure


def main(corpus_path: str, out_dir: str, ticket_id: str, model: str = "sonnet") -> dict:
    corpus = json.load(open(corpus_path))
    t = next(x for x in corpus["tickets"] if x["id"] == ticket_id)
    d = collect(corpus["repo"], t["base"], t.get("head", t["branch"]))
    st = structure(d)
    edits = scan(transcripts_from_glob(corpus["transcripts_glob"]), {f.path for f in d.files})
    prov = attribute(d, edits, t["branch"])
    t0 = time.monotonic()
    a = analyze(d, st, prov, {e.uuid: e for e in edits}, model=model)
    wall = round(time.monotonic() - t0, 1)
    ok = [c for c in a.calls if c.ok]
    steps = [s for ss in a.files.values() for s in ss]
    summary = {
        "id": ticket_id, "model": model, "wall_seconds": wall,
        "calls": len(a.calls), "failed_calls": len(a.calls) - len(ok),
        "cost_usd": round(sum(c.cost_usd or 0 for c in ok), 3),
        "input_tokens": sum(c.input_tokens or 0 for c in ok),
        "output_tokens": sum(c.output_tokens or 0 for c in ok),
        "slowest_call_s": max((c.seconds for c in a.calls), default=0),
        "groups": len(a.groups), "skim_files": len(a.skim),
        "files_with_steps": len(a.files), "steps": len(steps),
        "why_transcript_share": round(sum(1 for s in steps if s.get("why_source") == "transcript")
                                      / max(1, len(steps)), 3),
        "visuals": sum(1 for s in steps if (s.get("visual") or {}).get("kind") not in (None, "none")),
        "problems": len(a.problems),
    }
    os.makedirs(out_dir, exist_ok=True)
    summary["effort"] = dict(EFFORT)
    name = os.environ.get("CCD_REVIEW_RUN_NAME") or ticket_id
    with open(os.path.join(out_dir, f"{name}.analysis.json"), "w") as fh:
        json.dump({"summary": summary, "groups": a.groups, "skim": a.skim, "files": a.files,
                   "problems": a.problems, "calls": [asdict(c) for c in a.calls]}, fh, indent=1)
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    main(*sys.argv[1:])
