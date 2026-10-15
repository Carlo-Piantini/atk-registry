"""M0: compare the wizard's steps against real PR review findings.

Usage (from server/, with the plugin venv):
    ../.venv/bin/python -m review.m0_compare <corpus.json> <out_dir> <ticket_id> [model]

Needs <ticket_id>.json (m0_bench), <ticket_id>.analysis.json (m0_analyze) and
<ticket_id>.reviews.json: a list of {"who", "path", "line", "body", "commit"}
top-level review comments. For each finding with a path, a judge call reads
the wizard's steps for that file and classifies the finding:

  caught  - a step's risk note (or what/why) raises the same concern
  pointed - the wizard flags that code as needing care, but not this issue
  missed  - nothing in the steps would lead a reviewer to it

Each finding also records whether the file changed between the commit the
reviewer saw and the analyzed head: a fixed bug cannot be caught by reading
the fixed code, so those are reported separately.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter

from .analysis import _call

JUDGE_SYSTEM = """You evaluate a code-review assistant against real review \
findings. For each finding you get the reviewer's comment and the assistant's \
walkthrough steps for the same file (title, what, why, risk). Decide whether a \
reviewer following the walkthrough would have been led to the finding:

- "caught": a step explicitly raises the same concern (same bug, same risk).
- "pointed": a step tells the reviewer to scrutinize that exact logic or \
area, but does not name this concern.
- "missed": nothing in the steps leads to it.

Be strict: generic advice ("review carefully") is "missed"; "pointed" needs \
the same function or behavior. Reply with only JSON in <json></json> tags:
{"verdicts": [{"n": 1, "verdict": "caught", "evidence": "<= 25 words quoting \
or paraphrasing the step"}]}"""


def _changed_since(repo: str, commit: str | None, head: str, path: str) -> bool | None:
    if not commit:
        return None
    r = subprocess.run(["git", "-C", repo, "diff", "--quiet", commit, head, "--", path],
                       capture_output=True)
    return None if r.returncode not in (0, 1) else r.returncode == 1


def main(corpus_path: str, out_dir: str, ticket_id: str, model: str = "sonnet") -> dict:
    corpus = json.load(open(corpus_path))
    bench = json.load(open(os.path.join(out_dir, f"{ticket_id}.json")))
    run = os.environ.get("CCD_REVIEW_RUN_NAME") or ticket_id  # an experiment's analysis
    analysis = json.load(open(os.path.join(out_dir, f"{run}.analysis.json")))
    reviews = json.load(open(os.path.join(out_dir, f"{ticket_id}.reviews.json")))
    head = bench["diff"]["head_sha"]
    findings = [r for r in reviews if r.get("path")]

    blocks = []
    for i, f in enumerate(findings, 1):
        f["changed_since"] = _changed_since(corpus["repo"], f.get("commit"), head, f["path"])
        steps = analysis["files"].get(f["path"])
        if steps is None:
            in_skim = f["path"] in analysis["skim"]
            f["verdict"] = "skim" if in_skim else "no-steps"
            continue
        lines = "\n".join(
            f"  - {s['title']} | what: {s.get('what')} | why: {s.get('why')} | risk: {s.get('risk')}"
            for s in steps)
        blocks.append(f"<finding n=\"{i}\" file=\"{f['path']}\" line=\"{f.get('line')}\">\n"
                      f"{f['body'][:1500]}\n</finding>\n<steps>\n{lines}\n</steps>")

    metrics = None
    if blocks:
        data, metrics = _call("judge", JUDGE_SYSTEM, "\n\n".join(blocks), model)
        for v in (data or {}).get("verdicts", []):
            n = v.get("n")
            if isinstance(n, int) and 1 <= n <= len(findings):
                findings[n - 1]["verdict"] = v.get("verdict")
                findings[n - 1]["evidence"] = v.get("evidence")

    def tally(fs):
        return dict(Counter(f.get("verdict", "unjudged") for f in fs))

    summary = {
        "id": ticket_id,
        "run": run,
        "findings": len(findings),
        "all": tally(findings),
        "code_unchanged_since_review": tally([f for f in findings if f["changed_since"] is False]),
        "code_changed_since_review": tally([f for f in findings if f["changed_since"]]),
        "by_source": {who: tally([f for f in findings if f["who"] == who])
                      for who in sorted({f["who"] for f in findings})},
        "judge_cost_usd": metrics.cost_usd if metrics else 0,
    }
    with open(os.path.join(out_dir, f"{run}.compare.json"), "w") as fh:
        json.dump({"summary": summary, "findings": findings}, fh, indent=1)
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    main(*sys.argv[1:])
