"""M0: render one ticket's analysis as a static, self-contained wizard mock.

Usage (from server/):
    python3 -m review.m0_mock <out_dir> <ticket_id>

Reads <ticket_id>.json (m0_bench) and <ticket_id>.analysis.json (m0_analyze)
from out_dir and writes <ticket_id>.mock.html beside them. The page embeds the
ticket's code, so it stays wherever out_dir is (keep that out of git).
Verdicts persist in the browser's localStorage only — this is a UX mock, not
the M3 wizard.
"""

from __future__ import annotations

import json
import os
import sys

NARRATION_CHARS = 500


def build_data(bench: dict, analysis: dict, ticket_id: str) -> dict:
    diff = bench["diff"]
    structure = bench["structure"]["files"]
    edits = bench["edits"]
    files = {}
    hunks = {}
    for f in diff["files"]:
        files[f["path"]] = {
            "status": f["status"], "add": f["additions"], "del": f["deletions"],
            "kind": structure[f["path"]]["kind"],
        }
        for h in f["hunks"]:
            hunks[h["hash"]] = {
                "path": f["path"], "header": h["header"],
                "old": h["old_start"], "new": h["new_start"], "lines": h["lines"],
            }
    prov = {}
    for p in bench["provenance"]:
        e = edits.get(p["edits"][0]) if p["edits"] else None
        prov[p["hunk_hash"]] = {
            "status": p["status"],
            "tool": e["tool"] if e else None,
            "narration": (e["reasoning"][-NARRATION_CHARS:] if e and e["reasoning"] else ""),
            "subagent": bool(e and e["sidechain"]),
        }
    skim = [{"path": p, "reason": structure[p]["skim_reason"],
             "add": files[p]["add"], "del": files[p]["del"]} for p in analysis["skim"]]
    return {
        "id": ticket_id,
        "diff_hash": diff["diff_hash"],
        "totals": {"files": len(diff["files"]),
                   "hunks": sum(len(f["hunks"]) for f in diff["files"]),
                   "add": sum(f["additions"] for f in diff["files"]),
                   "del": sum(f["deletions"] for f in diff["files"])},
        "groups": [{"id": g["id"], "title": g["title"], "difficulty": g.get("difficulty", 3),
                    "summary": g.get("summary", ""), "files": g["files"],
                    "builds_on": g.get("builds_on", [])} for g in analysis["groups"]],
        "skim": skim,
        "files": files,
        "hunks": hunks,
        "steps": analysis["files"],
        "prov": prov,
        "edges": bench["structure"]["edges"],
        "cost": analysis["summary"].get("cost_usd"),
    }


def render(data: dict) -> str:
    payload = json.dumps(data).replace("</", "<\\/")
    return PAGE.replace("__DATA__", payload).replace("__TITLE__", f"Review · {data['id']}")


def main(out_dir: str, ticket_id: str) -> str:
    bench = json.load(open(os.path.join(out_dir, f"{ticket_id}.json")))
    analysis = json.load(open(os.path.join(out_dir, f"{ticket_id}.analysis.json")))
    path = os.path.join(out_dir, f"{ticket_id}.mock.html")
    with open(path, "w") as fh:
        fh.write(render(build_data(bench, analysis, ticket_id)))
    print(path)
    return path


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--mono:ui-monospace,SFMono-Regular,Menlo,monospace;--bg:#fafafa;--fg:#111827;--muted:#6b7280;
--border:#e5e7eb;--card:#fff;--code-bg:#f3f4f6;--accent:#4f46e5;--accent-fg:#1e3a8a;--accent-soft-bg:#e0e7ff;
--accent-soft-br:#c7d2fe;--ok:#10b981;--ok-fg:#065f46;--ok-soft-bg:#d1fae5;--ok-soft-br:#a7f3d0;--bad:#ef4444;
--bad-fg:#7f1d1d;--bad-soft-bg:#fee2e2;--bad-soft-br:#fecaca;--warn:#f59e0b;--warn-fg:#78350f;--warn-soft-bg:#fef3c7;
--warn-soft-br:#fde68a;--add-bg:#e6ffed;--del-bg:#ffeef0;--on-accent:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#0b1220;--fg:#e5e7eb;--muted:#94a3b8;--border:#1f2937;--card:#111827;
--code-bg:#0f172a;--accent:#818cf8;--accent-fg:#c7d2fe;--accent-soft-bg:#1e1b4b;--accent-soft-br:#312e81;--ok:#34d399;
--ok-fg:#6ee7b7;--ok-soft-bg:#052e2b;--ok-soft-br:#065f46;--bad:#f87171;--bad-fg:#fca5a5;--bad-soft-bg:#3f1212;
--bad-soft-br:#7f1d1d;--warn:#fbbf24;--warn-fg:#fcd34d;--warn-soft-bg:#3b2a05;--warn-soft-br:#78350f;
--add-bg:#0d2a1a;--del-bg:#2d1215;--on-accent:#0b1220}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:var(--accent);text-decoration:none}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px 80px}
.top{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.crumbs{color:var(--muted);font-size:13px;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.crumbs a{color:var(--muted)} .crumbs a:hover{color:var(--accent)}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:0 0 6px} h3{font-size:14px;margin:0 0 6px}
.muted{color:var(--muted)} .small{font-size:12px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.bar{height:6px;background:var(--border);border-radius:4px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--ok);transition:width .3s}
.chip{display:inline-flex;align-items:center;gap:4px;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;
border:1px solid var(--border);color:var(--muted);white-space:nowrap}
.chip.ok{background:var(--ok-soft-bg);border-color:var(--ok-soft-br);color:var(--ok-fg)}
.chip.bad{background:var(--bad-soft-bg);border-color:var(--bad-soft-br);color:var(--bad-fg)}
.chip.warn{background:var(--warn-soft-bg);border-color:var(--warn-soft-br);color:var(--warn-fg)}
.chip.acc{background:var(--accent-soft-bg);border-color:var(--accent-soft-br);color:var(--accent-fg)}
.fsq{display:inline-flex;flex-wrap:wrap;gap:3px;max-width:150px}
.fsq i{width:10px;height:10px;border-radius:2px;background:var(--muted);opacity:.35}
.fsq i.ok{background:var(--ok);opacity:1}
.hero{display:grid;grid-template-columns:1fr auto;gap:16px;align-items:end;margin-bottom:18px}
.stats{display:flex;gap:18px;flex-wrap:wrap}.stats b{display:block;font-size:20px}
.groups{display:flex;flex-direction:column;gap:10px}
.g{display:grid;grid-template-columns:34px 1fr 150px;gap:14px;align-items:center;cursor:pointer}
.g:hover{border-color:var(--accent-soft-br)}
.num{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-weight:700;background:var(--accent-soft-bg);color:var(--accent-fg)}
.num.done{background:var(--ok-soft-bg);color:var(--ok-fg)}
.g p{margin:2px 0 0;color:var(--muted);font-size:13px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.side{display:flex;flex-direction:column;gap:6px;align-items:stretch}
.btn{font:inherit;font-weight:600;font-size:13px;padding:7px 14px;border-radius:8px;border:1px solid var(--border);
background:var(--card);color:var(--fg);cursor:pointer}
.btn:hover{border-color:var(--accent)} .btn.primary{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}
.btn.okb{background:var(--ok);border-color:var(--ok);color:var(--on-accent)}
.btn.badb{border-color:var(--bad-soft-br);color:var(--bad-fg);background:var(--bad-soft-bg)}
kbd{font:11px var(--mono);border:1px solid var(--border);border-bottom-width:2px;border-radius:4px;padding:0 4px;color:var(--muted)}
.btn kbd{color:inherit;border-color:currentColor;opacity:.7;margin-left:4px}
.cols{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:16px;align-items:start}
.graph{min-height:120px;overflow:auto;text-align:center}.graph svg{max-width:100%;height:auto;max-height:460px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}
.fc{cursor:pointer;display:flex;flex-direction:column;gap:6px;padding:12px}
.fc:hover{border-color:var(--accent-soft-br)} .fc.done{border-color:var(--ok-soft-br)} .fc.changes{border-color:var(--bad-soft-br)}
.fc .nm{font-weight:600;word-break:break-all;font-size:13px} .fc .dir{font:11px var(--mono);color:var(--muted);word-break:break-all}
.fc .row{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.add{color:var(--ok-fg)} .del{color:var(--bad-fg)}
.dots{display:flex;gap:4px;flex-wrap:wrap} .dots a{width:22px;height:6px;border-radius:3px;background:var(--border)}
.dots a.on{background:var(--accent)} .dots a.approved{background:var(--ok)} .dots a.changes{background:var(--bad)}
.code{background:var(--code-bg);border:1px solid var(--border);border-radius:10px;overflow:auto;max-height:72vh;font:12px/1.55 var(--mono)}
.hh{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--border);padding:4px 10px;color:var(--muted);font-size:11px}
.ln{display:grid;grid-template-columns:44px 44px 1fr;white-space:pre}
.ln span{padding:0 8px} .ln .n{color:var(--muted);text-align:right;user-select:none;opacity:.7}
.ln.a{background:var(--add-bg)} .ln.d{background:var(--del-bg)}
.panel{display:flex;flex-direction:column;gap:12px}
.lbl{font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin-bottom:4px;display:flex;gap:6px;align-items:center}
.risk{background:var(--warn-soft-bg);border:1px solid var(--warn-soft-br);color:var(--warn-fg);border-radius:8px;padding:8px 10px}
.vis{overflow:auto} .vis table{border-collapse:collapse;font-size:12px;width:100%} .vis td,.vis th{border:1px solid var(--border);padding:4px 6px;text-align:left}
details{border:1px dashed var(--border);border-radius:8px;padding:6px 10px} summary{cursor:pointer;color:var(--muted);font-size:12px}
details pre{white-space:pre-wrap;font:12px/1.5 var(--mono);margin:6px 0 0;color:var(--muted)}
textarea{width:100%;min-height:80px;font:inherit;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--fg)}
.actions{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.verdict{border-radius:8px;padding:8px 10px;font-size:13px}
.verdict.approved{background:var(--ok-soft-bg);color:var(--ok-fg)} .verdict.changes{background:var(--bad-soft-bg);color:var(--bad-fg)}
.skim li{margin:2px 0} .feedback{white-space:pre-wrap;font:12px/1.5 var(--mono);background:var(--code-bg);padding:12px;border-radius:8px;max-height:50vh;overflow:auto}
.foot{margin-top:28px;color:var(--muted);font-size:12px}
@media (max-width:820px){.cols{grid-template-columns:1fr}.hero{grid-template-columns:1fr}.g{grid-template-columns:30px 1fr}.g .side{grid-column:2}}
</style></head>
<body><div class="wrap" id="app"></div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>
const D = __DATA__;
const KEY = "review-mock:" + D.diff_hash;
let S; try { S = JSON.parse(localStorage.getItem(KEY)) || {} } catch (e) { S = {} }
S.v = S.v || {}; S.skimmed = S.skimmed || false;
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(S)) } catch (e) {} };
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const md = s => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>");
const base = p => p.split("/").pop(), dir = p => p.split("/").slice(0, -1).join("/");
const stepKey = (p, i) => p + "#" + i;
const steps = p => D.steps[p] || [];
const fileState = p => { const ss = steps(p); if (!ss.length) return "nosteps";
  const vs = ss.map((_, i) => (S.v[stepKey(p, i)] || {}).v);
  if (vs.some(v => v === "changes")) return vs.every(Boolean) ? "changes" : "progress";
  return vs.every(v => v === "approved") ? "done" : vs.some(Boolean) ? "progress" : "todo"; };
const fileReviewed = p => ["done", "changes"].includes(fileState(p));
const groupDone = g => g.files.every(fileReviewed);
const allSteps = () => D.groups.flatMap(g => g.files.flatMap(p => steps(p).map((_, i) => stepKey(p, i))));
const gnum = id => D.groups.findIndex(g => g.id === id) + 1;
const buildsOn = g => (g.builds_on || []).length
  ? `Builds on ${g.builds_on.map(id => `<a href="#/g/${id}" onclick="event.stopPropagation()">${gnum(id)}</a>`).join(", ")}` : "Stands alone";
// One square per file in the group: gray until all its steps are approved, then green.
const fileSquares = g => `<span class="fsq">${g.files.map(p => {
  const ok = fileState(p) === "done";
  return `<i class="${ok ? "ok" : ""}" title="${esc(base(p))}${ok ? " · approved" : ""}"></i>`; }).join("")}</span>`;
const kindChip = k => `<span class="chip ${k === "test" || k === "test-support" ? "ok" : k === "api" ? "acc" : ""}">${esc(k)}</span>`;
if (window.mermaid) mermaid.initialize({ startOnLoad: false, securityLevel: "strict",
  theme: matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "default", flowchart: { useMaxWidth: true } });
let mid = 0;
async function drawMermaid(el, code) {
  if (!window.mermaid) { el.innerHTML = `<pre class="feedback">${esc(code)}</pre>`; return; }
  try { const { svg } = await mermaid.render("m" + (mid++), code); el.innerHTML = svg; }
  catch (e) { el.innerHTML = `<pre class="feedback">${esc(code)}</pre>`; }
}
function cleanHtml(html) {
  const doc = new DOMParser().parseFromString(html, "text/html");
  doc.querySelectorAll("script,style,iframe,link").forEach(n => n.remove());
  doc.querySelectorAll("*").forEach(n => [...n.attributes].forEach(a => { if (/^on|style/i.test(a.name)) n.removeAttribute(a.name) }));
  return doc.body.innerHTML;
}

function overview() {
  const keys = allSteps(), done = keys.filter(k => S.v[k]).length;
  const changes = keys.filter(k => (S.v[k] || {}).v === "changes").length;
  const skimAdd = D.skim.reduce((a, f) => a + f.add, 0);
  return `
  <div class="hero"><div>
    <div class="muted small">Code review · ${esc(D.id)}</div>
    <h1>${D.totals.files} files, walked in ${D.groups.length} groups</h1>
    <div class="muted small" style="margin-bottom:8px">Ordered so each group comes after the groups it builds on.</div>
    <div class="stats muted small">
      <span><b class="add">+${D.totals.add.toLocaleString()}</b>added</span>
      <span><b class="del">−${D.totals.del.toLocaleString()}</b>removed</span>
      <span><b>${keys.length}</b>steps to review</span>
      <span><b>${D.skim.length}</b>files to skim</span></div></div>
    <div style="min-width:220px"><div class="small muted" style="margin-bottom:4px">${done} of ${keys.length} steps reviewed${changes ? ` · <span class="del">${changes} change requests</span>` : ""}</div>
      <div class="bar"><i style="width:${keys.length ? 100 * done / keys.length : 0}%"></i></div>
      <div style="margin-top:10px;text-align:right"><a class="btn ${done === keys.length ? "primary" : ""}" href="#/summary">Finish review →</a></div></div></div>
  <div class="groups">${D.groups.map((g, i) => {
    const n = g.files.reduce((a, p) => a + steps(p).length, 0);
    const d = g.files.reduce((a, p) => a + steps(p).filter((_, j) => S.v[stepKey(p, j)]).length, 0);
    return `<div class="card g" onclick="location.hash='#/g/${g.id}'">
      <div class="num ${groupDone(g) ? "done" : ""}">${groupDone(g) ? "✓" : i + 1}</div>
      <div><h2>${esc(g.title)}</h2><p>${md(g.summary)}</p><div class="small muted" style="margin-top:4px">${buildsOn(g)}</div></div>
      <div class="side"><div style="display:flex;justify-content:space-between;align-items:center">${fileSquares(g)}<span class="small muted">${g.files.length} files</span></div>
        <div class="bar"><i style="width:${n ? 100 * d / n : 0}%"></i></div><span class="small muted">${d}/${n} steps</span></div></div>`; }).join("")}
    <div class="card"><div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap">
      <div><h2>Skim: generated &amp; bulky files</h2><div class="muted small">${D.skim.length} files, ${skimAdd.toLocaleString()} added lines. No line-by-line review; check they belong.</div></div>
      <button class="btn ${S.skimmed ? "okb" : ""}" onclick="S.skimmed=!S.skimmed;save();route()">${S.skimmed ? "✓ Skimmed" : "Mark skimmed"}</button></div>
      <ul class="skim small">${D.skim.map(f => `<li><code>${esc(f.path)}</code> <span class="muted">· ${esc(f.reason)} · <span class="add">+${f.add}</span> <span class="del">−${f.del}</span></span></li>`).join("")}</ul></div>
  </div>
  <div class="foot">Analysis cost $${D.cost} · diff ${esc(D.diff_hash)} · M0 mock: verdicts live in this browser only.</div>`;
}

function groupGraph(g) {
  const inG = new Set(g.files), ids = {}, lines = ["flowchart TD"];
  g.files.forEach((p, i) => { ids[p] = "n" + i; lines.push(`  n${i}["${base(p).replace(/"/g, "#quot;")}"]`); });
  const ext = {};
  D.edges.forEach(([a, b]) => {
    if (inG.has(a) && inG.has(b)) lines.push(`  ${ids[a]} --> ${ids[b]}`);
    else if (inG.has(a)) { const og = D.groups.find(x => x.files.includes(b)); if (og) {
      const k = "x" + og.id.replace(/\W/g, ""); if (!ext[k]) { ext[k] = og; lines.push(`  ${k}(["↗ ${og.title.replace(/"/g, "#quot;").slice(0, 40)}"])`); }
      const e = `  ${ids[a]} -.-> ${k}`; if (!lines.includes(e)) lines.push(e); } }
  });
  return lines.join("\n");
}

function group(gid) {
  const gi = D.groups.findIndex(g => g.id === gid), g = D.groups[gi];
  if (!g) return overview();
  const next = D.groups[gi + 1];
  setTimeout(() => drawMermaid(document.getElementById("graph"), groupGraph(g)));
  return `
  <div class="top"><div class="crumbs"><a href="#/">Review</a> / Group ${gi + 1} of ${D.groups.length}</div>
    ${next ? `<a class="btn ${groupDone(g) ? "primary" : ""}" href="#/g/${next.id}">Next group →</a>` : `<a class="btn primary" href="#/summary">Finish →</a>`}</div>
  <h1 style="margin-bottom:6px">${esc(g.title)}</h1>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">${fileSquares(g)}<span class="small muted">${g.files.filter(p => fileState(p) === "done").length} of ${g.files.length} files approved</span></div>
  <div class="card" style="margin-bottom:12px"><div class="lbl">What changes and why</div><div>${md(g.summary)}</div>
    <div class="small muted" style="margin-top:6px">${(g.builds_on || []).length ? "Builds on: " + g.builds_on.map(id => `<a href="#/g/${id}">${gnum(id)}. ${esc(D.groups[gnum(id) - 1].title)}</a>`).join(" · ") : "Builds on no other group in this change."}</div>
    <div style="margin-top:10px" class="small muted">${g.files.filter(fileReviewed).length} of ${g.files.length} files reviewed. Open each file to finish this group.</div></div>
  <div class="card" style="margin-bottom:16px"><div class="lbl">How these files relate</div><div class="graph" id="graph">Loading graph…</div>
    <div class="small muted">Solid: imports within this group. Dashed: imports into another group.</div></div>
  <div class="grid">${g.files.map(p => { const f = D.files[p], st = fileState(p), n = steps(p).length;
    const label = { done: '<span class="chip ok">✓ approved</span>', changes: '<span class="chip bad">changes requested</span>',
      progress: '<span class="chip warn">in progress</span>', todo: "", nosteps: '<span class="chip bad">analysis missing</span>' }[st];
    return `<div class="card fc ${st === "done" ? "done" : st === "changes" ? "changes" : ""}" onclick="location.hash='#/f/${gid}/${encodeURIComponent(p)}/0'">
      <div class="nm">${esc(base(p))}</div><div class="dir">${esc(dir(p))}</div>
      <div class="row">${kindChip(f.kind)}<span class="small"><span class="add">+${f.add}</span> <span class="del">−${f.del}</span></span>
        <span class="small muted">· ${n} step${n === 1 ? "" : "s"}</span>${label}</div></div>`; }).join("")}</div>`;
}

function hunkHtml(h) {
  let o = h.old, n = h.new;
  const rows = h.lines.map(l => { const t = l[0], body = esc(l.slice(1));
    if (t === "+") return `<div class="ln a"><span class="n"></span><span class="n">${n++}</span><span>+${body}</span></div>`;
    if (t === "-") return `<div class="ln d"><span class="n">${o++}</span><span class="n"></span><span>-${body}</span></div>`;
    return `<div class="ln"><span class="n">${o++}</span><span class="n">${n++}</span><span> ${body}</span></div>`; }).join("");
  return `<div class="hh">@@ −${h.old} +${h.new} @@ ${esc(h.header)}</div>${rows}`;
}

function provBadge(st) {
  const ps = st.hunks.map(h => D.prov[h]).filter(Boolean), att = ps.filter(p => p.status === "attributed");
  if (ps.length && att.length === ps.length) { const t = att[0].tool;
    return `<span class="chip ok" title="Every hunk in this step matches a transcript action">traced to ${t === "Bash" ? "a shell edit" : "an agent edit"}${att[0].subagent ? " (subagent)" : ""}</span>`; }
  if (att.length) return `<span class="chip warn">partly traced</span>`;
  return `<span class="chip bad" title="No transcript action wrote this code">not traced</span>`;
}

function file(gid, path, idx) {
  const g = D.groups.find(x => x.id === gid), ss = steps(path), f = D.files[path];
  if (!g || !f) return overview();
  idx = Math.max(0, Math.min(idx, ss.length - 1));
  const st = ss[idx];
  if (!st) return `<div class="top"><div class="crumbs"><a href="#/">Review</a> / <a href="#/g/${gid}">${esc(g.title)}</a> / ${esc(base(path))}</div></div><div class="card">No steps for this file.</div>`;
  const k = stepKey(path, idx), v = S.v[k] || {};
  const narr = st.hunks.map(h => (D.prov[h] || {}).narration).filter(Boolean);
  const vis = st.visual || {};
  if (vis.kind === "mermaid" && vis.body) setTimeout(() => drawMermaid(document.getElementById("vis"), vis.body));
  window.__step = { gid, path, idx, n: ss.length };
  return `
  <div class="top"><div class="crumbs"><a href="#/">Review</a> / <a href="#/g/${gid}">${esc(g.title)}</a> / <b>${esc(base(path))}</b></div>
    <span class="small muted">Step ${idx + 1} of ${ss.length}</span></div>
  <div class="dots" style="margin-bottom:12px">${ss.map((_, i) => `<a href="#/f/${gid}/${encodeURIComponent(path)}/${i}" class="${i === idx ? "on" : ""} ${(S.v[stepKey(path, i)] || {}).v || ""}"></a>`).join("")}</div>
  <div class="cols">
    <div class="code">${st.hunks.map(h => D.hunks[h] ? hunkHtml(D.hunks[h]) : "").join("")}</div>
    <div class="panel">
      <div class="card"><h2>${esc(st.title)}</h2><div class="small muted" style="word-break:break-all">${esc(path)}</div></div>
      <div class="card"><div class="lbl">What it does</div>${md(st.what)}</div>
      <div class="card"><div class="lbl">Why the agent did it <span class="chip ${st.why_source === "transcript" ? "acc" : ""}">${st.why_source === "transcript" ? "from transcript" : "inferred"}</span> ${provBadge(st)}</div>${md(st.why)}
        ${narr.length ? `<details style="margin-top:8px"><summary>Agent's words at the time</summary><pre>${esc(narr[0])}</pre></details>` : ""}</div>
      ${st.risk && !/^none\.?$/i.test(st.risk) ? `<div class="risk"><b>Check:</b> ${md(st.risk)}</div>` : ""}
      ${vis.kind === "mermaid" && vis.body ? `<div class="card"><div class="lbl">Visual</div><div class="vis" id="vis">Loading…</div></div>` : ""}
      ${vis.kind === "table" && vis.body ? `<div class="card"><div class="lbl">Visual</div><div class="vis">${cleanHtml(vis.body)}</div></div>` : ""}
      <div class="card">
        ${v.v ? `<div class="verdict ${v.v}" style="margin-bottom:10px">${v.v === "approved" ? "✓ Approved" : "Changes requested: " + esc(v.c)}</div>` : ""}
        <div id="cbox" style="display:none;margin-bottom:8px"><textarea id="ctext" placeholder="What should the agent change?">${esc(v.v === "changes" ? v.c : "")}</textarea>
          <div class="actions" style="margin-top:6px"><button class="btn badb" onclick="verdict('changes')">Save change request</button><button class="btn" onclick="document.getElementById('cbox').style.display='none'">Cancel</button></div></div>
        <div class="actions"><button class="btn okb" onclick="verdict('approved')">Approve <kbd>a</kbd></button>
          <button class="btn badb" onclick="openChanges()">Request changes <kbd>c</kbd></button>
          <span style="flex:1"></span>
          ${idx > 0 ? `<a class="btn" href="#/f/${gid}/${encodeURIComponent(path)}/${idx - 1}">← <kbd>k</kbd></a>` : ""}
          ${idx < ss.length - 1 ? `<a class="btn" href="#/f/${gid}/${encodeURIComponent(path)}/${idx + 1}"><kbd>j</kbd> →</a>` : `<a class="btn" href="#/g/${gid}">Back to group</a>`}</div></div>
    </div></div>`;
}

function openChanges() { const b = document.getElementById("cbox"); b.style.display = "block"; document.getElementById("ctext").focus(); }
function verdict(kind) {
  const s = window.__step; if (!s) return;
  const k = stepKey(s.path, s.idx);
  if (kind === "changes") { const c = document.getElementById("ctext").value.trim(); if (!c) return; S.v[k] = { v: "changes", c }; }
  else S.v[k] = { v: "approved" };
  save();
  location.hash = s.idx < s.n - 1 ? `#/f/${s.gid}/${encodeURIComponent(s.path)}/${s.idx + 1}` : `#/g/${s.gid}`;
}

function feedbackPrompt() {
  const items = [];
  D.groups.forEach(g => g.files.forEach(p => steps(p).forEach((st, i) => { const v = S.v[stepKey(p, i)];
    if (v && v.v === "changes") { const h = D.hunks[st.hunks[0]];
      items.push(`- ${p}${h ? ":" + h.new : ""} — "${st.title}"\n  ${v.c}`); } })));
  if (!items.length) return "";
  return `Code review feedback on ${D.id}. Please address each item, then summarise what you changed:\n\n${items.join("\n\n")}`;
}

function summary() {
  const keys = allSteps(), appr = keys.filter(k => (S.v[k] || {}).v === "approved").length;
  const ch = keys.filter(k => (S.v[k] || {}).v === "changes").length, left = keys.length - appr - ch;
  const fb = feedbackPrompt();
  return `<div class="top"><div class="crumbs"><a href="#/">Review</a> / Summary</div></div>
  <h1>Review summary</h1>
  <div class="stats" style="margin:10px 0 16px"><span><b class="add">${appr}</b>approved</span><span><b class="del">${ch}</b>changes requested</span>
    <span><b>${left}</b>not reviewed</span><span><b>${S.skimmed ? "✓" : "–"}</b>skim group</span></div>
  ${fb ? `<div class="card"><div class="lbl">Feedback for the agent</div><div class="feedback" id="fb">${esc(fb)}</div>
    <div class="actions" style="margin-top:10px"><button class="btn primary" onclick="navigator.clipboard.writeText(document.getElementById('fb').textContent).then(()=>this.textContent='✓ Copied')">Copy to clipboard</button>
    <span class="small muted">Paste into the chat. (Target: queued and injected on the next turn.)</span></div></div>`
    : `<div class="card">${left ? "No change requests yet." : "Everything approved. Nothing to send back."}</div>`}
  ${left ? `<div class="small muted" style="margin-top:10px">${left} steps still unreviewed. <a href="#/">Back to the groups</a></div>` : ""}`;
}

function route() {
  const h = location.hash.replace(/^#\/?/, "").split("/");
  const app = document.getElementById("app");
  if (h[0] === "g") app.innerHTML = group(h[1]);
  else if (h[0] === "f") app.innerHTML = file(h[1], decodeURIComponent(h[2]), +h[3] || 0);
  else if (h[0] === "summary") app.innerHTML = summary();
  else app.innerHTML = overview();
  if (h[0] !== "f") window.__step = null;
  window.scrollTo(0, 0);
}
addEventListener("hashchange", route);
addEventListener("keydown", e => {
  if (!window.__step || /TEXTAREA|INPUT/.test(e.target.tagName) || e.metaKey || e.ctrlKey) return;
  const s = window.__step;
  if (e.key === "a") verdict("approved");
  else if (e.key === "c") { e.preventDefault(); openChanges(); }
  else if (e.key === "j" && s.idx < s.n - 1) location.hash = `#/f/${s.gid}/${encodeURIComponent(s.path)}/${s.idx + 1}`;
  else if (e.key === "k" && s.idx > 0) location.hash = `#/f/${s.gid}/${encodeURIComponent(s.path)}/${s.idx - 1}`;
});
route();
</script></body></html>
"""


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
