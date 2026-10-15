"""Per-session dashboard regeneration: job registry + `claude -p` runner.

A regen job runs `claude -p` over one session's recent transcript to emit a
validated op-set (a delta) against the server-owned dashboard model; the server
folds it in, renders the HTML, and atomically writes the result. The Registry
owns at most one in-flight job per session_uuid, coalescing a trigger during an
in-flight run into a single follow-up.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import notices
from config import DEFAULT_MODEL, DEFAULT_REGEN_TIMEOUT
from logging_config import get_logger

import agent_io
import fold as _fold
import lineage as _lineage
import render as _render
from models import DashboardModel
from prompt import RegenPrompt, assemble_prompt

_log = get_logger("regen")

# ─── Constants ─────────────────────────────────────────────────────────

# Total attempts for a transient subprocess failure before it surfaces.
MAX_ATTEMPTS = 2

# In-flight age past which a job is treated as wedged and superseded; younger
# triggers coalesce into one follow-up instead.
STALE_SUPERSEDE_AFTER = 360.0

# Env marker so the subagent's own Stop hook exits early (else it recurses).
SUBAGENT_ENV_MARKER = "CLAUDE_DASHBOARD_SUBAGENT"

# Ambient vars that would hijack the subagent's subscription auth: API-key
# billing (API_KEY/AUTH_TOKEN) or a redirected endpoint (BASE_URL). Stripped so
# `claude -p` uses the keychain subscription against the default endpoint.
AMBIENT_AUTH_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")

# Appended to the user message on the corrective retry. Must name the same
# block format prompt._output_format declares, or the retry re-fails.
RETRY_INSTRUCTION = (
    "\n\n# Your previous output was REJECTED. Fix it and re-emit the FULL "
    "corrected output: the <update> block with the op-set JSON, then one "
    '<freeform ref="…"> block per freeform.upsert. Error:\n'
)


def build_subagent_env() -> dict:
    """Environment for every `claude -p` the plugin spawns: marks the subagent
    and strips ambient auth vars. Shared by invoke_claude and probe_auth."""
    env = {**os.environ, SUBAGENT_ENV_MARKER: "1"}
    for var in AMBIENT_AUTH_VARS:
        env.pop(var, None)
    return env


# ─── JSONL curation (pure functions) ───────────────────────────────────

def read_jsonl(path: Path) -> list:
    """Stream-parse a session JSONL into a list of dicts."""
    out = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def chat_title(events: list) -> str:
    """The chat's name as Claude Code shows it: its latest custom title, else its AI title."""
    custom = ai = ""
    for e in events:
        if e.get("type") == "custom-title":
            custom = str(e.get("customTitle") or "").strip()
        elif e.get("type") == "ai-title":
            ai = str(e.get("aiTitle") or "").strip()
    return custom or ai


def _is_pure_tool_result(user_event: dict) -> bool:
    content = user_event.get("message", {}).get("content")
    if not isinstance(content, list):
        return False
    for item in content:
        if not isinstance(item, dict):
            return False
        if item.get("type") != "tool_result":
            return False
    return True


def _is_turn_event(event: dict) -> bool:
    return event.get("type") in ("user", "assistant")


def split_into_turns(events: list) -> list:
    """Group consecutive user+assistant events into turn boundaries.

    A turn is one user message plus all assistant/tool_result events until the
    next non-tool_result user message.
    """
    turns = []
    current: list = []
    for event in events:
        if not _is_turn_event(event):
            continue
        if event.get("type") == "user" and current and not _is_pure_tool_result(event):
            turns.append(current)
            current = []
        current.append(event)
    if current:
        turns.append(current)
    return turns


# ─── Validation + atomic write ─────────────────────────────────────────

def atomic_write(session_dir: Path, body: str) -> None:
    """Write dashboard.html via a unique tmp + os.replace; the original stays intact unless the rename succeeds."""
    final = session_dir / "dashboard.html"
    fd, tmp_name = tempfile.mkstemp(dir=session_dir, prefix=".dashboard.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ─── Subprocess invocation ─────────────────────────────────────────────

def _spawn_claude(cmd: "list[str]") -> subprocess.Popen:
    """Start a `claude -p` child with piped stdio under the subagent env."""
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=build_subagent_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _kill_and_reap(proc: subprocess.Popen) -> str:
    """Kill and reap a child; return whatever partial stdout was buffered before the kill."""
    proc.kill()
    try:
        out, _ = proc.communicate(timeout=10)
        return out or ""
    except Exception:
        return ""


@dataclass
class RunResult:
    body: str
    elapsed_seconds: float
    returncode: int
    stderr: str
    input_tokens: "int | None" = None
    output_tokens: "int | None" = None
    cost_usd: "float | None" = None
    duration_ms: "int | None" = None
    model: "str | None" = None  # resolved model id (what the alias mapped to)


def parse_cli_json(stdout: str) -> "tuple[str, dict]":
    """Parse `claude -p --output-format json` stdout into (body, metrics), falling
    back to (raw_stdout, {}) when stdout isn't that envelope."""
    try:
        d = json.loads(stdout)
    except (ValueError, TypeError):
        return stdout, {}
    if not isinstance(d, dict) or "result" not in d:
        return stdout, {}
    usage = d.get("usage") if isinstance(d.get("usage"), dict) else {}
    inp = usage.get("input_tokens")
    if isinstance(inp, int):
        # Cached input counts toward the total prompt size.
        total_in = (
            inp
            + (usage.get("cache_creation_input_tokens") or 0)
            + (usage.get("cache_read_input_tokens") or 0)
        )
    else:
        total_in = None
    out = usage.get("output_tokens")
    # modelUsage maps each model id to its tokens; take the one with the most
    # (the CLI's helper calls ride along with tiny counts).
    model_usage = d.get("modelUsage")
    resolved_model = None
    if isinstance(model_usage, dict) and model_usage:
        def _tokens(mu: dict) -> int:
            return sum(v for v in mu.values() if isinstance(v, (int, float)))
        resolved_model = max(model_usage, key=lambda k: _tokens(model_usage[k] or {}))
    metrics = {
        "input_tokens": total_in,
        "output_tokens": out if isinstance(out, int) else None,
        "cost_usd": d.get("total_cost_usd"),
        "duration_ms": d.get("duration_ms"),
        "model": resolved_model,
    }
    return (d.get("result") or ""), metrics


def invoke_claude(
    *,
    system_prompt: str,
    user_message: str,
    model: str,
    timeout: float = DEFAULT_REGEN_TIMEOUT,
    on_proc: "Callable[[subprocess.Popen], None] | None" = None,
    effort: "str | None" = None,
) -> RunResult:
    """Run `claude -p` headlessly. `on_proc` hands the live Popen to the caller so
    the Registry can SIGTERM it on supersede.

    Flags: --setting-sources local (no user Stop hooks → no recursion),
    --no-session-persistence (no ghost .jsonl), --tools "" (text only, no FS).
    """
    cmd = [
        "claude",
        "-p",
        "--output-format", "json",
        "--setting-sources", "local",
        "--no-session-persistence",
        "--model", model,
        "--system-prompt", system_prompt,
        "--tools", "",
    ]
    if effort:
        # Hidden thinking dominates output tokens, hence wall time.
        cmd += ["--effort", effort]
    start = time.monotonic()
    proc = _spawn_claude(cmd)
    if on_proc is not None:
        on_proc(proc)
    try:
        stdout, stderr = proc.communicate(input=user_message, timeout=timeout)
    except subprocess.TimeoutExpired:
        # Non-retryable (a retry would just wait the full timeout again); the
        # partial-output size distinguishes an oversized op-set from a stuck socket.
        partial = _kill_and_reap(proc)
        elapsed = time.monotonic() - start
        partial_chars = len(partial)
        tail = f"; produced {partial_chars} chars before kill" if partial.strip() else "; produced nothing"
        raise RegenTimeout(
            f"claude -p exceeded {timeout:.0f}s wall-clock and was killed "
            f"(ran {elapsed:.1f}s){tail}",
            elapsed_seconds=elapsed,
            partial_chars=partial_chars,
        )
    except BaseException:
        # Any other communicate() failure must not leak the child.
        _kill_and_reap(proc)
        raise
    elapsed = time.monotonic() - start
    body, usage = parse_cli_json(stdout)
    return RunResult(
        body=body,
        elapsed_seconds=elapsed,
        returncode=proc.returncode,
        stderr=stderr,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cost_usd=usage.get("cost_usd"),
        duration_ms=usage.get("duration_ms"),
        model=usage.get("model"),
    )


def probe_auth(model: str = DEFAULT_MODEL,
               timeout_s: float = 25.0) -> "tuple[bool, str, str]":
    """Startup health check: run a tiny `claude -p` under the real auth policy.
    Returns (ok, kind, detail). `kind` names which way it failed so the notice
    catalog classifies it directly instead of parsing the diagnostic back."""
    cmd = [
        "claude", "-p",
        "--setting-sources", "local",
        "--no-session-persistence",
        "--model", model,
        "--tools", "",
    ]
    try:
        proc = _spawn_claude(cmd)
        stdout, stderr = proc.communicate(input="Reply with: ok", timeout=timeout_s)
    except FileNotFoundError:
        return False, "CliMissing", "claude CLI not found on PATH"
    except subprocess.TimeoutExpired:
        _kill_and_reap(proc)
        return False, "ProbeTimeout", f"probe timed out after {timeout_s:.0f}s"
    if proc.returncode == 0 and stdout.strip():
        return True, "", stdout.strip()[:200]
    detail = (stderr.strip() or stdout.strip() or f"claude -p exited {proc.returncode}")
    return False, "SubprocessFailed", detail[:300]


# ─── One full attempt ──────────────────────────────────────────────────

class OutputRejected(Exception):
    """The op-set is invalid after the corrective retry, or the render is too small."""


class SubprocessFailed(Exception):
    """Raised when `claude -p` exits non-zero. stderr is captured."""


class RegenTimeout(SubprocessFailed):
    """`claude -p` exceeded the wall-clock timeout and was killed. Non-retryable.
    Carries the run's elapsed time and partial-output size."""

    def __init__(
        self, message: str, *,
        elapsed_seconds: "float | None" = None,
        partial_chars: "int | None" = None,
    ):
        super().__init__(message)
        self.elapsed_seconds = elapsed_seconds
        self.partial_chars = partial_chars


class SessionGone(Exception):
    """The chat's JSONL no longer exists (deleted, or its worktree cleaned). Not a
    failure — the registry skips it quietly."""


def _safe_record(metrics: "DashboardStore | None", **fields) -> None:
    """Best-effort telemetry write; a failed insert never affects a regen."""
    if metrics is None:
        return
    try:
        metrics.record(**fields)
    except Exception:
        _log.warning("metrics record raised", exc_info=True)


def run_once(
    *,
    plugin_dir: Path,
    projects_root: Path,
    project_hash: str,
    session_uuid: str,
    chat_state: "ChatState",
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_REGEN_TIMEOUT,
    journey: bool = True,
    metrics: "DashboardStore | None" = None,
    attempts: int = 1,
    on_proc: "Callable[[subprocess.Popen], None] | None" = None,
) -> Path:
    """One complete regen: assemble the prompt, invoke `claude -p`, fold the op-set,
    render, and atomically write. Non-destructive — a bad op-set leaves the stored
    model and the live dashboard untouched. Raises SessionGone / FileNotFoundError /
    SubprocessFailed / OutputRejected."""
    system_prompt_path = plugin_dir / "SYSTEM.md"
    journey_prompt_path = plugin_dir / "SYSTEM_JOURNEY.md"
    jsonl_path = projects_root / project_hash / f"{session_uuid}.jsonl"
    session_dir = projects_root / project_hash / session_uuid

    if not jsonl_path.exists():
        raise SessionGone(f"jsonl not found: {jsonl_path}")
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"SYSTEM.md not found: {system_prompt_path}")
    if not journey_prompt_path.exists():
        raise FileNotFoundError(f"SYSTEM_JOURNEY.md not found: {journey_prompt_path}")
    session_dir.mkdir(parents=True, exist_ok=True)

    events = read_jsonl(jsonl_path)
    all_turns = split_into_turns(events)
    turn_no = len(all_turns)                        # absolute conversation depth
    # Claude Code's own chat title; it heads the dashboard ahead of the model's.
    title = chat_title(events)
    # A resumed, compacted, or forked chat replays its parent's events; seed
    # it once with the parent's state so the dashboard continues instead of
    # starting blank. Telemetry stays with the parent.
    if not (chat_state.snapshot(project_hash, session_uuid) or {}).get("parentChecked"):
        parent = _lineage.find_parent(jsonl_path)
        chat_state.fork_from(project_hash, session_uuid, parent,
                             branch_turn=turn_no)
        if parent:
            parent_model = chat_state.get_model(project_hash, parent) or {}
            parent_absolute = int(parent_model.get("turn_base") or 0) + \
                int(parent_model.get("turn") or 0)
            base = _fold.continuation_base(parent_absolute, turn_no)
            if base:
                chat_state.set_turn_base(project_hash, session_uuid, base)
            _log.info("regen %s/%s — continued from %s (turn base %d)",
                      project_hash, session_uuid, parent, base)

    snap = chat_state.snapshot(project_hash, session_uuid) or {}
    stored = snap.get("model")
    dash_model = DashboardModel.model_validate(stored) if stored else DashboardModel()
    verdicts = snap.get("verdicts") or {}
    acks = snap.get("acks") or {}
    if verdicts:
        dash_model = _fold.apply_verdicts(dash_model, verdicts)

    assembled = assemble_prompt(RegenPrompt(
        dashboard=dash_model,
        turns=all_turns,
        turn_no=turn_no,
        verdicts=verdicts,
        acks=acks,
        diagram_errors=snap.get("diagramErrors") or {},
        system_template=system_prompt_path.read_text(encoding="utf-8"),
        journey_template=journey_prompt_path.read_text(encoding="utf-8"),
        journey=journey,
    ))
    system_prompt, user_message = assembled.system, assembled.user

    _log.info(
        "regen %s/%s — events=%d turns=%d truncated=%s prompt_words≈%d",
        project_hash, session_uuid[:8], len(events), turn_no,
        assembled.truncated, assembled.transcript_words,
    )
    _log.debug(
        "regen %s/%s — PROMPT DUMP\n"
        "===== SYSTEM PROMPT (%d chars) =====\n%s\n"
        "===== USER MESSAGE (%d chars) =====\n%s\n"
        "===== END PROMPT DUMP =====",
        project_hash, session_uuid[:8],
        len(system_prompt), system_prompt, len(user_message), user_message,
    )

    prompt_words = assembled.transcript_words
    try:
        msg = user_message
        update = bodies = None
        last_err = last_body = ""
        for attempt in range(2):
            result = invoke_claude(
                system_prompt=system_prompt, user_message=msg,
                model=model, timeout=timeout, on_proc=on_proc,
            )
            if result.returncode != 0:
                err = result.stderr.strip()
                out = result.body.strip()
                raise SubprocessFailed("\n".join([
                    f"claude -p exited {result.returncode} after {result.elapsed_seconds:.1f}s",
                    "--- stderr ---", (err[:2000] if err else "(empty)"),
                    "--- stdout ---", (out[:2000] if out else "(empty)"),
                ]))
            try:
                update, bodies, notes = agent_io.parse_output(result.body, journey=journey)
                for note in notes:
                    _log.warning("regen %s/%s — %s", project_hash, session_uuid[:8], note)
                break
            except agent_io.AgentOutputError as e:
                last_err, last_body = str(e), result.body
                _log.warning(
                    "regen %s/%s — attempt %d rejected: %s | body %d chars, head=%r",
                    project_hash, session_uuid[:8], attempt + 1, last_err[:200],
                    len(result.body), result.body[:200],
                )
                msg = user_message + RETRY_INSTRUCTION + last_err
        if update is None:
            raise OutputRejected(
                f"op-set invalid after retry: {last_err} (first 200 chars: {last_body[:200]!r})"
            )

        new_model = _fold.apply_ops(dash_model, update, bodies, turn_no)
        if verdicts:
            # A user click beats any same-turn agent op on the item.
            new_model = _fold.apply_verdicts(new_model, verdicts)
        html = _render.render(new_model, title, journey=journey)
        if len(html) < 200:
            raise OutputRejected(f"render too small ({len(html)} bytes) — refusing to overwrite")
    except Exception as exc:
        # The registry reads this off the exception for failure telemetry.
        setattr(exc, "prompt_words", prompt_words)
        raise

    atomic_write(session_dir, html)
    chat_state.set_model(project_hash, session_uuid, new_model.model_dump(mode="json"))
    _log.info(
        "regen %s/%s — %d ops, wrote %d bytes in %.1fs",
        project_hash, session_uuid[:8], len(update.ops), len(html), result.elapsed_seconds,
    )
    _safe_record(
        metrics,
        project_hash=project_hash, session_uuid=session_uuid,
        model=result.model or model,  # resolved id when the CLI reports it, else the alias
        status="ok", kind="ok",
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        cost_usd=result.cost_usd, duration_ms=result.duration_ms,
        wall_ms=int(result.elapsed_seconds * 1000),
        prompt_words=prompt_words,
        output_bytes=len(html.encode("utf-8")),
        block_sizes=_render.block_sizes(new_model),
        attempts=attempts,
    )
    return session_dir / "dashboard.html"


# ─── Registry ──────────────────────────────────────────────────────────

@dataclass
class JobRecord:
    project_hash: str
    session_uuid: str
    started_at: float
    state: str  # "running" | "failed"
    proc: "subprocess.Popen | None" = None
    error: "str | None" = None
    last_attempt_at: "float | None" = None
    superseded: bool = False
    # A trigger arrived mid-run; schedule exactly one follow-up on completion.
    rerun_requested: bool = False


class Registry:
    """Tracks at most one in-flight regen per session_uuid and exposes
    state for the index UI. Thread-safe under a single RLock."""

    def __init__(
        self,
        *,
        plugin_dir: Path,
        projects_root: Path,
        chat_state: "ChatState",
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_REGEN_TIMEOUT,
        journey: bool = True,
        metrics: "DashboardStore | None" = None,
        on_success: "Callable[[str, str], None] | None" = None,
        on_failure: "Callable[[str, str, str, str], None] | None" = None,
    ):
        self._plugin_dir = plugin_dir
        self._projects_root = projects_root
        self._model = model
        self._timeout = timeout
        self._journey = journey
        self._metrics = metrics
        self._chat_state = chat_state
        self._lock = threading.RLock()
        self._jobs: "dict[str, JobRecord]" = {}
        self._on_success = on_success
        # Fired on a genuine failure, not a supersede:
        # (project_hash, session_uuid, kind, message).
        self._on_failure = on_failure

    # ─── Public API ────────────────────────────────────────────────

    def trigger(self, project_hash: str, session_uuid: str) -> dict:
        """Schedule a regen. A young in-flight job coalesces (flagged for one
        follow-up); otherwise any existing job is cancelled and a fresh one starts."""
        with self._lock:
            old = self._jobs.get(session_uuid)
            if old is not None and old.state == "running":
                age = time.time() - old.started_at
                if age < STALE_SUPERSEDE_AFTER:
                    old.rerun_requested = True
                    return self._snapshot_locked(old)
                # Fall through: the in-flight job is old enough to be wedged.
            if old is not None:
                self._cancel_locked(old)
            new = JobRecord(
                project_hash=project_hash,
                session_uuid=session_uuid,
                started_at=time.time(),
                state="running",
            )
            self._jobs[session_uuid] = new
            snapshot = self._snapshot_locked(new)

        thread = threading.Thread(
            target=self._run, args=(new,), daemon=True,
            name=f"regen-{session_uuid[:8]}",
        )
        thread.start()
        return snapshot

    def state_for(self, session_uuid: str) -> "dict | None":
        """Public state snapshot for one session, or None if no record."""
        with self._lock:
            r = self._jobs.get(session_uuid)
            return self._snapshot_locked(r) if r is not None else None

    def resolve_project_hash(self, session_uuid: str) -> "str | None":
        """The project_hash dir owning a session_uuid, found via
        `<projects_root>/*/<uuid>.jsonl`; None if absent."""
        for proj_dir in self._projects_root.iterdir():
            if not proj_dir.is_dir():
                continue
            if (proj_dir / f"{session_uuid}.jsonl").is_file():
                return proj_dir.name
        return None

    # ─── Internals ─────────────────────────────────────────────────

    def _current_timeout(self) -> float:
        """Resolve the wall-clock limit per run. serve.py passes a provider that
        reads live settings, so changing the limit applies to the next rebuild
        without a restart; tests pass a plain number."""
        limit = self._timeout
        return float(limit() if callable(limit) else limit)

    def _current_model(self) -> str:
        """Resolve the model alias per run; a provider makes a settings change
        apply to the next rebuild without a restart."""
        model = self._model
        return str(model() if callable(model) else model)

    def _current_journey(self) -> bool:
        """Resolve the journey switch per run; a provider makes a settings
        change apply to the next rebuild without a restart."""
        journey = self._journey
        return bool(journey() if callable(journey) else journey)

    def _snapshot_locked(self, r: JobRecord) -> dict:
        return {
            "state": r.state,
            "since": int(r.started_at),
            "lastAttemptAt": int(r.last_attempt_at) if r.last_attempt_at else None,
            "error": r.error,
        }

    def _cancel_locked(self, r: JobRecord) -> None:
        """Mark `r` superseded and SIGTERM its subprocess if still alive.
        Caller holds the lock."""
        r.superseded = True
        proc = r.proc
        if proc is not None and proc.poll() is None:
            _log.info(
                "supersede %s/%s — terminating pid %d",
                r.project_hash, r.session_uuid[:8], proc.pid,
            )
            try:
                proc.terminate()
            except OSError:
                pass
            # SIGKILL fallback off-thread so the lock-holder doesn't block.
            threading.Thread(
                target=self._sigkill_after_grace,
                args=(proc,),
                daemon=True,
            ).start()

    @staticmethod
    def _sigkill_after_grace(proc: subprocess.Popen, grace_s: float = 0.5) -> None:
        end = time.monotonic() + grace_s
        while time.monotonic() < end:
            if proc.poll() is not None:
                return
            time.sleep(0.05)
        try:
            proc.kill()
        except OSError:
            pass

    def _run(self, record: JobRecord) -> None:
        """Daemon-thread worker: run run_once (retrying transient failures) and
        update state under the lock. A superseded record leaves state to its
        replacement; on completion, honour a coalesced rerun request."""

        def attach_proc(p: subprocess.Popen) -> None:
            with self._lock:
                record.proc = p

        attempt = 0
        while True:
            attempt += 1
            # Bail if superseded or replaced before spending an attempt.
            with self._lock:
                if record.superseded or self._jobs.get(record.session_uuid) is not record:
                    return

            model = self._current_model()
            try:
                run_once(
                    plugin_dir=self._plugin_dir,
                    projects_root=self._projects_root,
                    project_hash=record.project_hash,
                    session_uuid=record.session_uuid,
                    chat_state=self._chat_state,
                    model=model,
                    timeout=self._current_timeout(),
                    journey=self._current_journey(),
                    metrics=self._metrics,
                    attempts=attempt,
                    on_proc=attach_proc,
                )
            except SessionGone as e:
                # Deleted between trigger and run — drop quietly, no failure/rerun.
                with self._lock:
                    if self._jobs.get(record.session_uuid) is record:
                        del self._jobs[record.session_uuid]
                _log.info(
                    "regen skipped %s/%s — session gone (%s)",
                    record.project_hash, record.session_uuid[:8], e,
                )
                return
            except Exception as e:
                error_kind = type(e).__name__
                # error_full is persisted (regenErrors); error_short is the chip tooltip.
                error_full = str(e).strip()
                error_short = _short_error(e)
                prompt_words = getattr(e, "prompt_words", None)
                wall_ms = output_bytes = None
                if isinstance(e, RegenTimeout):
                    if e.elapsed_seconds is not None:
                        wall_ms = int(e.elapsed_seconds * 1000)
                    output_bytes = e.partial_chars

                def record_outcome(status: str) -> None:
                    _safe_record(
                        self._metrics,
                        project_hash=record.project_hash,
                        session_uuid=record.session_uuid,
                        model=model,
                        status=status,
                        kind=error_kind,
                        prompt_words=prompt_words,
                        wall_ms=wall_ms,
                        output_bytes=output_bytes,
                        attempts=attempt,
                    )

                with self._lock:
                    superseded = record.superseded
                    if not superseded and self._jobs.get(record.session_uuid) is not record:
                        return
                    retry = (not superseded and attempt < MAX_ATTEMPTS
                             and _is_retryable(e))
                    if not superseded and not retry:
                        record.state = "failed"
                        record.error = error_short
                        record.last_attempt_at = time.time()
                        record.proc = None
                # A superseded run's failure is expected; record it, don't surface it.
                if superseded:
                    _log.info(
                        "supersede %s/%s — %s",
                        record.project_hash, record.session_uuid[:8], error_kind,
                    )
                    record_outcome("superseded")
                    return
                if retry:
                    _log.info(
                        "regen retry %s/%s — attempt %d/%d after %s: %s",
                        record.project_hash, record.session_uuid[:8],
                        attempt, MAX_ATTEMPTS, error_kind, error_short,
                    )
                    continue
                _log.warning(
                    "regen failed %s/%s — %s: %s",
                    record.project_hash, record.session_uuid[:8],
                    error_kind, error_short,
                )
                _log.info(
                    "regen-error detail %s/%s:\n%s",
                    record.project_hash, record.session_uuid[:8], error_full,
                )
                # Fires outside the lock; the callback writes to disk.
                if self._on_failure is not None:
                    try:
                        self._on_failure(
                            record.project_hash, record.session_uuid,
                            error_kind, error_full,
                        )
                    except Exception:
                        _log.exception("on_failure callback raised")
                record_outcome("failed")
                self._maybe_rerun(record)
                return

            # Success.
            fire_on_success = False
            with self._lock:
                if not record.superseded and self._jobs.get(record.session_uuid) is record:
                    # Clear the record so the UI infers "current" from file
                    # mtimes.
                    del self._jobs[record.session_uuid]
                    fire_on_success = True
            # Callbacks fire outside the lock (they may take other locks).
            if fire_on_success and self._on_success is not None:
                try:
                    self._on_success(record.project_hash, record.session_uuid)
                except Exception:
                    _log.exception("on_success callback raised")
            self._maybe_rerun(record)
            return

    def _maybe_rerun(self, record: JobRecord) -> None:
        """Start one fresh regen if a trigger was coalesced onto this job while it ran."""
        with self._lock:
            rerun = record.rerun_requested
            record.rerun_requested = False
        if rerun:
            _log.info(
                "regen coalesced-rerun %s/%s",
                record.project_hash, record.session_uuid[:8],
            )
            self.trigger(record.project_hash, record.session_uuid)


def _short_error(e: BaseException) -> str:
    """Trim error messages for the UI — full text is in the log file."""
    msg = str(e).strip()
    if len(msg) > 200:
        msg = msg[:197] + "..."
    return msg


def _is_retryable(e: BaseException) -> bool:
    """A transient SubprocessFailed is retried; RegenTimeout and OutputRejected
    are not, and neither is a failure that would fail the same way again, such
    as an oversized prompt or an expired sign in."""
    if isinstance(e, RegenTimeout):
        return False
    if not isinstance(e, SubprocessFailed):
        return False
    return not notices.is_permanent(notices.classify(type(e).__name__, str(e)))
