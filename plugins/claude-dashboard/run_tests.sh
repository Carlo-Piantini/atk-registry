#!/usr/bin/env bash
# Run all claude-dashboard unit tests. No network, no `claude -p`.
# Each test file is a self-contained stdlib script (python3 <file>); this just
# runs them all and reports an aggregate pass/fail.
#
# Usage: ./run_tests.sh
set -u

# Don't litter the plugin dir with __pycache__/*.pyc.
export PYTHONDONTWRITEBYTECODE=1

cd "$(dirname "$0")" || exit 2

# The server + its tests validate the dashboard model with pydantic, which lives
# in the plugin's pinned virtualenv (created by install.sh).
PY="$PWD/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: virtualenv missing at $PWD/.venv — run install.sh first." >&2
  exit 2
fi

fail=0

# run <dir> <test-name> — each test is a self-contained script.
run() {
  echo "=== ${2} ==="
  if ! ( cd "${1}" && "$PY" "${2}.py" ); then
    fail=1
  fi
  echo
}

run server  test_models
run server  test_agent_io
run server  test_fold
run server  test_render
run server  test_digest
run server  test_prompt
run server  test_chat_state
run server  test_regen
run server  test_run_once
run server  test_store
run server  test_config
run server  test_notices
run server  test_notices_html
run server  test_notice_center
run server  test_notice_discipline
run server  test_identity
run server  test_lineage
run server  test_nav
run server  test_serve_security
run server  test_serve_feed
run server  test_review
run preview test_preview
run .       test_manage

if [ "${fail}" -eq 0 ]; then
  echo "ALL TESTS PASSED"
else
  echo "SOME TESTS FAILED" >&2
fi
exit "${fail}"
