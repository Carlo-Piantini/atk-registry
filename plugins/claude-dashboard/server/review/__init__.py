"""Guided code review: turn a branch's diff into an ordered, explained walkthrough.

Deterministic passes (diff, structure, provenance) live here as stdlib-only
modules; the LLM analysis pass builds on their output. Unlike identity.py,
this package shells out to `git` — reading a diff has no file-only shortcut.
"""
