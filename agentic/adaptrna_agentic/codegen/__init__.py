"""Code generation: writing, verifying and landing new tasks and external wrappers.

The agents (`agents/toolsmith.py`, `agents/verifier.py`) are one model call each; the
loop, the staging, the sandbox and the verification harness in this package are plain
Python — the part that decides whether generated code is trustworthy is deterministic
(MASTER_PLAN §3.1).
"""
