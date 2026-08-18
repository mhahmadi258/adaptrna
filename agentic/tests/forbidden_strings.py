"""The canonical D1/D2 forbidden-string list (plans/PHASE_13_COLD_START_SINGLE_CSV.md §1):
no shipped task name, its datasets, or the ViennaRNA reference wrapper may appear anywhere
under `agentic/`.

This module is the one legitimate place these strings are written as literals — both
`test_no_shipped_task_knowledge.py`'s repo-wide sweep and `test_prompts.py`'s narrower
"never leaks into a generated prompt" check import from here rather than each keeping its
own copy, so there is exactly one list to keep in sync with the plan. The sweep excludes
this file by name for the same reason it excludes itself: a test that forbids a string
must be able to name it.
"""

FORBIDDEN = (
    "splice_site",
    "mrl",
    "sec_struct",
    "ncrna_classification",
    "Spliceator",
    "GS_1",
    "bpRNA",
    "vienna",
    "RNAfold",
)
