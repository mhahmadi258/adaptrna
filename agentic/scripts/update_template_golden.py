#!/usr/bin/env python
"""Regenerate the golden files `test_templates_render.py` checks rendered output against.

Run this after any change to `codegen/templates/` (the .j2 files, `render.py`, or
`TEMPLATE_VERSION`), then **read the diff** before committing it — the golden files are
the reviewable record of what the template actually emits, so a diff here is exactly the
change the template edit should have produced. `test_templates_render.py` will fail until
the golden files are regenerated to match.

    python agentic/scripts/update_template_golden.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agentic"))
sys.path.insert(0, str(REPO_ROOT / "agentic" / "tests"))

from adaptrna_agentic.codegen.templates import render as templates  # noqa: E402
from fixtures.template_specs import TEMPLATE_SPECS  # noqa: E402

GOLDEN_ROOT = REPO_ROOT / "agentic" / "tests" / "fixtures" / "golden" / "templates"


def main() -> int:
    for name, spec in TEMPLATE_SPECS.items():
        if not templates.covers(spec):
            print(f"skip {name}: template does not cover this spec", file=sys.stderr)
            continue

        files = templates.render(spec)
        case_dir = GOLDEN_ROOT / name
        case_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            (case_dir / filename).write_text(content)
        print(f"wrote {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
