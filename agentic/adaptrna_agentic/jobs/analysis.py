"""RunAnalyzer: did the training run actually work?

Deterministic checks over the engine's `metrics.csv`, each mapped to a documented failure
mode in the knowledge base — so a bad run comes back with the *reason* and the *remedy*,
not just a number. The LLM narrates this report; it does not judge the run itself.

Two rules this module exists to enforce:

* A **truncated** run (capped with `trainer.max_steps`) is never compared to reference
  metrics — a smoke test is not a result.
* A difference inside the task's tolerance is **not** a regression. FlashAttention's
  backward is non-deterministic: the same splice-site command and seed produced F1 95.21
  and 95.82 (MASTER_PLAN §7).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import math

from adaptrna_agentic.jobs.runner import latest_metrics_file
from adaptrna_agentic.knowledge import arm as arm_knowledge
from adaptrna_agentic.knowledge import task_knowledge

VERDICT_OK = "ok"
VERDICT_SUSPICIOUS = "suspicious"
VERDICT_FAILED = "failed"

#: Below this the primary metric is degenerate for every shipped task (F1 in percent,
#: R^2 in units): the model learned nothing, or the backbone was destroyed.
_DEGENERATE = 1e-6


def analyze_run(
    output_dir: Union[str, Path],
    task: Optional[str] = None,
    arm: str = "lora",
    plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze a finished (or in-flight) training run. Returns a report dict."""
    plan = plan or {}
    task = task or plan.get("task")
    arm = plan.get("arm", arm)

    report: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "task": task,
        "arm": arm,
        "truncated": bool(plan.get("quick_run") or plan.get("overrides", {}).get("trainer.max_steps")),
        "checks": [],
        "suggestions": [],
    }

    metrics_file = latest_metrics_file(output_dir)
    if metrics_file is None:
        report["verdict"] = VERDICT_FAILED
        report["checks"].append("no metrics.csv found — the run produced no metrics at all")
        report["suggestions"].append("Check the run log for an early crash (`job logs`).")
        return report

    import pandas as pd

    frame = pd.read_csv(metrics_file)
    metrics = _final_metrics(frame)
    report["metrics"] = metrics

    if task is None:
        report["verdict"] = VERDICT_SUSPICIOUS
        report["checks"].append("no task recorded — cannot compare against a reference")
        return report

    knowledge = task_knowledge(task)
    primary = knowledge["primary_metric"]
    value = metrics.get(primary)
    report["primary_metric"] = primary
    report["primary_value"] = value

    failures: List[str] = []
    warnings: List[str] = []

    # --- hard failures ----------------------------------------------------------
    nan_columns = _nonfinite_loss_columns(metrics_file)
    if nan_columns:
        failures.append(f"non-finite loss values in {nan_columns}")
        report["suggestions"].append(
            "NaN/inf loss means the run diverged — lower the learning rate and make sure "
            "trainer.gradient_clip_val is set."
        )

    if value is None:
        warnings.append(f"the primary metric '{primary}' was never logged")
        report["suggestions"].append(
            "The run may not have reached its test phase; check the log and whether it "
            "was cancelled."
        )
    elif not math.isfinite(value):
        failures.append(f"'{primary}' is not finite ({value})")
    elif value <= _DEGENERATE:
        failures.append(
            f"'{primary}' = {value:g} — the model learned nothing (degenerate output)"
        )
        report["suggestions"].append(_arm_remedy(arm))
    else:
        report["checks"].append(f"{primary} = {_fmt(value)}")

    # --- collapse signature ------------------------------------------------------
    if _looks_collapsed(frame):
        warnings.append(
            "training loss flatlined after an early drop — the constant-output collapse "
            "signature"
        )
        report["suggestions"].append(_arm_remedy(arm))

    # --- reference comparison ----------------------------------------------------
    reference = knowledge["reference"]
    band = reference.get("band")
    tolerance = float(reference.get("tolerance", 0))

    if report["truncated"]:
        report["checks"].append(
            "truncated run (capped by trainer.max_steps) — a smoke test, NOT comparable "
            "to the reference metrics for this task"
        )
    elif band and value is not None and math.isfinite(value):
        low, high = float(band[0]), float(band[1])
        if value < low - tolerance:
            warnings.append(
                f"{primary} = {_fmt(value)} is below the reference band "
                f"{low}–{high} (tolerance ±{tolerance})"
            )
            report["suggestions"].append(
                "Compare against the reference settings, and remember run-to-run variance: "
                "confirm with a second seed or data split before concluding anything."
            )
        elif value > high + tolerance:
            report["checks"].append(
                f"{primary} = {_fmt(value)} is above the reference band {low}–{high} — "
                f"good, but worth confirming there is no leakage between splits"
            )
        else:
            report["checks"].append(
                f"{primary} = {_fmt(value)} is within the reference band {low}–{high} "
                f"(±{tolerance} for run-to-run non-determinism)"
            )
    elif band is None:
        report["checks"].append("no in-repo reference for this task yet — nothing to compare")

    report["checks"].extend(f"FAIL: {message}" for message in failures)
    report["checks"].extend(f"WARN: {message}" for message in warnings)
    report["verdict"] = (
        VERDICT_FAILED if failures else VERDICT_SUSPICIOUS if warnings else VERDICT_OK
    )

    return report


# ---------------------------------------------------------------------- helpers

def _final_metrics(frame) -> Dict[str, float]:
    """The last logged value of every metric column (rows are sparse by design)."""
    metrics = {}
    for column in frame.columns:
        if column in ("epoch", "step"):
            continue
        series = frame[column].dropna()
        if not series.empty:
            metrics[column] = round(float(series.iloc[-1]), 6)

    return metrics


_NONFINITE_LITERALS = {"nan", "inf", "-inf", "+inf", "infinity", "-infinity"}


def _nonfinite_loss_columns(metrics_file: Path) -> List[str]:
    """Loss columns containing a *logged* NaN/inf.

    The engine's metrics.csv is sparse — a metric not logged on a given row is an empty
    cell — and pandas reads both an empty cell and a literal "nan" as NaN. So divergence
    can only be told apart from "not logged here" by reading the cells as text.
    """
    import pandas as pd

    raw = pd.read_csv(metrics_file, dtype=str, keep_default_na=False)

    return [
        column for column in raw.columns
        if "loss" in column
        and raw[column].str.strip().str.lower().isin(_NONFINITE_LITERALS).any()
    ]


def _looks_collapsed(frame) -> bool:
    """Constant-output collapse: the training loss stops moving entirely."""
    if "train/loss" not in frame.columns:
        return False

    values = frame["train/loss"].dropna()
    if len(values) < 4:
        return False

    tail = values.iloc[len(values) // 2:]
    return bool(float(tail.std()) < 1e-9)


def _arm_remedy(arm: str) -> str:
    modes = arm_knowledge(arm).get("failure_modes", [])
    if not modes:
        return "Review the training configuration against the knowledge base."

    mode = modes[0]
    return f"Known failure mode — {mode['setting']}: {mode['symptom']} {mode['remedy']}"


def _fmt(value: float) -> str:
    return f"{value:.4g}"
