from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceLocationCalibrationMethod:
    key: str
    label: str
    step: int
    output_dirname: str
    delay_strategy: str
    sigma_strategy: str
    mode_correlation_strategy: str
    learning_inventory_scope: str
    calibration_enabled: bool
    input_calibration_dirname: str | None = None
    allowed_phases: tuple[str, ...] | None = None  # None = all phases


_METHODS = (
    SourceLocationCalibrationMethod(
        key="step0_P_only",
        label="Step 0 - P-only baseline (diagnostic)",
        step=0,
        output_dirname="source_location_step0_P_only",
        delay_strategy="none",
        sigma_strategy="current",
        mode_correlation_strategy="current",
        learning_inventory_scope=(
            "No calibration. Only P arrivals are used; S picks are masked before "
            "inference. Diagnostic run to assess the contribution of uncalibrated "
            "S arrivals to location bias."
        ),
        calibration_enabled=False,
        allowed_phases=("P",),
    ),
    SourceLocationCalibrationMethod(
        key="step0_baseline",
        label="Step 0 - Baseline (uncalibrated)",
        step=0,
        output_dirname="source_location_step0_baseline",
        delay_strategy="none",
        sigma_strategy="current",
        mode_correlation_strategy="current",
        learning_inventory_scope=(
            "No learned calibration archive; direct inference with configured phase "
            "sigma and the configured mode-correlation setting."
        ),
        calibration_enabled=False,
    ),
    SourceLocationCalibrationMethod(
        key="step1_phase_only",
        label="Step 1 - Phase-only correction",
        step=1,
        output_dirname="source_location_step1_phase_only",
        delay_strategy="phase",
        sigma_strategy="phase_empirical",
        mode_correlation_strategy="current",
        learning_inventory_scope=(
            "Uses phase bias and empirical phase sigma learned from the larger "
            "solved residual inventory in "
            "results/source_location_step0_baseline/residual_calibration."
        ),
        calibration_enabled=True,
        input_calibration_dirname="source_location_step0_baseline/residual_calibration",
    ),
    SourceLocationCalibrationMethod(
        key="step2_additive",
        label="Step 2 - Additive phase+station correction",
        step=2,
        output_dirname="source_location_step2_additive",
        delay_strategy="additive",
        sigma_strategy="phase_after_additive_correction",
        mode_correlation_strategy="after_additive_correction",
        learning_inventory_scope=(
            "Uses additive phase and station offsets learned from the larger solved "
            "residual inventory in "
            "results/source_location_step0_baseline/residual_calibration."
        ),
        calibration_enabled=True,
        input_calibration_dirname="source_location_step0_baseline/residual_calibration",
    ),
    SourceLocationCalibrationMethod(
        key="step3_shrinkage",
        label="Step 3 - Shrinkage phase+station correction",
        step=3,
        output_dirname="source_location_step3_shrinkage",
        delay_strategy="shrinkage",
        sigma_strategy="phase_after_shrinkage_correction",
        mode_correlation_strategy="after_shrinkage_correction",
        learning_inventory_scope=(
            "Uses additive offsets plus empirical-Bayes station-phase shrinkage "
            "learned from the solved residual inventory in "
            "results/source_location_step2_additive/residual_calibration."
        ),
        calibration_enabled=True,
        input_calibration_dirname="source_location_step2_additive/residual_calibration",
    ),
)

_METHODS_BY_KEY = {method.key: method for method in _METHODS}
_METHODS_BY_OUTPUT_DIRNAME = {method.output_dirname: method for method in _METHODS}
_METHOD_ALIASES = {
    "step0_p_only": "step0_P_only",
    "p_only": "step0_P_only",
    "p-only": "step0_P_only",
    "baseline": "step0_baseline",
    "none": "step0_baseline",
    "step0": "step0_baseline",
    "step0_baseline": "step0_baseline",
    "phase": "step1_phase_only",
    "phase_empirical": "step1_phase_only",
    "phase_only": "step1_phase_only",
    "step1": "step1_phase_only",
    "step1_phase_only": "step1_phase_only",
    "additive": "step2_additive",
    "step2": "step2_additive",
    "step2_additive": "step2_additive",
    "shrinkage": "step3_shrinkage",
    "step3": "step3_shrinkage",
    "step3_shrinkage": "step3_shrinkage",
}
_METHODS_BY_STRATEGY = {
    (
        method.delay_strategy,
        method.sigma_strategy,
        method.mode_correlation_strategy,
    ): method
    for method in _METHODS
    if method.allowed_phases is None  # exclude phase-restricted variants
}


def available_source_location_methods() -> tuple[SourceLocationCalibrationMethod, ...]:
    return _METHODS


def normalize_source_location_method_key(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized not in _METHOD_ALIASES:
        options = ", ".join(method.key for method in _METHODS)
        raise ValueError(
            f"Unsupported source-location calibration method '{method}'. "
            f"Expected one of: {options}"
        )
    return _METHOD_ALIASES[normalized]


def get_source_location_method(method: str) -> SourceLocationCalibrationMethod:
    return _METHODS_BY_KEY[normalize_source_location_method_key(method)]


def normalize_source_location_method_keys(
    methods: Iterable[Any] | None,
    *,
    include_baseline: bool = False,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    if include_baseline:
        normalized.append("step0_baseline")
        seen.add("step0_baseline")

    if methods is None:
        return normalized

    if isinstance(methods, (str, bytes)):
        methods = [methods]

    for method in methods:
        method_key = get_source_location_method(str(method)).key
        if method_key in seen:
            continue
        normalized.append(method_key)
        seen.add(method_key)
    return normalized


def expand_source_location_execution_method_keys(
    methods: Iterable[Any] | None,
) -> list[str]:
    requested = normalize_source_location_method_keys(methods, include_baseline=True)
    required: set[str] = set()

    def add_prerequisites(method_key: str) -> None:
        if method_key in required:
            return

        method = get_source_location_method(method_key)
        if method.input_calibration_dirname is not None:
            prerequisite_output_dirname = str(method.input_calibration_dirname).replace(
                "\\",
                "/",
            )
            prerequisite_output_dirname = prerequisite_output_dirname.removesuffix(
                "/residual_calibration"
            )
            prerequisite_method = _METHODS_BY_OUTPUT_DIRNAME.get(
                prerequisite_output_dirname
            )
            if prerequisite_method is None:
                raise ValueError(
                    "Unsupported prerequisite calibration directory "
                    f"'{method.input_calibration_dirname}' for method '{method.key}'"
                )
            add_prerequisites(prerequisite_method.key)

        required.add(method_key)

    for method_key in requested:
        add_prerequisites(method_key)

    return [method.key for method in _METHODS if method.key in required]


def infer_source_location_method(
    *,
    calibration_enabled: bool,
    delay_strategy: Any = None,
    sigma_strategy: Any = None,
    mode_correlation_strategy: Any = None,
) -> SourceLocationCalibrationMethod:
    if not calibration_enabled:
        return get_source_location_method("step0_baseline")

    strategy_key = (
        str(delay_strategy or "none"),
        str(sigma_strategy or "current"),
        str(mode_correlation_strategy or "current"),
    )
    if strategy_key not in _METHODS_BY_STRATEGY:
        raise ValueError(
            "Unsupported source-location calibration strategy combination: "
            f"delay={strategy_key[0]}, sigma={strategy_key[1]}, "
            f"mode_correlation={strategy_key[2]}"
        )
    return _METHODS_BY_STRATEGY[strategy_key]


def resolve_source_location_method(
    calibration_cfg: dict[str, Any] | None,
) -> SourceLocationCalibrationMethod:
    if calibration_cfg is None:
        return get_source_location_method("step0_baseline")

    method = calibration_cfg.get("method")
    if method not in {None, ""}:
        return get_source_location_method(str(method))

    # Support methods list: use the last (most advanced) entry as the active method
    methods = calibration_cfg.get("methods")
    if methods is not None:
        if isinstance(methods, str):
            methods = [methods]
        if methods:
            return get_source_location_method(str(methods[-1]))

    return infer_source_location_method(
        calibration_enabled=bool(calibration_cfg.get("enabled", False)),
        delay_strategy=calibration_cfg.get("delay_strategy"),
        sigma_strategy=calibration_cfg.get("sigma_strategy"),
        mode_correlation_strategy=calibration_cfg.get("mode_correlation_strategy"),
    )


def default_source_location_results_dir(config: dict[str, Any]) -> Path:
    output_cfg = config.get("output", {})
    results_dir = output_cfg.get("results_dir")
    if results_dir not in {None, ""}:
        return Path(results_dir)

    source_location_cfg = config.get("source_location", {})
    output_dir = source_location_cfg.get("output_dir")
    if output_dir not in {None, ""}:
        return Path(output_dir).parent
    return Path.cwd() / "results"


def default_method_output_dir(
    results_dir: Path,
    method: SourceLocationCalibrationMethod,
) -> Path:
    return results_dir / method.output_dirname


def default_method_traveltimes_dir(results_dir: Path) -> Path:
    return results_dir / "source_location_step0_baseline" / "traveltimes"


def default_method_input_calibration_dir(
    results_dir: Path,
    method: SourceLocationCalibrationMethod,
) -> Path | None:
    if method.input_calibration_dirname is None:
        return None
    return results_dir / method.input_calibration_dirname


def default_method_output_calibration_dir(output_dir: Path) -> Path:
    return output_dir / "residual_calibration"


def default_method_comparison_output_dir(results_dir: Path) -> Path:
    return results_dir / "source_location_method_comparison"


def apply_source_location_method_config(
    config: dict[str, Any],
) -> SourceLocationCalibrationMethod:
    source_location_cfg = config.setdefault("source_location", {})
    calibration_cfg = source_location_cfg.setdefault("calibration", {})
    plots_cfg = source_location_cfg.setdefault("plots", {})

    method = resolve_source_location_method(calibration_cfg)
    results_dir = default_source_location_results_dir(config)
    output_dir = default_method_output_dir(results_dir, method)

    source_location_cfg["output_dir"] = str(output_dir)
    source_location_cfg["traveltimes_dir"] = str(
        default_method_traveltimes_dir(results_dir)
    )
    if isinstance(plots_cfg, dict):
        plots_cfg["output_dir"] = str(output_dir / "plots")

    calibration_cfg["method"] = method.key
    calibration_cfg["enabled"] = method.calibration_enabled
    calibration_cfg["delay_strategy"] = method.delay_strategy
    calibration_cfg["sigma_strategy"] = method.sigma_strategy
    calibration_cfg["mode_correlation_strategy"] = method.mode_correlation_strategy

    input_calibration_dir = default_method_input_calibration_dir(results_dir, method)
    if input_calibration_dir is None:
        calibration_cfg.pop("input_calibration_dir", None)
    else:
        calibration_cfg["input_calibration_dir"] = str(input_calibration_dir)
    calibration_cfg["output_calibration_dir"] = str(
        default_method_output_calibration_dir(output_dir)
    )

    # Phase filter: propagate allowed_phases from the method into uncertainty config
    uncertainty_cfg = source_location_cfg.setdefault("uncertainty", {})
    if method.allowed_phases is not None:
        uncertainty_cfg["allowed_phases"] = list(method.allowed_phases)
    else:
        uncertainty_cfg.pop("allowed_phases", None)

    return method


def resolve_source_location_pipeline_method_keys(
    calibration_cfg: dict[str, Any] | None,
) -> list[str]:
    """Return the prerequisite-expanded ordered list of method keys for a pipeline run.

    Reads ``calibration.methods`` (list) and expands any missing prerequisites.
    Falls back to ``calibration.method`` (single key) treated as a one-item list.
    Baseline (step0) is always included as the first entry.
    """
    if calibration_cfg is None:
        return ["step0_baseline"]

    methods = calibration_cfg.get("methods")
    if methods is not None:
        if isinstance(methods, str):
            methods = [methods]
        return expand_source_location_execution_method_keys(methods)

    method = calibration_cfg.get("method")
    if method not in {None, ""}:
        return expand_source_location_execution_method_keys([method])

    return ["step0_baseline"]


def resolve_method_comparison_method_keys(config: dict[str, Any]) -> list[str] | None:
    source_location_cfg = config.get("source_location", {})
    comparison_cfg = source_location_cfg.get("method_comparison", {})
    if not isinstance(comparison_cfg, dict):
        return ["step0_baseline"]

    methods = comparison_cfg.get("methods")
    if methods is None or methods == "":
        return ["step0_baseline"]

    return normalize_source_location_method_keys(methods, include_baseline=True)
