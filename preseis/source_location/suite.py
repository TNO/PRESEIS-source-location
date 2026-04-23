from __future__ import annotations

import logging
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from preseis.processing.config import load_config
from .methods import (
    expand_source_location_execution_method_keys,
    normalize_source_location_method_keys,
    resolve_method_comparison_method_keys,
)

_METHOD_STEP_SPECS = (
    {
        "script": "run_source_location.py",
        "supports_events": True,
        "supports_force": True,
    },
    {
        "script": "generate_source_location_catalogue.py",
        "supports_events": False,
        "supports_force": False,
    },
    {
        "script": "plot_source_locations.py",
        "supports_events": True,
        "supports_force": True,
    },
    {
        "script": "analyze_source_location_residuals.py",
        "supports_events": True,
        "supports_force": True,
    },
)


def resolve_requested_source_location_methods(
    config: dict[str, Any],
    *,
    methods_override: list[str] | None = None,
) -> list[str]:
    if methods_override:
        return normalize_source_location_method_keys(
            methods_override,
            include_baseline=True,
        )
    return resolve_method_comparison_method_keys(config) or ["step0_baseline"]


def _build_method_specific_config(
    config: dict[str, Any],
    *,
    method_key: str,
    comparison_methods: list[str],
) -> dict[str, Any]:
    method_config = deepcopy(config)
    source_location_cfg = method_config.setdefault("source_location", {})
    calibration_cfg = source_location_cfg.setdefault("calibration", {})
    comparison_cfg = source_location_cfg.setdefault("method_comparison", {})

    calibration_cfg["method"] = method_key
    calibration_cfg.pop("methods", None)
    comparison_cfg["methods"] = comparison_methods
    return method_config


def _build_script_command(
    *,
    script_dir: Path,
    step_spec: dict[str, Any],
    config_path: Path,
    base_dir: Path | None,
    events: list[str] | None,
    force: bool,
) -> list[str]:
    command = [sys.executable, str(script_dir / step_spec["script"]), str(config_path)]
    if base_dir is not None:
        command.extend(["--base-dir", str(base_dir)])
    if step_spec.get("supports_events") and events:
        command.extend(["--events", *events])
    if step_spec.get("supports_force") and force:
        command.append("--force")
    return command


def run_source_location_method_suite(
    *,
    config_path: Path,
    base_dir: Path | None = None,
    methods: list[str] | None = None,
    events: list[str] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or logging.getLogger(__name__)
    loaded_config = load_config(config_path, base_dir=base_dir)
    requested_methods = resolve_requested_source_location_methods(
        loaded_config,
        methods_override=methods,
    )
    execution_methods = expand_source_location_execution_method_keys(requested_methods)
    script_dir = Path(__file__).resolve().parents[3] / "scripts"
    run_cwd = base_dir or Path.cwd()

    active_logger.info(
        "Requested single-pass source-location methods: %s",
        ", ".join(requested_methods),
    )
    active_logger.info(
        "Execution order with prerequisites: %s",
        ", ".join(execution_methods),
    )

    with TemporaryDirectory(prefix="source_location_methods_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for method_key in execution_methods:
            method_config = _build_method_specific_config(
                loaded_config,
                method_key=method_key,
                comparison_methods=requested_methods,
            )
            method_config_path = temp_dir / f"{method_key}.yaml"
            method_config_path.write_text(
                yaml.safe_dump(method_config, sort_keys=False),
                encoding="utf-8",
            )
            active_logger.info(
                "Running one-pass source-location method: %s", method_key
            )
            for step_spec in _METHOD_STEP_SPECS:
                command = _build_script_command(
                    script_dir=script_dir,
                    step_spec=step_spec,
                    config_path=method_config_path,
                    base_dir=base_dir,
                    events=events,
                    force=force,
                )
                active_logger.info("Executing %s", " ".join(command))
                subprocess.run(command, check=True, cwd=run_cwd)

        export_config = _build_method_specific_config(
            loaded_config,
            method_key=requested_methods[0],
            comparison_methods=requested_methods,
        )
        export_config_path = temp_dir / "method_comparison.yaml"
        export_config_path.write_text(
            yaml.safe_dump(export_config, sort_keys=False),
            encoding="utf-8",
        )
        export_command = [
            sys.executable,
            str(script_dir / "export_source_location_method_comparison.py"),
            str(export_config_path),
        ]
        if base_dir is not None:
            export_command.extend(["--base-dir", str(base_dir)])
        active_logger.info("Executing %s", " ".join(export_command))
        subprocess.run(export_command, check=True, cwd=run_cwd)

    return {
        "requested_methods": requested_methods,
        "execution_methods": execution_methods,
    }
