"""Configuration and Stage-1/Stage-2 orchestration for immune-age runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_immune_age_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("task") != "immune_age":
        raise ValueError("Immune-age config requires task: immune_age")
    required = {
        "dataset_path",
        "data_root",
        "output_root",
        "clock_root",
        "seed",
        "scflowdiff",
        "split",
        "sexes",
        "directions",
        "external_cohorts",
        "flow_latent",
        "rna_counts_path",
        "datp",
    }
    if missing := required.difference(payload):
        raise ValueError(f"Immune-age config is missing {sorted(missing)}")
    model = payload["scflowdiff"]
    for key in ("stage1_steps", "stage2_steps", "batch_size"):
        if int(model.get(key, 0)) <= 0:
            raise ValueError(f"scflowdiff.{key} must be positive")
    split = payload.get("split", {})
    if split.get("strategy") != "frozen_donor":
        raise ValueError("Immune-age requires the frozen_donor split")
    if set(payload.get("directions", {})) != {"rna_to_atac", "atac_to_rna"}:
        raise ValueError("Both downstream directions must be configured")
    return payload


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_scflowdiff_environment(
    config: dict[str, Any], repository_root: str | Path, sex: str
) -> dict[str, str]:
    root = Path(repository_root).resolve()
    sex_config = config["sexes"][sex]
    data = _resolve(root, config["dataset_path"]).resolve()
    split = _resolve(root, sex_config["split_csv"]).resolve()
    feature = _resolve(root, sex_config["feature_contract"]).resolve()
    output = _resolve(root, config["output_root"]).resolve() / "shared" / "scflowdiff" / sex
    shared = {
        "SCFLOW_RNA_COUNTS_H5AD": str(_resolve(root, config["rna_counts_path"]).resolve()),
        "SCFLOW_SEED": "42",
        "SCFLOW_DONOR_SPLIT_CSV": str(split),
        "SCFLOW_DONOR_SPLIT_COLUMN": "benchmark_split",
        "SCFLOW_DONOR_TRAIN_LABEL": "train",
        "SCFLOW_DONOR_VALIDATION_LABEL": "validation",
        "SCFLOW_DONOR_TEST_LABEL": "sealed_test",
        "SCFLOW_FEATURE_CONTRACT": str(feature),
        "SCFLOW_SPLIT_MANIFEST": str(output / "contracts" / "split_manifest.json"),
    }
    return {
        **shared,
        "MMAE_H5MU": str(data),
        "MMAE_OUT_DIR": str(output / "stage1"),
        "MMAE_METRICS_PATH": str(output / "stage1" / "training_metrics.json"),
        "MMAE_MAX_STEPS": str(config["scflowdiff"]["stage1_steps"]),
        "MMAE_SEED": str(config["seed"]),
        "MMAE_BATCH_SIZE": str(config["scflowdiff"]["batch_size"]),
        "MMAE_PRECISION": str(config["scflowdiff"]["precision"]),
        "MMAE_EVAL_EVERY": str(config["scflowdiff"]["stage1_validation_every"]),
        "MMAE_EVAL_BATCHES_DURING_TRAIN": str(
            config["scflowdiff"]["stage1_validation_max_batches"]
        ),
        "MMAE_FINAL_EVAL_BATCHES": "0",
        "MMFLOW_H5MU": str(data),
        "MMFLOW_AE_CKPT": str(output / "stage1" / "best.ckpt"),
        "MMFLOW_OUT_DIR": str(output / "stage2"),
        "MMFLOW_LOG_JSON": str(output / "stage2" / "training_metrics.json"),
        "MMFLOW_MAX_STEPS": str(config["scflowdiff"]["stage2_steps"]),
        "MMFLOW_SEED": str(config["seed"]),
        "MMFLOW_BATCH_SIZE": str(config["scflowdiff"]["batch_size"]),
        "MMFLOW_VAL_EVERY": str(config["scflowdiff"]["stage2_validation_every"]),
        "MMFLOW_VAL_MAX_BATCHES": str(config["scflowdiff"]["stage2_validation_max_batches"]),
        "MMFLOW_RNA_TO_ATAC_PROB": str(config["scflowdiff"]["task_probabilities"]["rna_to_atac"]),
        "MMFLOW_ATAC_TO_RNA_PROB": str(config["scflowdiff"]["task_probabilities"]["atac_to_rna"]),
        "MMFLOW_JOINT_PROB": str(config["scflowdiff"]["task_probabilities"]["joint"]),
        "MMFLOW_SELECTION_RNA_TO_ATAC_WEIGHT": str(
            config["scflowdiff"]["checkpoint_selection"]["rna_to_atac_weight"]
        ),
        "MMFLOW_SELECTION_ATAC_TO_RNA_WEIGHT": str(
            config["scflowdiff"]["checkpoint_selection"]["atac_to_rna_weight"]
        ),
        "MMFLOW_SELECTION_MMD_CELLS": str(
            config["scflowdiff"]["checkpoint_selection"]["mmd_cells"]
        ),
        "MMFLOW_SELECTION_MMD_STEPS": str(
            config["scflowdiff"]["checkpoint_selection"]["mmd_steps"]
        ),
        "MMFLOW_SELECTION_MMD_WEIGHT": str(
            config["scflowdiff"]["checkpoint_selection"]["mmd_weight"]
        ),
    }


def run_scflowdiff(
    config_path: str | Path,
    *,
    repository_root: str | Path,
    sex: str,
    stage: str = "both",
) -> None:
    if sex not in {"female", "male"}:
        raise ValueError("sex must be female or male")
    if stage not in {"stage1", "stage2", "both"}:
        raise ValueError("stage must be stage1, stage2, or both")
    config = load_immune_age_config(config_path)
    environment = resolve_scflowdiff_environment(config, repository_root, sex)
    old = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        if stage in {"stage1", "both"}:
            from scflowdiff.training.stage1 import main as stage1_main

            stage1_main()
        if stage in {"stage2", "both"}:
            from scflowdiff.training.stage2 import main as stage2_main

            stage2_main()
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
