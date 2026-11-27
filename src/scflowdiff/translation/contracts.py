"""Configuration contract for paired-cell translation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def validate_translation_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("task") != "translation":
        raise ValueError("Translation config requires task: translation")
    required = {"dataset_path", "output_root", "seed", "split", "stage1", "stage2"}
    if missing := required.difference(config):
        raise ValueError(f"Translation config is missing {sorted(missing)}")
    for stage in ("stage1", "stage2"):
        settings = config[stage]
        for key in ("max_steps", "batch_size", "validation_every"):
            if int(settings.get(key, 0)) <= 0:
                raise ValueError(f"{stage}.{key} must be positive")
    probabilities = config["stage2"].get("task_probabilities", {})
    if (
        set(probabilities) != {"rna_to_atac", "atac_to_rna", "joint"}
        or abs(sum(float(value) for value in probabilities.values()) - 1.0) > 1e-9
    ):
        raise ValueError("Stage 2 task probabilities must contain both directions and sum to 1")
    split = config.get("split", {})
    if split.get("strategy") != "random_cell":
        raise ValueError("Translation requires a random_cell split")
    fractions = [float(split.get(key, -1)) for key in ("train", "validation", "test")]
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"Invalid translation split fractions: {fractions}")
    directions = set(config.get("directions", []))
    if directions != {"rna_to_atac", "atac_to_rna"}:
        raise ValueError("Translation must enable both RNA-to-ATAC and ATAC-to-RNA")
    return config


def load_translation_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return validate_translation_config(payload)
