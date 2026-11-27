"""Inference with trained sex- and direction-specific DATP bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .context import transform_design
from .directions import ATAC_TO_RNA, RNA_TO_ATAC, DirectionSpec, get_direction
from .features import transform_features


def load_datp(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(Path(path))
    required = {"states", "target_order", "architecture", "sex"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Invalid DATP bundle; missing {sorted(missing)}")
    bundle.setdefault("direction", "rna_to_atac")
    get_direction(bundle["direction"])
    return bundle


def load_flow_latent(path: str | Path, *, direction: str):
    """Flatten named endpoint token slots only after donor-state pooling."""
    with np.load(Path(path), allow_pickle=False) as payload:
        if payload["direction"].item() != direction:
            raise ValueError("Flow endpoint direction differs from DATP direction")
        latent = np.asarray(payload["target_latent"], dtype=np.float64)
        donors = payload["sample_id"].astype(str)
        states = payload["ct_id"]
    lookup = {(d, int(k)): i for i, (d, k) in enumerate(zip(donors, states, strict=True))}
    if (
        latent.ndim != 3
        or len(lookup) != len(donors)
        or len(donors) != len(latent)
        or not np.isfinite(latent).all()
    ):
        raise ValueError("Invalid donor-state endpoint latents")
    return latent.reshape(len(donors), latent.shape[1] * latent.shape[2]), lookup


def predict_targeted_modality(
    bundle: dict[str, Any],
    source_features: pd.DataFrame,
    flow_latent: np.ndarray,
    latent_lookup: dict[tuple[str, int], int],
    *,
    direction: str | DirectionSpec | None = None,
) -> pd.DataFrame:
    """Predict the contracted target without reading query targets or ages."""
    requested = get_direction(direction or bundle.get("direction", "rna_to_atac"))
    stored = get_direction(bundle.get("direction", "rna_to_atac"))
    if requested.name != stored.name:
        raise ValueError(
            f"DATP direction mismatch: bundle={stored.name}, requested={requested.name}"
        )
    source_features = source_features.copy()
    source_features.index = source_features.index.astype(str)
    donors = source_features.index.tolist()
    output = pd.DataFrame(index=source_features.index, columns=bundle["target_order"], dtype=float)
    donor_position = {donor: i for i, donor in enumerate(donors)}
    for state in bundle["states"]:
        cell_type_id = int(state["ct_id"])
        available = [donor for donor in donors if (donor, cell_type_id) in latent_lookup]
        if stored.name == ATAC_TO_RNA.name:
            output.loc[:, state["columns"]] = np.broadcast_to(
                state["observed_target_median"], (len(donors), len(state["columns"]))
            )
        if not available:
            continue
        context = transform_design(source_features, state["design_state"])
        flow_rows = np.asarray([latent_lookup[(donor, cell_type_id)] for donor in available])
        context_rows = np.asarray([donor_position[donor] for donor in available])
        features = transform_features(
            flow_latent[flow_rows], context[context_rows], state["feature_state"]
        )
        pred_z = np.asarray(state["model"].predict(features), dtype=np.float64)
        pred = state["target_scaler"]["mean"] + pred_z * state["target_scaler"]["sd"]
        output.loc[available, state["columns"]] = pred
    if np.isinf(output.to_numpy(float)).any():
        raise RuntimeError("Infinite DATP predictions")
    return output


def predict_targeted_atac(
    bundle: dict[str, Any],
    rna_features: pd.DataFrame,
    flow_latent: np.ndarray,
    latent_lookup: dict[tuple[str, int], int],
) -> pd.DataFrame:
    return predict_targeted_modality(
        bundle,
        rna_features,
        flow_latent,
        latent_lookup,
        direction=RNA_TO_ATAC,
    )


def predict_targeted_rna(
    bundle: dict[str, Any],
    atac_features: pd.DataFrame,
    flow_latent: np.ndarray,
    latent_lookup: dict[tuple[str, int], int],
) -> pd.DataFrame:
    return predict_targeted_modality(
        bundle,
        atac_features,
        flow_latent,
        latent_lookup,
        direction=ATAC_TO_RNA,
    )
