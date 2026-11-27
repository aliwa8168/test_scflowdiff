"""DATP Flow-latent and donor-context feature preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_feature_state(
    flow_latent: np.ndarray,
    glr: np.ndarray,
    *,
    pca_components: int = 32,
    whiten: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    flow_latent = np.asarray(flow_latent, dtype=np.float64)
    glr = np.asarray(glr, dtype=np.float64)
    if len(flow_latent) != len(glr) or len(flow_latent) < 3:
        raise ValueError(
            "Flow-latent and source-context rows must align and contain at least three donors"
        )
    flow_scaler = StandardScaler().fit(flow_latent)
    flow_z = flow_scaler.transform(flow_latent)
    n_components = min(int(pca_components), flow_z.shape[1], len(flow_z) - 1)
    flow_pca = PCA(n_components=n_components, whiten=whiten, svd_solver="full", random_state=seed)
    flow_pca.fit(flow_z)
    state = {
        "flow_scaler": flow_scaler,
        "flow_pca": flow_pca,
        "glr_scaler": StandardScaler().fit(glr),
        "flow_pca_components": n_components,
        "fused_dim": n_components + glr.shape[1],
    }
    return state


def transform_features(
    flow_latent: np.ndarray, glr: np.ndarray, state: dict[str, Any]
) -> np.ndarray:
    flow_pc = state["flow_pca"].transform(
        state["flow_scaler"].transform(np.asarray(flow_latent, dtype=np.float64))
    )
    glr_z = state["glr_scaler"].transform(np.asarray(glr, dtype=np.float64))
    result = np.concatenate([flow_pc, glr_z], axis=1)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite DATP features")
    return result
