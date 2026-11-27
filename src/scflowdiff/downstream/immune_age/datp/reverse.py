"""Serialized scaler type used by earlier ATAC-to-RNA DATP bundles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-8


@dataclass
class ArrayScaler:
    mean: np.ndarray
    sd: np.ndarray
    keep: np.ndarray
    clip: float = 8.0

    @classmethod
    def fit(cls, values: np.ndarray, clip: float = 8.0) -> ArrayScaler:
        values = np.asarray(values, dtype=np.float64)
        mean = np.nanmean(values, axis=0)
        sd = np.nanstd(values, axis=0)
        keep = np.isfinite(mean) & np.isfinite(sd) & (sd >= EPS)
        if not keep.any():
            raise RuntimeError("All reverse DATP context features failed variance filtering")
        return cls(mean=mean, sd=sd, keep=keep, clip=float(clip))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        result = (values[:, self.keep] - self.mean[self.keep]) / self.sd[self.keep]
        result = np.clip(result, -self.clip, self.clip)
        if not np.isfinite(result).all():
            raise RuntimeError("Non-finite reverse DATP context")
        return result
