import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from scflowdiff.core.cima_inputs import canonical_cell_types, check_paired_rows, read_rna_counts
from scflowdiff.core.constants import ModelEnum


class H5CSRRowMatrix:
    """Read one CSR row at a time from an H5MU modality without materializing it."""

    def __init__(self, path: Path, group_path: str):
        import h5py

        self.path = Path(path)
        self.group_path = group_path.strip("/")
        with h5py.File(self.path, "r") as handle:
            group = handle[self.group_path]
            self.shape = tuple(int(x) for x in group.attrs["shape"])
        self._handle: Any | None = None

    def _group(self) -> Any:
        import h5py

        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle[self.group_path]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def getrow(self, idx: int) -> Any:
        from scipy import sparse

        if not 0 <= int(idx) < self.shape[0]:
            raise IndexError(f"row index {idx} is outside {self.shape}")
        group = self._group()
        start = int(group["indptr"][idx])
        stop = int(group["indptr"][idx + 1])
        return sparse.csr_matrix(
            (
                np.asarray(group["data"][start:stop], dtype=np.float32),
                np.asarray(group["indices"][start:stop], dtype=np.int64),
                np.asarray([0, stop - start], dtype=np.int64),
            ),
            shape=(1, self.shape[1]),
        )

    def row_sum(self, idx: int) -> float:
        group = self._group()
        start = int(group["indptr"][idx])
        stop = int(group["indptr"][idx + 1])
        return float(np.asarray(group["data"][start:stop], dtype=np.float64).sum())


class MatrixView:
    """Row/column view over an in-memory or streamed sparse matrix."""

    def __init__(
        self,
        matrix: Any,
        *,
        rows: np.ndarray | None = None,
        columns: np.ndarray | None = None,
    ):
        self.matrix = matrix
        self.rows = (
            np.arange(matrix.shape[0], dtype=np.int64)
            if rows is None
            else np.asarray(rows, dtype=np.int64)
        )
        self.columns = (
            np.arange(matrix.shape[1], dtype=np.int64)
            if columns is None
            else np.asarray(columns, dtype=np.int64)
        )
        if len(np.unique(self.rows)) != len(self.rows):
            raise ValueError("MatrixView row indices must be unique")
        if len(np.unique(self.columns)) != len(self.columns):
            raise ValueError("MatrixView column indices must be unique")
        if len(self.rows) and (self.rows.min() < 0 or self.rows.max() >= matrix.shape[0]):
            raise IndexError("MatrixView row index is out of bounds")
        if len(self.columns) and (self.columns.min() < 0 or self.columns.max() >= matrix.shape[1]):
            raise IndexError("MatrixView column index is out of bounds")
        self.shape = (len(self.rows), len(self.columns))

    def getrow(self, idx: int) -> Any:
        if not 0 <= int(idx) < self.shape[0]:
            raise IndexError(f"row index {idx} is outside {self.shape}")
        physical = int(self.rows[int(idx)])
        if hasattr(self.matrix, "getrow"):
            return self.matrix.getrow(physical)[:, self.columns]
        from scipy import sparse

        return sparse.csr_matrix(np.asarray(self.matrix[physical, self.columns]).reshape(1, -1))

    def row_sum(self, idx: int) -> float:
        return float(self.getrow(idx).sum())


def _to_dense_float32(x: Any) -> np.ndarray:
    """Convert dense or sparse single-cell matrices to float32 NumPy arrays."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}")
    return arr


def _tokenize_binary_or_count_matrix(
    matrix: np.ndarray,
    token_ids: np.ndarray,
    seq_len: int | None,
    pad_token_idx: int,
    sample: Literal["expressed", "random", "none", "balanced", "top"] = "expressed",
    seed: int = 42,
    binarize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize each row into fixed-length feature/value arrays.

    For sparse modalities, non-zero features are emitted first and padded with
    ``pad_token_idx``. If a row has more non-zero features than ``seq_len``, a
    reproducible subset is sampled. This keeps the transformer sequence length
    bounded while preserving paired feature tokens and observed values.
    """
    values = (matrix > 0).astype(np.float32) if binarize else matrix.astype(np.float32)
    n_cells, n_features = values.shape
    if seq_len is None or seq_len <= 0:
        seq_len = n_features
    if token_ids.shape[0] != n_features:
        raise ValueError("token_ids length must match the matrix feature dimension")

    rng = np.random.default_rng(seed)
    out_tokens = np.full((n_cells, seq_len), pad_token_idx, dtype=np.int64)
    out_values = np.zeros((n_cells, seq_len), dtype=np.float32)

    for i in range(n_cells):
        if sample == "none":
            idx = np.arange(n_features)
        elif sample == "random":
            idx = rng.choice(n_features, size=min(seq_len, n_features), replace=False)
        elif sample == "expressed":
            idx = np.flatnonzero(values[i] > 0)
            if len(idx) == 0:
                idx = np.arange(min(seq_len, n_features))
            elif len(idx) > seq_len:
                idx = rng.choice(idx, size=seq_len, replace=False)
        elif sample == "top":
            idx = np.flatnonzero(values[i] > 0)
            if len(idx) == 0:
                idx = np.arange(min(seq_len, n_features))
            elif len(idx) > seq_len:
                top_order = np.argpartition(values[i, idx], -seq_len)[-seq_len:]
                idx = idx[top_order]
        elif sample == "balanced":
            positive_idx = np.flatnonzero(values[i] > 0)
            negative_idx = np.flatnonzero(values[i] <= 0)
            n_pos = min(len(positive_idx), max(1, seq_len // 2))
            n_neg = min(len(negative_idx), seq_len - n_pos)
            pos = (
                rng.choice(positive_idx, size=n_pos, replace=False)
                if n_pos > 0
                else np.empty(0, dtype=np.int64)
            )
            neg = (
                rng.choice(negative_idx, size=n_neg, replace=False)
                if n_neg > 0
                else np.empty(0, dtype=np.int64)
            )
            idx = np.concatenate([pos, neg])
            rng.shuffle(idx)
        else:
            raise ValueError(f"Unsupported multimodal sampling mode: {sample}")

        idx = np.asarray(idx[:seq_len], dtype=np.int64)
        length = len(idx)
        out_tokens[i, :length] = token_ids[idx]
        out_values[i, :length] = values[i, idx]

    return out_tokens, out_values


class MultimodalMuDataDataset(Dataset):
    """Paired RNA+ATAC token dataset backed by in-memory or row-streamed matrices."""

    def __init__(
        self,
        rna_matrix: Any,
        atac_matrix: Any,
        rna_seq_len: int | None,
        atac_seq_len: int | None,
        sample_features: Literal["expressed", "random", "none", "balanced", "top"] = "expressed",
        rna_sample_features: Literal["expressed", "random", "none", "balanced", "top"]
        | None = None,
        atac_sample_features: Literal["expressed", "random", "none", "balanced", "top"]
        | None = None,
        rna_encoder_seq_len: int | None = None,
        rna_encoder_sample_features: Literal["expressed", "random", "none", "balanced", "top"]
        | None = None,
        atac_encoder_seq_len: int | None = None,
        atac_encoder_sample_features: Literal["expressed", "random", "none", "balanced", "top"]
        | None = None,
        seed: int = 42,
        pad_token_idx: int = 0,
        cell_type: np.ndarray | None = None,
    ):
        if rna_matrix.shape[0] != atac_matrix.shape[0]:
            raise ValueError("RNA and ATAC matrices must contain the same number of cells")
        if len(rna_matrix.shape) != 2 or len(atac_matrix.shape) != 2:
            raise ValueError("RNA and ATAC matrices must both be 2D")
        self.rna_matrix = rna_matrix
        self.atac_matrix = atac_matrix
        self.rna_token_ids = np.arange(1, self.rna_matrix.shape[1] + 1, dtype=np.int64)
        self.atac_token_ids = np.arange(1, self.atac_matrix.shape[1] + 1, dtype=np.int64)
        self.pad_token_idx = pad_token_idx
        self.rna_seq_len = (
            self.rna_matrix.shape[1] if rna_seq_len is None or rna_seq_len <= 0 else rna_seq_len
        )
        self.atac_seq_len = (
            self.atac_matrix.shape[1] if atac_seq_len is None or atac_seq_len <= 0 else atac_seq_len
        )
        self.rna_sample_features = rna_sample_features or sample_features
        self.atac_sample_features = atac_sample_features or sample_features
        self.rna_encoder_seq_len = rna_encoder_seq_len
        self.rna_encoder_sample_features = rna_encoder_sample_features
        self.atac_encoder_seq_len = atac_encoder_seq_len
        self.atac_encoder_sample_features = atac_encoder_sample_features
        self.seed = seed
        self.rna_library_size = None
        if not hasattr(self.rna_matrix, "row_sum"):
            self.rna_library_size = np.asarray(
                self.rna_matrix.sum(axis=1), dtype=np.float32
            ).reshape(-1, 1)
            self.rna_library_size = np.maximum(self.rna_library_size, 1.0)
        self.cell_type = None if cell_type is None else np.asarray(cell_type, dtype=np.int64)
        if self.cell_type is not None and self.cell_type.shape[0] != self.rna_matrix.shape[0]:
            raise ValueError(
                "cell_type labels must have the same number of rows as RNA/ATAC matrices"
            )

    @property
    def n_genes(self) -> int:
        return int(self.rna_matrix.shape[1])

    @property
    def n_peaks(self) -> int:
        return int(self.atac_matrix.shape[1])

    def __len__(self) -> int:
        return int(self.rna_matrix.shape[0])

    def _tokenize_row(
        self,
        matrix: Any,
        idx: int,
        seq_len: int | None,
        token_ids: np.ndarray,
        binarize: bool,
        seed_offset: int,
        sample_features: Literal["expressed", "random", "none", "balanced", "top"],
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + seed_offset + idx)
        n_features = matrix.shape[1]
        if seq_len is None or seq_len <= 0:
            seq_len = n_features
        if hasattr(matrix, "getrow"):
            row = matrix.getrow(idx)
            nonzero_idx = row.indices.astype(np.int64, copy=False)
            row_values = row.data.astype(np.float32, copy=False)
            value_lookup = dict(zip(nonzero_idx.tolist(), row_values.tolist(), strict=False))
        else:
            dense_row = np.asarray(matrix[idx], dtype=np.float32).reshape(-1)
            nonzero_idx = np.flatnonzero(dense_row > 0)
            value_lookup = None

        if sample_features == "none":
            selected_idx = np.arange(min(seq_len, n_features), dtype=np.int64)
        elif sample_features == "random":
            selected_idx = rng.choice(
                n_features, size=min(seq_len, n_features), replace=False
            ).astype(np.int64)
        elif sample_features == "expressed":
            if len(nonzero_idx) == 0:
                selected_idx = np.arange(min(seq_len, n_features), dtype=np.int64)
            elif len(nonzero_idx) > seq_len:
                selected_idx = rng.choice(nonzero_idx, size=seq_len, replace=False).astype(np.int64)
            else:
                selected_idx = nonzero_idx[:seq_len].astype(np.int64, copy=False)
        elif sample_features == "top":
            if len(nonzero_idx) == 0:
                selected_idx = np.arange(min(seq_len, n_features), dtype=np.int64)
            elif len(nonzero_idx) > seq_len:
                if value_lookup is None:
                    values_for_rank = dense_row[nonzero_idx]
                else:
                    values_for_rank = row_values
                top_local = np.argpartition(values_for_rank, -seq_len)[-seq_len:]
                selected_idx = nonzero_idx[top_local].astype(np.int64, copy=False)
            else:
                selected_idx = nonzero_idx[:seq_len].astype(np.int64, copy=False)
        elif sample_features == "balanced":
            all_idx = np.arange(n_features, dtype=np.int64)
            if len(nonzero_idx) == 0:
                selected_idx = rng.choice(
                    all_idx, size=min(seq_len, n_features), replace=False
                ).astype(np.int64)
            else:
                zero_mask = np.ones(n_features, dtype=bool)
                zero_mask[nonzero_idx] = False
                zero_idx = np.flatnonzero(zero_mask).astype(np.int64, copy=False)
                n_pos = min(len(nonzero_idx), max(1, seq_len // 2))
                n_neg = min(len(zero_idx), seq_len - n_pos)
                pos = rng.choice(nonzero_idx, size=n_pos, replace=False).astype(np.int64)
                neg = rng.choice(zero_idx, size=n_neg, replace=False).astype(np.int64)
                selected_idx = np.concatenate([pos, neg])
                if len(selected_idx) < min(seq_len, n_features):
                    remaining = np.setdiff1d(all_idx, selected_idx, assume_unique=False)
                    extra = rng.choice(
                        remaining,
                        size=min(seq_len, n_features) - len(selected_idx),
                        replace=False,
                    ).astype(np.int64)
                    selected_idx = np.concatenate([selected_idx, extra])
                rng.shuffle(selected_idx)
        else:
            raise ValueError(f"Unsupported multimodal sampling mode: {sample_features}")

        out_tokens = np.full(seq_len, self.pad_token_idx, dtype=np.int64)
        out_values = np.zeros(seq_len, dtype=np.float32)
        length = len(selected_idx)
        out_tokens[:length] = token_ids[selected_idx]

        if sample_features == "none" and value_lookup is not None:
            valid = nonzero_idx[nonzero_idx < length]
            if len(valid) > 0:
                values = row_values[nonzero_idx < length]
                out_values[valid] = values
            if binarize:
                out_values = (out_values > 0).astype(np.float32)
        elif value_lookup is None:
            values = dense_row[selected_idx]
            if binarize:
                values = (values > 0).astype(np.float32)
            out_values[:length] = values
        else:
            values = np.asarray(
                [value_lookup.get(int(j), 0.0) for j in selected_idx], dtype=np.float32
            )
            if binarize:
                values = (values > 0).astype(np.float32)
            out_values[:length] = values
        return out_tokens, out_values

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rna_genes, rna_counts = self._tokenize_row(
            self.rna_matrix,
            idx,
            self.rna_seq_len,
            self.rna_token_ids,
            binarize=False,
            seed_offset=0,
            sample_features=self.rna_sample_features,
        )
        atac_peaks, atac_values = self._tokenize_row(
            self.atac_matrix,
            idx,
            self.atac_seq_len,
            self.atac_token_ids,
            binarize=True,
            seed_offset=100_000,
            sample_features=self.atac_sample_features,
        )
        if self.rna_library_size is None:
            rna_library_size = np.asarray(
                [max(self.rna_matrix.row_sum(idx), 1.0)], dtype=np.float32
            )
        else:
            rna_library_size = self.rna_library_size[idx].astype(np.float32, copy=True)
        output = {
            ModelEnum.RNA_COUNTS.value: torch.from_numpy(rna_counts),
            ModelEnum.RNA_GENES.value: torch.from_numpy(rna_genes),
            ModelEnum.RNA_LIBRARY_SIZE.value: torch.from_numpy(rna_library_size),
            ModelEnum.ATAC_VALUES.value: torch.from_numpy(atac_values),
            ModelEnum.ATAC_PEAKS.value: torch.from_numpy(atac_peaks),
        }
        if self.rna_encoder_seq_len is not None and self.rna_encoder_sample_features is not None:
            rna_genes_subset, rna_counts_subset = self._tokenize_row(
                self.rna_matrix,
                idx,
                self.rna_encoder_seq_len,
                self.rna_token_ids,
                binarize=False,
                seed_offset=200_000,
                sample_features=self.rna_encoder_sample_features,
            )
            output[ModelEnum.RNA_COUNTS_SUBSET.value] = torch.from_numpy(rna_counts_subset)
            output[ModelEnum.RNA_GENES_SUBSET.value] = torch.from_numpy(rna_genes_subset)
        if self.atac_encoder_seq_len is not None and self.atac_encoder_sample_features is not None:
            atac_peaks_subset, atac_values_subset = self._tokenize_row(
                self.atac_matrix,
                idx,
                self.atac_encoder_seq_len,
                self.atac_token_ids,
                binarize=True,
                seed_offset=300_000,
                sample_features=self.atac_encoder_sample_features,
            )
            output[ModelEnum.ATAC_VALUES_SUBSET.value] = torch.from_numpy(atac_values_subset)
            output[ModelEnum.ATAC_PEAKS_SUBSET.value] = torch.from_numpy(atac_peaks_subset)
        if self.cell_type is not None:
            output[ModelEnum.CELL_TYPE.value] = torch.tensor(
                int(self.cell_type[idx]), dtype=torch.long
            )
        return output


class MultimodalMuDataModule(LightningDataModule):
    """Minimal MuData loader for paired RNA and binary ATAC transformer batches."""

    def __init__(
        self,
        h5mu_path: Path,
        batch_size: int = 32,
        test_batch_size: int | None = None,
        rna_seq_len: int | None = 512,
        atac_seq_len: int | None = 1024,
        sample_features: Literal["expressed", "random", "none", "balanced", "top"] = "expressed",
        rna_sample_features: Literal["expressed", "random", "none", "balanced", "top"]
        | None = None,
        atac_sample_features: Literal["expressed", "random", "none", "balanced", "top"]
        | None = "balanced",
        rna_encoder_seq_len: int | None = None,
        rna_encoder_sample_features: Literal["expressed", "random", "none", "balanced", "top"]
        | None = None,
        atac_encoder_seq_len: int | None = None,
        atac_encoder_sample_features: Literal["expressed", "random", "none", "balanced", "top"]
        | None = None,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        split_manifest_path: Path | None = None,
        donor_split_csv: Path | None = None,
        donor_key: str = "sample_id",
        donor_split_column: str | None = None,
        donor_train_label: str = "train",
        donor_validation_label: str = "validation",
        donor_test_label: str = "sealed_test",
        outer_train_label: str = "train",
        inner_val_fraction: float = 0.1,
        feature_contract_path: Path | None = None,
        seed: int = 42,
        num_workers: int = 0,
        rna_counts_path: str | Path | None = None,
    ):
        super().__init__()
        self.h5mu_path = Path(h5mu_path)
        self.rna_counts_path = Path(rna_counts_path) if rna_counts_path else None
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.rna_seq_len = rna_seq_len
        self.atac_seq_len = atac_seq_len
        self.sample_features = sample_features
        self.rna_sample_features = rna_sample_features
        self.atac_sample_features = atac_sample_features
        self.rna_encoder_seq_len = rna_encoder_seq_len
        self.rna_encoder_sample_features = rna_encoder_sample_features
        self.atac_encoder_seq_len = atac_encoder_seq_len
        self.atac_encoder_sample_features = atac_encoder_sample_features
        self.train_fraction = float(train_fraction)
        self.val_fraction = val_fraction
        self.test_fraction = float(test_fraction)
        self.split_manifest_path = (
            None if split_manifest_path is None else Path(split_manifest_path)
        )
        self.donor_split_csv = None if donor_split_csv is None else Path(donor_split_csv)
        self.donor_key = donor_key
        self.donor_split_column = donor_split_column
        self.donor_train_label = donor_train_label
        self.donor_validation_label = donor_validation_label
        self.donor_test_label = donor_test_label
        self.outer_train_label = outer_train_label
        self.inner_val_fraction = float(inner_val_fraction)
        self.feature_contract_path = (
            None if feature_contract_path is None else Path(feature_contract_path)
        )
        self.feature_contract: dict[str, Any] | None = None
        self.seed = seed
        self.num_workers = num_workers
        self.n_genes: int | None = None
        self.n_peaks: int | None = None
        self.num_cell_types: int | None = None
        self.cell_type_mapping: dict[str, int] = {}
        self.split_info: dict[str, Any] = {}

        if self.donor_split_csv is None:
            fractions = np.asarray(
                [self.train_fraction, self.val_fraction, self.test_fraction],
                dtype=np.float64,
            )
            if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
                raise ValueError(
                    "train_fraction, val_fraction, and test_fraction must be "
                    f"positive and sum to 1.0, got {fractions.tolist()}"
                )
        elif self.donor_split_column is None and not 0.0 < self.inner_val_fraction < 1.0:
            raise ValueError("inner_val_fraction must be between 0 and 1")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _create_donor_split(
        self,
        donor_ids: np.ndarray,
        n_genes: int,
        n_peaks: int,
    ) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
        split_csv = self.donor_split_csv
        if split_csv is None or not split_csv.exists():
            raise FileNotFoundError(f"Donor split CSV not found: {split_csv}")
        with split_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        split_column = self.donor_split_column or "split"
        required = {self.donor_key, split_column}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Donor split CSV must contain columns {sorted(required)}")
        if len({row[self.donor_key] for row in rows}) != len(rows):
            raise ValueError(f"Donor split CSV contains duplicate {self.donor_key} values")

        if self.donor_split_column is not None:
            label_sets = {
                "train": {
                    row[self.donor_key]
                    for row in rows
                    if row[split_column] == self.donor_train_label
                },
                "validation": {
                    row[self.donor_key]
                    for row in rows
                    if row[split_column] == self.donor_validation_label
                },
                "test": {
                    row[self.donor_key]
                    for row in rows
                    if row[split_column] == self.donor_test_label
                },
            }
            if any(not values for values in label_sets.values()):
                raise ValueError(
                    f"Frozen donor split has an empty group: "
                    f"{ {key: len(value) for key, value in label_sets.items()} }"
                )
            if (
                label_sets["train"] & label_sets["validation"]
                or label_sets["train"] & label_sets["test"]
                or label_sets["validation"] & label_sets["test"]
            ):
                raise ValueError("Frozen donor groups overlap")
            observed_donors = set(map(str, donor_ids.tolist()))
            required_donors = set().union(*label_sets.values())
            missing = required_donors - observed_donors
            if missing:
                raise ValueError(
                    f"H5MU is missing {len(missing)} frozen-split donors; "
                    f"examples={sorted(missing)[:5]}"
                )
            donor_values = np.asarray(donor_ids, dtype=str)
            train_indices = (
                np.flatnonzero(np.isin(donor_values, list(label_sets["train"])))
                .astype(int)
                .tolist()
            )
            val_indices = (
                np.flatnonzero(np.isin(donor_values, list(label_sets["validation"])))
                .astype(int)
                .tolist()
            )
            test_indices = (
                np.flatnonzero(np.isin(donor_values, list(label_sets["test"]))).astype(int).tolist()
            )
            excluded_indices = (
                np.flatnonzero(~np.isin(donor_values, list(required_donors))).astype(int).tolist()
            )
            manifest = {
                "version": 3,
                "dataset": str(self.h5mu_path.resolve()),
                "seed": self.seed,
                "split_strategy": "frozen_donor_train_validation_sealed_test",
                "donor_key": self.donor_key,
                "split_column": split_column,
                "train_label": self.donor_train_label,
                "validation_label": self.donor_validation_label,
                "test_label": self.donor_test_label,
                "outer_split_csv": str(split_csv.resolve()),
                "outer_split_sha256": self._sha256(split_csv),
                "fractions": {"train": None, "validation": None, "test": None},
                "n_total": len(donor_ids),
                "n_train": len(train_indices),
                "n_val": len(val_indices),
                "n_test": len(test_indices),
                "n_excluded": len(excluded_indices),
                "n_rna_features": n_genes,
                "n_atac_features": n_peaks,
                "n_train_donors": len(label_sets["train"]),
                "n_validation_donors": len(label_sets["validation"]),
                "n_test_donors": len(label_sets["test"]),
                "train_donors": sorted(label_sets["train"]),
                "validation_donors": sorted(label_sets["validation"]),
                "test_donors": sorted(label_sets["test"]),
                "cell_sampling_applied": False,
                "feature_sampling_applied": self.feature_contract_path is not None,
                "feature_contract": (
                    None
                    if self.feature_contract_path is None
                    else str(self.feature_contract_path.resolve())
                ),
                "feature_contract_sha256": (
                    None
                    if self.feature_contract_path is None
                    else self._sha256(self.feature_contract_path)
                ),
                "test_is_validation": False,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "test_indices": test_indices,
                "excluded_indices": excluded_indices,
            }
            return train_indices, val_indices, test_indices, manifest

        outer_train_rows = [row for row in rows if row["split"] == self.outer_train_label]
        outer_train_donors = {row[self.donor_key] for row in outer_train_rows}
        outer_excluded_donors = {row[self.donor_key] for row in rows} - outer_train_donors
        observed_donors = set(map(str, donor_ids.tolist()))
        missing_train = outer_train_donors - observed_donors
        if missing_train:
            raise ValueError(
                f"Capped training H5MU is missing {len(missing_train)} outer-train donors; "
                f"examples={sorted(missing_train)[:5]}"
            )

        strata: dict[str, list[str]] = {}
        for row in outer_train_rows:
            stratum = row.get("age_bin") or "all"
            strata.setdefault(stratum, []).append(row[self.donor_key])
        rng = np.random.default_rng(self.seed)
        inner_val_donors: set[str] = set()
        for stratum in sorted(strata):
            values = np.asarray(sorted(strata[stratum]), dtype=object)
            rng.shuffle(values)
            n_val = max(1, int(round(len(values) * self.inner_val_fraction)))
            if n_val >= len(values) and len(values) > 1:
                n_val = len(values) - 1
            inner_val_donors.update(map(str, values[:n_val].tolist()))
        inner_train_donors = outer_train_donors - inner_val_donors
        if not inner_train_donors or not inner_val_donors:
            raise ValueError("Donor-level inner train/validation split is empty")

        donor_ids_str = np.asarray(donor_ids, dtype=str)
        train_indices = (
            np.flatnonzero(np.isin(donor_ids_str, list(inner_train_donors))).astype(int).tolist()
        )
        val_indices = (
            np.flatnonzero(np.isin(donor_ids_str, list(inner_val_donors))).astype(int).tolist()
        )
        excluded_indices = (
            np.flatnonzero(~np.isin(donor_ids_str, list(outer_train_donors))).astype(int).tolist()
        )
        if not train_indices or not val_indices:
            raise ValueError("No cells were assigned to donor-level train or validation")
        manifest = {
            "version": 2,
            "dataset": str(self.h5mu_path.resolve()),
            "seed": self.seed,
            "split_strategy": "outer_train_only_donor_stratified_inner_validation",
            "donor_key": self.donor_key,
            "outer_train_label": self.outer_train_label,
            "outer_split_csv": str(split_csv.resolve()),
            "outer_split_sha256": self._sha256(split_csv),
            "inner_val_fraction": self.inner_val_fraction,
            "fractions": {
                "inner_train": 1.0 - self.inner_val_fraction,
                "inner_validation": self.inner_val_fraction,
                "test": "validation_alias",
            },
            "n_total": len(donor_ids),
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "n_test": len(val_indices),
            "n_excluded": len(excluded_indices),
            "n_rna_features": n_genes,
            "n_atac_features": n_peaks,
            "n_outer_train_donors": len(outer_train_donors),
            "n_outer_excluded_donors": len(outer_excluded_donors),
            "n_inner_train_donors": len(inner_train_donors),
            "n_inner_val_donors": len(inner_val_donors),
            "outer_train_donors": sorted(outer_train_donors),
            "outer_excluded_donors": sorted(outer_excluded_donors),
            "inner_train_donors": sorted(inner_train_donors),
            "inner_val_donors": sorted(inner_val_donors),
            "cell_sampling_applied": False,
            "feature_sampling_applied": False,
            "test_is_validation": True,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": val_indices,
            "excluded_indices": excluded_indices,
        }
        return train_indices, val_indices, val_indices, manifest

    def _load_or_create_split(
        self,
        *,
        n_cells: int,
        n_genes: int,
        n_peaks: int,
        donor_ids: np.ndarray | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        manifest_path = self.split_manifest_path
        if manifest_path is not None and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "n_total": n_cells,
                "n_rna_features": n_genes,
                "n_atac_features": n_peaks,
                "seed": self.seed,
            }
            observed = {key: manifest.get(key) for key in expected}
            if observed != expected:
                raise ValueError(
                    f"Split manifest does not match the current dataset: {observed} != {expected}"
                )
            if self.donor_split_csv is not None:
                if self.donor_split_column is None:
                    donor_expected = {
                        "split_strategy": "outer_train_only_donor_stratified_inner_validation",
                        "outer_split_sha256": self._sha256(self.donor_split_csv),
                        "donor_key": self.donor_key,
                        "outer_train_label": self.outer_train_label,
                        "inner_val_fraction": self.inner_val_fraction,
                    }
                else:
                    donor_expected = {
                        "split_strategy": "frozen_donor_train_validation_sealed_test",
                        "outer_split_sha256": self._sha256(self.donor_split_csv),
                        "donor_key": self.donor_key,
                        "split_column": self.donor_split_column,
                        "train_label": self.donor_train_label,
                        "validation_label": self.donor_validation_label,
                        "test_label": self.donor_test_label,
                    }
                donor_observed = {key: manifest.get(key) for key in donor_expected}
                if donor_observed != donor_expected:
                    raise ValueError(
                        "Existing split manifest violates the donor-isolation contract: "
                        f"{donor_observed} != {donor_expected}"
                    )
            train_indices = [int(x) for x in manifest["train_indices"]]
            val_indices = [int(x) for x in manifest["val_indices"]]
            test_indices = [int(x) for x in manifest["test_indices"]]
            if self.donor_split_csv is not None:
                if donor_ids is None:
                    raise ValueError("donor_ids are required to validate donor-level indices")
                donor_values = np.asarray(donor_ids, dtype=str)
                observed_train_donors = set(donor_values[train_indices].tolist())
                observed_val_donors = set(donor_values[val_indices].tolist())
                if self.donor_split_column is None:
                    if observed_train_donors != set(manifest["inner_train_donors"]):
                        raise ValueError(
                            "Stored training indices no longer match inner_train_donors"
                        )
                    if observed_val_donors != set(manifest["inner_val_donors"]):
                        raise ValueError(
                            "Stored validation indices no longer match inner_val_donors"
                        )
                else:
                    observed_test_donors = set(donor_values[test_indices].tolist())
                    if observed_train_donors != set(manifest["train_donors"]):
                        raise ValueError("Stored training indices no longer match train_donors")
                    if observed_val_donors != set(manifest["validation_donors"]):
                        raise ValueError(
                            "Stored validation indices no longer match validation_donors"
                        )
                    if observed_test_donors != set(manifest["test_donors"]):
                        raise ValueError("Stored test indices no longer match test_donors")
        else:
            if self.donor_split_csv is not None:
                if donor_ids is None:
                    raise ValueError("donor_ids are required for donor-level splitting")
                train_indices, val_indices, test_indices, manifest = self._create_donor_split(
                    donor_ids, n_genes, n_peaks
                )
                if manifest_path is not None:
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            else:
                rng = np.random.default_rng(self.seed)
                permutation = rng.permutation(n_cells)
                n_train = int(round(n_cells * self.train_fraction))
                n_val = int(round(n_cells * self.val_fraction))
                n_test = n_cells - n_train - n_val
                if min(n_train, n_val, n_test) <= 0:
                    raise ValueError(
                        f"Dataset with {n_cells} cells is too small for "
                        f"{self.train_fraction}/{self.val_fraction}/{self.test_fraction}"
                    )
                train_indices = permutation[:n_train].astype(int).tolist()
                val_indices = permutation[n_train : n_train + n_val].astype(int).tolist()
                test_indices = permutation[n_train + n_val :].astype(int).tolist()

                manifest = {
                    "version": 1,
                    "dataset": str(self.h5mu_path.resolve()),
                    "seed": self.seed,
                    "fractions": {
                        "train": self.train_fraction,
                        "validation": self.val_fraction,
                        "test": self.test_fraction,
                    },
                    "n_total": n_cells,
                    "n_train": len(train_indices),
                    "n_val": len(val_indices),
                    "n_test": len(test_indices),
                    "n_rna_features": n_genes,
                    "n_atac_features": n_peaks,
                    "cell_sampling_applied": False,
                    "feature_sampling_applied": False,
                    "test_is_validation": False,
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "test_indices": test_indices,
                }
                if manifest_path is not None:
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2),
                        encoding="utf-8",
                    )

        index_sets = [
            set(train_indices),
            set(val_indices),
            set(test_indices),
        ]
        test_is_validation = bool(manifest.get("test_is_validation", False))
        if index_sets[0] & index_sets[1] or index_sets[0] & index_sets[2]:
            raise ValueError("Train, validation, and test split indices overlap")
        if not test_is_validation and index_sets[1] & index_sets[2]:
            raise ValueError("Validation and test split indices overlap")
        if test_is_validation and index_sets[1] != index_sets[2]:
            raise ValueError("test_is_validation requires identical validation and test indices")
        combined = index_sets[0] | index_sets[1] | index_sets[2]
        expected_covered = set(range(n_cells)) - set(map(int, manifest.get("excluded_indices", [])))
        if combined != expected_covered:
            raise ValueError("Split indices do not cover the intended non-excluded cells")

        self.split_info = {
            "dataset": str(self.h5mu_path.resolve()),
            "split_manifest": None if manifest_path is None else str(manifest_path),
            "seed": self.seed,
            "fractions": manifest.get(
                "fractions",
                {
                    "train": self.train_fraction,
                    "validation": self.val_fraction,
                    "test": self.test_fraction,
                },
            ),
            "n_total": n_cells,
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "n_test": len(test_indices),
            "n_rna_features": n_genes,
            "n_atac_features": n_peaks,
            "cell_sampling_applied": False,
            "feature_sampling_applied": self.feature_contract_path is not None,
            "feature_contract": (
                None
                if self.feature_contract_path is None
                else str(self.feature_contract_path.resolve())
            ),
            "feature_contract_sha256": (
                None
                if self.feature_contract_path is None
                else self._sha256(self.feature_contract_path)
            ),
            "test_is_validation": test_is_validation,
            "split_strategy": manifest.get("split_strategy", "random_cell"),
            "n_excluded": int(manifest.get("n_excluded", 0)),
            "n_outer_train_donors": manifest.get("n_outer_train_donors"),
            "n_outer_excluded_donors": manifest.get("n_outer_excluded_donors"),
            "n_inner_train_donors": manifest.get("n_inner_train_donors"),
            "n_inner_val_donors": manifest.get("n_inner_val_donors"),
            "outer_split_csv": manifest.get("outer_split_csv"),
            "outer_split_sha256": manifest.get("outer_split_sha256"),
        }
        return train_indices, val_indices, test_indices

    def setup(self, stage: str | None = None) -> None:
        if not self.h5mu_path.exists():
            raise FileNotFoundError(f"MuData file not found: {self.h5mu_path}")
        feature_contract = None
        if self.feature_contract_path is not None:
            if not self.feature_contract_path.is_file():
                raise FileNotFoundError(f"Feature contract not found: {self.feature_contract_path}")
            feature_contract = json.loads(self.feature_contract_path.read_text(encoding="utf-8"))
            required = {
                "rna_feature_names",
                "rna_feature_indices",
                "atac_feature_names",
                "atac_feature_indices",
                "cell_types",
            }
            if not required.issubset(feature_contract):
                raise ValueError(
                    f"Feature contract is missing {sorted(required - set(feature_contract))}"
                )
            self.feature_contract = feature_contract
        lazy_setting = os.environ.get("SCFLOW_H5MU_LAZY", "auto").strip().lower()
        if lazy_setting not in {"auto", "0", "1"}:
            raise ValueError("SCFLOW_H5MU_LAZY must be one of auto, 0, or 1")
        use_lazy = lazy_setting == "1" or (
            lazy_setting == "auto" and self.h5mu_path.stat().st_size >= 32 * 1024**3
        )

        if use_lazy:
            import h5py

            rna = H5CSRRowMatrix(self.h5mu_path, "mod/rna/X")
            atac = H5CSRRowMatrix(self.h5mu_path, "mod/atac/X")
            if rna.shape[0] != atac.shape[0]:
                raise ValueError("RNA and ATAC H5MU modalities have unequal cell counts")
            with h5py.File(self.h5mu_path, "r") as handle:
                cell_type_group = handle.get("mod/rna/obs/cell_type")
                if cell_type_group is not None and "codes" in cell_type_group:
                    categories = [
                        x.decode("utf-8") if isinstance(x, bytes) else str(x)
                        for x in cell_type_group["categories"][:]
                    ]
                    cell_type_codes = np.asarray(cell_type_group["codes"][:], dtype=np.int64)
                    cell_type_values = np.asarray(categories, dtype=object)[cell_type_codes]
                    cell_type_encoded = cell_type_codes
                    self.cell_type_mapping = {
                        label: int(index) for index, label in enumerate(categories)
                    }
                    self.num_cell_types = int(len(categories))
                else:
                    cell_type_encoded = None
                    cell_type_values = None
                    self.num_cell_types = None
                    self.cell_type_mapping = {}
                donor_group = handle.get(f"mod/rna/obs/{self.donor_key}")
                if (
                    donor_group is not None
                    and hasattr(donor_group, "keys")
                    and "codes" in donor_group
                ):
                    donor_categories = np.asarray(
                        [
                            x.decode("utf-8") if isinstance(x, bytes) else str(x)
                            for x in donor_group["categories"][:]
                        ]
                    )
                    donor_ids = donor_categories[
                        np.asarray(donor_group["codes"][:], dtype=np.int64)
                    ]
                elif donor_group is not None:
                    donor_ids = np.asarray(
                        [
                            x.decode("utf-8") if isinstance(x, bytes) else str(x)
                            for x in donor_group[:]
                        ]
                    )
                else:
                    donor_ids = None
                rna_index_name = handle["mod/rna/var"].attrs["_index"]
                atac_index_name = handle["mod/atac/var"].attrs["_index"]
                rna_feature_names = np.asarray(
                    [
                        x.decode("utf-8") if isinstance(x, bytes) else str(x)
                        for x in handle[f"mod/rna/var/{rna_index_name}"][:]
                    ],
                    dtype=object,
                )
                atac_feature_names = np.asarray(
                    [
                        x.decode("utf-8") if isinstance(x, bytes) else str(x)
                        for x in handle[f"mod/atac/var/{atac_index_name}"][:]
                    ],
                    dtype=object,
                )
        else:
            import muon as mu

            mdata = mu.read(self.h5mu_path)
            if "rna" not in mdata.mod or "atac" not in mdata.mod:
                raise ValueError("Expected MuData modalities named 'rna' and 'atac'")
            rna = mdata.mod["rna"].X
            atac = mdata.mod["atac"].X
            cell_type_encoded = None
            if "cell_type" in mdata.mod["rna"].obs:
                cell_type_values = np.asarray(mdata.mod["rna"].obs["cell_type"].astype(str).values)
                classes, inverse = np.unique(cell_type_values, return_inverse=True)
                self.cell_type_mapping = {str(label): int(i) for i, label in enumerate(classes)}
                self.num_cell_types = int(len(classes))
                cell_type_encoded = inverse.astype(np.int64)
            else:
                self.num_cell_types = None
                self.cell_type_mapping = {}
                cell_type_values = None
            donor_ids = (
                np.asarray(mdata.mod["rna"].obs[self.donor_key].astype(str).values)
                if self.donor_key in mdata.mod["rna"].obs
                else None
            )
            rna_feature_names = np.asarray(mdata.mod["rna"].var_names.astype(str), dtype=object)
            atac_feature_names = np.asarray(mdata.mod["atac"].var_names.astype(str), dtype=object)

        selected_rows = None
        if feature_contract is not None:
            rna_columns = np.asarray(feature_contract["rna_feature_indices"], dtype=np.int64)
            atac_columns = np.asarray(feature_contract["atac_feature_indices"], dtype=np.int64)
            expected_rna = np.asarray(feature_contract["rna_feature_names"], dtype=str)
            expected_atac = np.asarray(feature_contract["atac_feature_names"], dtype=str)
            if rna_feature_names[rna_columns].astype(str).tolist() != expected_rna.tolist():
                raise ValueError("RNA physical feature order violates the frozen contract")
            if atac_feature_names[atac_columns].astype(str).tolist() != expected_atac.tolist():
                raise ValueError("ATAC physical feature order violates the frozen contract")
            if donor_ids is None or cell_type_values is None:
                raise ValueError(
                    "Feature-contracted training requires donor and cell-type metadata"
                )
            cell_type_values = canonical_cell_types(cell_type_values)
            allowed_cell_types = canonical_cell_types(feature_contract["cell_types"]).tolist()
            if len(set(allowed_cell_types)) != len(allowed_cell_types):
                raise ValueError("Duplicate canonical cell-state name in feature contract")
            allowed_donors = None
            if self.donor_split_csv is not None:
                with self.donor_split_csv.open(newline="", encoding="utf-8") as handle:
                    split_rows = list(csv.DictReader(handle))
                allowed_donors = {str(row[self.donor_key]) for row in split_rows}
            keep = np.isin(np.asarray(cell_type_values, dtype=str), allowed_cell_types)
            if allowed_donors is not None:
                keep &= np.isin(np.asarray(donor_ids, dtype=str), list(allowed_donors))
            selected_rows = np.flatnonzero(keep).astype(np.int64)
            if not len(selected_rows):
                raise ValueError("No cells remain after applying the frozen feature contract")
            rna = MatrixView(rna, rows=selected_rows, columns=rna_columns)
            atac = MatrixView(atac, rows=selected_rows, columns=atac_columns)
            donor_ids = np.asarray(donor_ids, dtype=str)[selected_rows]
            cell_type_values = np.asarray(cell_type_values, dtype=str)[selected_rows]
            self.cell_type_mapping = {
                label: index for index, label in enumerate(allowed_cell_types)
            }
            self.num_cell_types = len(allowed_cell_types)
            cell_type_encoded = np.asarray(
                [self.cell_type_mapping[value] for value in cell_type_values],
                dtype=np.int64,
            )
        if self.rna_counts_path:
            import h5py

            with h5py.File(self.h5mu_path, "r") as handle:
                check_paired_rows(handle)
                original_ids = handle["mod/rna/obs/original_cell_id"].asstr()[:]
            if selected_rows is not None:
                original_ids = original_ids[selected_rows]
            genes = expected_rna if feature_contract is not None else rna_feature_names
            rna = read_rna_counts(self.rna_counts_path, original_ids, genes)
        dataset = MultimodalMuDataDataset(
            rna,
            atac,
            rna_seq_len=self.rna_seq_len,
            atac_seq_len=self.atac_seq_len,
            sample_features=self.sample_features,
            rna_sample_features=self.rna_sample_features,
            atac_sample_features=self.atac_sample_features,
            rna_encoder_seq_len=self.rna_encoder_seq_len,
            rna_encoder_sample_features=self.rna_encoder_sample_features,
            atac_encoder_seq_len=self.atac_encoder_seq_len,
            atac_encoder_sample_features=self.atac_encoder_sample_features,
            seed=self.seed,
            cell_type=cell_type_encoded,
        )
        self.n_genes = dataset.n_genes
        self.n_peaks = dataset.n_peaks

        train_indices, val_indices, test_indices = self._load_or_create_split(
            n_cells=len(dataset),
            n_genes=dataset.n_genes,
            n_peaks=dataset.n_peaks,
            donor_ids=donor_ids,
        )
        if feature_contract is not None:
            self.split_info["cell_type_counts"] = {}
            for split_name, indices in (("train", train_indices), ("validation", val_indices)):
                counts = np.bincount(cell_type_encoded[indices], minlength=len(allowed_cell_types))
                self.split_info["cell_type_counts"][split_name] = dict(
                    zip(allowed_cell_types, counts.tolist())
                )
            missing = [
                state for state, n in self.split_info["cell_type_counts"]["train"].items() if n == 0
            ]
            if missing:
                raise ValueError(f"Contracted states absent from backbone training: {missing}")
        self.train_dataset = torch.utils.data.Subset(dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
