"""Shared input conventions for CIMA cell encoding."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def canonical_cell_types(values):
    return np.char.replace(np.asarray(values, dtype=str), " ", "_")


def check_paired_rows(handle):
    """Check CIMA pseudopair identity before sharing a physical row index."""
    rna, atac = handle["mod/rna/obs"], handle["mod/atac/obs"]
    for field in ("sample_id", "cell_type", "pair_id"):
        columns = []
        for obs in (rna, atac):
            node = obs[field]
            if isinstance(node, h5py.Group):
                codes = node["codes"][:]
                if (codes < 0).any():
                    raise ValueError(f"Missing {field} in paired CIMA metadata")
                values = node["categories"].asstr()[:][codes]
            else:
                values = node.asstr()[:]
            columns.append(canonical_cell_types(values) if field == "cell_type" else values)
        if not np.array_equal(*columns):
            raise ValueError(f"CIMA RNA and ATAC rows disagree on {field}")


def _count_rows(obs, ids):
    """Locate requested cells in the source without loading millions of string IDs at once."""
    requested = pd.Index(ids).unique()
    positions = np.full(len(requested), -1, dtype=np.int64)
    index = obs[obs.attrs["_index"]].asstr()
    for start in range(0, len(index), 131072):
        found = requested.get_indexer(index[start : start + 131072])
        hits = np.flatnonzero(found >= 0)
        slots = found[hits]
        if np.any(positions[slots] >= 0) or len(np.unique(slots)) != len(slots):
            raise ValueError("Duplicate requested cell ID in the original RNA counts")
        positions[slots] = start + hits
    if np.any(positions < 0):
        raise ValueError(f"RNA counts missing for cell {requested[positions < 0][0]}")
    return positions[requested.get_indexer(ids)]


def read_rna_counts(path: Path, cell_ids, genes):
    """Read integer UMIs in requested cell/gene order, preserving repeated pseudopair rows."""
    with h5py.File(path, "r") as raw:
        var = raw["var"]
        vocab = pd.Index(var[var.attrs["_index"]].asstr()[:])
        columns = vocab.get_indexer(genes)
        if np.any(columns < 0) or len(np.unique(columns)) != len(columns):
            raise ValueError("Missing or duplicated RNA gene in counts vocabulary")
        if not len(cell_ids):
            return sparse.csr_matrix((0, len(columns)), dtype=np.float32)
        rows, restore = np.unique(_count_rows(raw["obs"], cell_ids), return_inverse=True)
        counts = raw["layers/counts"]
        if counts.attrs["encoding-type"] != "csr_matrix":
            raise ValueError("Expected a CSR RNA counts layer")
        width = int(counts.attrs["shape"][1])
        blocks = []
        # Read nearby physical rows together, even when requested cells are interleaved.
        for selected in np.split(rows, np.flatnonzero(np.diff(rows // 512)) + 1):
            first, last = selected[0], selected[-1]
            ptr = counts["indptr"][first : last + 2].astype(np.int64)
            section = slice(ptr[0], ptr[-1])
            block = sparse.csr_matrix(
                (counts["data"][section], counts["indices"][section], ptr - ptr[0]),
                shape=(last - first + 1, width),
            )[selected - first][:, columns].astype(np.float32)
            values = block.data
            if not np.all(np.isfinite(values) & (values >= 0) & (values == np.floor(values))):
                raise ValueError("RNA counts layer contains invalid UMI values")
            blocks.append(block)
    return sparse.vstack(blocks, format="csr")[restore]
