import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scflowdiff.core.cima_inputs import canonical_cell_types, read_rna_counts
from scflowdiff.downstream.immune_age.datp import fit_datp_bundle, load_flow_latent


def test_matched_umi_rows_and_gene_order(tmp_path):
    path = tmp_path / "raw.h5ad"
    with h5py.File(path, "w") as h:
        for group, names in (("obs", [b"B", b"A"]), ("var", [b"g2", b"g1", b"g3"])):
            h.create_group(group).attrs["_index"] = "name"
            h[group]["name"] = names
        matrix = sparse.csr_matrix([[1, 0, 5], [0, 3, 8]])
        layer = h.create_group("layers/counts")
        layer.attrs.update({"encoding-type": "csr_matrix", "shape": matrix.shape})
        for field in ("data", "indptr", "indices"):
            layer[field] = getattr(matrix, field)
    actual = read_rna_counts(path, ["A", "B", "A"], ["g3", "g1"])
    np.testing.assert_array_equal(actual.toarray(), [[8, 3], [5, 0], [8, 3]])
    assert canonical_cell_types(["CD4 Naïve", "CD4_Naïve"]).tolist() == ["CD4_Naïve"] * 2


def test_endpoint_preserves_token_order_and_direction(tmp_path):
    path = tmp_path / "endpoint.npz"
    z = np.arange(2 * 16 * 32).reshape(2, 16, 32)
    np.savez(path, target_latent=z, direction="atac_to_rna", sample_id=["A", "B"], ct_id=[1, 1])
    latent, lookup = load_flow_latent(path, direction="atac_to_rna")
    np.testing.assert_array_equal(latent[lookup[("B", 1)]], z[1].ravel())
    with pytest.raises(ValueError, match="direction"):
        load_flow_latent(path, direction="rna_to_atac")


def test_imputed_rna_never_trains_or_scores_reverse_pls():
    rng = np.random.default_rng(42)
    donors = [f"d{n}" for n in range(18)]
    source = pd.DataFrame(
        rng.normal(size=(18, 5)), index=donors, columns=[f"ATAC__CD4__p{n}" for n in range(5)]
    )
    source["Missing__ATAC__CD4"] = 0
    source.iloc[0, 0] = np.nan
    target = pd.DataFrame({"CD4__g": rng.normal(size=18), "Missing__CD4": 0}, index=donors)
    target.loc["d0", "Missing__CD4"] = 1
    latent = rng.normal(size=(18, 8))
    lookup = {(d, 0): i for i, d in enumerate(donors)}
    table = pd.DataFrame({"feature": ["CD4__g"], "cell_type": ["CD4"]})

    def fit():
        return fit_datp_bundle(
            source,
            target,
            latent,
            lookup,
            table,
            ["CD4"],
            sex="female",
            direction="atac_to_rna",
            folds=3,
            pls_components=(1,),
            global_components=2,
            local_components=2,
            flow_pca_components=2,
        )

    before = fit()
    target.loc["d0", "CD4__g"] = 1e10
    after = fit()
    assert before["states"][0]["n_development"] == 17
    assert before["selection"] == after["selection"]
    np.testing.assert_array_equal(
        before["states"][0]["model"].coef_, after["states"][0]["model"].coef_
    )
