"""Tests for eQTL GraphREML: I/O, matching, batching, and refactoring equivalence."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphld.io import read_ldgm_metadata, load_ldgm, merge_snplists, merge_alleles
from graphld.eqtl_io import (
    load_egenes,
    load_eqtl_sumstats,
    compute_eqtl_sample_size,
    map_genes_to_blocks,
    compute_link_fn_denominator,
)
from graphld.eqtl_reml import EqtlGraphREML
from graphld.heritability import GraphREML, ModelOptions, MethodOptions


# ---------------------------------------------------------------------------
# Paths — skip tests if data is not available
# ---------------------------------------------------------------------------

EGENES_PATH = Path("/Users/noah/data/gtex_v10/GTEx_Analysis_v10_eQTL_updated/Adipose_Visceral_Omentum.v10.eGenes.txt.gz")
PARQUET_DIR = Path("/Users/noah/data/gtex_v10")
PARQUET_CHR21 = PARQUET_DIR / "Adipose_Subcutaneous.v10.allpairs.chr21.parquet"
LDGM_METADATA = Path("/Users/noah/bin/graphld/data/ldgms/metadata.csv")
ANNOT_DIR = Path("/Users/noah/bin/graphld/data/annot")

# Test data bundled with graphld
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "test"
TEST_METADATA = TEST_DATA_DIR / "metadata.csv"

has_gtex = EGENES_PATH.exists() and PARQUET_CHR21.exists()
has_ldgms = LDGM_METADATA.exists() and ANNOT_DIR.exists()

skip_no_gtex = pytest.mark.skipif(not has_gtex, reason="GTEx data not available")
skip_no_ldgms = pytest.mark.skipif(not has_ldgms, reason="LDGM data not available")


def _sksparse_works():
    """Check whether scikit-sparse Cholesky works (fails on 0.5.0)."""
    try:
        from sksparse.cholmod import cholesky
        from scipy.sparse import eye
        M = eye(3, format="csc") * 2.0
        F = cholesky(M)
        F(np.ones(3))  # 0.5.0 returns tuple, not callable
        return True
    except (TypeError, ImportError):
        return False


skip_no_cholesky = pytest.mark.skipif(
    not _sksparse_works(),
    reason="scikit-sparse Cholesky broken (need 0.4.16)",
)


# ===================================================================
# Test 1: Vectorized matching equivalence
# ===================================================================

def _old_matching_loop(vi_positions, unique_indices, vi_anc, vi_der, gene_df, M):
    """Original Python-loop Z-score matching (reference implementation)."""
    z = np.zeros(M)
    gene_pos = gene_df.get_column("POS").to_numpy()
    gene_z = gene_df.get_column("Z").to_numpy()
    gene_ref = gene_df.get_column("REF")
    gene_alt = gene_df.get_column("ALT")

    gene_lookup = {}
    for j in range(len(gene_pos)):
        pos = int(gene_pos[j])
        gene_lookup.setdefault(pos, []).append(
            (gene_z[j], gene_ref[j], gene_alt[j])
        )

    matched = 0
    for k in range(len(vi_positions)):
        pos = int(vi_positions[k])
        if pos not in gene_lookup:
            continue
        for z_val, ref_val, alt_val in gene_lookup[pos]:
            anc = str(vi_anc[k]).lower() if vi_anc[k] is not None else ""
            der = str(vi_der[k]).lower() if vi_der[k] is not None else ""
            ref_lower = str(ref_val).lower()
            alt_lower = str(alt_val).lower()
            if anc == ref_lower and der == alt_lower:
                z[unique_indices[k]] = z_val
                matched += 1
                break
            elif anc == alt_lower and der == ref_lower:
                z[unique_indices[k]] = -z_val
                matched += 1
                break
    return z, matched


def _new_matching_vectorized(ref_df, gene_df, M):
    """New Polars-join Z-score matching (under test)."""
    matched = gene_df.join(
        ref_df, left_on="POS", right_on="position", how="inner",
    )
    if matched.is_empty():
        return np.zeros(M), 0

    phase = merge_alleles(
        matched["anc_alleles"], matched["deriv_alleles"],
        matched["REF"], matched["ALT"],
    )
    matched = matched.with_columns(phase.alias("phase"))
    matched = matched.filter(pl.col("phase") != 0)
    if matched.is_empty():
        return np.zeros(M), 0

    z = np.zeros(M)
    idx = matched["ldgm_idx"].to_numpy()
    z_vals = matched["Z"].to_numpy() * matched["phase"].to_numpy()
    _, first_occ = np.unique(idx, return_index=True)
    z[idx[first_occ]] = z_vals[first_occ]
    return z, len(first_occ)


@skip_no_gtex
@skip_no_ldgms
def test_vectorized_matching_equivalence():
    """Old Python-loop and new Polars-join matching must produce identical Z vectors."""
    # Load real data for one block in the middle of chr21
    metadata = read_ldgm_metadata(str(LDGM_METADATA), populations="EUR", chromosomes=21)
    egenes = load_egenes(str(EGENES_PATH))
    egenes_chr21 = egenes.filter(pl.col("chrom") == 21)
    eqtl = load_eqtl_sumstats(str(PARQUET_DIR), "Adipose_Subcutaneous", egenes_chr21["gene_id"], chromosomes=[21])
    block_to_genes, _ = map_genes_to_blocks(egenes_chr21, metadata)

    # Pick a block in the middle with genes
    block_idx = 10
    block_meta = metadata.row(block_idx, named=True)
    gene_ids = block_to_genes.get(block_idx, [])
    assert len(gene_ids) > 0, "Block 10 should have genes"

    # Load LDGM and merge annotations
    from graphld.io import load_annotations
    annotations = load_annotations(str(ANNOT_DIR), chromosome=21, add_positions=False)
    annot_cols = [c for c in annotations.columns if c not in ["SNP", "CHR", "POS", "CM", "BP"]]

    ldgm = load_ldgm(str(LDGM_METADATA.parent / block_meta["name"]))
    ldgm, _ = merge_snplists(
        ldgm, annotations, match_by_position=True, pos_col="POS",
        add_cols=annot_cols, add_allelic_cols=[], modify_in_place=True,
    )

    vi = ldgm.variant_info
    first_mask = vi.select(pl.col("index").is_first_distinct()).to_numpy().flatten()
    unique_vi = vi.filter(first_mask)
    unique_indices = unique_vi.get_column("index").to_numpy()
    vi_positions = unique_vi.get_column("position").to_numpy()
    vi_anc = unique_vi.get_column("anc_alleles")
    vi_der = unique_vi.get_column("deriv_alleles")

    ref_df = pl.DataFrame({
        "position": unique_vi.get_column("position"),
        "ldgm_idx": unique_indices,
        "anc_alleles": unique_vi.get_column("anc_alleles"),
        "deriv_alleles": unique_vi.get_column("deriv_alleles"),
    })

    M = ldgm.shape[0]

    # Test on multiple genes
    tested = 0
    for gene_id in gene_ids[:5]:
        gene_df = eqtl.filter(
            (pl.col("gene_id") == gene_id)
            & (pl.col("CHR") == block_meta["chrom"])
            & (pl.col("POS") >= block_meta["chromStart"])
            & (pl.col("POS") < block_meta["chromEnd"])
        ).select("POS", "REF", "ALT", "Z")

        if gene_df.is_empty():
            continue

        z_old, n_old = _old_matching_loop(vi_positions, unique_indices, vi_anc, vi_der, gene_df, M)
        z_new, n_new = _new_matching_vectorized(ref_df, gene_df, M)

        assert np.allclose(z_old, z_new, atol=1e-6), (
            f"Gene {gene_id}: Z vectors differ. "
            f"Old matched {n_old}, new matched {n_new}, "
            f"max diff = {np.max(np.abs(z_old - z_new))}"
        )
        assert n_old == n_new, f"Gene {gene_id}: match counts differ ({n_old} vs {n_new})"
        tested += 1

    assert tested >= 2, f"Expected at least 2 genes tested, got {tested}"


# ===================================================================
# Test 2: eqtl_io round-trip
# ===================================================================

class TestLoadEgenes:
    """Tests for load_egenes."""

    @skip_no_gtex
    def test_basic_load(self):
        egenes = load_egenes(str(EGENES_PATH))
        assert len(egenes) > 0
        assert set(egenes.columns) == {"gene_id", "chrom", "tss"}

    @skip_no_gtex
    def test_qval_filter(self):
        strict = load_egenes(str(EGENES_PATH), qval_threshold=0.01)
        relaxed = load_egenes(str(EGENES_PATH), qval_threshold=0.05)
        assert len(strict) < len(relaxed)

    @skip_no_gtex
    def test_no_chrx(self):
        """Autosomes only — no chrX/chrY genes."""
        egenes = load_egenes(str(EGENES_PATH))
        chroms = egenes["chrom"].to_list()
        assert all(1 <= c <= 22 for c in chroms)

    @skip_no_gtex
    def test_tss_strand_logic(self):
        """TSS should differ for + vs - strand genes."""
        raw = pl.read_csv(str(EGENES_PATH), separator="\t")
        raw = raw.filter(pl.col("qval") <= 0.05)
        # Filter to autosomes
        raw = raw.filter(pl.col("gene_chr").str.contains(r"^\d+$") == False)
        raw = raw.filter(~pl.col("gene_chr").str.contains("X|Y|M"))

        # Find a + strand gene and a - strand gene from the full data
        plus_genes = raw.filter(pl.col("strand") == "+")
        minus_genes = raw.filter(pl.col("strand") == "-")
        if len(plus_genes) > 0 and len(minus_genes) > 0:
            p = plus_genes.row(0, named=True)
            m = minus_genes.row(0, named=True)
            egenes = load_egenes(str(EGENES_PATH))
            p_tss = egenes.filter(pl.col("gene_id") == p["gene_id"])["tss"][0]
            m_tss = egenes.filter(pl.col("gene_id") == m["gene_id"])["tss"][0]
            assert p_tss == p["gene_start"], "TSS for + strand should be gene_start"
            assert m_tss == m["gene_end"], "TSS for - strand should be gene_end"

    def test_with_synthetic_data(self, tmp_path):
        """Test load_egenes with a tiny synthetic file."""
        content = (
            "gene_id\tgene_name\tbiotype\tgene_chr\tgene_start\tgene_end\tstrand\tqval\n"
            "ENSG1\tGENE1\tpc\tchr1\t1000\t2000\t+\t0.01\n"
            "ENSG2\tGENE2\tpc\tchr2\t3000\t4000\t-\t0.04\n"
            "ENSG3\tGENE3\tpc\tchr3\t5000\t6000\t+\t0.10\n"  # filtered by qval
            "ENSG4\tGENE4\tpc\tchrX\t7000\t8000\t+\t0.01\n"  # filtered by chrX
        )
        f = tmp_path / "test_egenes.txt"
        f.write_text(content)
        egenes = load_egenes(str(f), qval_threshold=0.05)
        assert len(egenes) == 2
        assert set(egenes["gene_id"].to_list()) == {"ENSG1", "ENSG2"}
        # TSS: ENSG1 (+strand) -> gene_start=1000, ENSG2 (-strand) -> gene_end=4000
        e1 = egenes.filter(pl.col("gene_id") == "ENSG1")
        e2 = egenes.filter(pl.col("gene_id") == "ENSG2")
        assert e1["tss"][0] == 1000
        assert e2["tss"][0] == 4000
        assert e1["chrom"][0] == 1
        assert e2["chrom"][0] == 2


class TestLoadEqtlSumstats:
    """Tests for load_eqtl_sumstats."""

    @skip_no_gtex
    def test_basic_load(self):
        egenes = load_egenes(str(EGENES_PATH))
        eqtl = load_eqtl_sumstats(str(PARQUET_DIR), "Adipose_Subcutaneous", egenes["gene_id"], chromosomes=[21])
        assert len(eqtl) > 0
        assert set(eqtl.columns) == {"gene_id", "CHR", "POS", "REF", "ALT", "Z"}

    @skip_no_gtex
    def test_z_is_finite(self):
        egenes = load_egenes(str(EGENES_PATH))
        eqtl = load_eqtl_sumstats(str(PARQUET_DIR), "Adipose_Subcutaneous", egenes["gene_id"], chromosomes=[21])
        assert eqtl["Z"].is_finite().all()

    @skip_no_gtex
    def test_chr_parsing(self):
        egenes = load_egenes(str(EGENES_PATH))
        eqtl = load_eqtl_sumstats(str(PARQUET_DIR), "Adipose_Subcutaneous", egenes["gene_id"], chromosomes=[21])
        assert (eqtl["CHR"] == 21).all()

    @skip_no_gtex
    def test_gene_filtering(self):
        """Only requested gene_ids should appear."""
        egenes = load_egenes(str(EGENES_PATH))
        # Get genes on chr21 so they match the available parquet
        chr21_genes = egenes.filter(pl.col("chrom") == 21)
        subset = chr21_genes.head(5)["gene_id"]
        eqtl = load_eqtl_sumstats(str(PARQUET_DIR), "Adipose_Subcutaneous", subset, chromosomes=[21])
        assert eqtl["gene_id"].n_unique() <= 5
        # All returned genes should be in the requested subset
        returned = set(eqtl["gene_id"].unique().to_list())
        assert returned.issubset(set(subset.to_list()))


class TestComputeSampleSize:
    @skip_no_gtex
    def test_reasonable_n(self):
        N = compute_eqtl_sample_size(str(PARQUET_CHR21))
        assert 100 < N < 2000, f"N={N} outside reasonable range for GTEx"


class TestMapGenesToBlocks:
    @skip_no_ldgms
    @skip_no_gtex
    def test_basic_mapping(self):
        egenes = load_egenes(str(EGENES_PATH))
        egenes_chr21 = egenes.filter(pl.col("chrom") == 21)
        metadata = read_ldgm_metadata(str(LDGM_METADATA), populations="EUR", chromosomes=21)
        b2g, g2b = map_genes_to_blocks(egenes_chr21, metadata)
        # Every gene should map to at least one block (within 1Mb of any block)
        n_unmapped = sum(1 for v in g2b.values() if not v)
        assert n_unmapped <= len(egenes_chr21) * 0.1, "Too many unmapped genes"

    def test_boundary_gene(self):
        """Gene at block boundary maps to correct overlapping blocks."""
        egenes = pl.DataFrame({"gene_id": ["G1"], "chrom": [1], "tss": [2_500_000]})
        metadata = pl.DataFrame({
            "chrom": [1, 1],
            "chromStart": [1_000_000, 2_000_000],
            "chromEnd": [2_000_000, 3_000_000],
        })
        b2g, g2b = map_genes_to_blocks(egenes, metadata, window=1_000_000)
        # TSS=2.5M with 1Mb window → [1.5M, 3.5M] overlaps both blocks
        assert len(g2b["G1"]) == 2

    def test_isolated_gene(self):
        """Gene far from any block gets empty mapping."""
        egenes = pl.DataFrame({"gene_id": ["G1"], "chrom": [1], "tss": [100_000_000]})
        metadata = pl.DataFrame({
            "chrom": [1],
            "chromStart": [1_000_000],
            "chromEnd": [2_000_000],
        })
        b2g, g2b = map_genes_to_blocks(egenes, metadata, window=1_000_000)
        assert g2b["G1"] == []


class TestComputeLinkFnDenominator:
    def test_basic(self):
        eqtl = pl.DataFrame({
            "gene_id": ["G1", "G1", "G2", "G2", "G2"],
            "CHR": [1, 1, 1, 1, 1],
            "POS": [100, 200, 100, 200, 300],
        })
        # G1 has 2 unique variants, G2 has 3 → total = 5
        assert compute_link_fn_denominator(eqtl) == 5


# ===================================================================
# Test 3: Batched vs per-gene equivalence
# ===================================================================

@skip_no_ldgms
@skip_no_cholesky
def test_batched_vs_pergene_equivalence():
    """Batched computation must produce same results as per-gene loop."""
    metadata = read_ldgm_metadata(str(LDGM_METADATA), populations="EUR", chromosomes=21)

    # Pick a block in the middle of chr21
    block_idx = 10
    block_meta = metadata.row(block_idx, named=True)
    ldgm = load_ldgm(str(LDGM_METADATA.parent / block_meta["name"]))

    # Merge with annotations
    from graphld.io import load_annotations
    annotations = load_annotations(str(ANNOT_DIR), chromosome=21, add_positions=False)
    annot_cols = [c for c in annotations.columns if c not in ["SNP", "CHR", "POS", "CM", "BP"]]

    ldgm, _ = merge_snplists(
        ldgm, annotations, match_by_position=True, pos_col="POS",
        add_cols=annot_cols, add_allelic_cols=[], modify_in_place=True,
    )
    ldgm.factor()

    M = ldgm.shape[0]
    p = len(annot_cols)

    # Create synthetic Pz vectors for 5 "genes"
    np.random.seed(42)
    Pz_genes = {f"gene_{i}": np.random.randn(M, 1) * 0.01 for i in range(5)}

    # Build a dummy del_M_del_a
    annot = ldgm.variant_info.select(annot_cols).to_numpy()
    params = np.zeros((p, 1))

    from graphld.heritability import _get_softmax_link_function
    link_fn, link_fn_grad, _ = _get_softmax_link_function(6e6)
    del_h2_del_a = link_fn_grad(annot, params)
    del_M_del_a = np.zeros((M, p))
    np.add.at(del_M_del_a, ldgm.variant_indices, del_h2_del_a)

    num_samples = 10
    seed = 123

    # --- Per-gene loop (reference) ---
    ref_L = 0.0
    ref_g = np.zeros(p)
    ref_H = np.zeros((p, p))
    for gid, Pz in Pz_genes.items():
        L, g, H = GraphREML._compute_gene_contribution(
            Pz, ldgm, del_M_del_a, num_samples, False, seed,
        )
        ref_L += L
        ref_g += g
        ref_H += H

    # --- Batched (under test) ---
    # Need to reset the factorization state since per-gene calls may have changed it
    ldgm.del_factor()
    ldgm.factor()
    bat_L, bat_g, bat_H = EqtlGraphREML._compute_batched_contributions(
        Pz_genes, ldgm, del_M_del_a, num_samples, False, seed,
    )

    # Compare
    assert np.isclose(ref_L, bat_L, rtol=1e-6), (
        f"Likelihood: ref={ref_L}, batched={bat_L}, diff={abs(ref_L - bat_L)}"
    )
    assert np.allclose(ref_g, bat_g, rtol=1e-4), (
        f"Gradient max diff: {np.max(np.abs(ref_g - bat_g))}"
    )
    assert np.allclose(ref_H, bat_H, rtol=1e-4), (
        f"Hessian max diff: {np.max(np.abs(ref_H - bat_H))}"
    )


@skip_no_ldgms
@skip_no_cholesky
def test_batched_likelihood_only():
    """Batched likelihood-only mode should match per-gene sum."""
    metadata = read_ldgm_metadata(str(LDGM_METADATA), populations="EUR", chromosomes=21)
    block_meta = metadata.row(10, named=True)
    ldgm = load_ldgm(str(LDGM_METADATA.parent / block_meta["name"]))

    from graphld.io import load_annotations
    annotations = load_annotations(str(ANNOT_DIR), chromosome=21, add_positions=False)
    annot_cols = [c for c in annotations.columns if c not in ["SNP", "CHR", "POS", "CM", "BP"]]
    ldgm, _ = merge_snplists(
        ldgm, annotations, match_by_position=True, pos_col="POS",
        add_cols=annot_cols, add_allelic_cols=[], modify_in_place=True,
    )
    ldgm.factor()

    M = ldgm.shape[0]
    p = len(annot_cols)
    np.random.seed(99)
    Pz_genes = {f"gene_{i}": np.random.randn(M, 1) * 0.01 for i in range(3)}
    del_M_del_a = np.zeros((M, p))  # dummy

    # Per-gene
    ref_L = 0.0
    for Pz in Pz_genes.values():
        L, _, _ = GraphREML._compute_gene_contribution(
            Pz, ldgm, del_M_del_a, 10, True, 42,
        )
        ref_L += L

    ldgm.del_factor()
    ldgm.factor()
    bat_L, bat_g, bat_H = EqtlGraphREML._compute_batched_contributions(
        Pz_genes, ldgm, del_M_del_a, 10, True, 42,
    )

    assert np.isclose(ref_L, bat_L, rtol=1e-6)
    assert bat_g is None
    assert bat_H is None


# ===================================================================
# Test 4: Refactored _compute_block_likelihood equivalence
# ===================================================================

@skip_no_cholesky
def test_refactored_compute_block_likelihood():
    """_compute_block_likelihood wrapper must match manual _update + _contribute + del_factor."""
    # Use bundled test data (chr1 blocks)
    metadata = read_ldgm_metadata(str(TEST_METADATA), populations="EUR")
    block_meta = metadata.row(0, named=True)
    ldgm = load_ldgm(str(TEST_DATA_DIR / block_meta["name"]))

    # Build synthetic data matching the LDGM
    vi = ldgm.variant_info
    M = ldgm.shape[0]

    # Simple annotation: just 'base' (all ones)
    n_variants = len(vi)
    annot = np.ones((n_variants, 1))
    params = np.zeros((1, 1))
    old_h2 = np.zeros(n_variants).reshape(-1, 1)

    np.random.seed(42)
    z = np.random.randn(M, 1) * 0.01
    ldgm.factor()
    Pz = ldgm._matrix @ z  # raw sparse matvec (avoid Schur complement)

    link_fn_denom = 6e6
    num_samples = 10
    seed = 42

    # --- Method A: wrapper ---
    ldgm_a = load_ldgm(str(TEST_DATA_DIR / block_meta["name"]))
    ldgm_a.factor()
    Pz_a = ldgm_a._matrix @ z
    L_a, g_a, H_a, h2_a = GraphREML._compute_block_likelihood(
        ldgm_a, Pz_a, annot, params, link_fn_denom, old_h2, num_samples, False, seed,
    )

    # --- Method B: manual split ---
    ldgm_b = load_ldgm(str(TEST_DATA_DIR / block_meta["name"]))
    ldgm_b.factor()
    Pz_b = ldgm_b._matrix @ z
    h2_b, del_M_b = GraphREML._update_block_model(
        ldgm_b, annot, params, link_fn_denom, old_h2,
    )
    L_b, g_b, H_b = GraphREML._compute_gene_contribution(
        Pz_b, ldgm_b, del_M_b, num_samples, False, seed,
    )
    ldgm_b.del_factor()

    # Compare
    assert np.isclose(L_a, L_b, rtol=1e-10), f"Likelihood: {L_a} vs {L_b}"
    assert np.allclose(g_a, g_b, rtol=1e-10), f"Gradient max diff: {np.max(np.abs(g_a - g_b))}"
    assert np.allclose(H_a, H_b, rtol=1e-10), f"Hessian max diff: {np.max(np.abs(H_a - H_b))}"
    assert np.allclose(h2_a, h2_b, rtol=1e-10), f"h2 max diff: {np.max(np.abs(h2_a - h2_b))}"


@skip_no_cholesky
def test_refactored_likelihood_only():
    """Likelihood-only mode should also match between wrapper and manual split."""
    metadata = read_ldgm_metadata(str(TEST_METADATA), populations="EUR")
    block_meta = metadata.row(0, named=True)

    vi_len = None
    M = None

    # Method A
    ldgm_a = load_ldgm(str(TEST_DATA_DIR / block_meta["name"]))
    ldgm_a.factor()
    M = ldgm_a.shape[0]
    np.random.seed(42)
    z = np.random.randn(M, 1) * 0.01
    Pz = ldgm_a._matrix @ z
    vi_len = len(ldgm_a.variant_info)
    annot = np.ones((vi_len, 1))
    params = np.zeros((1, 1))
    old_h2 = np.zeros(vi_len).reshape(-1, 1)

    L_a, g_a, H_a, h2_a = GraphREML._compute_block_likelihood(
        ldgm_a, Pz, annot, params, 6e6, old_h2, 10, True, 42,
    )
    assert g_a is None
    assert H_a is None

    # Method B
    ldgm_b = load_ldgm(str(TEST_DATA_DIR / block_meta["name"]))
    ldgm_b.factor()
    Pz_b = ldgm_b._matrix @ z
    h2_b, del_M_b = GraphREML._update_block_model(
        ldgm_b, annot, params, 6e6, old_h2,
    )
    L_b, g_b, H_b = GraphREML._compute_gene_contribution(
        Pz_b, ldgm_b, del_M_b, 10, True, 42,
    )
    ldgm_b.del_factor()

    assert np.isclose(L_a, L_b, rtol=1e-10)
    assert g_b is None
    assert H_b is None
    assert np.allclose(h2_a, h2_b, rtol=1e-10)
