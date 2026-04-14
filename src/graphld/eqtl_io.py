"""I/O functions for eQTL summary statistics and gene-to-block mapping."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl


def load_egenes(
    egenes_path: str,
    qval_threshold: float = 0.05,
) -> pl.DataFrame:
    """Load eGenes file and filter to significant eGenes.

    Args:
        egenes_path: Path to .eGenes.txt.gz file
        qval_threshold: Maximum q-value for eGene significance

    Returns:
        DataFrame with columns: gene_id (str), chrom (Int64), tss (Int64)
    """
    df = pl.read_csv(egenes_path, separator="\t")

    df = df.filter(pl.col("qval") <= qval_threshold)

    # Parse chromosome, filtering to autosomes (1-22)
    chrom_str = df["gene_chr"].str.replace("chr", "")
    is_autosome = chrom_str.str.contains(r"^\d+$")
    df = df.filter(is_autosome)

    df = df.with_columns(
        pl.when(pl.col("strand") == "+")
        .then(pl.col("gene_start"))
        .otherwise(pl.col("gene_end"))
        .alias("tss"),
        pl.col("gene_chr")
        .str.replace("chr", "")
        .cast(pl.Int64)
        .alias("chrom"),
    )

    return df.select("gene_id", "chrom", "tss").unique(subset=["gene_id"])


def _parse_variant_id(expr: pl.Expr) -> dict:
    """Build column expressions to parse GTEx variant_id format (chr21_5033246_G_T_b38)."""
    split = expr.str.split("_")
    return {
        "CHR": split.list.get(0).str.replace("chr", "").cast(pl.Int64),
        "POS": split.list.get(1).cast(pl.Int64),
        "REF": split.list.get(2),
        "ALT": split.list.get(3),
    }


def load_eqtl_sumstats(
    parquet_dir: str,
    tissue: str,
    egene_ids: pl.Series,
    chromosomes: Optional[List[int]] = None,
) -> pl.DataFrame:
    """Load eQTL summary statistics for specified eGenes.

    Args:
        parquet_dir: Directory containing {tissue}.v10.allpairs.chr{N}.parquet
        tissue: Tissue name (e.g., "Adipose_Subcutaneous")
        egene_ids: Series of gene_id values to include
        chromosomes: Optional list of chromosomes to load (default: 1-22)

    Returns:
        DataFrame with columns: gene_id, CHR, POS, REF, ALT, Z
    """
    if chromosomes is None:
        chromosomes = list(range(1, 23))

    parquet_dir = Path(parquet_dir)
    egene_set = set(egene_ids.to_list())
    frames = []

    for chrom in chromosomes:
        path = parquet_dir / f"{tissue}.v10.allpairs.chr{chrom}.parquet"
        if not path.exists():
            continue

        df = (
            pl.scan_parquet(path)
            .filter(pl.col("gene_id").is_in(egene_set))
            .select("gene_id", "variant_id", "slope", "slope_se")
            .collect()
        )

        if df.is_empty():
            continue

        parsed = _parse_variant_id(pl.col("variant_id"))
        df = df.with_columns(
            **parsed,
            Z=(pl.col("slope") / pl.col("slope_se")),
        ).select("gene_id", "CHR", "POS", "REF", "ALT", "Z")

        # Drop non-finite Z
        df = df.filter(pl.col("Z").is_finite())
        frames.append(df)

    if not frames:
        raise ValueError(f"No eQTL data found for tissue {tissue} in {parquet_dir}")

    return pl.concat(frames)


def compute_eqtl_sample_size(
    parquet_path: str,
    n_rows: int = 100_000,
) -> float:
    """Estimate tissue sample size from eQTL parquet as median(ma_count / (2 * af)).

    Args:
        parquet_path: Path to one allpairs parquet file
        n_rows: Number of rows to sample for estimation

    Returns:
        Estimated sample size (float)
    """
    df = pl.read_parquet(parquet_path, n_rows=n_rows, columns=["af", "ma_count"])
    df = df.filter((pl.col("af") > 0) & (pl.col("af") < 1))
    n_estimates = df.select(
        (pl.col("ma_count") / (2.0 * pl.col("af"))).alias("N")
    )
    return float(n_estimates["N"].median())


def map_genes_to_blocks(
    egenes: pl.DataFrame,
    metadata: pl.DataFrame,
    window: int = 1_000_000,
) -> Tuple[Dict[int, List[str]], Dict[str, List[int]]]:
    """Map eGenes to LDGM blocks based on TSS proximity.

    A gene maps to a block if the block's genomic interval overlaps the gene's
    cis-window [TSS - window, TSS + window].

    Args:
        egenes: DataFrame with columns: gene_id, chrom, tss
        metadata: LDGM metadata DataFrame with columns: chrom, chromStart, chromEnd
        window: Distance in bp from TSS to include (default 1Mb)

    Returns:
        Tuple of:
        - block_to_genes: dict mapping block_index -> list of gene_ids
        - gene_to_blocks: dict mapping gene_id -> list of block_indices
    """
    block_to_genes: Dict[int, List[str]] = {}
    gene_to_blocks: Dict[str, List[int]] = {}

    # Group blocks by chromosome for efficient lookup
    block_chroms = metadata.get_column("chrom").to_numpy()
    block_starts = metadata.get_column("chromStart").to_numpy()
    block_ends = metadata.get_column("chromEnd").to_numpy()

    chrom_block_indices: Dict[int, np.ndarray] = {}
    for chrom in np.unique(block_chroms):
        chrom_block_indices[int(chrom)] = np.where(block_chroms == chrom)[0]

    for row in egenes.iter_rows(named=True):
        gene_id = row["gene_id"]
        chrom = row["chrom"]
        tss = row["tss"]
        cis_start = tss - window
        cis_end = tss + window

        block_idxs = chrom_block_indices.get(chrom, np.array([], dtype=int))
        if len(block_idxs) == 0:
            gene_to_blocks[gene_id] = []
            continue

        # Block overlaps cis-window if: block_end > cis_start AND block_start < cis_end
        overlapping = block_idxs[
            (block_ends[block_idxs] > cis_start)
            & (block_starts[block_idxs] < cis_end)
        ]

        gene_to_blocks[gene_id] = overlapping.tolist()
        for bi in overlapping:
            block_to_genes.setdefault(int(bi), []).append(gene_id)

    return block_to_genes, gene_to_blocks


def compute_link_fn_denominator(eqtl_sumstats: pl.DataFrame) -> int:
    """Compute link function denominator as total cis-variant count across genes.

    For each gene, counts its unique variants, then sums across all genes.
    This represents the effective number of SNP-observations in the pooled model.

    Args:
        eqtl_sumstats: DataFrame with gene_id, CHR, POS columns

    Returns:
        Total variant-gene observation count
    """
    per_gene = eqtl_sumstats.group_by("gene_id").agg(
        pl.struct("CHR", "POS").n_unique().alias("n_variants")
    )
    return int(per_gene["n_variants"].sum())
