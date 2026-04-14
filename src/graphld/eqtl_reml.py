"""eQTL GraphREML: per-gene cis-eQTL heritability estimation pooled across genes."""

from multiprocessing import Value
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from .heritability import FLAGS, GraphREML, ModelOptions, MethodOptions, _get_softmax_link_function
from .io import merge_snplists, merge_alleles, partition_variants
from .likelihood import gaussian_likelihood
from .precision import PrecisionOperator
from .multiprocessing_template import SharedData


class EqtlGraphREML(GraphREML):
    """GraphREML subclass for pooled cis-eQTL analysis.

    Each LD block may be used by multiple genes. The LDGM and Cholesky
    factorization are shared across genes at each block; only the Z-scores
    (and thus Pz vectors) differ per gene. Likelihood, gradient, and hessian
    contributions are summed across genes at each block.
    """

    @staticmethod
    def create_shared_memory(
        metadata: pl.DataFrame, block_data: list, **kwargs
    ) -> SharedData:
        """Create shared memory arrays for eQTL REML.

        Uses 'annotations' key instead of 'sumstats' from block_data.
        """
        num_params = kwargs.get("num_params")
        num_blocks = len(metadata)
        num_variants = sum(
            len(d["annotations"]) for d in block_data if d["annotations"] is not None
        )
        return SharedData({
            "params": num_params,
            "variant_data": num_variants,
            "likelihood": num_blocks,
            "gradient": num_blocks * num_params,
            "hessian": num_blocks * num_params**2,
        })

    @classmethod
    def prepare_block_data(cls, metadata: pl.DataFrame, **kwargs) -> list:
        """Prepare per-block data including annotations and per-gene Z-scores.

        Args:
            metadata: LDGM metadata DataFrame
            **kwargs: Must include:
                annotation_data: pl.DataFrame with CHR, POS, annotation columns
                annotation_columns: List[str]
                eqtl_sumstats: pl.DataFrame with gene_id, CHR, POS, REF, ALT, Z
                block_to_genes: Dict[int, List[str]]
                method: MethodOptions

        Returns:
            List of dicts per block with keys: annotations, gene_zscores,
            variant_offset, block_index, Pz_genes, block_name
        """
        annotation_data: pl.DataFrame = kwargs["annotation_data"]
        eqtl_sumstats: pl.DataFrame = kwargs["eqtl_sumstats"]
        block_to_genes: Dict[int, List[str]] = kwargs["block_to_genes"]
        method: MethodOptions = kwargs.get("method")

        # Partition annotations by block
        annot_blocks = partition_variants(metadata, annotation_data)

        # Pre-index eQTL data by (gene_id, chrom) for efficient filtering
        block_chroms = metadata.get_column("chrom").to_list()
        block_starts = metadata.get_column("chromStart").to_list()
        block_ends = metadata.get_column("chromEnd").to_list()

        # Build block data
        cumulative_variants = 0
        block_data = []
        block_names = metadata.get_column("name").to_list() if "name" in metadata.columns else []

        for i, annot_block in enumerate(annot_blocks):
            gene_ids = block_to_genes.get(i, [])

            # Collect per-gene Z-scores for this block
            gene_zscores = {}
            if gene_ids:
                chrom = block_chroms[i]
                bstart = block_starts[i]
                bend = block_ends[i]

                # Filter eQTL data to this block's region first
                block_eqtl = eqtl_sumstats.filter(
                    (pl.col("CHR") == chrom)
                    & (pl.col("POS") >= bstart)
                    & (pl.col("POS") < bend)
                )

                for gene_id in gene_ids:
                    gene_df = block_eqtl.filter(pl.col("gene_id") == gene_id)
                    if not gene_df.is_empty():
                        gene_zscores[gene_id] = gene_df.select("POS", "REF", "ALT", "Z")

            block_data.append({
                "annotations": annot_block,
                "gene_zscores": gene_zscores,
                "variant_offset": cumulative_variants,
                "block_index": i,
                "Pz_genes": None,  # populated during INITIALIZE
                "block_name": block_names[i] if i < len(block_names) else None,
            })
            cumulative_variants += len(annot_block)

        if method and method.verbose:
            total_gene_block_pairs = sum(len(d["gene_zscores"]) for d in block_data)
            blocks_with_genes = sum(1 for d in block_data if d["gene_zscores"])
            print(f"{total_gene_block_pairs} gene-block pairs across {blocks_with_genes} blocks")
            print(f"{cumulative_variants} total annotation variants")

        return block_data

    @staticmethod
    def _initialize_block_annotations(
        ldgm: PrecisionOperator,
        annot_df: pl.DataFrame,
        annotation_columns: list,
        gene_zscores: Dict[str, pl.DataFrame],
        match_by_position: bool,
        verbose: bool = False,
    ) -> Tuple[Optional[PrecisionOperator], Dict[str, np.ndarray], Optional[np.ndarray]]:
        """Initialize block by merging annotations and computing per-gene Pz vectors.

        1. Merges annotations with the LDGM (no Z-scores).
        2. For each gene, matches Z-scores to the merged variant list,
           applies allele flipping, and computes Pz_g = ldgm @ z_g.

        Args:
            ldgm: LDGM PrecisionOperator
            annot_df: Annotation DataFrame (CHR, POS, annotation columns, no Z)
            annotation_columns: List of annotation column names
            gene_zscores: Dict mapping gene_id -> DataFrame with POS, REF, ALT, Z
            match_by_position: Whether to match by position
            verbose: Print diagnostics

        Returns:
            Tuple of:
            - ldgm: Modified PrecisionOperator (or None if no variants)
            - Pz_genes: Dict mapping gene_id -> Pz vector
            - annot_indices: Array mapping merged variants to annotation rows
        """
        if annot_df.is_empty():
            return None, {}, None

        # Ensure required columns exist for position matching
        # merge_snplists needs a POS column; annotation data should have it
        ldgm, annot_indices = merge_snplists(
            ldgm,
            annot_df,
            match_by_position=match_by_position,
            pos_col="POS",
            add_cols=annotation_columns,
            add_allelic_cols=[],
            modify_in_place=True,
        )

        if len(ldgm.variant_info) == 0:
            if verbose:
                print("No variants after merging annotations with LDGM - skipping block")
            return None, {}, None

        # Build reference table from unique LDGM variants for vectorized matching
        vi = ldgm.variant_info
        first_mask = vi.select(pl.col("index").is_first_distinct()).to_numpy().flatten()
        unique_vi = vi.filter(first_mask)
        unique_indices = unique_vi.get_column("index").to_numpy()

        ref_df = pl.DataFrame({
            "position": unique_vi.get_column("position"),
            "ldgm_idx": unique_indices,
            "anc_alleles": unique_vi.get_column("anc_alleles"),
            "deriv_alleles": unique_vi.get_column("deriv_alleles"),
        })

        # Compute Pz for each gene using vectorized join + allele matching
        Pz_genes = {}
        for gene_id, gene_df in gene_zscores.items():
            if gene_df.is_empty():
                continue

            # Join gene Z-scores to LDGM variants on position
            matched = gene_df.join(
                ref_df, left_on="POS", right_on="position", how="inner",
            )
            if matched.is_empty():
                continue

            # Vectorized allele phase computation
            phase = merge_alleles(
                matched["anc_alleles"],
                matched["deriv_alleles"],
                matched["REF"],
                matched["ALT"],
            )
            # Filter to matched alleles (phase != 0)
            matched = matched.with_columns(phase.alias("phase"))
            matched = matched.filter(pl.col("phase") != 0)

            if matched.is_empty():
                continue

            # Build z vector with allele-flipped Z-scores
            z = np.zeros(ldgm.shape[0])
            idx = matched["ldgm_idx"].to_numpy()
            z_vals = matched["Z"].to_numpy() * matched["phase"].to_numpy()
            # If multiple gene variants map to the same LDGM index, take first
            _, first_occ = np.unique(idx, return_index=True)
            z[idx[first_occ]] = z_vals[first_occ]

            Pz_g = ldgm @ z.reshape(-1, 1)
            Pz_genes[gene_id] = Pz_g

            if verbose:
                print(f"  Gene {gene_id}: {len(first_occ)}/{len(gene_df)} variants matched")

        return ldgm, Pz_genes, annot_indices

    @staticmethod
    def _compute_batched_contributions(
        Pz_genes: Dict[str, np.ndarray],
        ldgm: PrecisionOperator,
        del_M_del_a: np.ndarray,
        num_samples: int,
        likelihood_only: bool,
        seed: Optional[int] = None,
    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute summed likelihood/gradient/hessian across all genes in one block.

        Optimizations vs per-gene loop:
        - ONE batched solve for all Pz vectors (instead of 3N separate solves)
        - ONE inverse_diagonal call (shared across genes, not repeated N times)
        - ONE gradient matmul after summing node_grads (not N separate matmuls)
        - ONE batched solve for all hessian b_scaled matrices

        Args:
            Pz_genes: Dict mapping gene_id -> Pz vector (M, 1)
            ldgm: PrecisionOperator with current diagonal
            del_M_del_a: Derivative of precision diagonal wrt parameters (M, p)
            num_samples: Number of samples for xdiag trace estimator
            likelihood_only: If True, skip gradient and hessian
            seed: Random seed for trace estimator

        Returns:
            (total_likelihood, total_gradient_or_None, total_hessian_or_None)
        """
        gene_ids = list(Pz_genes.keys())
        N_genes = len(gene_ids)
        M = ldgm.shape[0]

        # Stack all Pz into matrix: (M, N_genes)
        Pz_all = np.column_stack([Pz_genes[g].ravel() for g in gene_ids])

        # ONE batched solve: b_all = M^{-1} @ Pz_all
        b_all = ldgm.solve(Pz_all)  # (M, N_genes)

        # Likelihoods: L_g = -0.5 * (n*log(2pi) + logdet + Pz_g^T b_g)
        logdet = ldgm.logdet()
        quads = np.sum(Pz_all * b_all, axis=0)  # (N_genes,)
        total_likelihood = float(np.sum(
            -0.5 * (M * np.log(2 * np.pi) + logdet + quads)
        ))

        if likelihood_only:
            return total_likelihood, None, None

        p = del_M_del_a.shape[1]

        # ONE inverse_diagonal call (shared — depends only on M, not Pz)
        minv_diag = ldgm.inverse_diagonal(
            method="xdiag", n_samples=num_samples, seed=seed,
        ).flatten()  # (M,)

        # Gradient: sum node_grads THEN multiply by del_M_del_a once
        # node_grad_g = -0.5 * (minv_diag - b_g^2)
        # total_node_grad = -0.5 * (N*minv_diag - sum_g(b_g^2))
        sum_b_sq = np.sum(b_all ** 2, axis=1)  # (M,)
        total_node_grad = -0.5 * (N_genes * minv_diag - sum_b_sq)
        total_gradient = total_node_grad @ del_M_del_a  # (p,)

        # Hessian: batch all b_scaled, ONE solve
        # b_scaled_g = b_g * del_M_del_a, shape (M, p) per gene
        # Stack into (M, N_genes*p), solve once
        b_scaled_all = b_all[:, :, None] * del_M_del_a[:, None, :]  # (M, N, p)
        b_scaled_all = b_scaled_all.reshape(M, N_genes * p)  # (M, N*p)
        minv_b_scaled = ldgm.solve(b_scaled_all)  # (M, N*p)

        # Sum per-gene hessians: H_g = -0.5 * b_scaled_g^T @ minv_b_scaled_g
        total_hessian = np.zeros((p, p))
        for j in range(N_genes):
            s = slice(j * p, (j + 1) * p)
            total_hessian += b_scaled_all[:, s].T @ minv_b_scaled[:, s]
        total_hessian *= -0.5

        return total_likelihood, total_gradient, total_hessian

    @classmethod
    def process_block(
        cls,
        ldgm: PrecisionOperator,
        flag: Value,
        shared_data: SharedData,
        block_offset: int,
        block_data: Any = None,
        worker_params: Tuple[ModelOptions, MethodOptions] = None,
    ):
        """Process a single block with multiple gene Z-score vectors.

        On INITIALIZE: merges annotations with LDGM and computes per-gene Pz.
        On COMPUTE_ALL: updates diagonal once, then sums likelihood/gradient/hessian
        across all genes sharing this block.
        """
        model_options, method_options = worker_params
        seed = None
        if method_options.gradient_seed is not None:
            seed = method_options.gradient_seed + block_data["block_index"]
            np.random.seed(seed)

        if flag.value == FLAGS["INITIALIZE"]:
            ldgm, Pz_genes, annot_indices = cls._initialize_block_annotations(
                ldgm,
                block_data["annotations"],
                model_options.annotation_columns,
                block_data["gene_zscores"],
                method_options.match_by_position,
                method_options.verbose,
            )

            if ldgm is not None and Pz_genes:
                # Scale to effect-size units
                for gene_id in Pz_genes:
                    Pz_genes[gene_id] /= np.sqrt(model_options.sample_size)
                ldgm.times_scalar(model_options.intercept / model_options.sample_size)

                # Store annot_indices in variant_info for process_block indexing
                if annot_indices is not None and "annot_indices" not in ldgm.variant_info.columns:
                    ldgm.variant_info = ldgm.variant_info.with_columns(
                        pl.Series("annot_indices", annot_indices)
                    )

            block_data["Pz_genes"] = Pz_genes if Pz_genes else None
            return  # ldgm is modified in place and reused

        # Non-INITIALIZE iterations
        Pz_genes = block_data.get("Pz_genes")
        if Pz_genes is None or len(Pz_genes) == 0:
            return

        # Score test flags not supported for eQTL v1
        if flag.value in (FLAGS["COMPUTE_VARIANT_SCORE"], FLAGS["COMPUTE_VARIANT_HESSIAN"],
                          FLAGS["WRITE_VARIANT_INFO"]):
            return

        # Setup: same as base class
        annot_indices = ldgm.variant_info.select("annot_indices").to_numpy().flatten()
        max_index = np.max(annot_indices) + 1 if len(annot_indices) > 0 else 0
        variant_offset = block_data["variant_offset"]
        block_variants = slice(variant_offset, variant_offset + max_index)
        annot = ldgm.variant_info.select(model_options.annotation_columns).to_numpy()
        params = shared_data["params"].reshape(-1, 1)
        block_index = block_data["block_index"]
        num_annot = len(model_options.annotation_columns)

        old_variant_h2 = shared_data["variant_data", block_variants][annot_indices]
        likelihood_only = flag.value == FLAGS["COMPUTE_LIKELIHOOD_ONLY"]

        # Step 1: Update block model once (shared across genes)
        per_variant_h2, del_M_del_a = cls._update_block_model(
            ldgm, annot, params, model_options.link_fn_denominator, old_variant_h2,
        )

        # Step 2: Batched computation across all genes
        total_likelihood, total_gradient, total_hessian = cls._compute_batched_contributions(
            Pz_genes, ldgm, del_M_del_a,
            method_options.gradient_num_samples, likelihood_only, seed,
        )

        ldgm.del_factor()  # Free memory after all genes processed

        # Store results
        shared_data["likelihood", block_index] = total_likelihood
        variant_h2_padded = np.zeros(max_index)
        variant_h2_padded[annot_indices] = per_variant_h2.ravel()
        shared_data["variant_data", block_variants] = variant_h2_padded

        if likelihood_only:
            return

        gradient_slice = slice(block_index * num_annot, (block_index + 1) * num_annot)
        shared_data["gradient", gradient_slice] = total_gradient.flatten()

        hessian_slice = slice(block_index * num_annot**2, (block_index + 1) * num_annot**2)
        shared_data["hessian", hessian_slice] = total_hessian.flatten()


    @classmethod
    def _get_block_annotation_df(cls, block_data: list, annotation_columns: list) -> pl.DataFrame:
        """Get concatenated annotation DataFrame from block data.

        Overrides base class pattern which uses block_data[i]["sumstats"].
        eQTL block data uses "annotations" key instead.
        """
        SPECIAL_COLNAMES = ["SNP", "CHR", "POS"]
        frames = []
        for d in block_data:
            annot = d.get("annotations")
            if annot is not None and len(annot) > 0:
                # Select only the columns that exist in this DataFrame
                available = [c for c in annotation_columns + SPECIAL_COLNAMES if c in annot.columns]
                frames.append(annot.select(available))
        return pl.concat(frames) if frames else pl.DataFrame()

    @classmethod
    def supervise(cls, manager, shared_data, block_data, **kwargs):
        """Override supervise to use 'annotations' key instead of 'sumstats'."""
        # Temporarily add "sumstats" keys pointing to "annotations" so the
        # parent supervise method can access them
        for d in block_data:
            if "sumstats" not in d:
                d["sumstats"] = d.get("annotations", pl.DataFrame())

        return super().supervise(manager, shared_data, block_data, **kwargs)


def run_eqtl_graphREML(
    model_options: ModelOptions,
    method_options: MethodOptions,
    annotation_data: pl.DataFrame,
    eqtl_sumstats: pl.DataFrame,
    block_to_genes: Dict[int, List[str]],
    ldgm_metadata_path: str,
    populations: Union[str, List[str]] = None,
    chromosomes: Optional[Union[int, List[int]]] = None,
):
    """Run eQTL GraphREML: pooled cis-eQTL heritability estimation.

    Args:
        model_options: Model configuration (annotations, params, sample_size, etc.)
        method_options: Method configuration (iterations, convergence, etc.)
        annotation_data: Variant annotations DataFrame (CHR, POS, annotation columns)
        eqtl_sumstats: eQTL summary stats (gene_id, CHR, POS, REF, ALT, Z)
        block_to_genes: Mapping from block index to list of gene_ids
        ldgm_metadata_path: Path to LDGM metadata CSV
        populations: Population(s) to use (e.g., "EUR")
        chromosomes: Chromosome(s) to include

    Returns:
        Dictionary with parameters, heritability, enrichment, jackknife estimates, etc.
    """
    if populations is None:
        raise ValueError("Populations must be provided")

    # Force settings appropriate for eQTL
    method_options.match_by_position = True
    method_options.use_surrogate_markers = False

    if method_options.verbose:
        n_genes = eqtl_sumstats["gene_id"].n_unique()
        n_variants = len(eqtl_sumstats)
        n_blocks_used = len(block_to_genes)
        print(f"eQTL REML: {n_genes} genes, {n_variants} variant-gene pairs, {n_blocks_used} blocks")
        print(f"Sample size N={model_options.sample_size}")
        print(f"link_fn_denominator={model_options.link_fn_denominator}")

    run_fn = EqtlGraphREML.run_serial if method_options.run_serial else EqtlGraphREML.run
    return run_fn(
        ldgm_metadata_path,
        populations=populations,
        chromosomes=chromosomes,
        num_processes=method_options.num_processes,
        worker_params=(model_options, method_options),
        num_params=len(model_options.annotation_columns),
        model=model_options,
        method=method_options,
        num_iterations=method_options.num_iterations,
        verbose=method_options.verbose,
        convergence_tol=method_options.convergence_tol,
        sample_size=model_options.sample_size,
        # eQTL-specific kwargs consumed by prepare_block_data
        annotation_data=annotation_data,
        annotation_columns=model_options.annotation_columns,
        eqtl_sumstats=eqtl_sumstats,
        block_to_genes=block_to_genes,
    )
