"""
Agent 2 — TF Activity Inference
=========================================
Infers transcription factor activities from expression data or precomputed
DEG fold-changes. Two primary entry points:

  infer_tf_activity()           — contrast-based VIPER (primary method)
  infer_tf_activity_from_degs() — DEG weighted mean fallback (no expression matrix needed)

Primary method: VIPER + CollecTRI on a contrast vector (condition A vs condition B).
  1. Compute per-gene t-statistics (IgAN vs Control) across all samples
  2. Build a 1-row contrast matrix (1 × n_genes)
  3. Run dc.mt.viper() on the contrast — this is the aREA algorithm from Alvarez 2016
  4. Select top N TFs by |NES| for CARNIVAL input

Why contrast-based VIPER (not per-sample ULM):
  - Liu et al. 2019 (the IgAN validation paper) used VIPER on a differential signature
  - Per-sample averaging of ULM scores is biologically meaningless for CARNIVAL input
  - VIPER's aREA algorithm with pleiotropy correction is more appropriate for
    weak distributed signals in complex disease / microarray data
  - dc.mt.viper is available in decoupler 2.x — no extra packages required

Key design decisions:
- Uses decoupler 2.x API exclusively (dc.op.collectri, dc.mt.viper)
- Caches CollecTRI locally to survive OmniPath server outages
- Input expression matrix: genes × samples (standard bioinformatics convention)
  — transposed internally to samples × genes for decoupler
- Metadata CSV must have columns: sample, condition
- Outputs a measurements.csv file

Output file format (measurements.csv):
    TF,NES
    RELA,2.34
    TP53,-1.87
    ...

Author: Giselle ventura
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# Default cache location (relative to this file's directory)
_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data"
_COLLECTRI_CACHE_FILENAME = "collectri_human.csv"


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def infer_tf_activity(
    expression_matrix_path: str,
    deg_file_path: str,
    output_dir: str,
    top_n_tfs: int = 50,
    confidence_levels: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    metadata_path: Optional[str] = None,
    condition_a: str = "IgAN",
    condition_b: str = "Control",
    regulon_db: str = "collectri",
) -> Tuple[pd.DataFrame, str, dict]:
    """
    Infer TF activities from expression data using CollecTRI + VIPER (contrast-based).

    Primary method: contrast-based VIPER on IgAN vs Control differential signature.
    Fallback: per-sample ULM (used when metadata_path is None or conditions not found). # should give a message

    Contrast-based VIPER workflow (preferred):
      1. Load expression matrix and sample metadata
      2. Compute per-gene t-statistics: condition_a vs condition_b
      3. Build 1-row contrast matrix (condition_a_vs_condition_b × genes)
      4. Run dc.mt.viper() — aREA algorithm with pleiotropy correction
      5. Select top N TFs by |NES|, write measurements.csv


    Args:
        expression_matrix_path: Path to expression matrix CSV.
            Standard bioinformatics orientation: genes × samples (rows=genes, cols=samples).
            Also accepts samples × genes — auto-detected.
        deg_file_path: Path to DEG results CSV. Used only to confirm gene overlap.
        output_dir: Directory to write measurements.csv.
        top_n_tfs: Number of top TFs to pass to CARNIVAL (by absolute NES). Default 50.
        confidence_levels: CollecTRI confidence levels. Default: ["A", "B", "C"] (all).
        cache_dir: Directory for CollecTRI cache. Defaults to ../data/ from agent2/.
        metadata_path: Path to sample metadata CSV with columns: sample, condition.
            If None, falls back to per-sample ULM (less accurate).
        condition_a: Disease condition label in metadata (numerator). Default: "IgAN".
        condition_b: Control condition label in metadata (denominator). Default: "Control".
        regulon_db: Regulon database to use. "collectri" (default, 1185 TFs, no confidence
            levels) or "dorothea" (429 TFs, filtered by confidence_levels A/B/C).

    Returns:
        Tuple of:
          - tf_scores_df: DataFrame of TF scores (1 row = contrast, columns = TFs)
          - measurements_path: Absolute path to measurements.csv
          - stats: Dict with diagnostic information including method_used

    Raises:
        FileNotFoundError: If expression matrix not found.
        RuntimeError: If CollecTRI cannot be loaded from any source.
        ValueError: If expression matrix has no gene overlap with CollecTRI.
    """
    if confidence_levels is None:
        confidence_levels = ["A", "B", "C"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load regulon network
    logger.info(f"Step B.1: Loading {regulon_db} regulons...")
    if regulon_db == "dorothea":
        import decoupler as dc
        levels = confidence_levels if confidence_levels else ["A", "B", "C"]
        collectri_net = dc.op.dorothea(organism="human", levels=levels)
        # DoRothEA has fractional weights (0.33, 0.5, 1.0) — don't clip to sign 
        # Just verify required columns exist.
        required = {"source", "target", "weight"}
        if not required.issubset(collectri_net.columns):
            raise ValueError(f"DoRothEA missing columns. Got: {list(collectri_net.columns)}")
        logger.info(f"DoRothEA loaded (levels={levels})")
    else:
        collectri_net = _load_collectri(
            confidence_levels=confidence_levels,
            cache_dir=cache_dir or str(_DEFAULT_CACHE_DIR),
        )
    n_interactions_raw = len(collectri_net)
    logger.info(
        f"{regulon_db} loaded: {n_interactions_raw} interactions, "
        f"{collectri_net['source'].nunique()} TFs, "
        f"regulon_db={regulon_db}"
    )

    # Step 2: Load and orient expression matrix (genes × samples → samples × genes)
    logger.info("Step B.2: Loading expression matrix...")
    expr_samples_x_genes = _load_and_orient_expression(expression_matrix_path)
    logger.info(
        f"Expression matrix oriented: {expr_samples_x_genes.shape[0]} samples × "
        f"{expr_samples_x_genes.shape[1]} genes"
    )

    # Step 3: Check gene overlap
    gene_overlap = set(expr_samples_x_genes.columns) & set(collectri_net["target"].unique())
    if len(gene_overlap) < 10:
        raise ValueError(
            f"Only {len(gene_overlap)} genes overlap between expression matrix "
            f"and CollecTRI targets. "
            f"NEXT STEP: Check that expression matrix uses HGNC gene symbols "
            f"(e.g., 'TGFB1', not Ensembl IDs like 'ENSG00000105329'). "
            f"Expression matrix has {expr_samples_x_genes.shape[1]} genes. "
            f"CollecTRI has {collectri_net['target'].nunique()} unique targets."
        )
    logger.info(
        f"Gene overlap: {len(gene_overlap)}/{expr_samples_x_genes.shape[1]} genes "
        f"({len(gene_overlap)/expr_samples_x_genes.shape[1]*100:.0f}%) match CollecTRI targets"
    )

    # Step 4: Run VIPER (contrast-based) or ULM (per-sample fallback)
    method_used = "viper_contrast"
    tf_scores_df, nes_df = _infer_with_contrast_viper(
        expr_samples_x_genes=expr_samples_x_genes,
        collectri_net=collectri_net,
        metadata_path=metadata_path,
        condition_a=condition_a,
        condition_b=condition_b,
        output_dir=str(output_path),
    )

    if tf_scores_df is None:
        # Fallback: per-sample ULM (metadata missing or conditions not found)
        logger.warning(
            "Falling back to per-sample ULM. "
            "NEXT STEP: Provide metadata_path with 'sample' and 'condition' columns "
            f"containing labels '{condition_a}' and '{condition_b}' for contrast-based VIPER which is more accurate."
        )
        method_used = "ulm_per_sample_fallback"
        ulm_scores = _run_ulm(expr_samples_x_genes, collectri_net)
        nes_series = ulm_scores.mean(axis=0)
        nes_series.name = "NES"
        nes_df = nes_series.reset_index()
        nes_df.columns = ["TF", "NES"]
        nes_df = nes_df.dropna(subset=["NES"])
        nes_df["abs_NES"] = nes_df["NES"].abs()
        nes_df = nes_df.sort_values("abs_NES", ascending=False).reset_index(drop=True)
        # Reshape to match contrast format: 1 row × TFs
        tf_scores_df = nes_df[["TF", "NES"]].set_index("TF").T.copy()
        tf_scores_df.index = ["per_sample_mean"]

    logger.info(
        f"TF scoring complete (method={method_used}): "
        f"{(nes_df['NES'] > 0).sum()} activated, "
        f"{(nes_df['NES'] < 0).sum()} repressed"
    )

    # Step 5: Select top N TFs
    top_tfs_df = nes_df.head(top_n_tfs)[["TF", "NES"]].copy()
    logger.info(
        f"Selected top {len(top_tfs_df)} TFs by |NES| "
        f"(range: {top_tfs_df['NES'].min():.2f} to {top_tfs_df['NES'].max():.2f})"
    )

    # Step 6: Write measurements.csv (top N TFs for CARNIVAL)
    measurements_path = output_path / "measurements.csv"
    top_tfs_df.to_csv(measurements_path, index=False)
    logger.info(f"measurements.csv written: {measurements_path}")

    # Save full VIPER/ULM output (all TFs, not just top N) for audit/analysis
    all_tfs_path = output_path / "tf_activity_all.csv"
    nes_df.to_csv(all_tfs_path, index=False)
    logger.info(f"Full TF activity scores ({len(nes_df)} TFs) saved: {all_tfs_path}")

    # Step 7: Build stats dict for schema
    stats = {
        "n_tfs_total":              len(nes_df),
        "n_tfs_selected":           len(top_tfs_df),
        "top_tfs_up":               nes_df[nes_df["NES"] > 0].head(10)["TF"].tolist(),
        "top_tfs_down":             nes_df[nes_df["NES"] < 0].head(10)["TF"].tolist(),
        "measurements_path":        str(measurements_path.resolve()),
        "regulon_db":               regulon_db,
        "regulon_n_interactions":   n_interactions_raw,
        "method_used":              method_used,
        "gene_overlap_count":       len(gene_overlap),
    }

    return tf_scores_df, str(measurements_path.resolve()), stats

# =============================================================================
# PUBLIC ENTRY POINT (DEG-based — no expression matrix needed) - I may get rid of this after testing
# =============================================================================

def infer_tf_activity_from_degs(
    deg_file_path: str,
    output_dir: str,
    top_n_tfs: int = 50,
    confidence_levels: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, dict]:
    """
    Infer TF activities from precomputed DEG fold-changes using CollecTRI.

    Use this when a full expression matrix is unavailable but you have
    precomputed DEG results with log2FC values. Scores each TF by the
    weighted mean log2FC of its CollecTRI target genes found in the DEG list.

    Biologically equivalent to ULM for CARNIVAL input purposes. CARNIVAL
    only needs relative TF activity direction, not absolute NES values.

    Args:
        deg_file_path: CSV with columns: gene, log2fc (extras ignored).
            Accepted column aliases: gene_name, gene_symbol, log2FoldChange, logfc.
        output_dir:    Directory to write measurements.csv.
        top_n_tfs:     Number of top TFs by |score| to pass to CARNIVAL.
        confidence_levels: CollecTRI confidence levels (default: all ["A","B","C"]).
        cache_dir:     CollecTRI cache directory.

    Returns:
        Tuple of:
          - tf_scores_df: DataFrame (1 row × n_tfs), index="deg_comparison"
          - measurements_path: Absolute path to measurements.csv
          - stats: Dict matching infer_tf_activity() stats structure

    Raises:
        FileNotFoundError: If deg_file_path does not exist.
        ValueError: If required columns missing or no TFs could be scored.
    """
    if confidence_levels is None:
        confidence_levels = ["A", "B", "C"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Validate input file
    deg_path = Path(deg_file_path)
    if not deg_path.exists():
        raise FileNotFoundError(
            f"DEG file not found: {deg_file_path}. "
            f"NEXT STEP: Check that Agent 1 wrote deg_file_path correctly in DEGResults."
        )

    # Step 2: Load CollecTRI
    logger.info("Step B.1 (DEG mode): Loading CollecTRI regulons...")
    collectri_net = _load_collectri(
        confidence_levels=confidence_levels,
        cache_dir=cache_dir or str(_DEFAULT_CACHE_DIR),
    )
    logger.info(
        f"CollecTRI loaded: {len(collectri_net)} interactions, "
        f"{collectri_net['source'].nunique()} TFs"
    )

    # Step 3: Load DEG file and normalize column names
    logger.info(f"Step B.2 (DEG mode): Loading DEG file: {deg_file_path}")
    deg_df = pd.read_csv(deg_file_path)

    col_map = {}
    for col in deg_df.columns:
        if col.lower() in ("gene", "gene_name", "gene_symbol", "symbol"):
            col_map[col] = "gene"
        elif col.lower() in ("log2fc", "log2foldchange", "lfc", "logfc"):
            col_map[col] = "log2fc"
    if col_map:
        logger.info(f"Renaming DEG columns: {col_map}")
        deg_df = deg_df.rename(columns=col_map)

    missing = {"gene", "log2fc"} - set(deg_df.columns)
    if missing:
        raise ValueError(
            f"DEG file missing required columns: {missing}. "
            f"Found columns: {list(deg_df.columns)}. "
            f"NEXT STEP: Ensure DEG file has 'gene' and 'log2fc' columns."
        )

    deg_df = deg_df.dropna(subset=["gene", "log2fc"])
    deg_scores = dict(zip(deg_df["gene"], deg_df["log2fc"]))
    logger.info(f"Loaded {len(deg_scores)} genes with log2FC scores")

    # Step 4: Score TFs — weighted mean log2FC of their CollecTRI targets
    logger.info("Step B.3 (DEG mode): Scoring TFs via weighted mean log2FC...")
    tf_scores: dict[str, float] = {}
    collectri_targets = set(collectri_net["target"].values)
    gene_overlap = set(deg_scores.keys()) & collectri_targets

    logger.info(
        f"Gene overlap: {len(gene_overlap)}/{len(deg_scores)} DEGs "
        f"found in CollecTRI targets"
    )

    if len(gene_overlap) < 5:
        raise ValueError(
            f"Only {len(gene_overlap)} DEGs overlap with CollecTRI targets. "
            f"Too few to score TFs reliably. "
            f"NEXT STEP: Check that DEG file uses HGNC gene symbols (e.g. 'TGFB1')."
        )

    for tf, group in collectri_net.groupby("source"):
        targets = group["target"].values
        weights = group["weight"].values  # +1 activation, -1 repression

        matched = [
            (t, w) for t, w in zip(targets, weights)
            if t in deg_scores
        ]
        if len(matched) < 3:  # minimum targets for reliable score
            continue

        score = sum(deg_scores[t] * w for t, w in matched) / len(matched)
        tf_scores[tf] = score

    if not tf_scores:
        raise ValueError(
            f"No TFs could be scored — no TF had ≥3 targets in the DEG list. "
            f"DEG list has {len(deg_scores)} genes, "
            f"CollecTRI overlap: {len(gene_overlap)}. "
            f"NEXT STEP: Use a less stringent DEG threshold to get more genes."
        )

    logger.info(f"Scored {len(tf_scores)} TFs from DEG fold-changes")

    # Step 5: Build sorted NES DataFrame
    nes_df = (
        pd.DataFrame.from_dict(tf_scores, orient="index", columns=["NES"])
        .reset_index()
        .rename(columns={"index": "TF"})
    )
    nes_df["abs_NES"] = nes_df["NES"].abs()
    nes_df = nes_df.sort_values("abs_NES", ascending=False).reset_index(drop=True)

    logger.info(
        f"TF scoring summary: "
        f"{(nes_df['NES'] > 0).sum()} activated, "
        f"{(nes_df['NES'] < 0).sum()} repressed"
    )

    # Step 6: Select top N TFs
    top_tfs_df = nes_df.head(top_n_tfs)[["TF", "NES"]].copy()
    logger.info(
        f"Selected top {len(top_tfs_df)} TFs by |score| "
        f"(range: {top_tfs_df['NES'].min():.2f} to {top_tfs_df['NES'].max():.2f})"
    )

    # Step 7: Write measurements.csv
    measurements_path = output_path / "measurements.csv"
    top_tfs_df.to_csv(measurements_path, index=False)
    logger.info(f"measurements.csv written: {measurements_path}")

    # Step 8: Build stats dict — same structure as infer_tf_activity()
    stats = {
        "n_tfs_total":              len(nes_df),
        "n_tfs_selected":           len(top_tfs_df),
        "top_tfs_up":               nes_df[nes_df["NES"] > 0].head(10)["TF"].tolist(),
        "top_tfs_down":             nes_df[nes_df["NES"] < 0].head(10)["TF"].tolist(),
        "measurements_path":        str(measurements_path.resolve()),
        "collectri_n_interactions": len(collectri_net),
        "method_used":              "deg_weighted_mean",
        "gene_overlap_count":       len(gene_overlap),
    }

    # Return ALL scored TFs (not just top_n) to match infer_tf_activity() behavior.
    # The top_n filter is applied downstream in _run_and_package().
    # Shape: 1 × n_all_tfs, index = comparison label
    tf_scores_df = nes_df[["TF", "NES"]].set_index("TF").T.copy()
    tf_scores_df.index = ["deg_comparison"]

    return tf_scores_df, str(measurements_path.resolve()), stats
# =============================================================================
# COLLECTRI LOADING WITH CACHE
# =============================================================================

def _load_collectri(
    organism: str = "human",
    confidence_levels: Optional[List[str]] = None,
    cache_dir: str = str(_DEFAULT_CACHE_DIR),
) -> pd.DataFrame:
    """
    Load CollecTRI regulons from cache or download via decoupler 2.x.

    Cache strategy: Download once, save to CSV, load from CSV on subsequent runs.
    This makes the pipeline robust to OmniPath server outages.

    Returns:
        DataFrame with columns: source | target | weight
        source = TF (HGNC symbol), target = regulated gene, weight = +1 or -1
    """
    cache_path = Path(cache_dir) / _COLLECTRI_CACHE_FILENAME
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Try cache first
    if cache_path.exists():
        logger.info(f"Loading CollecTRI from cache: {cache_path}")
        net = pd.read_csv(cache_path)
        net = _validate_and_filter_collectri(net, confidence_levels)
        logger.info(
            f"CollecTRI from cache: {len(net)} interactions "
            f"(after confidence filter: {confidence_levels})"
        )
        return net

    # Download via decoupler 2.x
    logger.info("CollecTRI not cached. Downloading via decoupler 2.x...")
    net = _download_collectri(organism=organism)

    # Save full network to cache (before confidence filtering)
    net.to_csv(cache_path, index=False)
    logger.info(f"CollecTRI cached to: {cache_path} ({len(net)} interactions)")

    # Apply confidence filter
    net = _validate_and_filter_collectri(net, confidence_levels)
    return net


def _download_collectri(organism: str = "human") -> pd.DataFrame:
    """
    Download CollecTRI via decoupler 2.x API.

    Expects decoupler 2.1.1+ with dc.op.collectri().
    The full network is returned (no confidence filtering here — done separately).

    Returns:
        DataFrame with columns including: source | target | weight
        CollecTRI 2.x also returns: resources | references | sign_decision
    """
    try:
        import decoupler as dc
        logger.info(f"decoupler version: {dc.__version__}")

        # decoupler 2.x: resources under dc.op.*
        if hasattr(dc, "op") and hasattr(dc.op, "collectri"):
            logger.info("Using dc.op.collectri() [decoupler 2.x]")
            net = dc.op.collectri(organism=organism)
            return _standardize_collectri_columns(net)

        # decoupler 1.x fallback (shouldn't be needed in your environment)
        if hasattr(dc, "get_collectri"):
            logger.warning(
                "Using dc.get_collectri() [decoupler 1.x — unexpected in this environment]. "
                "Expected dc.op.collectri(). Check: dc.__version__"
            )
            net = dc.get_collectri(organism=organism)
            return _standardize_collectri_columns(net)

        raise RuntimeError(
            f"decoupler {dc.__version__} has neither dc.op.collectri nor dc.get_collectri. "
            f"NEXT STEP: Run `pip install --upgrade decoupler --break-system-packages` "
            f"or install omnipath package as fallback."
        )

    except ImportError:
        raise RuntimeError(
            "decoupler is not installed. "
            "NEXT STEP: pip install decoupler --break-system-packages"
        )


def _standardize_collectri_columns(net: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure CollecTRI DataFrame has standard columns: source | target | weight.

    CollecTRI 2.x returns 6 columns: source | target | weight | resources | references | sign_decision
    We keep all columns in cache (for future filtering options) but enforce the 3 required ones exist.
    """
    required = {"source", "target", "weight"}
    if not required.issubset(net.columns):
        raise ValueError(
            f"CollecTRI DataFrame missing required columns. "
            f"Got: {list(net.columns)}. Expected at minimum: {required}. "
            f"NEXT STEP: Check decoupler version with `import decoupler as dc; print(dc.__version__)` "
            f"and inspect dc.op.collectri() output columns."
        )

    # Verify weight values are +1/-1
    unique_weights = net["weight"].dropna().unique()
    valid_weights = set(unique_weights).issubset({-1.0, 1.0, -1, 1})
    if not valid_weights:
        logger.warning(
            f"CollecTRI weight column has unexpected values: {unique_weights[:10]}. "
            f"Expected only {{-1, 1}}. Will clip to sign."
        )
        net["weight"] = np.sign(net["weight"])
        net = net[net["weight"] != 0]

    logger.info(
        f"CollecTRI columns validated: {len(net)} interactions, "
        f"{net['source'].nunique()} TFs, {net['target'].nunique()} targets"
    )
    return net


def _validate_and_filter_collectri(
    net: pd.DataFrame,
    confidence_levels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply confidence level filtering if the 'resources' column is present.

    CollecTRI doesn't have an explicit A/B/C confidence column like DoRothEA.
    The 'confidence' filtering here maps to the 'resources' column:
    - "A" resources: TRRUST, GTRD (high-quality, manually curated)
    - "B" resources: ChEA, ENCODE (medium quality, ChIP-seq based)
    - "C" resources: Reactome, KEGG (pathway-based, lower specificity)

    NOTE: If the CollecTRI download doesn't include a parseable confidence column,
    all interactions are used (no filtering). This is the safe fallback.
    """
    if confidence_levels is None or set(confidence_levels) == {"A", "B", "C"}:
        # No filtering needed — use all
        logger.info(f"CollecTRI: using all {len(net)} interactions (confidence filter: all)")
        return net[["source", "target", "weight"]].dropna()

    # If 'confidence' column exists (some versions of CollecTRI include it)
    if "confidence" in net.columns:
        before = len(net)
        net = net[net["confidence"].isin(confidence_levels)]
        logger.info(
            f"CollecTRI confidence filter ({confidence_levels}): "
            f"{before} → {len(net)} interactions"
        )
    else:
        logger.info(
            f"CollecTRI has no 'confidence' column. "
            f"Using all {len(net)} interactions. "
            f"(Confidence filtering only works if decoupler includes this column.)"
        )

    return net[["source", "target", "weight"]].dropna()


# =============================================================================
# EXPRESSION MATRIX LOADING AND ORIENTATION
# =============================================================================

def _load_and_orient_expression(expression_matrix_path: str) -> pd.DataFrame:
    """
    Load expression matrix and ensure it's oriented as samples × genes.

    dc.mt.ulm() expects: data = samples × genes (rows=samples, columns=genes).
    This is the OPPOSITE of standard bioinformatics convention (genes × samples).

    Auto-detection strategy:
      1. Load the CSV
      2. Check if index values look like gene symbols (e.g., contain letters)
         or like sample IDs (e.g., barcodes, numeric)
      3. Transpose if index appears to be genes (genes × samples → samples × genes)

    Args:
        expression_matrix_path: Path to expression matrix CSV.

    Returns:
        DataFrame oriented as samples × genes.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If orientation cannot be determined.
    """
    path = Path(expression_matrix_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expression matrix not found: {expression_matrix_path}. "
            f"NEXT STEP: Check that Agent 1 wrote expression_matrix_path correctly "
            f"in Agent1Result."
        )

    logger.info(f"Loading expression matrix from: {expression_matrix_path}")
    expr = pd.read_csv(expression_matrix_path, index_col=0)
    logger.info(f"Raw matrix shape: {expr.shape} (rows × columns)")

    # Determine orientation
    orientation = _detect_orientation(expr)

    if orientation == "genes_x_samples":
        logger.info(
            f"Detected genes × samples orientation. Transposing to samples × genes "
            f"(required by dc.mt.ulm)."
        )
        expr = expr.T
    elif orientation == "samples_x_genes":
        logger.info("Detected samples × genes orientation. No transpose needed.")
    else:
        # Ambiguous — make a guess based on shape
        if expr.shape[0] > expr.shape[1]:
            logger.warning(
                f"Orientation ambiguous (rows={expr.shape[0]} > cols={expr.shape[1]}). "
                f"Assuming genes × samples and transposing. "
                f"Verify by checking if resulting row names look like sample IDs."
            )
            expr = expr.T
        else:
            logger.warning(
                f"Orientation ambiguous (rows={expr.shape[0]} ≤ cols={expr.shape[1]}). "
                f"Assuming already samples × genes. "
                f"Verify by checking if column names look like gene symbols."
            )

    # Final sanity check
    logger.info(
        f"Final orientation: {expr.shape[0]} samples × {expr.shape[1]} genes. "
        f"Sample names (first 3): {list(expr.index[:3])}. "
        f"Gene names (first 3): {list(expr.columns[:3])}."
    )

    return expr


def _detect_orientation(expr: pd.DataFrame) -> str:
    """
    Heuristic to detect whether a matrix is genes × samples or samples × genes.

    Gene symbols are typically uppercase letters + optional numbers (TP53, TGFB1).
    Sample/cell barcodes typically contain hyphens, underscores, or are long strings.

    Returns: "genes_x_samples" | "samples_x_genes" | "ambiguous"
    """
    def _looks_like_gene_symbols(index_values: pd.Index, sample_size: int = 20) -> bool:
        """Check if index values look like HGNC gene symbols."""
        sample = list(index_values[:sample_size])
        gene_pattern_count = sum(
            1 for v in sample
            if isinstance(v, str)
            and v.replace("-", "").replace(".", "").isalnum()  # HGNC symbols: letters + digits
            and len(v) <= 15   # Gene symbols are short
            and v.upper() == v  # Gene symbols are uppercase (mostly)
            and not v.startswith("AAAC")  # Cell barcodes start like this
        )
        return gene_pattern_count / max(len(sample), 1) > 0.6

    index_looks_like_genes = _looks_like_gene_symbols(expr.index)
    cols_look_like_genes = _looks_like_gene_symbols(expr.columns)

    if index_looks_like_genes and not cols_look_like_genes:
        return "genes_x_samples"
    elif cols_look_like_genes and not index_looks_like_genes:
        return "samples_x_genes"
    else:
        return "ambiguous"


# =============================================================================
# CONTRAST-BASED VIPER INFERENCE (PRIMARY METHOD)
# =============================================================================

def _infer_with_contrast_viper(
    expr_samples_x_genes: pd.DataFrame,
    collectri_net: pd.DataFrame,
    metadata_path: Optional[str],
    condition_a: str,
    condition_b: str,
    output_dir: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Attempt contrast-based VIPER. Returns (None, None) if metadata is unavailable
    or conditions cannot be found — caller falls back to per-sample ULM.

    Workflow:
      1. Load metadata CSV (sample, condition columns)
      2. Split samples into condition_a and condition_b groups
      3. Compute per-gene t-statistics (condition_a vs condition_b)
      4. Build 1-row contrast matrix
      5. Run dc.mt.viper() on contrast

    Returns:
        (tf_scores_df, nes_df) where:
          tf_scores_df: DataFrame, shape 1 × n_tfs, index = 'condition_a_vs_condition_b'
          nes_df: DataFrame with columns [TF, NES, abs_NES], sorted by abs_NES descending
        OR (None, None) if contrast cannot be computed.
    """
    if metadata_path is None:
        logger.warning(
            "metadata_path not provided. Cannot compute contrast-based VIPER. "
            "NEXT STEP: Pass metadata_path to infer_tf_activity() for accurate results. "
            "Metadata CSV must have columns: sample, condition."
        )
        return None, None

    meta_path = Path(metadata_path)
    if not meta_path.exists():
        logger.warning(
            f"Metadata file not found: {metadata_path}. "
            "Falling back to per-sample ULM. "
            "NEXT STEP: Check that metadata_path is correct."
        )
        return None, None

    # Load metadata
    meta = pd.read_csv(meta_path)

    # Normalise column names
    meta.columns = [c.strip().lower() for c in meta.columns]
    if "sample" not in meta.columns or "condition" not in meta.columns:
        logger.warning(
            f"Metadata file must have 'sample' and 'condition' columns. "
            f"Found: {list(meta.columns)}. Falling back to per-sample ULM."
        )
        return None, None

    meta = meta.set_index("sample")

    # Get sample lists for each condition
    a_samples = meta[meta["condition"] == condition_a].index.tolist()
    b_samples = meta[meta["condition"] == condition_b].index.tolist()

    # Filter to samples present in expression matrix
    a_samples = [s for s in a_samples if s in expr_samples_x_genes.index]
    b_samples = [s for s in b_samples if s in expr_samples_x_genes.index]

    if len(a_samples) == 0 or len(b_samples) == 0:
        logger.warning(
            f"Could not find samples for conditions '{condition_a}' (n={len(a_samples)}) "
            f"or '{condition_b}' (n={len(b_samples)}) in expression matrix. "
            f"Available conditions: {meta['condition'].unique().tolist()}. "
            f"Falling back to per-sample ULM. "
            f"NEXT STEP: Check condition labels match metadata exactly (case-sensitive)."
        )
        return None, None

    logger.info(
        f"Step B.3: Computing contrast: {condition_a} (n={len(a_samples)}) "
        f"vs {condition_b} (n={len(b_samples)})"
    )

    # Compute per-gene t-statistics
    t_stats, p_vals = scipy_stats.ttest_ind(
        expr_samples_x_genes.loc[a_samples].values,
        expr_samples_x_genes.loc[b_samples].values,
        axis=0,
        equal_var=False,  # Welch's t-test — robust to unequal group sizes (27 vs 9)
    )

    # Replace NaN t-stats (from zero-variance genes) with 0.0 — no variance = no signal
    n_nan = np.isnan(t_stats).sum()
    if n_nan > 0:
        logger.warning(f"{n_nan} genes had NaN t-statistics (zero variance) — replaced with 0.0")
        t_stats = np.nan_to_num(t_stats, nan=0.0)

    contrast_label = f"{condition_a}_vs_{condition_b}"
    contrast = pd.DataFrame(
        [t_stats],
        columns=expr_samples_x_genes.columns,
        index=[contrast_label],
    )

    logger.info(
        f"Contrast computed: {contrast_label}. "
        f"t-stat range: [{t_stats.min():.2f}, {t_stats.max():.2f}]. "
        f"Genes with |t|>2: {(abs(t_stats) > 2).sum()}"
    )

    # Save contrast t-statistics for audit/analysis
    if output_dir:
        contrast_path = Path(output_dir) / "contrast_tstats.csv"
        contrast.T.to_csv(contrast_path, header=["t_stat"])
        logger.info(f"Contrast t-statistics saved: {contrast_path}")

    # Run VIPER on contrast
    logger.info("Step B.4: Running dc.mt.viper() on contrast vector...")
    tf_acts_df = _run_viper(contrast, collectri_net)
    logger.info(f"VIPER complete: {tf_acts_df.shape[1]} TFs scored")

    # Build nes_df (TF, NES, abs_NES) sorted by abs_NES
    nes_series = tf_acts_df.iloc[0]  # single row = contrast
    nes_series.name = "NES"
    nes_df = nes_series.reset_index()
    nes_df.columns = ["TF", "NES"]
    nes_df = nes_df.dropna(subset=["NES"])
    nes_df["abs_NES"] = nes_df["NES"].abs()
    nes_df = nes_df.sort_values("abs_NES", ascending=False).reset_index(drop=True)

    return tf_acts_df, nes_df


def _run_viper(
    contrast: pd.DataFrame,
    collectri_net: pd.DataFrame,
    tmin: int = 5,
) -> pd.DataFrame:
    """
    Run VIPER (aREA algorithm) for TF activity inference on a contrast vector.

    Uses decoupler 2.x API: dc.mt.viper(data=contrast, net=collectri_net).
    Input contrast must be a 1-row DataFrame: index = contrast label, columns = genes.

    VIPER advantages over ULM for contrast-based analysis:
      - Rank-based enrichment: less sensitive to outlier genes
      - Separate scoring of activation vs repression targets 
      - Pleiotropy correction: accounts for TFs sharing target genes
      - Matches methodology of Liu et al. 2019 (original IgAN CARNIVAL paper)

    Args:
        contrast: DataFrame, shape 1 × n_genes. Values are t-statistics or similar.
        collectri_net: CollecTRI regulons with source | target | weight columns.
        tmin: Minimum targets per TF to score (default 5).

    Returns:
        DataFrame, shape 1 × n_tfs (NES scores).

    Raises:
        RuntimeError: If decoupler is not installed or VIPER fails.
    """
    try:
        import decoupler as dc
    except ImportError:
        raise RuntimeError(
            "decoupler is not installed. "
            "NEXT STEP: pip install decoupler --break-system-packages"
        )

    logger.info(
        f"Running dc.mt.viper(): "
        f"contrast shape={contrast.shape}, "
        f"{collectri_net['source'].nunique()} TFs in network, "
        f"tmin={tmin}"
    )

    try:
        acts, pvals = dc.mt.viper(
            data=contrast,
            net=collectri_net,
            tmin=tmin,
        )
    except TypeError as e:
        raise RuntimeError(
            f"dc.mt.viper() failed with TypeError: {e}. "
            f"NEXT STEP: Verify decoupler 2.x API — first arg is 'data', second is 'net'. "
            f"Run: import decoupler as dc; print(dc.__version__)"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"dc.mt.viper() failed: {type(e).__name__}: {e}. "
            f"NEXT STEP: Check that contrast DataFrame has gene symbols as column names "
            f"and numeric t-statistics as values."
        ) from e

    if acts is None or acts.empty:
        raise ValueError(
            "dc.mt.viper() returned empty results. "
            "NEXT STEP: Check gene overlap — CollecTRI needs genes in contrast columns. "
            "Run: set(contrast.columns) & set(collectri_net['target'])"
        )

    logger.info(
        f"VIPER results: {acts.shape[1]} TFs scored. "
        f"NES range: [{acts.values.min():.2f}, {acts.values.max():.2f}]"
    )

    return acts


# =============================================================================
# ULM INFERENCE (FALLBACK — per-sample, used when metadata unavailable, not optimal but can get some signal, especially for strong perturbations or pseudobulk data)
# =============================================================================

def _run_ulm(
    expr_samples_x_genes: pd.DataFrame,
    collectri_net: pd.DataFrame,
    tmin: int = 5,
) -> pd.DataFrame:
    """
    Run Univariate Linear Model (ULM) for TF activity inference.

    FALLBACK METHOD, used only when metadata_path is not provided or conditions
    cannot be matched. The primary method is contrast-based VIPER (_run_viper).

    Per-sample ULM produces one NES per TF per sample. These are averaged across
    samples in the caller, which is less accurate than contrast-based VIPER because:
      - Averaging mixes disease and control samples
      - Does not represent the differential signature
      - Weaker signal for complex disease / microarray data

    Use this when: no metadata is available, or for scRNA-seq pseudobulk where
    a single pseudobulk sample is already a contrast-equivalent.

    Uses decoupler 2.x API: dc.mt.ulm(data=..., net=...).
    Returns a DataFrame of shape: samples × TFs (normalized enrichment scores).
    """
    try:
        import decoupler as dc
    except ImportError:
        raise RuntimeError(
            "decoupler is not installed. "
            "NEXT STEP: pip install decoupler --break-system-packages"
        )

    logger.info(
        f"Running dc.mt.ulm(): "
        f"{expr_samples_x_genes.shape[0]} samples × {expr_samples_x_genes.shape[1]} genes, "
        f"{collectri_net['source'].nunique()} TFs in network, "
        f"tmin={tmin}"
    )

    try:
        # decoupler 2.x: methods under dc.mt.*
        # Returns tuple: (acts DataFrame, pvals DataFrame), both samples × TFs
        acts, pvals = dc.mt.ulm(
            data=expr_samples_x_genes,
            net=collectri_net,
            tmin=tmin,
        )
    except TypeError as e:
        # Catch common API mismatch: if someone accidentally uses 1.x kwargs
        raise RuntimeError(
            f"dc.mt.ulm() failed with TypeError: {e}. "
            f"NEXT STEP: Verify decoupler 2.x API — first arg is 'data' not 'mat'. "
            f"Run: import decoupler as dc; print(dc.__version__)"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"dc.mt.ulm() failed: {type(e).__name__}: {e}. "
            f"NEXT STEP: Check that expression matrix contains numeric values only "
            f"and has gene symbols as column names."
        ) from e

    if acts is None or acts.empty:
        raise ValueError(
            f"dc.mt.ulm() returned empty results. "
            f"NEXT STEP: Check gene overlap — CollecTRI needs genes in your expression matrix. "
            f"Run: set(expr.columns) & set(collectri_net['target'])"
        )

    logger.info(
        f"ULM results: {acts.shape[0]} samples × {acts.shape[1]} TFs scored. "
        f"NES range: [{acts.values.min():.2f}, {acts.values.max():.2f}]"
    )

    return acts


# =============================================================================
# CONVENIENCE: QUICK DIAGNOSTIC
# =============================================================================

def run_collectri_diagnostic(cache_dir: Optional[str] = None) -> None:
    """
    Print a diagnostic report for CollecTRI loading.
    Run this in your environment to verify everything is working.

    Usage:
        from agent2.core.tf_activity import run_collectri_diagnostic
        run_collectri_diagnostic()
    """
    print("=" * 60)
    print("CollecTRI / decoupler Diagnostic")
    print("=" * 60)

    try:
        import decoupler as dc
        print(f" decoupler version: {dc.__version__}")
        print(f"   Has dc.op: {hasattr(dc, 'op')}")
        print(f"   Has dc.op.collectri: {hasattr(dc, 'op') and hasattr(dc.op, 'collectri')}")
        print(f"   Has dc.mt: {hasattr(dc, 'mt')}")
        print(f"   Has dc.mt.ulm: {hasattr(dc, 'mt') and hasattr(dc.mt, 'ulm')}")
    except ImportError:
        print(" decoupler NOT installed. Run: pip install decoupler --break-system-packages")
        return

    print()
    try:
        net = _load_collectri(
            confidence_levels=["A", "B", "C"],
            cache_dir=cache_dir or str(_DEFAULT_CACHE_DIR),
        )
        print(f" CollecTRI loaded: {len(net)} interactions")
        print(f"   TFs: {net['source'].nunique()}")
        print(f"   Targets: {net['target'].nunique()}")
        print(f"   Columns: {list(net.columns)}")
        print(f"   Weight values: {sorted(net['weight'].unique())}")
        print(f"   Sample rows:")
        print(net.head(3).to_string(index=False))
    except Exception as e:
        print(f" CollecTRI loading failed: {e}")


if __name__ == "__main__":
    run_collectri_diagnostic()
