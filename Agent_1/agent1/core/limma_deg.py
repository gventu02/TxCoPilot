"""
LIMMA Differential Expression Analysis for Microarray Data (via inmoose)
=========================================================================
Pure-Python limma using the inmoose library — no R, no rpy2.

Why limma and NOT PyDESeq2?
    - PyDESeq2 models raw integer counts with a negative-binomial distribution.
    - Microarray data is already normalised log2 intensity — limma's moderated
      t-test is the statistically correct model for this data type.
    - Applying DESeq2 to log2-normalised microarray data produces wrong results.

Statistical workflow (mirrors R limma exactly):
    1. Build a ~0 + condition design matrix (patsy) — explicit group columns,
       no intercept, so contrasts are unambiguous.
    2. lmFit()         — fit a gene-wise linear model to every probe at once.
    3. makeContrasts() — define disease − control comparison as a contrast vector.
    4. contrasts_fit() — re-express the fit in terms of that contrast.
    5. eBayes()        — empirical Bayes shrinkage of gene-wise variances.
    6. topTable()      — extract moderated t-statistics + BH-adjusted p-values
                         for ALL genes (number=np.inf).

Install:
    pip install inmoose patsy

References:
    Ritchie et al. (2015) Nucleic Acids Research 43(7).
    Colange et al. (2025) BMC Bioinformatics 26:160.
"""

import logging

import numpy as np
import pandas as pd
import patsy

from inmoose.limma import (
    eBayes,
    lmFit,
    makeContrasts,
    topTable,
    contrasts_fit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API — the only thing you need to import
# ---------------------------------------------------------------------------

def run_limma_deg(
    expr_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    disease_label: str,
    control_label: str,
    condition_column: str = "condition",
    padj_threshold: float = 0.05,
    log2fc_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run limma DEG analysis on microarray (log2-normalised) data via inmoose.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Genes x samples log2 expression matrix.
        - Index   : gene / probe identifiers (strings).
        - Columns : sample IDs matching metadata_df rows.
        - Values  : log2-normalised floats. Do NOT pass raw integer counts.
    metadata_df : pd.DataFrame
        Sample metadata table.
        - Index (or column 'sample_id') must match expr_df column names.
        - Must contain `condition_column` with `disease_label` / `control_label`.
    disease_label : str
        Label for the disease / case group in condition_column.
    control_label : str
        Label for the reference / control group.
    condition_column : str
        Column in metadata_df holding group labels. Default: "condition".
    padj_threshold : float
        BH-adjusted p-value cutoff. Default: 0.05.
    log2fc_threshold : float
        Minimum |log2FC| cutoff. Default: 0.5.

    Returns
    -------
    pd.DataFrame
        Significant DEGs with columns:
            gene           - gene identifier
            log2FoldChange - log2(disease / control)
            pvalue         - raw moderated t-test p-value
            padj           - BH-adjusted p-value
        Sorted by padj ascending. Returns empty DataFrame if nothing passes.

    Raises
    ------
    ImportError  - inmoose or patsy not installed.
    ValueError   - malformed inputs (wrong labels, empty overlap, integer data).
    RuntimeError - inmoose call failed.

    Example
    -------
    >>> deg_df = run_limma_deg(
    ...     expr_df=expr,           # shape (20000, 36), log2 intensities
    ...     metadata_df=meta,       # shape (36, 3), has "condition" column
    ...     disease_label="IgAN",
    ...     control_label="Control",
    ...     padj_threshold=0.05,
    ...     log2fc_threshold=0.5,
    ... )
    >>> deg_df.columns.tolist()
    ['gene', 'log2FoldChange', 'pvalue', 'padj']
    """
    # Step 1: validate
    _validate_inputs(expr_df, metadata_df, condition_column,
                     disease_label, control_label)

    # Step 2: align samples
    meta = _align_metadata(expr_df, metadata_df, condition_column,
                           disease_label, control_label)
    expr = expr_df[meta.index.tolist()]  # restrict + re-order columns to match meta

    n_disease = (meta[condition_column] == disease_label).sum()
    n_control = (meta[condition_column] == control_label).sum()
    logger.info(
        f"limma | {disease_label} n={n_disease}, {control_label} n={n_control} "
        f"| {expr.shape[0]} genes"
    )

    # Step 3: run limma
    results_df = _run_limma(expr, meta, condition_column,
                            disease_label, control_label)

    # Step 4: filter to significant DEGs
    sig_mask = (
        (results_df["padj"] < padj_threshold)
        & (results_df["log2FoldChange"].abs() > log2fc_threshold)
    )
    deg_df = results_df[sig_mask].sort_values("padj").reset_index(drop=True)

    logger.info(
        f"limma complete | {len(deg_df)} significant DEGs "
        f"(padj<{padj_threshold}, |log2FC|>{log2fc_threshold}) "
        f"out of {len(results_df)} genes tested."
    )
    if deg_df.empty:
        logger.warning(
            "No significant DEGs found. "
            "Consider relaxing padj_threshold or log2fc_threshold."
        )

    return deg_df[["gene", "log2FoldChange", "pvalue", "padj"]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    expr_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    condition_column: str,
    disease_label: str,
    control_label: str,
) -> None:
    """Fail fast with an actionable error message."""
    if expr_df.empty:
        raise ValueError(
            "expr_df is empty. Expected a genes x samples log2 expression matrix."
        )
    if metadata_df.empty:
        raise ValueError(
            "metadata_df is empty. Expected a sample metadata table."
        )
    if condition_column not in metadata_df.columns:
        raise ValueError(
            f"condition_column '{condition_column}' not found in metadata_df. "
            f"Available columns: {metadata_df.columns.tolist()}"
        )

    labels = metadata_df[condition_column].unique().tolist()
    if disease_label not in labels:
        raise ValueError(
            f"disease_label '{disease_label}' not in metadata_df['{condition_column}']. "
            f"Found: {labels}"
        )
    if control_label not in labels:
        raise ValueError(
            f"control_label '{control_label}' not in metadata_df['{condition_column}']. "
            f"Found: {labels}"
        )
    if disease_label == control_label:
        raise ValueError(
            "disease_label and control_label must refer to different groups."
        )

    # Guard: integer dtype strongly suggests raw counts — wrong for limma
    if np.issubdtype(expr_df.values.dtype, np.integer):
        raise ValueError(
            "expr_df contains integer values. limma requires log2-normalised "
            "microarray intensities. For raw RNA-seq counts use PyDESeq2 instead."
        )

    # Soft warning for suspiciously large floats
    max_val = float(np.nanmax(expr_df.values))
    if max_val > 30:
        logger.warning(
            f"expr_df max value is {max_val:.1f} — unusually high for log2 data. "
            "Verify this is log2-normalised (not raw counts)."
        )


def _align_metadata(
    expr_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    condition_column: str,
    disease_label: str,
    control_label: str,
) -> pd.DataFrame:
    """
    Return metadata indexed by sample ID, restricted to disease + control
    samples that also appear in expr_df columns.
    """
    meta = metadata_df.copy()

    # Normalise index: accept index-as-sample-ID or a 'sample_id' column
    if "sample_id" in meta.columns and not set(meta.index).issubset(set(expr_df.columns)):
        meta = meta.set_index("sample_id")

    # Keep only the two groups of interest
    pre_filter_values = meta[condition_column].value_counts(dropna=False).to_dict()
    meta = meta[meta[condition_column].isin([disease_label, control_label])]
    logger.info(
        f"_align_metadata: pre-filter condition values={pre_filter_values}, "
        f"post-filter samples={len(meta)}"
    )

    # Intersect with expression columns (preserves expr_df column order)
    common = [s for s in expr_df.columns if s in meta.index]
    if not common:
        raise ValueError(
            "No overlap between expr_df column names and metadata_df index. "
            "Check that sample IDs match exactly (case-sensitive)."
        )

    dropped = set(expr_df.columns) - set(meta.index)
    if dropped:
        logger.warning(
            f"{len(dropped)} expr_df samples not in metadata and excluded: "
            f"{sorted(dropped)[:5]}{'...' if len(dropped) > 5 else ''}"
        )

    meta = meta.loc[common]

    # Enforce minimum replicate count
    for label in (disease_label, control_label):
        n = (meta[condition_column] == label).sum()
        if n < 2:
            raise ValueError(
                f"Only {n} sample(s) for '{label}'. "
                "limma requires >= 2 replicates per group."
            )

    return meta


def _run_limma(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    condition_column: str,
    disease_label: str,
    control_label: str,
) -> pd.DataFrame:
    """
    Core inmoose limma pipeline.
    Returns unfiltered results DataFrame with columns:
        gene, log2FoldChange, pvalue, padj
    """
    # ── Design matrix via patsy (~0 + condition = no intercept) ────────────
    # Control is listed first so it becomes the reference level.
    condition_cat = pd.Categorical(
        meta[condition_column].values,
        categories=[control_label, disease_label],
    )
    design_df = pd.DataFrame(
        {"condition": condition_cat},
        index=meta.index,
    )

    # patsy produces columns like "condition[T.IgAN]" or "conditionControl"
    # depending on whether intercept is suppressed. With "0 +" it gives
    # "condition[Control]" and "condition[IgAN]" — we strip brackets.
    design = patsy.dmatrix("0 + condition", data=design_df, return_type="dataframe")
    design.columns = [
        c.replace("[", "").replace("]", "").replace("T.", "")
        for c in design.columns
    ]

    disease_col = f"condition{disease_label}"
    control_col = f"condition{control_label}"

    if disease_col not in design.columns or control_col not in design.columns:
        raise RuntimeError(
            f"Design matrix columns {design.columns.tolist()} do not contain "
            f"expected '{disease_col}' and '{control_col}'. "
            "Verify that disease_label / control_label exactly match condition_column values."
        )

    logger.info(
        f"Design matrix: shape={design.shape}, columns={design.columns.tolist()}, "
        f"condition values={meta[condition_column].unique().tolist()}"
    )

    # ── lmFit ───────────────────────────────────────────────────────────────
    # inmoose lmFit expects: obj shape = (genes x samples), design shape = (samples x groups)
    fit = lmFit(expr, design)
    fit.index = expr.index.tolist()   # attach gene names for topTable
    # inmoose renames columns to "column0", "column1", ... but contrasts_fit
    # uses pandas @ which requires column/index name alignment. Restore names
    # on all DataFrame attributes that participate in matrix multiplication.
    for attr in ("coefficients", "stdev_unscaled"):
        df = getattr(fit, attr, None)
        if isinstance(df, pd.DataFrame) and len(df.columns) == len(design.columns):
            df.columns = design.columns
    if isinstance(fit.cov_coefficients, pd.DataFrame):
        fit.cov_coefficients.columns = design.columns
        fit.cov_coefficients.index = design.columns

    # ── makeContrasts ───────────────────────────────────────────────────────
    contrast_str = f"{disease_col}-{control_col}"
    cont_matrix = makeContrasts(
        contrasts=[contrast_str],
        levels=design.columns.tolist(),
    )

    logger.info(
        f"contrasts_fit: fit.coefficients shape={fit.coefficients.shape}, "
        f"cont_matrix shape={cont_matrix.shape}"
    )

    # ── contrasts_fit + eBayes ──────────────────────────────────────────────
    fit2 = contrasts_fit(fit, cont_matrix)
    fit2 = eBayes(fit2)

    # ── topTable: ALL genes, BH ─────────────────────────────────────────────
    tt = topTable(
        fit2,
        coef=contrast_str,   # use contrast name (not int — pandas columns are named)
        number=np.inf,       # return all genes
        adjust_method="fdr_bh",
        p_value=1.0,         # no pre-filtering
    )

    # inmoose topTable returns a DEResults subclass with strict column validation.
    # Convert to plain DataFrame before manipulating to avoid constructor errors.
    # Column names vary by inmoose version:
    #   Older: logFC, AveExpr, t, P.Value, adj.P.Val, B
    #   Newer: log2FoldChange, lfcSE, AveExpr, stat, pvalue, adj_pvalue, B
    tt = pd.DataFrame(tt)
    tt = tt.reset_index().rename(columns={"index": "gene"})

    # Fallback: if first column is unnamed or differently named
    if "gene" not in tt.columns:
        tt = tt.rename(columns={tt.columns[0]: "gene"})

    # Normalise column names to our standard (handles both old and new inmoose)
    tt = tt.rename(columns={
        "logFC":       "log2FoldChange",
        "P.Value":     "pvalue",
        "adj.P.Val":   "padj",
        "adj_pvalue":  "padj",
    })

    tt = tt.dropna(subset=["pvalue", "padj"])
    logger.info(f"topTable returned {len(tt)} genes.")

    return tt[["gene", "log2FoldChange", "pvalue", "padj"]]