"""
Agent 2 — PROGENy Pathway Scoring
==========================================
Scores pathway activity using PROGENy (top 200 responsive genes per pathway)
and computes per-gene importance scores for CARNIVAL soft objective.

The Saez lab approach (modeling-with-carnival.ipynb):
  1. Run PROGENy (dc.mt.mlm) to get pathway activity scores per sample
  2. Test differential pathway activity (disease vs control)
  3. For significant pathways, compute gene importance = |progeny_weight| * |DE_t_stat|
  4. These scores become a soft objective in CARNIVAL via P.add_objective()

This biases CARNIVAL toward activating pathway member genes without
hard-constraining them, the solver is incentivized, not forced.

Reference: Liu et al. 2019 used γ (weightObj) for per-node penalty reduction.
CORNETO has no γ equivalent, so I added P.add_objective() as the a substitute.

I communicated with Saez lab and they confirmed this approach is valid.

Giselle Ventura

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default PROGENy model path (relative to Agent_2/)
_DEFAULT_PROGENY_PATH = Path(__file__).parent.parent / "data" / "progeny_human_top200.csv"


def score_pathways(
    expression_matrix_path: str,
    output_dir: str,
    metadata_path: Optional[str] = None,
    condition_a: str = "disease",
    condition_b: str = "control",
    progeny_model_path: Optional[str] = None,
) -> dict:
    """
    Score pathway activity using PROGENy + MLM.

    Args:
        expression_matrix_path: Path to expression matrix CSV (genes × samples).
        output_dir: Directory to write pathway_scores.csv.
        metadata_path: Path to metadata CSV with sample,condition columns.
            If provided, runs t-test (disease vs control) for significance.
            If None, runs one-sample t-test (scores vs 0).
        condition_a: Disease condition label (must match values in metadata 'condition' column).
        condition_b: Control condition label (must match values in metadata 'condition' column).
        progeny_model_path: Path to PROGENy model CSV. Default: data/progeny_human_top200.csv.

    Returns:
        Dict with keys:
            pathway_scores: DataFrame (samples × pathways) of activity scores
            significant_pathways: Dict {pathway: {score, pvalue, direction}}
            n_pathways: Total pathways scored
            n_significant: Number with p < 0.05
    """
    import decoupler as dc

    # Load PROGENy model
    progeny_path = Path(progeny_model_path) if progeny_model_path else _DEFAULT_PROGENY_PATH
    if not progeny_path.exists():
        raise FileNotFoundError(f"PROGENy model not found: {progeny_path}")
    progeny = pd.read_csv(progeny_path)
    logger.info(f"PROGENy model loaded: {len(progeny)} gene-pathway associations, "
                f"{progeny['source'].nunique()} pathways")

    # Load expression matrix and orient to samples × genes (required by dc.mt.mlm)
    from core.tf_activity import _load_and_orient_expression
    expr = _load_and_orient_expression(expression_matrix_path)
    logger.info(f"Expression matrix: {expr.shape[0]} samples × {expr.shape[1]} genes")

    # Run PROGENy via MLM (multivariate linear model)
    logger.info("Running dc.mt.mlm() for pathway activity scoring...")
    acts, pvals = dc.mt.mlm(
        data=expr,
        net=progeny,
        tmin=5,
    )
    logger.info(f"PROGENy scoring complete: {acts.shape[0]} samples × {acts.shape[1]} pathways")

    # Test significance: differential (with metadata) or one-sample (without)
    from scipy import stats as scipy_stats

    significant_pathways = {}
    pvalue_threshold = 0.05
    used_differential = False

    if metadata_path and Path(metadata_path).exists():
        meta = pd.read_csv(metadata_path)
        meta.columns = [c.strip().lower() for c in meta.columns]
        if "condition" in meta.columns and "sample" in meta.columns:
            a_samples = meta[meta["condition"] == condition_a]["sample"].tolist()
            b_samples = meta[meta["condition"] == condition_b]["sample"].tolist()
            a_in_acts = [s for s in a_samples if s in acts.index]
            b_in_acts = [s for s in b_samples if s in acts.index]

            if len(a_in_acts) >= 2 and len(b_in_acts) >= 2:
                logger.info(f"Differential pathway test: {len(a_in_acts)} {condition_a} vs "
                            f"{len(b_in_acts)} {condition_b}")
                used_differential = True
                for pathway in acts.columns:
                    a_scores = acts.loc[a_in_acts, pathway].values
                    b_scores = acts.loc[b_in_acts, pathway].values
                    t_stat, p_val = scipy_stats.ttest_ind(a_scores, b_scores, equal_var=False)
                    mean_diff = a_scores.mean() - b_scores.mean()
                    if p_val < pvalue_threshold:
                        significant_pathways[pathway] = {
                            "score": float(mean_diff),
                            "pvalue": float(p_val),
                            "direction": "up" if mean_diff > 0 else "down",
                        }
                if not significant_pathways:
                    logger.info(
                        "Differential test found 0 significant pathways (p < 0.05). "
                        "This is a valid biological result — no pathways are differentially active."
                    )
            else:
                logger.warning(
                    f"Not enough samples for differential t-test: "
                    f"{condition_a}={len(a_in_acts)}, {condition_b}={len(b_in_acts)}. "
                    f"Falling back to one-sample test (less accurate — mixes disease and control samples)."
                )

    if not used_differential:
        # One-sample fallback: tests whether pathway scores are consistently non-zero.
        # NOTE: This mixes disease + control samples, so it answers "is this pathway
        # active at all?" rather than "is it differentially active?" Results may include
        # pathways that are active in both conditions (false positives for differential analysis).
        logger.warning(
            "Using one-sample pathway test (scores vs 0). This is less accurate than "
            "a differential test. For best results, provide metadata with 'sample' and "
            "'condition' columns so disease vs control can be compared directly."
        )
        for pathway in acts.columns:
            scores = acts[pathway].values
            t_stat, p_val = scipy_stats.ttest_1samp(scores, 0)
            mean_score = scores.mean()
            if p_val < pvalue_threshold:
                significant_pathways[pathway] = {
                    "score": float(mean_score),
                    "pvalue": float(p_val),
                    "direction": "up" if mean_score > 0 else "down",
                }

    # Log results
    for pw, info in sorted(significant_pathways.items(), key=lambda x: x[1]["pvalue"]):
        logger.info(f"  {pw}: score={info['score']:.3f}, p={info['pvalue']:.4f}, {info['direction']}")

    # Save pathway scores
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    acts.to_csv(out_path / "pathway_scores.csv")
    logger.info(f"Pathway scores saved: {out_path / 'pathway_scores.csv'}")

    return {
        "pathway_scores": acts,
        "significant_pathways": significant_pathways,
        "n_pathways": acts.shape[1],
        "n_significant": len(significant_pathways),
    }


def get_progeny_gene_scores(
    significant_pathways: dict,
    contrast_tstats_path: Optional[str] = None,
    tf_gene_set: Optional[set] = None,
    progeny_model_path: Optional[str] = None,
) -> dict:
    """
    Compute per-gene importance scores for CARNIVAL soft objective.

    For each significant pathway, identifies member genes that are also
    differentially expressed, then scores them by:
        importance = abs(progeny_weight) * abs(DE_t_statistic)

    Saez lab approach (modeling-with-carnival.ipynb):
    - Targets pathway MEMBER genes (scattered throughout network)
    - NOT entry-point receptors (avoids mechanistic assumptions)
    - Dual evidence (PROGENy + DE) = higher confidence

    Args:
        significant_pathways: Dict from score_pathways() {pathway: {score, pvalue, direction}}.
        contrast_tstats_path: Path to contrast_tstats.csv (gene, t_stat).
            If None, uses PROGENy weights alone (less rigorous).
        tf_gene_set: Set of TF gene symbols — excluded to avoid double-counting
            (TFs are already in the objective as measurements).
        progeny_model_path: Path to PROGENy model CSV.

    Returns:
        Dict {gene_symbol: importance_score} for genes in significant pathways.
        Empty dict if no significant pathways or no gene overlap.
    """
    if not significant_pathways:
        logger.info("No significant pathways — skipping PROGENy gene scoring")
        return {}

    # Load PROGENy model
    progeny_path = Path(progeny_model_path) if progeny_model_path else _DEFAULT_PROGENY_PATH
    progeny = pd.read_csv(progeny_path)

    # Load contrast t-statistics if available
    de_stats = {}
    if contrast_tstats_path and Path(contrast_tstats_path).exists():
        contrast_df = pd.read_csv(contrast_tstats_path, index_col=0)
        col = contrast_df.columns[0]  # "t_stat"
        de_stats = contrast_df[col].to_dict()
        logger.info(f"Loaded {len(de_stats)} DE t-statistics from {contrast_tstats_path}")

    tf_set = tf_gene_set or set()

    # Score genes in significant pathways
    gene_scores = {}
    for pathway, info in significant_pathways.items():
        pw_genes = progeny[progeny["source"] == pathway]
        for _, row in pw_genes.iterrows():
            gene = row["target"]
            progeny_weight = abs(row["weight"])

            # Skip TFs (already in objective as measurements)
            if gene in tf_set:
                continue

            # Compute importance score
            if gene in de_stats:
                # Dual evidence: PROGENy weight × DE t-statistic
                importance = progeny_weight * abs(de_stats[gene])
            else:
                # PROGENy weight only (gene not in DE results or no contrast file)
                importance = progeny_weight

            # Keep highest score if gene appears in multiple pathways
            if gene not in gene_scores or importance > gene_scores[gene]:
                gene_scores[gene] = importance

    logger.info(
        f"PROGENy gene scores: {len(gene_scores)} genes from "
        f"{len(significant_pathways)} significant pathways "
        f"({len(tf_set)} TFs excluded)"
    )

    return gene_scores
