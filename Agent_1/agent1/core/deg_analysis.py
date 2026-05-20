"""
Differential Expression Analysis
==================================
PyDESeq2 wrapper for identifying differentially expressed genes.
Also supports loading pre-computed DEG results.

Why PyDESeq2?
    - Python-native implementation of DESeq2 (the gold standard for bulk RNA-seq DE)
    - No R bridge needed for this step
    - Handles library size normalization, dispersion estimation, and shrinkage

Input:
    Expression matrix (genes × samples, raw counts) + metadata with condition column

Output:
    DEGResults schema with summary stats + CSV file of full results

Usage:
    from agent1.core.deg_analysis import DEGAnalyzer
    
    analyzer = DEGAnalyzer(config)
    deg_results = analyzer.run_pydeseq2(
        counts_df=counts,
        metadata_df=meta,
        condition_column="condition",
        disease_label="IPF",
        control_label="Control",
    )
    
    # Or load pre-computed:
    deg_results = analyzer.load_precomputed(deg_df, "IPF vs Control")
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agent1.schemas_agent1 import Agent1Config, DEGResults, DEG_COLUMN_ALIASES

logger = logging.getLogger(__name__)


class DEGAnalyzer:
    """
    Differential expression analysis using PyDESeq2 or pre-computed results.
    """
    
    def __init__(self, config: Agent1Config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_pydeseq2(
        self,
        counts_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        condition_column: str = "condition",
        disease_label: str = "disease",
        control_label: str = "control",
        padj_threshold: Optional[float] = None,
        log2fc_threshold: Optional[float] = None,
    ) -> DEGResults:
        """
        Run DESeq2 via PyDESeq2.
        
        Args:
            counts_df: genes × samples raw count matrix (integers)
            metadata_df: sample metadata with condition column
            condition_column: which column has disease/control labels
            disease_label: label for disease group
            control_label: label for control group
            padj_threshold: adjusted p-value cutoff (default from config)
            log2fc_threshold: log2 fold change cutoff (default from config)
            
        Returns:
            DEGResults with summary and path to full results CSV
        """
        try:
            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.ds import DeseqStats
        except ImportError:
            raise ImportError(
                "PyDESeq2 required for DEG analysis. "
                "Install with: pip install pydeseq2"
            )
        
        padj_thr = padj_threshold or self.config.padj_threshold
        lfc_thr = log2fc_threshold or self.config.log2fc_threshold

        # Guard: reject log2-normalised microarray data (PyDESeq2 requires raw counts)
        values = counts_df.values
        if (values > 0).all() and values.max() < 20 and (values == 0).mean() < 0.01:
            raise ValueError(
                "Input looks like log2-normalised microarray data (all positive, "
                f"max={values.max():.1f}, <1% zeros). PyDESeq2 requires raw integer "
                "counts. Use run_limma_deg() for microarray data instead."
            )

        logger.info(
            f"Running PyDESeq2: {disease_label} vs {control_label} "
            f"(padj < {padj_thr}, |log2FC| > {lfc_thr})"
        )

        # ── Step 1: Prepare data for PyDESeq2 ──
        # PyDESeq2 expects: samples × genes (transposed from our genes × samples)
        counts_t = counts_df.T

        # Ensure counts are integers (PyDESeq2 requirement)
        counts_t = counts_t.round().astype(int)
        
        # Align metadata index with counts columns
        common_samples = counts_t.index.intersection(metadata_df.index)
        if len(common_samples) == 0:
            raise ValueError(
                "No matching sample names between expression matrix columns "
                "and metadata index. Check that sample IDs match."
            )
        
        counts_t = counts_t.loc[common_samples]
        meta_aligned = metadata_df.loc[common_samples].copy()

        # Keep only disease + control samples (exclude stray labels like "Nan")
        valid_mask = meta_aligned[condition_column].isin([disease_label, control_label])
        if not valid_mask.all():
            n_dropped = (~valid_mask).sum()
            stray = meta_aligned.loc[~valid_mask, condition_column].unique().tolist()
            logger.warning(
                f"Dropping {n_dropped} sample(s) with unexpected condition values: "
                f"{stray} (keeping only '{disease_label}' and '{control_label}')"
            )
            meta_aligned = meta_aligned[valid_mask]
            counts_t = counts_t.loc[meta_aligned.index]

        logger.info(
            f"Data aligned: {counts_t.shape[0]} samples × {counts_t.shape[1]} genes"
        )
        
        # ── Step 2: Filter low-count genes ──
        min_counts = self.config.min_counts_per_gene
        min_samples = self.config.min_samples_per_gene
        
        genes_before = counts_t.shape[1]
        gene_mask = (counts_t >= min_counts).sum(axis=0) >= min_samples
        counts_t = counts_t.loc[:, gene_mask]
        genes_after = counts_t.shape[1]
        
        logger.info(
            f"Gene filtering: {genes_before} → {genes_after} "
            f"(>= {min_counts} counts in >= {min_samples} samples)"
        )
        
        # ── Step 3: Run PyDESeq2 ──
        logger.info("Fitting DESeq2 model...")
        
        dds = DeseqDataSet(
            counts=counts_t,
            metadata=meta_aligned,
            design=f"~{condition_column}",
        )
        
        dds.deseq2()
        
        # ── Step 4: Extract results ──
        logger.info(f"Extracting results: {disease_label} vs {control_label}")
        
        stat_res = DeseqStats(
            dds,
            contrast=[condition_column, disease_label, control_label],
        )
        stat_res.summary()
        
        # Get results DataFrame
        results_df = stat_res.results_df.copy()
        results_df = results_df.reset_index()
        results_df = results_df.rename(columns={"index": "gene"})
        
        # ── Step 5: Classify DEGs ──
        results_df["significant"] = (
            (results_df["padj"] < padj_thr)
            & (results_df["log2FoldChange"].abs() > lfc_thr)
        )
        results_df["direction"] = np.where(
            results_df["log2FoldChange"] > 0, "up", "down"
        )
        
        # Sort by significance
        results_df = results_df.sort_values("padj")
        
        # ── Step 6: Save full results ──
        output_path = (self.output_dir / "deg_results.csv").resolve()
        results_df.to_csv(output_path, index=False)
        logger.info(f"DEG results saved: {output_path}")
        
        # ── Step 7: Build summary ──
        sig = results_df[results_df["significant"]]
        n_up = (sig["direction"] == "up").sum()
        n_down = (sig["direction"] == "down").sum()
        
        top_up = (
            sig[sig["direction"] == "up"]
            .nlargest(10, "log2FoldChange")["gene"]
            .tolist()
        )
        top_down = (
            sig[sig["direction"] == "down"]
            .nsmallest(10, "log2FoldChange")["gene"]
            .tolist()
        )
        
        # Build warnings
        warnings = []
        # ENHANCED: adaptive_deg_threshold — In future, if n_significant is too
        # high or too low, the LLM-guided enhanced tool will iteratively adjust
        # thresholds (max 2 iterations), then fall back to top 100 DEGs.
        if len(sig) < 50:
            warnings.append(
                f"Only {len(sig)} significant DEGs found. Consider relaxing thresholds "
                f"(current: padj < {padj_thr}, |log2FC| > {lfc_thr})."
            )
        elif len(sig) > 2000:
            warnings.append(
                f"{len(sig)} significant DEGs found — this is a large number. "
                f"Consider tightening thresholds for a more focused CARNIVAL analysis. "
                f"(current: padj < {padj_thr}, |log2FC| > {lfc_thr})."
            )
        
        if results_df["padj"].isna().sum() > genes_after * 0.5:
            warnings.append(
                "Many genes have NA adjusted p-values. This may indicate low counts "
                "or high dispersion. Results should be interpreted cautiously."
            )
        
        deg_results = DEGResults(
            source="pydeseq2",
            comparison=f"{disease_label} vs {control_label}",
            padj_threshold=padj_thr,
            log2fc_threshold=lfc_thr,
            total_genes_tested=genes_after,
            n_significant=len(sig),
            n_upregulated=n_up,
            n_downregulated=n_down,
            deg_file_path=str(output_path),
            top_upregulated=top_up,
            top_downregulated=top_down,
            warnings=warnings,
        )
        
        logger.info(
            f"DEG analysis complete: {len(sig)} significant "
            f"({n_up} up, {n_down} down) out of {genes_after} tested"
        )
        
        return deg_results
    
    def load_precomputed(
        self,
        deg_df: pd.DataFrame,
        comparison: str,
        padj_threshold: Optional[float] = None,
        log2fc_threshold: Optional[float] = None,
    ) -> DEGResults:
        """
        Load pre-computed DEG results (from external DESeq2, edgeR, limma, etc).
        
        Args:
            deg_df: DataFrame with columns: gene, log2FoldChange, padj
            comparison: Description of comparison, e.g., "IPF vs Control"
            padj_threshold: adjusted p-value cutoff
            log2fc_threshold: log2 fold change cutoff
            
        Returns:
            DEGResults summary
        """
        padj_thr = padj_threshold or self.config.padj_threshold
        lfc_thr = log2fc_threshold or self.config.log2fc_threshold
        
        logger.info(f"Loading pre-computed DEGs: {comparison}")
        
        rename_map = {}
        for standard_name, aliases in DEG_COLUMN_ALIASES.items():
            if standard_name not in deg_df.columns:
                for alias in aliases:
                    matched = [c for c in deg_df.columns if c.lower() == alias.lower()]
                    if matched:
                        rename_map[matched[0]] = standard_name
                        break
        
        if rename_map:
            logger.info(f"Renaming columns: {rename_map}")
            deg_df = deg_df.rename(columns=rename_map)
        
        # Validate required columns
        required = {"gene", "log2FoldChange", "padj"}
        missing = required - set(deg_df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {list(deg_df.columns)}. "
                f"Expected: gene, log2FoldChange (or log2fc), padj (or fdr)."
            )
        
        # Classify
        deg_df = deg_df.copy()
        deg_df["significant"] = (
            (deg_df["padj"] < padj_thr)
            & (deg_df["log2FoldChange"].abs() > lfc_thr)
        )
        deg_df["direction"] = np.where(
            deg_df["log2FoldChange"] > 0, "up", "down"
        )
        
        # Save copy
        output_path = (self.output_dir / "deg_results_precomputed.csv").resolve()
        deg_df.to_csv(output_path, index=False)
        
        sig = deg_df[deg_df["significant"]]
        n_up = (sig["direction"] == "up").sum()
        n_down = (sig["direction"] == "down").sum()
        
        top_up = (
            sig[sig["direction"] == "up"]
            .nlargest(10, "log2FoldChange")["gene"]
            .tolist()
        )
        top_down = (
            sig[sig["direction"] == "down"]
            .nsmallest(10, "log2FoldChange")["gene"]
            .tolist()
        )
        
        warnings = []
        if len(sig) < 50:
            warnings.append(
                f"Only {len(sig)} significant DEGs in pre-computed results."
            )
        elif len(sig) > 2000:
            warnings.append(
                f"{len(sig)} significant DEGs — consider tightening thresholds."
            )
        
        return DEGResults(
            source="precomputed",
            comparison=comparison,
            padj_threshold=padj_thr,
            log2fc_threshold=lfc_thr,
            total_genes_tested=len(deg_df),
            n_significant=len(sig),
            n_upregulated=n_up,
            n_downregulated=n_down,
            deg_file_path=str(output_path),
            top_upregulated=top_up,
            top_downregulated=top_down,
            warnings=warnings,
        )
