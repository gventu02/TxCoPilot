"""
Quality Assessment
===================
Deterministic dataset quality metrics for Agent 2 parameter selection.

NO LLM involved — pure computation. The LLM interprets these results
in the output_explanation module.

Metrics Computed:
    1. Library size statistics (mean, CV across samples)
    2. Housekeeping gene CV (primary noise indicator)
    3. Gene detection rate
    4. Gene filtering statistics
    5. Overall quality tier classification

How Agent 2 Uses This:
    - noise_level → DoRothEA confidence levels (microarray) or CollecTRI selection (RNA-seq/scRNA-seq), CARNIVAL beta
    - n_samples → statistical power assessment
    - n_genes → network size expectations
    - quality_tier → overall parameter aggressiveness

Usage:
    from agent1.core.quality_assessment import QualityAssessor
    
    assessor = QualityAssessor(config)
    report = assessor.assess(counts_df, metadata_df)
    print(report.noise_level)   # "low" / "medium" / "high"
    print(report.quality_tier)  # "high" / "medium" / "low"
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from agent1.schemas_agent1 import (
    Agent1Config,
    DataType,
    NoiseLevel,
    PseudobulkInfo,
    QualityReport,
    QualityTier,
)

logger = logging.getLogger(__name__)


class QualityAssessor:
    """
    Computes deterministic quality metrics from expression data.
    All methods are pure functions — no LLM, no network calls.
    """
    
    def __init__(self, config: Agent1Config):
        self.config = config
    
    def assess(
        self,
        counts_df: pd.DataFrame,
        metadata_df: Optional[pd.DataFrame] = None,
        data_type: DataType = DataType.BULK,
        pseudobulk_info: Optional[PseudobulkInfo] = None,
    ) -> QualityReport:
        """
        Run full quality assessment on expression data.
        
        Args:
            counts_df: genes × samples expression matrix (raw or pseudobulk counts)
            metadata_df: sample metadata (optional, for additional context)
            data_type: detected data type from validator
            pseudobulk_info: aggregation details if scRNA input
            
        Returns:
            QualityReport with all metrics and classified quality tier
        """
        logger.info(
            f"Quality assessment: {counts_df.shape[0]} genes × {counts_df.shape[1]} samples"
        )
        
        warnings = []
        
        # ── 1. Basic dimensions ──
        n_genes, n_samples = counts_df.shape
        
        if n_samples < 3:
            warnings.append(
                f"Very few samples (n={n_samples}). DEG analysis requires at least "
                "3 per group. Results will have low statistical power."
            )
        
        if n_genes < 5000:
            warnings.append(
                f"Low gene count ({n_genes}). Expected 15,000-25,000 for a "
                "typical RNA-seq dataset. Check filtering or data completeness."
            )
        
        # ── 2. Library size statistics ──
        library_sizes = counts_df.sum(axis=0)
        mean_lib_size = float(library_sizes.mean())
        cv_lib_size = float(library_sizes.std() / library_sizes.mean()) if mean_lib_size > 0 else 0.0
        
        if cv_lib_size > 1.0:
            warnings.append(
                f"High variability in library sizes (CV={cv_lib_size:.2f}). "
                "This may indicate batch effects or technical issues."
            )
        
        # ── 3. Gene filtering ──
        genes_before = n_genes

        if data_type == DataType.MICROARRAY:
            # Microarray values are log2 intensities — count-based filtering is invalid
            genes_after = n_genes
            gene_detection_rate = float((counts_df > 0).any(axis=1).mean())
            warnings.append(
                f"Microarray data detected (log2 range "
                f"{counts_df.values.min():.1f}\u2013{counts_df.values.max():.1f}). "
                "Skipping count-based gene filtering. All genes retained."
            )
        else:
            min_counts = self.config.min_counts_per_gene
            min_samples = self.config.min_samples_per_gene

            gene_mask = (counts_df >= min_counts).sum(axis=1) >= min(min_samples, n_samples)
            genes_after = int(gene_mask.sum())

            # Gene detection rate (non-zero in at least 1 sample)
            gene_detection_rate = float((counts_df > 0).any(axis=1).mean())

            if genes_after < genes_before * 0.3:
                warnings.append(
                    f"Heavy gene filtering: {genes_before} → {genes_after} "
                    f"({genes_after/genes_before*100:.0f}% retained). "
                    "Many genes have very low counts."
                )
        
        # ── 4. Housekeeping gene CV (primary noise indicator) ──
        cv_housekeeping, hk_genes_found = self._calculate_housekeeping_cv(counts_df)
        
        if cv_housekeeping is not None:
            noise_level = self._classify_noise(cv_housekeeping)
        else:
            # No housekeeping genes found — fall back to overall CV
            noise_level = self._fallback_noise_estimate(counts_df)
            warnings.append(
                "No housekeeping genes found in dataset. "
                "Noise level estimated from overall expression variability."
            )
        
        # ── 5. Overall quality tier ──
        quality_tier = self._classify_quality(
            n_samples=n_samples,
            noise_level=noise_level,
            gene_detection_rate=gene_detection_rate,
            cv_lib_size=cv_lib_size,
        )
        
        report = QualityReport(
            n_genes=n_genes,
            n_samples=n_samples,
            data_type=data_type,
            mean_library_size=mean_lib_size,
            cv_library_size=cv_lib_size,
            cv_housekeeping=cv_housekeeping,
            noise_level=noise_level,
            housekeeping_genes_found=hk_genes_found,
            genes_before_filtering=genes_before,
            genes_after_filtering=genes_after,
            gene_detection_rate=gene_detection_rate,
            quality_tier=quality_tier,
            warnings=warnings,
            pseudobulk_info=pseudobulk_info,
        )
        
        logger.info(
            f"Quality: tier={quality_tier.value}, noise={noise_level.value}, "
            f"genes={genes_after}/{genes_before}, "
            f"lib_size_cv={cv_lib_size:.2f}"
        )
        
        return report
    
    # =========================================================================
    # NOISE ESTIMATION
    # =========================================================================
    
    def _calculate_housekeeping_cv(
        self, counts_df: pd.DataFrame
    ) -> tuple[Optional[float], List[str]]:
        """
        Calculate coefficient of variation of housekeeping genes.
        
        Housekeeping genes should have stable expression across samples.
        High CV in these genes indicates noisy data.
        
        Returns:
            (cv_value, list_of_found_genes)
            cv_value is None if no housekeeping genes found
        """
        hk_genes = self.config.housekeeping_genes
        
        # Find which housekeeping genes are in the data (case-insensitive)
        gene_index_lower = {g.lower(): g for g in counts_df.index}
        found_genes = []
        
        for hk in hk_genes:
            if hk.lower() in gene_index_lower:
                found_genes.append(gene_index_lower[hk.lower()])
        
        if not found_genes:
            logger.warning(
                f"No housekeeping genes found. Looked for: {hk_genes}"
            )
            return None, []
        
        logger.info(f"Housekeeping genes found: {found_genes}")
        
        # Calculate CV for each housekeeping gene across samples
        hk_expr = counts_df.loc[found_genes]
        
        # Normalize by library size to avoid confounding
        lib_sizes = counts_df.sum(axis=0)
        hk_normalized = hk_expr.div(lib_sizes, axis=1) * 1e6  # CPM normalization
        
        # CV per gene, then take median across genes
        cvs = []
        for gene in found_genes:
            gene_values = hk_normalized.loc[gene]
            mean_val = gene_values.mean()
            if mean_val > 0:
                cv = float(gene_values.std() / mean_val)
                cvs.append(cv)
                logger.debug(f"  {gene}: CV = {cv:.3f}")
        
        if not cvs:
            return None, found_genes
        
        median_cv = float(np.median(cvs))
        logger.info(f"Housekeeping gene median CV: {median_cv:.3f}")
        
        return median_cv, found_genes
    
    @staticmethod
    def _classify_noise(cv: float) -> NoiseLevel:
        """
        Classify noise level from housekeeping gene CV.
        
        Thresholds (from architecture spec):
            CV < 0.2  → LOW noise: clean data, aggressive parameters OK
            CV 0.2-0.5 → MEDIUM noise: standard parameters
            CV > 0.5  → HIGH noise: conservative parameters needed
        """
        if cv < 0.2:
            return NoiseLevel.LOW
        elif cv <= 0.5:
            return NoiseLevel.MEDIUM
        else:
            return NoiseLevel.HIGH
    
    @staticmethod
    def _fallback_noise_estimate(counts_df: pd.DataFrame) -> NoiseLevel:
        """
        Fallback noise estimation when no housekeeping genes found.
        Uses overall inter-sample correlation as a proxy.
        """
        # Sample a subset of genes for speed
        n_sample = min(1000, len(counts_df))
        sampled = counts_df.sample(n=n_sample, random_state=42)
        
        # Normalize
        lib_sizes = counts_df.sum(axis=0)
        normalized = sampled.div(lib_sizes, axis=1) * 1e6
        
        # Calculate pairwise correlation between samples
        corr_matrix = normalized.corr()
        
        # Average off-diagonal correlation
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        avg_corr = corr_matrix.values[mask].mean()
        
        # High correlation → low noise
        if avg_corr > 0.9:
            return NoiseLevel.LOW
        elif avg_corr > 0.7:
            return NoiseLevel.MEDIUM
        else:
            return NoiseLevel.HIGH
    
    # =========================================================================
    # QUALITY CLASSIFICATION
    # =========================================================================
    
    @staticmethod
    def _classify_quality(
        n_samples: int,
        noise_level: NoiseLevel,
        gene_detection_rate: float,
        cv_lib_size: float,
    ) -> QualityTier:
        """
        Classify overall dataset quality tier.
        
        This is a composite score based on multiple factors.
        Agent 2 uses this for overall parameter aggressiveness.
        """
        score = 0
        
        # Sample count contribution
        if n_samples >= 10:
            score += 2
        elif n_samples >= 6:
            score += 1
        # else: 0 points
        
        # Noise contribution
        if noise_level == NoiseLevel.LOW:
            score += 2
        elif noise_level == NoiseLevel.MEDIUM:
            score += 1
        
        # Gene detection
        if gene_detection_rate > 0.7:
            score += 1
        
        # Library size consistency
        if cv_lib_size < 0.5:
            score += 1
        
        # Classify: max score = 6
        if score >= 5:
            return QualityTier.HIGH
        elif score >= 3:
            return QualityTier.MEDIUM
        else:
            return QualityTier.LOW
