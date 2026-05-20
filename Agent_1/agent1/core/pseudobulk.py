"""
Pseudobulk Aggregation
=======================
Aggregate single-cell RNA-seq data into pseudobulk samples per cell type and donor.

Why Pseudobulk?
    - CARNIVAL is designed for bulk-like data
    - Proper biological replication (donor-level, not cell-level)
    - Reduces false discoveries vs naive single-cell DE methods
    - See: Squair et al. 2021, Crowell et al. 2020 (muscat)

Input:
    Expression DataFrame (cells × genes) + metadata with cell type and donor columns
    
Output:
    Expression DataFrame (genes × pseudobulk_samples) + metadata for DEG analysis

Usage:
    from agent1.core.pseudobulk import PseudobulkAggregator
    
    aggregator = PseudobulkAggregator(min_cells=10)
    counts_df, meta_df, info = aggregator.aggregate(
        expression_df=expr,    # cells × genes
        metadata_df=meta,
        cell_type_column="celltype",
        cell_type="Hepatic Stellate Cells",
        sample_column="Sample_Name",
        condition_column="Diagnosis",
    )
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from agent1.schemas_agent1 import PseudobulkInfo

logger = logging.getLogger(__name__)


class PseudobulkAggregator:
    """
    Aggregates scRNA-seq cells into pseudobulk samples.
    
    For each donor, sums raw counts across all cells of the target cell type
    to create one pseudobulk sample. This produces a bulk-like expression
    matrix suitable for DESeq2 and CARNIVAL.
    """
    
    def __init__(self, min_cells: int = 10):
        """
        Args:
            min_cells: Minimum cells per donor to form a pseudobulk sample.
                       Donors with fewer cells are excluded.
        """
        self.min_cells = min_cells
    
    def aggregate(
        self,
        expression_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        cell_type_column: str,
        cell_type: str,
        sample_column: str,
        condition_column: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, PseudobulkInfo]:
        """
        Aggregate cells into pseudobulk for a specific cell type.
        
        Args:
            expression_df: cells × genes expression matrix (raw counts)
            metadata_df: Cell-level metadata (index aligned with expression_df)
            cell_type_column: Column name for cell type annotations
            cell_type: Which cell type to aggregate
            sample_column: Column for biological replicate grouping (donor/patient)
            condition_column: Column for condition labels (disease/control)
            
        Returns:
            Tuple of:
                - counts_df: genes × pseudobulk_samples (summed counts)
                - meta_df: Sample-level metadata for DEG analysis
                - info: PseudobulkInfo with aggregation statistics
                
        Raises:
            ValueError: If cell type not found or insufficient donors
        """
        logger.info(f"Pseudobulk aggregation: '{cell_type}' from '{cell_type_column}'")
        
        # ── Step 1: Filter to target cell type ──
        if cell_type_column not in metadata_df.columns:
            raise ValueError(
                f"Cell type column '{cell_type_column}' not found. "
                f"Available: {list(metadata_df.columns)}"
            )
        
        cell_mask = metadata_df[cell_type_column] == cell_type
        n_cells_total = cell_mask.sum()
        
        if n_cells_total == 0:
            available = metadata_df[cell_type_column].unique().tolist()
            raise ValueError(
                f"No cells found for cell type '{cell_type}'. "
                f"Available types: {available}"
            )
        
        logger.info(f"Found {n_cells_total} cells of type '{cell_type}'")
        
        # Verify expression and metadata indices are aligned
        if not expression_df.index.equals(metadata_df.index):
            shared = expression_df.index.intersection(metadata_df.index)
            if len(shared) == 0:
                raise ValueError(
                    "Expression and metadata have no shared indices — cannot align."
                )
            logger.warning(
                f"Index mismatch: expression has {len(expression_df)} rows, "
                f"metadata has {len(metadata_df)} rows, {len(shared)} shared. "
                f"Aligning to shared indices."
            )
            expression_df = expression_df.loc[shared]
            metadata_df = metadata_df.loc[shared]
            cell_mask = metadata_df[cell_type_column] == cell_type
            n_cells_total = cell_mask.sum()

        # Filter expression and metadata to this cell type
        ct_expr = expression_df.loc[cell_mask]
        ct_meta = metadata_df.loc[cell_mask]
        
        # ── Step 2: Aggregate by donor ──
        pseudobulk_counts = []
        pseudobulk_metadata = []
        cells_per_donor = {}
        donors_excluded = []
        
        for donor in ct_meta[sample_column].unique():
            donor_mask = ct_meta[sample_column] == donor
            donor_cells = ct_expr.loc[donor_mask]
            n_cells = len(donor_cells)
            
            if n_cells < self.min_cells:
                logger.warning(
                    f"Donor '{donor}': only {n_cells} cells "
                    f"(< {self.min_cells} minimum). Excluding."
                )
                donors_excluded.append(str(donor))
                continue
            
            # Sum counts across cells for this donor
            summed = donor_cells.sum(axis=0)
            pseudobulk_counts.append(summed)
            cells_per_donor[str(donor)] = n_cells
            
            # Get condition for this donor (should be the same for all cells)
            donor_condition = ct_meta.loc[donor_mask, condition_column].iloc[0]
            
            pseudobulk_metadata.append({
                "sample": f"{cell_type}_{donor}",
                "condition": donor_condition,
                "donor": str(donor),
                "cell_type": cell_type,
                "n_cells_aggregated": n_cells,
            })
        
        # ── Step 3: Build output DataFrames ──
        if len(pseudobulk_counts) < 2:
            raise ValueError(
                f"Only {len(pseudobulk_counts)} donors passed the minimum cell threshold "
                f"({self.min_cells} cells). Need at least 2 for DEG analysis. "
                f"Excluded donors: {donors_excluded}"
            )
        
        sample_names = [m["sample"] for m in pseudobulk_metadata]
        
        # genes × samples (standard bulk format)
        counts_df = pd.DataFrame(
            np.array([s.values for s in pseudobulk_counts]).T,
            index=ct_expr.columns,   # gene names
            columns=sample_names,
        )
        
        meta_df = pd.DataFrame(pseudobulk_metadata)
        meta_df = meta_df.set_index("sample")
        
        # ── Step 4: Build info object ──
        info = PseudobulkInfo(
            cell_type_analyzed=cell_type,
            cell_type_column=cell_type_column,
            sample_column=sample_column,
            n_cells_total=n_cells_total,
            n_donors=len(pseudobulk_counts),
            cells_per_donor=cells_per_donor,
            donors_excluded=donors_excluded,
            min_cells_threshold=self.min_cells,
        )
        
        logger.info(
            f"Pseudobulk complete: {counts_df.shape[0]} genes × "
            f"{counts_df.shape[1]} samples ({info.n_donors} donors, "
            f"{len(donors_excluded)} excluded)"
        )
        
        # Log condition distribution
        condition_counts = meta_df["condition"].value_counts()
        for cond, count in condition_counts.items():
            logger.info(f"  {cond}: {count} donors")
        
        return counts_df, meta_df, info
