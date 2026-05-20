"""
Data Validator
===============
Validates input data and detects data type (bulk RNA-seq vs scRNA-seq).
Identifies key metadata columns for downstream analysis.

This is the FIRST module that runs in Agent 1's pipeline.
Its job is to answer: "What did the user give us, and is it usable?"

Key Responsibilities:
    1. Detect input format (CSV, TSV, H5AD)
    2. Classify as bulk RNA-seq, scRNA-seq, or pre-computed DEGs
    3. Validate expression matrix shape and values
    4. Auto-detect metadata columns (condition, cell type, sample/donor)
    5. Return ClarificationRequest if anything is ambiguous

Usage:
    from agent1.core.data_validator import DataValidator
    
    validator = DataValidator(config)
    result = validator.validate(user_query)
    
    if result.needs_clarification:
        # Ask user and re-run
    else:
        # Proceed with result.expression_df, result.metadata_df, etc.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent1.schemas_agent1 import (
    Agent1Config,
    ClarificationRequest,
    DataType,
    UserQuery,
)

from agent1.schemas_agent1 import DEG_COLUMN_ALIASES

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS: Known column names for auto-detection
# =============================================================================

# Common condition column names (case-insensitive matching)
CONDITION_COLUMN_CANDIDATES = [
    "condition", "diagnosis", "disease", "group", "treatment",
    "status", "phenotype", "class", "label",
]

# Common cell type column names
CELLTYPE_COLUMN_CANDIDATES = [
    "celltype", "cell_type", "cell_annotation", "cluster",
    "cell_label", "population", "leiden", "louvain",
]

# Common sample/donor column names (for pseudobulk grouping)
SAMPLE_COLUMN_CANDIDATES = [
    "sample_name", "sample", "donor", "patient", "subject",
    "individual", "sample_id", "donor_id", "patient_id",
    "orig.ident",
]


# =============================================================================
# VALIDATION RESULT: Internal data class (not part of Agent 1 → Agent 2 contract)
# =============================================================================

@dataclass
class ValidationResult:
    """Internal result from data validation. NOT serialized to Agent 2."""
    
    # Core data
    expression_df: Optional[pd.DataFrame] = None  # genes × samples
    metadata_df: Optional[pd.DataFrame] = None
    precomputed_degs: Optional[pd.DataFrame] = None
    
    # Detection results
    data_type: DataType = DataType.BULK
    input_format: str = ""  # "csv", "tsv", "h5ad"
    
    # Auto-detected columns
    condition_column: Optional[str] = None
    cell_type_column: Optional[str] = None
    sample_column: Optional[str] = None
    
    # Condition labels
    disease_label: Optional[str] = None
    control_label: Optional[str] = None
    
    # Available cell types (only for scRNA)
    available_cell_types: Dict[str, int] = field(default_factory=dict)
    
    # Clarifications needed
    clarifications: List[ClarificationRequest] = field(default_factory=list)
    
    # Validation warnings
    warnings: List[str] = field(default_factory=list)
    
    @property
    def needs_clarification(self) -> bool:
        return len(self.clarifications) > 0


# =============================================================================
# DATA VALIDATOR
# =============================================================================

class DataValidator:
    """
    Validates and characterizes input data for Agent 1 pipeline.
    
    Responsibilities:
        1. Load and validate expression data
        2. Detect bulk vs scRNA
        3. Auto-detect metadata columns
        4. Request clarification when ambiguous
    """
    
    def __init__(self, config: Agent1Config):
        self.config = config
    
    def validate(self, user_query: UserQuery) -> ValidationResult:
        """
        Main entry point: validate all input data.
        
        Args:
            user_query: User's query with file paths and optional context
            
        Returns:
            ValidationResult with loaded data and detected properties.
            Check result.needs_clarification before proceeding.
        """
        result = ValidationResult()
        
        # ── Step 1: Handle pre-computed DEGs ──
        if user_query.precomputed_deg_path:
            result = self._load_precomputed_degs(user_query, result)
            if result.needs_clarification:
                return result
            
            # Still need expression data for Agent 2 (TF activity inference)
            if user_query.expression_path:
                result = self._load_expression_data(user_query, result)
                if result.needs_clarification:
                    return result
                
                # Also load metadata if available (for quality assessment context)
                # but do NOT require condition/celltype detection — the DEGs
                # already represent a completed comparison
                if user_query.metadata_path:
                    result = self._load_metadata(user_query, result)
                    result.clarifications.clear()

                    # Detect condition info from metadata — Agent 2 needs
                    # disease/control labels for VIPER and PROGENy even when
                    # DEGs are pre-computed
                    if result.metadata_df is not None:
                        result = self._detect_condition_info(user_query, result)
                        # Don't block on clarifications — DEGs are already done
                        result.clarifications.clear()
            
            logger.info(
                f"Validation complete (precomputed DEGs): "
                f"{result.precomputed_degs.shape[0]} DEGs loaded"
                + (f", expression: {result.expression_df.shape[0]} genes × "
                   f"{result.expression_df.shape[1]} samples"
                   if result.expression_df is not None else "")
            )
            return result
        
        # ── Step 2: Load expression data ──
        if not user_query.expression_path:
            result.clarifications.append(ClarificationRequest(
                field="expression_path",
                question="No expression data provided. Please upload an expression matrix.",
                context=(
                    "Accepted formats: CSV/TSV (genes × samples) or "
                    "H5AD (AnnData object from scanpy/Seurat)"
                ),
            ))
            return result
        
        result = self._load_expression_data(user_query, result)
        if result.needs_clarification:
            return result
        
        # ── Step 3: Load metadata ──
        result = self._load_metadata(user_query, result)
        if result.needs_clarification:
            return result
        
        # ── Step 4: Detect condition column + labels ──
        result = self._detect_condition_info(user_query, result)
        if result.needs_clarification:
            return result
        
        # ── Step 5: If scRNA, detect cell type info ──
        if result.data_type == DataType.SINGLE_CELL:
            result = self._detect_celltype_info(user_query, result)
            if result.needs_clarification:
                return result
            
            result = self._detect_sample_column(user_query, result)
            if result.needs_clarification:
                return result
        
        logger.info(
            f"Validation complete: {result.data_type.value}, "
            f"{result.expression_df.shape[0]} genes × {result.expression_df.shape[1]} samples, "
            f"condition='{result.condition_column}', "
            f"{len(result.warnings)} warnings"
        )
        
        return result
    
    # =========================================================================
    # PRIVATE: Loading methods
    # =========================================================================
    
    def _load_expression_data(
        self, query: UserQuery, result: ValidationResult
    ) -> ValidationResult:
        """Load expression matrix and detect data type."""
        path = Path(query.expression_path)
        
        if not path.exists():
            result.clarifications.append(ClarificationRequest(
                field="expression_path",
                question=f"Expression file not found: {path}",
                context="Please check the file path and try again.",
            ))
            return result
        
        suffix = path.suffix.lower()
        
        try:
            if suffix == ".h5ad":
                result = self._load_h5ad(path, result)
            elif suffix in (".csv", ".tsv", ".txt"):
                sep = "\t" if suffix in (".tsv", ".txt") else ","
                result = self._load_tabular(path, sep, result)
            else:
                result.clarifications.append(ClarificationRequest(
                    field="expression_path",
                    question=f"Unsupported file format: {suffix}",
                    options=[".csv", ".tsv", ".h5ad"],
                    context="Please provide data in one of the supported formats.",
                ))
                return result
        except Exception as e:
            result.clarifications.append(ClarificationRequest(
                field="expression_path",
                question=f"Failed to load expression data: {str(e)}",
                context="Check that the file is not corrupted and matches the expected format.",
            ))
            return result
        
        # Validate expression values
        result = self._validate_expression_values(result)
        
        return result
    
    def _load_h5ad(self, path: Path, result: ValidationResult) -> ValidationResult:
        """Load H5AD (AnnData) file — this is scRNA-seq data."""
        try:
            import anndata as ad
        except ImportError:
            raise ImportError(
                "anndata package required for H5AD files. "
                "Install with: pip install anndata scanpy"
            )
        
        logger.info(f"Loading H5AD file: {path}")
        adata = ad.read_h5ad(path)
        
        result.data_type = DataType.SINGLE_CELL
        result.input_format = "h5ad"
        
        # Extract expression matrix (genes × cells → will be transposed later if needed)
        # AnnData stores cells × genes in .X
        if hasattr(adata.X, "toarray"):
            # Sparse matrix → dense
            expr_matrix = pd.DataFrame(
                adata.X.toarray(),
                index=adata.obs_names,
                columns=adata.var_names,
            )
        else:
            expr_matrix = pd.DataFrame(
                adata.X,
                index=adata.obs_names,
                columns=adata.var_names,
            )
        
        result.expression_df = expr_matrix  # cells × genes for scRNA
        result.metadata_df = adata.obs.copy()
        
        logger.info(
            f"Loaded scRNA-seq: {expr_matrix.shape[0]} cells × {expr_matrix.shape[1]} genes"
        )
        
        return result
    
    def _load_tabular(
        self, path: Path, sep: str, result: ValidationResult
    ) -> ValidationResult:
        """Load CSV/TSV expression matrix."""
        logger.info(f"Loading tabular file: {path}")
        
        df = pd.read_csv(path, sep=sep, index_col=0)
        result.input_format = "csv" if sep == "," else "tsv"
        
        # Heuristic: detect bulk vs scRNA based on dimensions
        # Bulk RNA-seq: typically 20,000-60,000 genes × 3-100 samples
        # scRNA-seq: typically 20,000-30,000 genes × 1,000-100,000 cells
        n_rows, n_cols = df.shape
        
        # Check orientation: genes should be rows for our convention
        # If more columns than rows, data is likely transposed (samples × genes)
        if n_cols > n_rows and n_cols > 1000:
            logger.info("Detected transposed matrix (samples × genes). Transposing to genes × samples.")
            df = df.T
            n_rows, n_cols = df.shape
        
        if n_cols > 500:
            # Likely scRNA (many cells as columns)
            result.data_type = DataType.SINGLE_CELL
            logger.info(f"Detected scRNA-seq based on dimensions: {n_rows} genes × {n_cols} cells")
            result.warnings.append(
                f"Detected as scRNA-seq ({n_cols} columns). "
                "If this is bulk RNA-seq, please let me know."
            )
        elif self._is_microarray(df):
            result.data_type = DataType.MICROARRAY
            logger.info(
                f"Detected microarray data: {n_rows} probes × {n_cols} samples "
                f"(log2 range {df.values.min():.1f}–{df.values.max():.1f})"
            )
        else:
            result.data_type = DataType.BULK
            logger.info(f"Detected bulk RNA-seq: {n_rows} genes × {n_cols} samples")

        result.expression_df = df

        return result

    @staticmethod
    def _is_microarray(expr_df: pd.DataFrame) -> bool:
        """
        Detect if data is log2-normalised microarray vs raw RNA-seq counts.

        Microarray: continuous float values, range ~3-15, no zeros.
        Raw counts: non-negative integers, many zeros, large max values.
        """
        values = expr_df.values
        all_positive = (values > 0).all()
        max_value = values.max()
        zero_fraction = (values == 0).mean()
        is_float = not np.issubdtype(values.dtype, np.integer)

        if all_positive and max_value < 20 and zero_fraction < 0.01 and is_float:
            return True
        return False
    
    def _validate_expression_values(self, result: ValidationResult) -> ValidationResult:
        """
        Check expression values are valid (numeric, reasonable NaN levels, non-negative).

        Failure modes detected:
          1. Non-numeric text in expression cells (e.g. "N/A", gene names in data area)
          2. High NaN fraction across the whole matrix (corrupt or incomplete data)
          3. Entire rows (genes) or columns (samples) that are all NaN/zero
          4. Negative values (log-transformed data fed as raw counts)
          5. Suspiciously large values (unnormalized TPM/FPKM)
        """
        df = result.expression_df

        if df is None:
            return result

        n_genes, n_samples = df.shape
        total_cells = n_genes * n_samples

        # ----- 1. Count pre-existing NaN/blank cells BEFORE coercion -----
        pre_nan_count = int(df.isna().sum().sum())

        # ----- 2. Coerce non-numeric text → NaN -----
        if not np.issubdtype(df.values.dtype, np.number):
            try:
                coerced = df.apply(pd.to_numeric, errors="coerce")
                text_coerced_count = int(coerced.isna().sum().sum()) - pre_nan_count
                if text_coerced_count > 0:
                    result.warnings.append(
                        f"Found {text_coerced_count:,} non-numeric values in the expression "
                        f"matrix (e.g. text, 'N/A', or gene names in the data area). "
                        f"These were converted to zero expression."
                    )
                df = coerced
            except Exception:
                result.clarifications.append(ClarificationRequest(
                    field="expression_path",
                    question=(
                        "The expression matrix contains values that could not be "
                        "converted to numbers. This usually means the file has a "
                        "formatting issue — for example, gene annotations mixed into "
                        "the data columns, or a delimiter mismatch."
                    ),
                    context=(
                        "Expected format: a numeric matrix where rows are genes and "
                        "columns are samples. The first column (or row index) should "
                        "contain gene symbols, and every other cell should be a number."
                    ),
                ))
                return result

        # ----- 3. Overall NaN fraction check -----
        total_nan = int(df.isna().sum().sum())
        nan_pct = 100 * total_nan / total_cells if total_cells > 0 else 0

        if nan_pct > 50:
            # Severely corrupt — ask user to fix
            result.clarifications.append(ClarificationRequest(
                field="expression_path",
                question=(
                    f"The expression matrix has {nan_pct:.0f}% missing values "
                    f"({total_nan:,} of {total_cells:,} cells are blank or NaN). "
                    f"This level of missingness will produce unreliable DEG results. "
                    f"The data may be corrupt or in the wrong format."
                ),
                context=(
                    "Common causes:\n"
                    "- Wrong delimiter (e.g. tab-separated file saved as .csv)\n"
                    "- Gene annotation columns included in the numeric matrix\n"
                    "- Incomplete download or truncated file\n"
                    "- Sparse scRNA-seq data that should be in H5AD format\n\n"
                    "Please check your file and re-upload, or provide data in H5AD format."
                ),
            ))
            return result

        if nan_pct > 20:
            result.warnings.append(
                f"The expression matrix has {nan_pct:.1f}% missing values "
                f"({total_nan:,} cells). Missing values will be treated as zero "
                f"expression, which may bias DEG analysis. Consider checking your "
                f"data for incomplete rows or formatting issues."
            )
        elif total_nan > 0 and pre_nan_count > 0:
            result.warnings.append(
                f"The expression matrix has {total_nan:,} missing/blank cells "
                f"({nan_pct:.1f}%). These will be treated as zero expression."
            )

        # ----- 4. All-NaN sample columns (bad samples) -----
        sample_nan_frac = df.isna().mean(axis=0)  # fraction NaN per column
        bad_samples = sample_nan_frac[sample_nan_frac > 0.9].index.tolist()
        if bad_samples:
            n_bad = len(bad_samples)
            examples = ", ".join(str(s) for s in bad_samples[:5])
            if n_bad > n_samples * 0.5:
                result.clarifications.append(ClarificationRequest(
                    field="expression_path",
                    question=(
                        f"{n_bad} of {n_samples} samples have >90% missing values "
                        f"(e.g. {examples}). This suggests the data file is corrupt "
                        f"or the matrix orientation is wrong (genes and samples may "
                        f"be swapped)."
                    ),
                    context=(
                        "Expected format: rows = genes, columns = samples. "
                        "If your file has samples as rows, the pipeline will try to "
                        "auto-detect and transpose, but this failed. "
                        "Please verify and re-upload."
                    ),
                ))
                return result
            else:
                result.warnings.append(
                    f"{n_bad} sample(s) have >90% missing values ({examples}). "
                    f"These samples may produce unreliable results — consider "
                    f"removing them from your data."
                )

        # ----- 5. All-NaN gene rows (empty genes) -----
        gene_nan_frac = df.isna().mean(axis=1)
        bad_genes = gene_nan_frac[gene_nan_frac > 0.9].index.tolist()
        if bad_genes:
            n_bad_genes = len(bad_genes)
            if n_bad_genes > n_genes * 0.3:
                result.warnings.append(
                    f"{n_bad_genes:,} genes ({100*n_bad_genes/n_genes:.0f}%) have "
                    f">90% missing values. This is unusually high — the data may "
                    f"contain annotation rows or non-gene entries."
                )
            # Drop all-NaN gene rows silently (they contribute nothing)
            all_nan_genes = gene_nan_frac[gene_nan_frac == 1.0].index.tolist()
            if all_nan_genes:
                df = df.drop(index=all_nan_genes)
                logger.info(f"Dropped {len(all_nan_genes)} all-NaN gene rows")

        # ----- 6. Fill remaining NaN → 0 -----
        if df.isna().any().any():
            df = df.fillna(0)

        result.expression_df = df

        # ----- 7. Negative values (log-transformed data) -----
        if (df.values < 0).any():
            n_neg = int((df.values < 0).sum())
            result.warnings.append(
                f"Expression matrix contains {n_neg:,} negative values. "
                "This suggests log-transformed or normalized data. "
                "PyDESeq2 requires raw counts — DEG analysis may not be appropriate. "
                "If you have pre-computed DEGs, provide them via the precomputed_deg_path."
            )

        # ----- 8. Suspiciously large values -----
        max_val = df.values.max()
        if max_val > 1e8:
            result.warnings.append(
                f"Maximum expression value is very large ({max_val:.0f}). "
                "Ensure this is raw counts, not TPM/FPKM."
            )

        # ----- 9. Constant rows (zero-variance genes — will cause NaN t-stats) -----
        gene_std = df.std(axis=1)
        n_constant = int((gene_std == 0).sum())
        if n_constant > n_genes * 0.5:
            result.warnings.append(
                f"{n_constant:,} of {n_genes:,} genes have zero variance across all "
                f"samples ({100*n_constant/n_genes:.0f}%). This is unusually high and "
                f"may indicate the data is filtered too aggressively or contains "
                f"only a single sample's values repeated."
            )

        return result
    
    def _load_metadata(
        self, query: UserQuery, result: ValidationResult
    ) -> ValidationResult:
        """Load metadata, or check if already loaded from H5AD."""
        
        # If loaded from H5AD, metadata is already in result.metadata_df
        if result.metadata_df is not None and result.input_format == "h5ad":
            logger.info(
                f"Metadata loaded from H5AD obs: {result.metadata_df.shape[0]} rows, "
                f"columns: {list(result.metadata_df.columns)}"
            )
            return result
        
        # Need external metadata file
        if not query.metadata_path:
            result.clarifications.append(ClarificationRequest(
                field="metadata_path",
                question="No metadata file provided. Please upload sample metadata.",
                context=(
                    "Metadata should be a CSV with at least two columns: "
                    "'sample' (matching expression column names) and "
                    "'condition' (e.g., 'disease' vs 'control')."
                ),
            ))
            return result
        
        path = Path(query.metadata_path)
        if not path.exists():
            result.clarifications.append(ClarificationRequest(
                field="metadata_path",
                question=f"Metadata file not found: {path}",
            ))
            return result
        
        try:
            meta_df = pd.read_csv(path, index_col=0)
            
            # Handle index mismatch: expression and metadata may use different
            # cell IDs (e.g., Cell_001 vs F00208_TGCCCTAGTCGCGTGT) but align
            # positionally. This is common with Seurat-exported data.
            if result.expression_df is not None:
                # For bulk data, samples are COLUMNS; for scRNA, samples are ROWS.
                # Check both axes to determine alignment.
                expr_index = set(result.expression_df.index)
                expr_columns = set(result.expression_df.columns)
                meta_index = set(meta_df.index)

                row_overlap = len(expr_index & meta_index)
                col_overlap = len(expr_columns & meta_index)

                if col_overlap > row_overlap and col_overlap > 0:
                    # Bulk data: metadata samples match expression COLUMNS — correct layout
                    logger.info(
                        f"Metadata-expression alignment: {col_overlap}/{len(meta_index)} "
                        f"samples matched via expression columns (bulk layout)"
                    )
                elif row_overlap == 0 and col_overlap == 0 and len(result.expression_df) == len(meta_df):
                    logger.warning(
                        "No matching cell IDs between expression and metadata, "
                        "but row counts match. Aligning by position."
                    )
                    # Preserve original metadata index as a column for reference
                    meta_df["original_barcode"] = meta_df.index.copy()
                    meta_df.index = result.expression_df.index.copy()
                    result.warnings.append(
                        "Expression and metadata cell IDs don't match. "
                        "Aligned by row position (both have "
                        f"{len(result.expression_df)} rows). "
                        "Verify that files are in the same cell order."
                    )
                elif row_overlap < len(result.expression_df) * 0.5 and col_overlap == 0:
                    # Some overlap but not much — try positional if same length
                    if len(result.expression_df) == len(meta_df):
                        logger.warning(
                            f"Only {row_overlap}/{len(result.expression_df)} cell IDs match. "
                            "Aligning by position since row counts match."
                        )
                        meta_df["original_barcode"] = meta_df.index.copy()
                        meta_df.index = result.expression_df.index.copy()
                    else:
                        result.warnings.append(
                            f"Poor index overlap ({row_overlap} matching cells) and "
                            "different row counts. Results may be unreliable."
                        )
            
            result.metadata_df = meta_df
            logger.info(
                f"Metadata loaded: {meta_df.shape[0]} rows, "
                f"columns: {list(meta_df.columns)}"
            )
        except Exception as e:
            result.clarifications.append(ClarificationRequest(
                field="metadata_path",
                question=f"Failed to load metadata: {str(e)}",
            ))
        
        return result
    
    def _load_precomputed_degs(
        self, query: UserQuery, result: ValidationResult
    ) -> ValidationResult:
        """Load pre-computed DEG results."""
        path = Path(query.precomputed_deg_path)
        
        if not path.exists():
            result.clarifications.append(ClarificationRequest(
                field="precomputed_deg_path",
                question=f"DEG file not found: {path}",
            ))
            return result
        
        try:
            deg_df = pd.read_csv(path)
        except Exception as e:
            result.clarifications.append(ClarificationRequest(
                field="precomputed_deg_path",
                question=f"Failed to load DEG file: {str(e)}",
            ))
            return result
        
        # Validate expected columns
        required_cols = {"gene", "log2FoldChange", "padj"}
        # Try case-insensitive matching
        col_map = {}
        for req_col in required_cols:
            matched = [c for c in deg_df.columns if c.lower() == req_col.lower()]
            if matched:
                col_map[matched[0]] = req_col
            else:
                found = False
                for alias in DEG_COLUMN_ALIASES.get(req_col, []):
                    matched = [c for c in deg_df.columns if c.lower() == alias.lower()]
                    if matched:
                        col_map[matched[0]] = req_col
                        found = True
                        break
                
                if not found:
                    result.clarifications.append(ClarificationRequest(
                        field="precomputed_deg_path",
                        question=(
                            f"DEG file missing required column: '{req_col}'. "
                            f"Found columns: {list(deg_df.columns)}"
                        ),
                        context=(
                            "Expected columns: 'gene' (symbol), "
                            "'log2FoldChange' (fold change), 'padj' (adjusted p-value)"
                        ),
                    ))
                    return result
        
        # Rename columns to standard names
        if col_map:
            deg_df = deg_df.rename(columns=col_map)
        
        result.precomputed_degs = deg_df
        result.data_type = DataType.PRECOMPUTED_DEGS
        
        logger.info(f"Loaded {len(deg_df)} pre-computed DEGs")
        
        return result
    
    # =========================================================================
    # PRIVATE: Column detection methods
    # =========================================================================
    
    def _detect_condition_info(
        self, query: UserQuery, result: ValidationResult
    ) -> ValidationResult:
        """Detect which metadata column contains condition labels (disease vs control)."""
        meta = result.metadata_df
        
        if meta is None:
            return result
        
        # If user specified the column, use it
        if query.condition_column and query.condition_column in meta.columns:
            result.condition_column = query.condition_column
        else:
            # Auto-detect: find column matching known condition names
            result.condition_column = self._find_column(
                meta, CONDITION_COLUMN_CANDIDATES
            )
        
        if result.condition_column is None:
            result.clarifications.append(ClarificationRequest(
                field="condition_column",
                question="Which column contains the condition labels (e.g., disease vs control)?",
                options=[c for c in meta.columns if meta[c].nunique() <= 10],
                context=f"Available columns: {list(meta.columns)}",
            ))
            return result
        
        # Fill blank/NaN condition labels if user already provided a disease label
        # (this runs on the second pass after clarification)
        col = result.condition_column
        _NA_STRINGS = {"nan", "na", "n/a", "none", "null", "missing", ""}
        if query.disease_label:
            blank_mask = (
                meta[col].isna()
                | meta[col].astype(str).str.strip().str.lower().isin(_NA_STRINGS)
            )
            if blank_mask.any():
                meta.loc[blank_mask, col] = query.disease_label
                result.metadata_df = meta
                logger.info(
                    f"Filled {blank_mask.sum()} blank/NA condition labels "
                    f"with '{query.disease_label}'"
                )

        # Detect disease/control labels within the condition column
        unique_labels = meta[result.condition_column].unique().tolist()

        # Check for blank/NaN labels — ask user to identify unlabeled samples
        blank_labels = [
            l for l in unique_labels
            if l is None or (isinstance(l, float) and np.isnan(l))
            or (isinstance(l, str) and l.strip().lower() in _NA_STRINGS)
        ]
        if blank_labels:
            named_labels = [str(l) for l in unique_labels if l not in blank_labels]
            n_blank = int(meta[result.condition_column].isin(blank_labels).sum()
                          + meta[result.condition_column].isna().sum())
            result.clarifications.append(ClarificationRequest(
                field="disease_label",
                question=(
                    f"The column '{result.condition_column}' has {n_blank} samples "
                    f"with blank or missing labels. The labeled groups are: "
                    f"{named_labels}. What condition do the unlabeled samples represent?"
                ),
                context=(
                    "For example, if the labeled group is 'Control' and the unlabeled "
                    "samples are the disease group, type the disease name (e.g., 'NASH', "
                    "'IPF'). The pipeline will assign this label to the blank rows."
                ),
            ))
            return result

        label_options = [str(l) for l in unique_labels]

        # Single condition — cannot do differential analysis
        if len(unique_labels) < 2:
            result.clarifications.append(ClarificationRequest(
                field="condition_column",
                question=(
                    f"The column '{result.condition_column}' has only one group: "
                    f"{unique_labels}. Differential expression requires at least "
                    f"two groups (e.g., disease vs control). Please select a "
                    f"different condition column, or provide pre-computed DEGs."
                ),
                options=[c for c in meta.columns if meta[c].nunique() >= 2 and meta[c].nunique() <= 10],
                context=(
                    "If your data truly has only one condition, you can still "
                    "analyze it by providing pre-computed DEG results from an "
                    "external comparison."
                ),
            ))
            return result

        # Use user-provided labels if both are present
        if query.disease_label and query.control_label:
            result.disease_label = query.disease_label
            result.control_label = query.control_label
        elif len(unique_labels) == 2:
            # Binary comparison — auto-detect which is disease, which is control
            result.disease_label, result.control_label = self._guess_disease_control(
                unique_labels
            )
            if result.disease_label is None:
                # Can't auto-detect — ask for both
                result.clarifications.append(ClarificationRequest(
                    field="disease_label",
                    question=(
                        f"Found two groups in '{result.condition_column}': {unique_labels}. "
                        "Which is the disease/treatment group?"
                    ),
                    options=label_options,
                ))
                result.clarifications.append(ClarificationRequest(
                    field="control_label",
                    question="Which is the control/reference group?",
                    options=label_options,
                ))
                return result
        else:
            # More than 2 groups — ask for both disease and control in one round
            # Check if user already provided one of the two
            need_disease = not query.disease_label
            need_control = not query.control_label
            
            if need_disease:
                result.clarifications.append(ClarificationRequest(
                    field="disease_label",
                    question=(
                        f"Found {len(unique_labels)} groups in '{result.condition_column}': "
                        f"{unique_labels}. Which is the disease group to compare?"
                    ),
                    options=label_options,
                ))
            else:
                result.disease_label = query.disease_label
            
            if need_control:
                result.clarifications.append(ClarificationRequest(
                    field="control_label",
                    question=(
                        f"Which group is the control/reference? "
                        f"Groups: {unique_labels}"
                    ),
                    options=label_options,
                    default="Control" if "Control" in label_options else None,
                ))
            else:
                result.control_label = query.control_label
            
            if result.clarifications:
                return result
        
        logger.info(
            f"Condition: '{result.condition_column}' → "
            f"disease='{result.disease_label}' vs control='{result.control_label}'"
        )
        
        return result
    
    def _detect_celltype_info(
        self, query: UserQuery, result: ValidationResult
    ) -> ValidationResult:
        """Detect cell type column and available types (scRNA only)."""
        meta = result.metadata_df
        
        if meta is None:
            return result
        
        # If user specified column, use it
        if query.cell_type_column and query.cell_type_column in meta.columns:
            result.cell_type_column = query.cell_type_column
        else:
            result.cell_type_column = self._find_column(
                meta, CELLTYPE_COLUMN_CANDIDATES
            )
        
        if result.cell_type_column is None:
            result.clarifications.append(ClarificationRequest(
                field="cell_type_column",
                question="Which column contains cell type annotations?",
                options=[
                    c for c in meta.columns
                    if 5 <= meta[c].nunique() <= 50  # Likely cell type range
                ],
                context=f"Available columns: {list(meta.columns)}",
            ))
            return result
        
        # Enumerate available cell types with counts
        cell_counts = meta[result.cell_type_column].value_counts().to_dict()
        result.available_cell_types = cell_counts
        
        # Check if user specified or mentioned a cell type
        target_cell_type = query.cell_type
        
        if target_cell_type:
            # Fuzzy match against available types
            matched = self._fuzzy_match_celltype(target_cell_type, list(cell_counts.keys()))
            if matched:
                query.cell_type = matched
                logger.info(f"Matched cell type '{target_cell_type}' → '{matched}'")
            else:
                result.clarifications.append(ClarificationRequest(
                    field="cell_type",
                    question=(
                        f"Cell type '{target_cell_type}' not found in data. "
                        "Which cell type do you want to analyze?"
                    ),
                    options=[
                        f"{ct} ({count} cells)"
                        for ct, count in sorted(cell_counts.items(), key=lambda x: -x[1])
                    ],
                ))
                return result
        else:
            # No cell type specified — ask user
            result.clarifications.append(ClarificationRequest(
                field="cell_type",
                question="Which cell type do you want to analyze?",
                options=[
                    f"{ct} ({count} cells)"
                    for ct, count in sorted(cell_counts.items(), key=lambda x: -x[1])
                ],
                context=(
                    "For CARNIVAL analysis, we need to focus on one cell type at a time. "
                    "The data will be aggregated into pseudobulk per donor for this cell type."
                ),
                default=max(cell_counts, key=cell_counts.get),
            ))
            return result
        
        return result
    
    def _detect_sample_column(
        self, query: UserQuery, result: ValidationResult
    ) -> ValidationResult:
        """Detect the biological replicate/donor column for pseudobulk."""
        meta = result.metadata_df
        
        if meta is None:
            return result
        
        if query.sample_column and query.sample_column in meta.columns:
            result.sample_column = query.sample_column
        else:
            result.sample_column = self._find_column(
                meta, SAMPLE_COLUMN_CANDIDATES
            )
        
        if result.sample_column is None:
            # Try to infer: look for columns with moderate cardinality (3-50 unique values)
            candidates = [
                c for c in meta.columns
                if 3 <= meta[c].nunique() <= 50
                and c != result.condition_column
                and c != result.cell_type_column
            ]
            
            if candidates:
                result.clarifications.append(ClarificationRequest(
                    field="sample_column",
                    question=(
                        "Which column identifies biological replicates (donors/patients)? "
                        "This is needed to aggregate cells into pseudobulk samples."
                    ),
                    options=candidates,
                    context=(
                        "Pseudobulk requires grouping cells by donor to create "
                        "independent biological replicates for statistical analysis."
                    ),
                ))
            else:
                result.clarifications.append(ClarificationRequest(
                    field="sample_column",
                    question=(
                        "Cannot identify a donor/patient column. "
                        "Which column should be used to group cells into pseudobulk samples?"
                    ),
                    options=list(meta.columns),
                ))
            return result
        
        n_donors = meta[result.sample_column].nunique()
        logger.info(f"Sample column: '{result.sample_column}' ({n_donors} donors)")
        
        if n_donors < 3:
            result.warnings.append(
                f"Only {n_donors} biological replicates found. "
                "DEG analysis requires at least 3 per group for statistical power. "
                "Results may be unreliable."
            )
        
        return result
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """
        Find a column in the DataFrame matching candidate names.
        Case-insensitive, returns first match.
        """
        def _norm(s: str) -> str:
            return s.lower().replace(" ", "_").replace("-", "_").replace(".", "_")

        df_cols_lower = {_norm(c): c for c in df.columns}

        for candidate in candidates:
            if _norm(candidate) in df_cols_lower:
                return df_cols_lower[_norm(candidate)]
        
        return None
    
    @staticmethod
    def _guess_disease_control(labels: list) -> Tuple[Optional[str], Optional[str]]:
        """
        Guess which of two labels is disease and which is control.
        Returns (disease_label, control_label) or (None, None) if unsure.
        """
        control_keywords = [
            "control", "ctrl", "healthy", "normal", "wt", "wildtype",
            "wild-type", "baseline", "untreated", "sham",
        ]
        
        label_lower = [str(l).lower() for l in labels]
        
        for i, ll in enumerate(label_lower):
            if any(kw in ll for kw in control_keywords):
                control_idx = i
                disease_idx = 1 - i
                return str(labels[disease_idx]), str(labels[control_idx])
        
        # Can't determine — need to ask user
        return None, None
    
    @staticmethod
    def _fuzzy_match_celltype(query: str, available: List[str]) -> Optional[str]:
        """
        Fuzzy match a cell type query against available types.
        Handles common variations (e.g., 'stellate' → 'Hepatic Stellate Cells').
        """
        query_lower = query.lower().strip()
        
        # Exact match (case-insensitive)
        for ct in available:
            if ct.lower() == query_lower:
                return ct
        
        # Substring match
        for ct in available:
            if query_lower in ct.lower() or ct.lower() in query_lower:
                return ct
        
        # Word overlap match
        query_words = set(query_lower.split())
        best_match = None
        best_overlap = 0
        
        for ct in available:
            ct_words = set(ct.lower().replace("_", " ").replace("-", " ").split())
            overlap = len(query_words & ct_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = ct
        
        if best_overlap > 0:
            return best_match
        
        return None
