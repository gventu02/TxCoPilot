"""
Agent 1 Schemas: Data Preparation & Quality Assessment
========================================================
Pydantic models defining the contract between Agent 1 → Agent 2.

These schemas serve three purposes:
1. VALIDATION: Catch malformed data before it reaches CARNIVAL
2. SERIALIZATION: All models convert to JSON via .model_dump_json()
3. DOCUMENTATION: The schema IS the interface spec

Usage:
    result = Agent1Result(...)
    result.model_dump_json()  # → JSON string for Agent 2
    result.model_dump()       # → Python dict

Author: Giselle
Date: February 2026
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS: Constrained choices for type safety
# =============================================================================

class DataType(str, Enum):
    """Type of input expression data."""
    BULK = "bulk"
    SINGLE_CELL = "single_cell"
    MICROARRAY = "microarray"
    PRECOMPUTED_DEGS = "precomputed_degs"


class NoiseLevel(str, Enum):
    """Dataset noise classification based on CV of housekeeping genes."""
    LOW = "low"        # CV < 0.2: clean data, aggressive parameters OK
    MEDIUM = "medium"  # CV 0.2-0.5: standard parameters
    HIGH = "high"      # CV > 0.5: conservative parameters needed


class QualityTier(str, Enum):
    """Overall dataset quality assessment."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Agent1Status(str, Enum):
    """Agent 1 pipeline completion status."""
    SUCCESS = "success"                          # All steps completed
    SUCCESS_WITH_WARNINGS = "success_with_warnings"  # Completed but issues found
    NEEDS_CLARIFICATION = "needs_clarification"  # Cannot proceed without user input
    FAILED = "failed"                            # Unrecoverable error

# =============================================================================
# SHARED CONSTANTS: Column name aliases for DEG file compatibility
# =============================================================================

DEG_COLUMN_ALIASES: dict[str, list[str]] = {
    "gene": ["gene_symbol", "gene_name", "symbol", "Gene", "GENE"],
    "log2FoldChange": ["log2fc", "logfc", "lfc", "log2_fold_change", "logFC", "Log2FC"],
    "padj": ["p_adj", "fdr", "adjusted_pvalue", "adj_pval", "qvalue", "FDR", "q_value"],
}
"""
Mapping of standard column names → known aliases.
Used by data_validator.py and deg_analysis.py for pre-computed DEG files.
Single source of truth — add new aliases here, both modules pick them up.
"""

# =============================================================================
# CLARIFICATION: When Agent 1 needs to ask the user a question
# =============================================================================

class ClarificationRequest(BaseModel):
    """
    Returned when Agent 1 cannot proceed without user input.

    Two modes:
        1. LLM-generated (biological decisions): `issue` + `details` are provided,
           and the LLM generates a conversational `question` + `context`.
        2. Hardcoded (data validation): `question` is set directly by the rule
           that detected the problem (no LLM call needed).

    The UI renders `question` + `options` regardless of how they were produced.
    After user responds, Agent 1 is re-invoked with the clarification filled in.
    """
    field: str = Field(
        description="Which parameter needs clarification (e.g., 'cell_type', 'condition_column')"
    )
    # Structured input for LLM question generation (biological clarifications)
    issue: Optional[str] = Field(
        default=None,
        description=(
            "Machine-readable issue type for LLM question generation. "
            "e.g., 'out_of_scope_query', 'within_disease_comparison', 'disease_mismatch'"
        )
    )
    details: dict = Field(
        default_factory=dict,
        description="Structured facts for LLM context (labels, counts, reasons, etc.)"
    )
    # User-facing output (filled by LLM or hardcoded)
    question: str = Field(
        default="",
        description="Human-readable question to show the user (LLM-generated or hardcoded)"
    )
    options: Optional[List[str]] = Field(
        default=None,
        description="Available choices (None = free-text input expected)"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context to help user decide (LLM-generated or hardcoded)"
    )
    default: Optional[str] = Field(
        default=None,
        description="Suggested default if user doesn't want to choose"
    )


# =============================================================================
# INPUT SCHEMAS: What Agent 1 receives
# =============================================================================

class UserQuery(BaseModel):
    """
    User's natural language query + uploaded data references.
    This is the top-level input to Agent 1.
    """
    query_text: str = Field(
        description="User's natural language question, e.g., 'What TFs drive fibrosis in stellate cells?'"
    )
    expression_path: Optional[str] = Field(
        default=None,
        description="Path to expression matrix file (CSV/TSV/H5AD)"
    )
    metadata_path: Optional[str] = Field(
        default=None,
        description="Path to metadata file (CSV). Not needed if expression is H5AD with obs."
    )
    precomputed_deg_path: Optional[str] = Field(
        default=None,
        description="Path to pre-computed DEG results (CSV: gene, log2FoldChange, padj)"
    )
    
    # Optional user-provided context (from previous clarification or UI)
    cell_type: Optional[str] = Field(
        default=None,
        description="Cell type to analyze (for scRNA). If None, Agent 1 will detect and ask."
    )
    condition_column: Optional[str] = Field(
        default=None,
        description="Metadata column with disease/control labels. Auto-detected if None."
    )
    disease_label: Optional[str] = Field(
        default=None,
        description="Label for disease group in condition column (e.g., 'IPF')"
    )
    control_label: Optional[str] = Field(
        default=None,
        description="Label for control group in condition column (e.g., 'Control')"
    )
    cell_type_column: Optional[str] = Field(
        default=None,
        description="Metadata column with cell type annotations. Auto-detected if None."
    )
    sample_column: Optional[str] = Field(
        default=None,
        description="Metadata column identifying biological replicates/donors for pseudobulk."
    )
    disease_mismatch_confirmed: bool = Field(
        default=False,
        description="User confirmed query disease differs from metadata disease intentionally."
    )
    threshold_confirmed: bool = Field(
        default=False,
        description="User already confirmed threshold choice for within-disease comparison."
    )
    log2fc_threshold_override: Optional[float] = Field(
        default=None,
        description="User-chosen log2FC threshold (e.g., 0.5 for subtle signal). Overrides config default."
    )


# =============================================================================
# QUALITY REPORT: Deterministic assessment of dataset quality
# =============================================================================

class QualityReport(BaseModel):
    """
    Deterministic quality metrics computed from the expression data.
    Agent 2 uses this to select CARNIVAL parameters.
    
    NO LLM involved — pure computation.
    """
    # Dataset dimensions
    n_genes: int = Field(description="Number of genes in expression matrix")
    n_samples: int = Field(description="Number of samples (or pseudobulk replicates)")
    
    # Data type detection
    data_type: DataType = Field(description="Detected input data type")
    
    # Library size statistics
    mean_library_size: float = Field(description="Mean total counts per sample")
    cv_library_size: float = Field(description="Coefficient of variation of library sizes")
    
    # Noise estimation
    cv_housekeeping: Optional[float] = Field(
        default=None,
        description="CV of housekeeping genes (ACTB, GAPDH, B2M). Primary noise indicator."
    )
    noise_level: NoiseLevel = Field(
        description="Classified noise level: low (<0.2), medium (0.2-0.5), high (>0.5)"
    )
    housekeeping_genes_found: List[str] = Field(
        default_factory=list,
        description="Which housekeeping genes were present in the data"
    )
    
    # Gene filtering stats
    genes_before_filtering: int = Field(description="Genes before low-count filtering")
    genes_after_filtering: int = Field(description="Genes after low-count filtering")
    gene_detection_rate: float = Field(
        description="Fraction of genes with non-zero expression in at least 1 sample"
    )
    
    # Overall quality
    quality_tier: QualityTier = Field(description="Overall quality assessment")
    
    # Warnings (human-readable)
    warnings: List[str] = Field(
        default_factory=list,
        description="Quality warnings, e.g., 'Low sample count (n=4), results may be underpowered'"
    )
    
    # Pseudobulk-specific (only if scRNA)
    pseudobulk_info: Optional[PseudobulkInfo] = None


class PseudobulkInfo(BaseModel):
    """Metadata about the pseudobulk aggregation (only for scRNA input)."""
    cell_type_analyzed: str = Field(description="Which cell type was selected")
    cell_type_column: str = Field(description="Metadata column used for cell type")
    sample_column: str = Field(description="Metadata column used for donor/sample grouping")
    n_cells_total: int = Field(description="Total cells of this type across all donors")
    n_donors: int = Field(description="Number of biological replicates (donors)")
    cells_per_donor: Dict[str, int] = Field(
        description="Number of cells per donor, e.g., {'donor1': 150, 'donor2': 230}"
    )
    donors_excluded: List[str] = Field(
        default_factory=list,
        description="Donors excluded due to too few cells (< min_cells threshold)"
    )
    min_cells_threshold: int = Field(
        default=10,
        description="Minimum cells required per donor for pseudobulk"
    )


# Need to update QualityReport's forward reference now that PseudobulkInfo is defined
QualityReport.model_rebuild()


# =============================================================================
# DEG RESULTS: Output from differential expression analysis
# =============================================================================

class DEGResults(BaseModel):
    """
    Differential expression analysis results.
    Either computed by PyDESeq2 or loaded from pre-computed file.
    
    Agent 2 uses these as CARNIVAL measurements.
    """
    # How were DEGs produced?
    source: str = Field(
        description="'pydeseq2' or 'precomputed'"
    )
    comparison: str = Field(
        description="Comparison description, e.g., 'IPF vs Control'"
    )
    
    # Thresholds used
    padj_threshold: float = Field(default=0.05) # 0.05 is standard, but can be relaxed for testing or if data is noisy
    log2fc_threshold: float = Field(default=1.0) # 1.0 is standard, but can be relaxed to capture more DEGs for testing or if data is noisy
    
    # Summary statistics
    total_genes_tested: int = Field(description="Total genes in DEG test")
    n_significant: int = Field(description="Genes passing both thresholds")
    n_upregulated: int = Field(description="Significant + log2FC > 0")
    n_downregulated: int = Field(description="Significant + log2FC < 0")
    
    # File reference (the actual DEG table is saved to disk, not embedded)
    deg_file_path: str = Field(
        description="Path to full DEG results CSV (gene, log2FoldChange, padj, baseMean, stat)"
    )
    
    # Top genes for quick reference (not exhaustive)
    top_upregulated: List[str] = Field(
        default_factory=list,
        description="Top 10 upregulated genes by log2FC"
    )
    top_downregulated: List[str] = Field(
        default_factory=list,
        description="Top 10 downregulated genes by log2FC"
    )
    
    # Warnings
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "DEG-specific warnings, e.g., "
            "'Only 23 DEGs found — consider relaxing thresholds'"
        )
    )

    @field_validator("n_significant")
    @classmethod
    def check_deg_count(cls, v: int) -> int:
        """Warn if DEG count is suspiciously low or high."""
        if v < 10:
            logger.warning(
                f"Very few DEGs ({v}). CARNIVAL may produce a sparse network. "
                "Consider relaxing thresholds."
            )
        elif v > 5000:
            logger.warning(
                f"Very many DEGs ({v}). CARNIVAL network may be too large. "
                "Consider tightening thresholds."
            )
        return v


# =============================================================================
# ENTITY EXTRACTION: LLM-extracted biological entities from user query
# =============================================================================

class ExtractedEntities(BaseModel):
    """
    Biological entities extracted from user's natural language query.
    Produced by a single LLM call in Agent 1.
    
    Agent 2 uses these for context-aware parameter selection.
    Agent 3 uses these for focused interpretation.
    """
    disease: Optional[str] = Field(
        default=None,
        description="Disease mentioned, e.g., 'IPF', 'liver fibrosis', 'NASH'"
    )
    genes_of_interest: List[str] = Field(
        default_factory=list,
        description="Specific genes mentioned, e.g., ['TGFB1', 'SMAD3', 'TP53']"
    )
    pathways_mentioned: List[str] = Field(
        default_factory=list,
        description="Pathways mentioned, e.g., ['TGF-beta signaling', 'Wnt pathway']"
    )
    cell_type_mentioned: Optional[str] = Field(
        default=None,
        description="Cell type mentioned, e.g., 'hepatic stellate cells', 'macrophages'"
    )
    query_intent: Optional[str] = Field(
        default=None,
        description="Inferred intent: 'discovery', 'validation', 'comparison', 'mechanism', 'out_of_scope'"
    )
    comparison_type: Optional[str] = Field(
        default=None,
        description=(
            "Type of comparison: 'disease_vs_control' (standard, e.g., IPF vs healthy), "
            "'within_disease' (staging/grading, e.g., F0-F1 vs F3-F4, early vs late), "
            "or null if unclear. Within-disease comparisons may need relaxed fold-change thresholds."
        )
    )

    # Validated gene symbols (after OmniPath/cache check)
    validated_genes: List[str] = Field(
        default_factory=list,
        description="Genes confirmed to exist in OmniPath/gene database"
    )
    invalid_genes: List[str] = Field(
        default_factory=list,
        description="Genes from query that could not be validated (possible typos)"
    )


# =============================================================================
# LLM EXPLANATION: Agent 1's summary of what it did and why
# =============================================================================

class Agent1Explanation(BaseModel):
    """
    LLM-generated explanation of Agent 1's actions.
    This is the 'showcase LLM reasoning' feature.
    
    Shown to user AND included in Agent 2's context.
    """
    summary: str = Field(
        description=(
            "2-3 sentence summary of what Agent 1 did. "
            "Example: 'Detected single-cell RNA-seq data with 15 cell types. "
            "Aggregated 1,247 hepatic stellate cells across 8 donors into pseudobulk. "
            "Identified 487 DEGs (312 up, 175 down) comparing IPF vs Control.'"
        )
    )
    quality_interpretation: str = Field(
        description=(
            "What the quality metrics mean for downstream analysis. "
            "Example: 'Medium noise level (CV=0.34) suggests moderate confidence in "
            "differential expression calls. 8 biological replicates provides "
            "reasonable statistical power.'"
        )
    )
    context_for_agent2: str = Field(
        description=(
            "Factual context that Agent 2 uses for parameter selection. "
            "Reports DATA FACTS only — Agent 2's rules handle the decisions. "
            "Example: '487 DEGs (312 up, 175 down), noise_level=medium (CV=0.34), "
            "n_samples=16 (8 IPF, 8 Control), data_type=pseudobulk from scRNA-seq.'"
        )
    )
    caveats: List[str] = Field(
        default_factory=list,
        description="Important limitations or concerns the user should know about"
    )


# =============================================================================
# MAIN OUTPUT: The complete Agent 1 result
# =============================================================================

class Agent1Result(BaseModel):
    """
    Complete output from Agent 1, consumed by Agent 2.
    
    This is the PRIMARY contract between the two agents.
    Agent 2 should be able to run its entire pipeline from this object alone.
    
    Serialization:
        result.model_dump_json()   → JSON string
        result.model_dump()        → Python dict
        Agent1Result(**json_dict)  → Reconstruct from JSON
    """
    # Status
    status: Agent1Status = Field(description="Pipeline completion status")
    
    # Clarification (only if status == NEEDS_CLARIFICATION)
    clarifications_needed: List[ClarificationRequest] = Field(
        default_factory=list,
        description="Questions for the user (only when status is 'needs_clarification')"
    )
    
    # Core outputs (populated when status is SUCCESS or SUCCESS_WITH_WARNINGS)
    quality_report: Optional[QualityReport] = Field(
        default=None,
        description="Deterministic quality assessment"
    )
    deg_results: Optional[DEGResults] = Field(
        default=None,
        description="DEG analysis results"
    )
    extracted_entities: Optional[ExtractedEntities] = Field(
        default=None,
        description="LLM-extracted biological entities from user query"
    )
    explanation: Optional[Agent1Explanation] = Field(
        default=None,
        description="LLM-generated explanation of Agent 1's actions"
    )
    
    # File paths for Agent 2 to consume
    # (Agent 2 loads these files directly rather than passing DataFrames through JSON)
    expression_matrix_path: Optional[str] = Field(
        default=None,
        description="Path to clean expression matrix (genes × samples CSV)"
    )
    metadata_path: Optional[str] = Field(
        default=None,
        description="Path to metadata CSV with condition labels"
    )
    
    # Original user input (passed through for Agent 2/3 context)
    original_query: Optional[str] = Field(
        default=None,
        description="The user's original query text"
    )
    
    # Processing metadata
    processing_time_seconds: Optional[float] = None
    tokens_used: Optional[int] = Field(
        default=None,
        description="Total LLM tokens used by Agent 1 (entity extraction + explanation)"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error details when status is FAILED. Shown to user and logged."
    )

    @model_validator(mode="after")
    def validate_completeness(self) -> "Agent1Result":
        """Ensure successful results have all required fields."""
        if self.status in (Agent1Status.SUCCESS, Agent1Status.SUCCESS_WITH_WARNINGS):
            missing = []
            if self.quality_report is None:
                missing.append("quality_report")
            if self.deg_results is None:
                missing.append("deg_results")
            if self.expression_matrix_path is None:
                missing.append("expression_matrix_path")
            if missing:
                raise ValueError(
                    f"Status is '{self.status}' but missing required fields: {missing}"
                )
        
        if self.status == Agent1Status.NEEDS_CLARIFICATION:
            if not self.clarifications_needed:
                raise ValueError(
                    "Status is 'needs_clarification' but no clarifications provided"
                )
        
        return self


# =============================================================================
# CONFIGURATION: Agent 1 runtime settings
# =============================================================================
# NOTE: Requires `pydantic-settings` package:
#   pip install pydantic-settings
#   or: conda install pydantic-settings
#
# BaseSettings reads environment variables automatically.
# With env_prefix="AGENT1_", set vars like:
#   export AGENT1_PADJ_THRESHOLD=0.01
#   export AGENT1_LLM_MODEL=claude-sonnet-4-6
#
# Priority order (highest wins):
#   1. Explicitly passed kwargs:  Agent1Config(padj_threshold=0.01)
#   2. Environment variables:     AGENT1_PADJ_THRESHOLD=0.01
#   3. Default values in schema:  padj_threshold: float = 0.05
# =============================================================================

from pydantic_settings import BaseSettings

class Agent1Config(BaseSettings):
    """
    Runtime configuration for Agent 1.
    
    Reads from environment variables with AGENT1_ prefix,
    or can be instantiated directly with kwargs.
    
    Examples:
        # From env vars (Docker-friendly):
        # export AGENT1_PADJ_THRESHOLD=0.01
        config = Agent1Config()  # padj_threshold will be 0.01
        
        # Direct instantiation (testing):
        config = Agent1Config(padj_threshold=0.01)
        
        # Mixed (env var for API key, defaults for rest):
        # export AGENT1_ANTHROPIC_API_KEY=sk-ant-...
        config = Agent1Config()
    """
    model_config = ConfigDict(
        env_prefix="AGENT1_",
        # If an env var is set but can't be parsed, use the default instead of crashing
        # Useful during development when env might have stale values
        extra="ignore",
    )
    
    # API settings
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key. Falls back to AGENT1_ANTHROPIC_API_KEY env var."
    )
    llm_model: str = Field(
        default="claude-sonnet-4-6",
        description="Which Claude model to use for LLM calls"
    )
    
    # DEG analysis defaults
    padj_threshold: float = Field(default=0.05)
    log2fc_threshold: float = Field(default=1.0)
    min_counts_per_gene: int = Field(
        default=10,
        description="Minimum total counts for a gene to be included"
    )
    min_samples_per_gene: int = Field(
        default=3,
        description="Gene must have >= min_counts in at least this many samples"
    )
    
    # Pseudobulk settings
    min_cells_per_pseudobulk: int = Field(
        default=10,
        description="Minimum cells required to form a pseudobulk sample"
    )
    
    # Quality assessment
    housekeeping_genes: List[str] = Field(
        default=["ACTB", "GAPDH", "B2M", "RPL13A", "RPLP0"],
        description="Housekeeping genes for noise estimation"
    )
    
    # Output paths (relative to project root)
    output_dir: str = Field(
        default="agent1_output",
        description="Directory for Agent 1 output files"
    )
    
    # Gene validation
    gene_cache_path: str = Field(
        default="data/gene_symbols_cache.json",
        description="Cached gene symbol list (fallback when OmniPath is unavailable)"
    )
    omnipath_timeout: int = Field(
        default=30,
        description="Timeout in seconds for OmniPath API calls"
    )
