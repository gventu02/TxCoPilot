"""
Agent 1 Main Orchestrator
==========================
Chains all Agent 1 modules into a single pipeline:

    UserQuery → validate → (pseudobulk) → quality → DEG → extract entities 
              → validate genes → explain → Agent1Result

This is the ONLY file that Agent 2 (or Streamlit UI) needs to call.

Design:
    - Deterministic pipeline with 2 optional LLM calls
    - Returns early if clarification is needed (user must respond first)
    - Gracefully degrades if LLM is unavailable (fallback extraction + templates)
    - All file paths are relative (Docker-portable)

Usage:
    from agent1.agent1_main import Agent1Pipeline
    from agent1.config import get_config
    from agent1.schemas_agent1 import UserQuery
    
    pipeline = Agent1Pipeline(get_config())
    
    query = UserQuery(
        query_text="What TFs drive fibrosis in IPF AT2 cells?",
        expression_path="data/expression_matrix.csv",
        metadata_path="data/metadata.csv",
        cell_type="AT2",
        condition_column="Diagnosis",
        disease_label="IPF",
        control_label="Control",
        sample_column="Sample_Name",
    )
    
    result = pipeline.run(query)
    
    if result.status == "needs_clarification":
        # Show result.clarifications_needed to user, update query, re-run
        ...
    else:
        # result is ready for Agent 2
        result.model_dump_json()  # serialize for handoff
"""

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from agent1.config import get_config
from agent1.schemas_agent1 import (
    Agent1Config,
    Agent1Result,
    Agent1Status,
    ClarificationRequest,
    DataType,
    UserQuery,
)

# =============================================================================
# MODELS: Defined in dependency order. If adding new models that reference
# others, place the referenced model ABOVE the one that uses it.
# =============================================================================

# Core modules (Phase 1 — deterministic)
from agent1.core.data_validator import DataValidator
from agent1.core.pseudobulk import PseudobulkAggregator
from agent1.core.quality_assessment import QualityAssessor # this refernces pseudo bulk, must always come after it
from agent1.core.deg_analysis import DEGAnalyzer
from agent1.core.gene_id_normalizer import GeneIDNormalizer
from agent1.core.limma_deg import run_limma_deg

# LLM modules (Phase 2 — gracefully degrade)
from agent1.llm.entity_extraction import EntityExtractor
from agent1.llm.output_explanation import ExplanationGenerator
from agent1.llm.clarification_generator import ClarificationGenerator
from agent1.validation.gene_validation import GeneValidator

logger = logging.getLogger(__name__)


class Agent1Pipeline:
    """
    Complete Agent 1 pipeline orchestrator.
    
    Chains: validate → (pseudobulk) → quality → DEG → entity extraction 
            → gene validation → explanation → Agent1Result
    
    Returns Agent1Result which is the contract with Agent 2.
    """
    
    def __init__(self, config: Optional[Agent1Config] = None):
        self.config = config or get_config()
        
        # Initialize all modules
        self.validator = DataValidator(self.config)
        self.pseudobulk = PseudobulkAggregator(
            min_cells=self.config.min_cells_per_pseudobulk
        )
        self.quality_assessor = QualityAssessor(self.config)
        self.deg_analyzer = DEGAnalyzer(self.config)
        self.entity_extractor = EntityExtractor(self.config)
        self.gene_validator = GeneValidator(self.config)
        self.explanation_generator = ExplanationGenerator(self.config)
        self.clarification_generator = ClarificationGenerator(self.config)
        
        # Track LLM token usage across the pipeline
        self._total_tokens = 0
    
    def run(self, query: UserQuery) -> Agent1Result:
        """
        Execute the full Agent 1 pipeline.
        
        Args:
            query: User's query with file paths and optional context
            
        Returns:
            Agent1Result — the complete output for Agent 2.
            Check result.status to determine next action:
                - SUCCESS / SUCCESS_WITH_WARNINGS → pass to Agent 2
                - NEEDS_CLARIFICATION → show clarifications to user, re-run
                - FAILED → show error to user
        """
        start_time = time.time()
        self._total_tokens = 0
        
        logger.info(f"{'='*60}")
        logger.info(f"AGENT 1 PIPELINE START")
        logger.info(f"Query: {query.query_text[:80]}...")
        logger.info(f"{'='*60}")
        
        try:
            # ==============================================================
            # STEP 1: Data Validation
            # ==============================================================
            logger.info("Step 1: Data validation")
            validation = self.validator.validate(query)
            
            if validation.needs_clarification:
                logger.info(
                    f"Clarification needed: {len(validation.clarifications)} questions"
                )
                return Agent1Result(
                    status=Agent1Status.NEEDS_CLARIFICATION,
                    clarifications_needed=validation.clarifications,
                    original_query=query.query_text,
                )
            
            # ==============================================================
            # STEP 2: Pseudobulk Aggregation (scRNA only)
            # ==============================================================
            counts_df = validation.expression_df
            metadata_df = validation.metadata_df
            pseudobulk_info = None
            
            if (
                validation.data_type == DataType.SINGLE_CELL
                and not query.precomputed_deg_path
            ):
                logger.info("Step 2: Pseudobulk aggregation (scRNA detected)")
                
                counts_df, metadata_df, pseudobulk_info = self.pseudobulk.aggregate(
                    expression_df=validation.expression_df,
                    metadata_df=validation.metadata_df,
                    cell_type_column=validation.cell_type_column,
                    cell_type=query.cell_type,
                    sample_column=validation.sample_column,
                    condition_column=validation.condition_column,
                )
            else:
                logger.info("Step 2: Skipped (bulk data, no pseudobulk needed)")

            # ==============================================================
            # STEP 2b: Gene ID Normalization (if not already HGNC)
            # ==============================================================
            normalizer = GeneIDNormalizer()

            if validation.input_format == "h5ad":
                # H5AD: genes are column headers → transpose so genes are in index
                norm_result = normalizer.normalize(counts_df.T)
                if norm_result.normalization_needed:
                    counts_df = norm_result.mapped_df.T
                    logger.info(f"Gene ID normalization (H5AD columns): {norm_result.summary}")
                    if norm_result.warnings:
                        for w in norm_result.warnings:
                            logger.warning(w)
            else:
                # CSV/TSV: genes already in index
                norm_result = normalizer.normalize(counts_df)
                if norm_result.normalization_needed:
                    counts_df = norm_result.mapped_df
                    logger.info(f"Gene ID normalization (index): {norm_result.summary}")
                    if norm_result.warnings:
                        for w in norm_result.warnings:
                            logger.warning(w)

            if not norm_result.normalization_needed:
                logger.info("Step 2b: Gene IDs already HGNC symbols, no mapping needed")

            # ==============================================================
            # STEP 3: Quality Assessment
            # ==============================================================
            logger.info("Step 3: Quality assessment")
            quality_report = self.quality_assessor.assess(
                counts_df=counts_df,
                metadata_df=metadata_df,
                data_type=validation.data_type,
                pseudobulk_info=pseudobulk_info,
            )

            # Propagate data validation warnings into quality report
            # so they surface to the user via Agent 1's explanation caveats
            if validation.warnings:
                if not quality_report.warnings:
                    quality_report.warnings = []
                for w in validation.warnings:
                    quality_report.warnings.append(w)
                    logger.warning(f"Data validation: {w}")

            # ==============================================================
            # STEP 4a: Entity Extraction (moved before DEG analysis)
            # ==============================================================
            logger.info("Step 4a: Entity extraction")
            extracted_entities = self._extract_entities(query.query_text)

            # STEP 4b: Scope Guard — refuse unanswerable queries
            if (
                extracted_entities
                and extracted_entities.query_intent == "out_of_scope"
            ):
                logger.warning(
                    f"Query classified as out_of_scope: '{query.query_text[:80]}'"
                )
                clarification = self._generate_clarification(
                    ClarificationRequest(
                        field="query_text",
                        issue="out_of_scope_query",
                        details={
                            "user_query": query.query_text,
                            "pipeline_capabilities": [
                                "gene expression analysis",
                                "differential expression",
                                "signaling network construction",
                                "transcription factor identification",
                                "pathway activity assessment",
                            ],
                        },
                    )
                )
                return Agent1Result(
                    status=Agent1Status.NEEDS_CLARIFICATION,
                    clarifications_needed=[clarification],
                    original_query=query.query_text,
                )

            # ==============================================================
            # STEP 4c: Within-disease comparison detection
            # ==============================================================
            if not query.threshold_confirmed:
                from agent1.llm.entity_extraction import detect_within_disease_comparison

                heuristic_flag = detect_within_disease_comparison(
                    query_text=query.query_text,
                    disease_label=validation.disease_label or query.disease_label,
                    control_label=validation.control_label or query.control_label,
                )
                llm_flag = (
                    extracted_entities
                    and extracted_entities.comparison_type == "within_disease"
                )

                if heuristic_flag or llm_flag:
                    import re as _re
                    reasons = []
                    if heuristic_flag:
                        reasons.append("metadata labels suggest no healthy control group")
                    if llm_flag:
                        reasons.append("query describes within-disease staging/grading")

                    # Detect if the prompt implies merging multiple stages on one side
                    # e.g. "F3 and F4 against F0-F1" — Agent 1 cannot merge at runtime
                    _MULTI_STAGE_RE = _re.compile(
                        r"[FfTtNnGg][0-9].{0,10}(and|&|\+|,).{0,10}[FfTtNnGg][0-9]"
                        r"|stages?\s+[IiVv0-9]+\s*(and|&|\+|,)",
                        _re.IGNORECASE,
                    )
                    multi_stage = bool(_MULTI_STAGE_RE.search(query.query_text))

                    details = {
                        "user_query": query.query_text,
                        "disease_label": validation.disease_label or query.disease_label,
                        "control_label": validation.control_label or query.control_label,
                        "detection_reasons": reasons,
                        "disease": extracted_entities.disease if extracted_entities else None,
                    }
                    if multi_stage:
                        details["grouping_note"] = (
                            "The prompt references multiple stages on one side "
                            "(e.g., 'F3 and F4'). This pipeline compares exactly two "
                            "labeled groups at a time and cannot merge stages at runtime. "
                            "To compare combined stages, upload a metadata file where those "
                            "samples already share one label (e.g., F3 and F4 both labeled "
                            "'NASH_advanced'). The disease_label field must match that single label exactly."
                        )

                    logger.info(
                        f"Within-disease comparison detected ({', '.join(reasons)})"
                        f"{' — multi-stage grouping implied' if multi_stage else ''}. "
                        f"Requesting threshold confirmation from user."
                    )
                    clarification = self._generate_clarification(
                        ClarificationRequest(
                            field="log2fc_threshold_choice",
                            issue="within_disease_comparison",
                            details=details,
                            options=[
                                "Relaxed threshold (log2FC > 0.5) — recommended for staging/grading",
                                "Standard threshold (log2FC > 1.0) — use if you expect strong signal",
                            ],
                            default="Relaxed threshold (log2FC > 0.5) — recommended for staging/grading",
                        )
                    )
                    return Agent1Result(
                        status=Agent1Status.NEEDS_CLARIFICATION,
                        clarifications_needed=[clarification],
                        original_query=query.query_text,
                        extracted_entities=extracted_entities,
                    )

            # ==============================================================
            # STEP 4d: DEG Analysis
            # ==============================================================
            logger.info("Step 4d: DEG analysis")
            # Normalize DEG gene IDs if needed (reuse same normalizer for cache)
            if validation.precomputed_degs is not None:
                deg_norm = normalizer.normalize(
                    validation.precomputed_degs, column='gene', sort_by='padj'
                )
                if deg_norm.normalization_needed:
                    validation.precomputed_degs = deg_norm.mapped_df
                    logger.info(f"DEG gene ID normalization: {deg_norm.summary}")
                    if deg_norm.warnings:
                        for w in deg_norm.warnings:
                            logger.warning(w)

            # Determine log2FC threshold: user override > microarray auto-adjust > config default
            if query.log2fc_threshold_override is not None:
                lfc_threshold = query.log2fc_threshold_override
                logger.info(f"Using user-selected log2FC threshold: {lfc_threshold}")
            else:
                lfc_threshold = self.config.log2fc_threshold
            # Adjust for microarray (compressed dynamic range) if still at default
            if validation.data_type == DataType.MICROARRAY and lfc_threshold >= 1.0:
                lfc_threshold = 0.5
                logger.info(
                    f"Microarray detected: lowering log2FC threshold from "
                    f"{self.config.log2fc_threshold} to {lfc_threshold} "
                    f"(microarray has compressed dynamic range vs RNA-seq)"
                )

            if validation.precomputed_degs is not None:
                # Use pre-computed DEGs
                deg_results = self.deg_analyzer.load_precomputed(
                    deg_df=validation.precomputed_degs,
                    comparison=(
                        f"{validation.disease_label or 'disease'} vs "
                        f"{validation.control_label or 'control'}"
                    ),
                    log2fc_threshold=lfc_threshold,
                )
            elif validation.data_type == DataType.MICROARRAY:
                # Limma via inmoose (correct model for log2 microarray data)
                logger.info("Step 4: Using limma for microarray DEG analysis")
                condition_col = validation.condition_column
                disease_lbl = validation.disease_label or query.disease_label
                control_lbl = validation.control_label or query.control_label

                limma_deg_df = run_limma_deg(
                    expr_df=counts_df,
                    metadata_df=metadata_df,
                    disease_label=disease_lbl,
                    control_label=control_lbl,
                    condition_column=condition_col,
                    padj_threshold=self.config.padj_threshold,
                    log2fc_threshold=lfc_threshold,
                )
                # Wrap in DEGResults via load_precomputed (saves CSV for Agent 2)
                deg_results = self.deg_analyzer.load_precomputed(
                    deg_df=limma_deg_df,
                    comparison=f"{disease_lbl} vs {control_lbl} (limma)",
                    padj_threshold=self.config.padj_threshold,
                    log2fc_threshold=lfc_threshold,
                )
            else:
                # PyDESeq2 (correct model for raw count data)
                deg_results = self.deg_analyzer.run_pydeseq2(
                    counts_df=counts_df,
                    metadata_df=metadata_df,
                    condition_column="condition" if pseudobulk_info else validation.condition_column,
                    disease_label=validation.disease_label or query.disease_label,
                    control_label=validation.control_label or query.control_label,
                    padj_threshold=self.config.padj_threshold,
                    log2fc_threshold=lfc_threshold,
                )
            
            # ==============================================================
            # STEP 5: Gene Validation
            # ==============================================================
            if extracted_entities and extracted_entities.genes_of_interest:
                logger.info("Step 5: Gene validation")
                validated, invalid = self.gene_validator.validate(
                    extracted_entities.genes_of_interest
                )
                extracted_entities.validated_genes = validated
                extracted_entities.invalid_genes = invalid
            else:
                logger.info("Step 5: Skipped (no genes to validate)")

            # ==============================================================
            # STEP 5b: Disease info validation
            # ==============================================================
            # Agent 2 needs disease/condition info for differential TF activity
            # (VIPER contrast) and differential pathway scoring (PROGENy).
            # Without it, Agent 2 falls back to less accurate methods (ULM,
            # one-sample t-test) that mix disease and control samples.
            has_disease_from_query = (
                extracted_entities and extracted_entities.disease
                and extracted_entities.disease.strip("<> ").upper() != "UNKNOWN"
            )
            has_disease_from_metadata = bool(validation.disease_label)
            has_disease_from_user_input = bool(
                query.disease_label and query.disease_label.strip()
            )

            # Also accept user-typed disease label from the UI
            if not has_disease_from_metadata and has_disease_from_user_input:
                validation.disease_label = query.disease_label
                validation.control_label = query.control_label
                has_disease_from_metadata = True
                logger.info(
                    f"Disease backfilled from user input: '{query.disease_label}'"
                )

            # Disease mismatch detection: query mentions one disease but
            # metadata contains a different one. Flag for user confirmation.
            if (
                has_disease_from_query
                and has_disease_from_metadata
                and not query.disease_mismatch_confirmed
            ):
                query_disease = (extracted_entities.disease or "").strip().lower()
                meta_disease = (validation.disease_label or "").strip().lower()
                if query_disease != meta_disease:
                    logger.warning(
                        f"Disease mismatch: query='{extracted_entities.disease}' "
                        f"vs metadata='{validation.disease_label}'"
                    )
                    clarification = self._generate_clarification(
                        ClarificationRequest(
                            field="disease_mismatch_confirmed",
                            issue="disease_mismatch",
                            details={
                                "user_query": query.query_text,
                                "query_disease": extracted_entities.disease,
                                "metadata_disease_label": validation.disease_label,
                                "metadata_control_label": validation.control_label,
                            },
                            options=["Yes, proceed with dataset groups",
                                     "No, I uploaded the wrong data"],
                        )
                    )
                    return Agent1Result(
                        status=Agent1Status.NEEDS_CLARIFICATION,
                        clarifications_needed=[clarification],
                        original_query=query.query_text,
                    )

            if extracted_entities and not has_disease_from_query and has_disease_from_metadata:
                extracted_entities.disease = validation.disease_label
                logger.info(
                    f"Disease backfilled from metadata: '{validation.disease_label}'"
                )

            if not has_disease_from_query and not has_disease_from_metadata:
                missing_disease_warning = (
                    "No disease/condition identified from query or metadata. "
                    "Agent 2 will use less accurate methods (ULM instead of VIPER, "
                    "one-sample pathway test instead of differential). "
                    "For best results, re-run with a query that mentions the disease "
                    "or provide metadata with a labeled condition column."
                )
                logger.warning(missing_disease_warning)
                if not quality_report.warnings:
                    quality_report.warnings = []
                quality_report.warnings.append(missing_disease_warning)
            elif has_disease_from_query and not has_disease_from_metadata:
                logger.info(
                    f"Disease '{extracted_entities.disease}' found in query but not "
                    f"in metadata. Agent 2 will attempt to match conditions from metadata."
                )

            # ==============================================================
            # STEP 7: Save expression matrix for Agent 2
            # ==============================================================
            logger.info("Step 7: Saving outputs")
            output_dir = Path(self.config.output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            expr_path = output_dir / "expression_matrix_clean.csv"
            counts_df.to_csv(expr_path)
            
            meta_path = output_dir / "metadata_clean.csv"
            if metadata_df is not None:
                metadata_df.to_csv(meta_path)
            
            # ==============================================================
            # STEP 8: Generate Explanation (LLM call 2)
            # ==============================================================
            logger.info("Step 8: Generating explanation")
            explanation = self._generate_explanation(
                query_text=query.query_text,
                quality_report=quality_report,
                deg_results=deg_results,
                extracted_entities=extracted_entities,
            )
            
            # ==============================================================
            # STEP 9: Build final result
            # ==============================================================
            elapsed = time.time() - start_time
            
            # Determine status based on warnings
            has_warnings = bool(quality_report.warnings or deg_results.warnings)
            status = (
                Agent1Status.SUCCESS_WITH_WARNINGS
                if has_warnings
                else Agent1Status.SUCCESS
            )
            
            result = Agent1Result(
                status=status,
                quality_report=quality_report,
                deg_results=deg_results,
                extracted_entities=extracted_entities,
                explanation=explanation,
                expression_matrix_path=str(expr_path),
                metadata_path=str(meta_path) if metadata_df is not None else None,
                original_query=query.query_text,
                processing_time_seconds=round(elapsed, 2),
                tokens_used=self._total_tokens if self._total_tokens > 0 else None,
            )
            
            # Save the full result as JSON
            result_path = output_dir / "agent1_result.json"
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            
            logger.info(f"{'='*60}")
            logger.info(f"AGENT 1 PIPELINE COMPLETE")
            logger.info(f"Status: {result.status.value}")
            logger.info(f"DEGs: {deg_results.n_significant} significant")
            logger.info(f"Time: {elapsed:.1f}s")
            logger.info(f"Tokens: {self._total_tokens}")
            logger.info(f"Result saved: {result_path}")
            logger.info(f"{'='*60}")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Pipeline failed after {elapsed:.1f}s: {e}", exc_info=True)

            # Build a user-friendly error explanation
            error_type = type(e).__name__
            raw_msg = str(e)

            # Categorize common failures with actionable guidance
            if "API" in error_type or "api" in raw_msg.lower():
                friendly = (
                    f"The AI service returned an error: {raw_msg}\n\n"
                    "This usually means the API key is invalid or the service "
                    "is temporarily unavailable. Check that ANTHROPIC_API_KEY "
                    "is set correctly and try again."
                )
            elif "FileNotFoundError" in error_type or "No such file" in raw_msg:
                friendly = (
                    f"A required file was not found: {raw_msg}\n\n"
                    "Please verify that all uploaded files exist at the "
                    "specified paths and haven't been moved or deleted."
                )
            elif "KeyError" in error_type or "column" in raw_msg.lower():
                friendly = (
                    f"A required column was missing from your data: {raw_msg}\n\n"
                    "This can happen if the expression matrix or metadata file "
                    "has unexpected column names. Check that your files match "
                    "the expected format (genes as rows, samples as columns "
                    "for expression; sample IDs and condition labels for metadata)."
                )
            elif "MemoryError" in error_type or "memory" in raw_msg.lower():
                friendly = (
                    f"The dataset is too large to process in available memory.\n\n"
                    "Try reducing the dataset size (e.g., filter low-expression "
                    "genes) or provide pre-computed DEGs instead."
                )
            elif "singular matrix" in raw_msg.lower() or "deseq" in raw_msg.lower():
                friendly = (
                    f"DEG analysis failed: {raw_msg}\n\n"
                    "This often happens when there are too few samples per group, "
                    "duplicate samples, or when the expression matrix contains "
                    "constant rows. Check that you have at least 3 samples per "
                    "condition and that the data isn't pre-filtered."
                )
            else:
                friendly = (
                    f"An unexpected error occurred during analysis: "
                    f"{error_type}: {raw_msg}\n\n"
                    "Things to check:\n"
                    "1. Expression file: Should be genes (rows) x samples (columns) "
                    "with numeric values\n"
                    "2. Metadata file: Should have a column matching expression "
                    "sample names and a condition column (e.g., 'disease' vs 'control')\n"
                    "3. If using pre-computed DEGs: needs 'gene', 'log2FoldChange', "
                    "and 'padj' columns\n\n"
                    "Check the terminal logs for the full error traceback."
                )

            return Agent1Result(
                status=Agent1Status.FAILED,
                original_query=query.query_text,
                processing_time_seconds=round(elapsed, 2),
                error_message=friendly,
                # Pydantic validator requires these for non-FAILED status,
                # but for FAILED they can be None — the model_validator allows it
            )
    
    # =========================================================================
    # LLM WRAPPERS: Graceful degradation when API is unavailable
    # =========================================================================
    
    def _extract_entities(self, query_text: str):
        """Extract entities with LLM, fall back to keyword matching."""
        try:
            entities, tokens = self.entity_extractor.extract(query_text)
            self._total_tokens += tokens
            return entities
        except (ImportError, ValueError, RuntimeError) as e:
            logger.warning(f"LLM entity extraction unavailable: {e}")
            logger.warning("Falling back to keyword-based extraction.")
            return self.entity_extractor.extract_without_llm(query_text)
    
    def _generate_explanation(self, query_text, quality_report, deg_results,
                              extracted_entities):
        """Generate explanation with LLM, fall back to templates."""
        try:
            explanation, tokens = self.explanation_generator.generate(
                query_text=query_text,
                quality_report=quality_report,
                deg_results=deg_results,
                extracted_entities=extracted_entities,
            )
            self._total_tokens += tokens
            return explanation
        except (ImportError, ValueError, RuntimeError) as e:
            logger.warning(f"LLM explanation unavailable: {e}")
            logger.warning("Falling back to template-based explanation.")
            return self.explanation_generator.generate_without_llm(
                query_text=query_text,
                quality_report=quality_report,
                deg_results=deg_results,
                extracted_entities=extracted_entities,
            )

    def _generate_clarification(self, request: ClarificationRequest) -> ClarificationRequest:
        """Use LLM to generate a conversational clarification question."""
        try:
            request, tokens = self.clarification_generator.generate(request)
            self._total_tokens += tokens
            return request
        except (ImportError, ValueError, RuntimeError) as e:
            logger.warning(f"LLM clarification generation unavailable: {e}")
            return request  # fallback already applied inside generator


# =============================================================================
# CLI ENTRY POINT: For quick testing
# =============================================================================

def main():
    """
    Quick CLI entry point for testing.
    
    Usage:
        python -m agent1.agent1_main \\
            --query "What TFs drive fibrosis in IPF AT2 cells?" \\
            --expression data/expression_matrix.csv \\
            --metadata data/metadata.csv \\
            --cell-type AT2 \\
            --disease IPF \\
            --control Control
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent 1 Pipeline")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--expression", help="Expression matrix path")
    parser.add_argument("--metadata", help="Metadata file path")
    parser.add_argument("--precomputed-degs", help="Pre-computed DEGs path")
    parser.add_argument("--cell-type", help="Cell type (scRNA)")
    parser.add_argument("--condition-column", help="Metadata condition column")
    parser.add_argument("--disease", help="Disease label")
    parser.add_argument("--control", help="Control label")
    parser.add_argument("--sample-column", help="Sample/donor column")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup
    from agent1.config import setup_logging
    setup_logging(args.log_level)
    
    # Build query
    query = UserQuery(
        query_text=args.query,
        expression_path=args.expression,
        metadata_path=args.metadata,
        precomputed_deg_path=args.precomputed_degs,
        cell_type=args.cell_type,
        condition_column=args.condition_column,
        disease_label=args.disease,
        control_label=args.control,
        sample_column=args.sample_column,
    )
    
    # Run
    pipeline = Agent1Pipeline()
    result = pipeline.run(query)
    
    # Output
    print(f"\nStatus: {result.status.value}")
    if result.status == Agent1Status.NEEDS_CLARIFICATION:
        for c in result.clarifications_needed:
            print(f"  Q: {c.question}")
            if c.options:
                print(f"  Options: {c.options}")
    elif result.explanation:
        print(f"\n{result.explanation.summary}")
        print(f"\nContext for Agent 2: {result.explanation.context_for_agent2}")


if __name__ == "__main__":
    main()
