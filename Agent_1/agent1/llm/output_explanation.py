"""
Output Explanation
===================
LLM-generated explanation of what Agent 1 did and why.

This is the "showcase LLM reasoning" module — it demonstrates that the system
doesn't just run a pipeline, it understands the biological context and can
explain its actions in natural language.

The explanation connects three things:
    1. What the USER asked (query intent, entities of interest)
    2. What Agent 1 FOUND (data quality, DEG results)
    3. What Agent 2 NEEDS to know (factual context for parameter selection)

This is shown to the user AND passed to Agent 2 as context.

Usage:
    from agent1.llm.output_explanation import ExplanationGenerator
    
    generator = ExplanationGenerator(config)
    explanation, tokens = generator.generate(
        query_text="What TFs drive fibrosis in IPF AT2 cells?",
        quality_report=quality_report,
        deg_results=deg_results,
        extracted_entities=entities,
    )
    print(explanation.summary)
    print(explanation.context_for_agent2)
"""

import logging
import time
from typing import Optional, Tuple

from agent1.schemas_agent1 import (
    Agent1Config,
    Agent1Explanation,
    DEGResults,
    ExtractedEntities,
    QualityReport,
)

logger = logging.getLogger(__name__)

# Lazy import: same pattern as entity_extraction.py
try:
    from anthropic import Anthropic, APIError, APITimeoutError
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


# =============================================================================
# TOOL SCHEMA: Structured explanation output
# =============================================================================

EXPLANATION_TOOL = {
    "name": "generate_explanation",
    "description": (
        "Generate a structured explanation of Agent 1's data preparation results. "
        "Connect the user's original question to what was found in the data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "2-3 sentence summary connecting what the user asked to what Agent 1 did. "
                    "Start with the user's question context, then describe the key actions taken "
                    "and results found. Be specific with numbers."
                ),
            },
            "quality_interpretation": {
                "type": "string",
                "description": (
                    "1-2 sentences explaining what the quality metrics mean "
                    "for the user's analysis. Translate technical metrics into "
                    "practical implications (e.g., 'medium noise means moderate "
                    "confidence in the DEG calls'). Do NOT recommend specific "
                    "parameters — just explain the data quality."
                ),
            },
            "context_for_agent2": {
                "type": "string",
                "description": (
                    "Factual summary of key metrics for downstream parameter selection. "
                    "Include: DEG count (up/down), noise level + CV value, sample count, "
                    "data type, and any quality warnings. "
                    "Report DATA FACTS only — no recommendations."
                ),
            },
            "caveats": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of important limitations the user should know about. "
                    "Examples: low sample size, subset data, high noise, "
                    "many genes filtered out, no housekeeping genes found. "
                    "Empty list if no significant concerns."
                ),
            },
        },
        "required": ["summary", "quality_interpretation", "context_for_agent2", "caveats"],
    },
}


# =============================================================================
# SYSTEM PROMPT: Guides the LLM's explanation behavior
# =============================================================================

EXPLANATION_SYSTEM_PROMPT = """\
You are a bioinformatics pipeline assistant explaining data preparation results.

Your job: generate a clear, concise explanation of what the data preparation 
pipeline did and what the results mean for the user's analysis question.

Rules:
- Connect the user's original question to the pipeline results
- Be specific with numbers (e.g., "487 DEGs" not "many DEGs")
- Explain quality metrics in plain language
- In context_for_agent2, report ONLY factual metrics — no recommendations
- Flag genuine concerns in caveats, but don't be alarmist about normal results
- Keep the summary under 4 sentences
- Keep quality_interpretation under 3 sentences
"""


# =============================================================================
# EXPLANATION GENERATOR
# =============================================================================

class ExplanationGenerator:
    """
    Generates LLM-powered explanations of Agent 1's pipeline results.
    
    Takes the user query + all pipeline outputs and produces a human-readable
    explanation that demonstrates AI reasoning about biological data.
    """
    
    def __init__(self, config: Agent1Config):
        self.config = config
        self._client = None
    
    @property
    def client(self) -> "Anthropic":
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            if not _HAS_ANTHROPIC:
                raise ImportError(
                    "anthropic package required for LLM explanation. "
                    "Install with: pip install anthropic. "
                    "Or use generate_without_llm() for template-based fallback."
                )
            if not self.config.anthropic_api_key:
                raise ValueError(
                    "Anthropic API key required. "
                    "Set ANTHROPIC_API_KEY environment variable."
                )
            self._client = Anthropic(api_key=self.config.anthropic_api_key)
        return self._client
    
    def generate(
        self,
        query_text: str,
        quality_report: QualityReport,
        deg_results: DEGResults,
        extracted_entities: Optional[ExtractedEntities] = None,
    ) -> Tuple[Agent1Explanation, int]:
        """
        Generate an explanation of Agent 1's results.
        
        Args:
            query_text: User's original natural language question
            quality_report: Computed quality metrics
            deg_results: DEG analysis results
            extracted_entities: LLM-extracted entities (optional, enriches context)
            
        Returns:
            Tuple of:
                - Agent1Explanation with all fields populated
                - tokens_used: total tokens consumed
        """
        logger.info("Generating pipeline explanation...")
        start_time = time.time()
        
        # Build the context message that tells the LLM what happened
        context = self._build_context(
            query_text, quality_report, deg_results, extracted_entities
        )
        
        # Call LLM with tool use
        raw_explanation, tokens_used = self._call_llm(context)
        
        explanation = Agent1Explanation(
            summary=raw_explanation.get("summary", ""),
            quality_interpretation=raw_explanation.get("quality_interpretation", ""),
            context_for_agent2=raw_explanation.get("context_for_agent2", ""),
            caveats=raw_explanation.get("caveats", []),
        )
        
        # Merge pipeline warnings into caveats deterministically.
        # The LLM may have captured some/all of these, but we ensure
        # none are lost by adding any that aren't already mentioned.
        pipeline_warnings = list(quality_report.warnings) + list(deg_results.warnings)
        existing_caveats_lower = {c.lower() for c in explanation.caveats}
        for warning in pipeline_warnings:
            # Only add if not already captured (avoid duplicates)
            if warning.lower() not in existing_caveats_lower:
                explanation.caveats.append(warning)
                existing_caveats_lower.add(warning.lower())
        
        elapsed = time.time() - start_time
        logger.info(
            f"Explanation generated ({elapsed:.1f}s, {tokens_used} tokens)"
        )
        
        return explanation, tokens_used
    
    def _build_context(
        self,
        query_text: str,
        quality_report: QualityReport,
        deg_results: DEGResults,
        extracted_entities: Optional[ExtractedEntities],
    ) -> str:
        """
        Build the structured context message for the LLM.
        
        This is where we assemble all pipeline outputs into a prompt
        that guides the LLM to produce a coherent explanation.
        """
        sections = []
        
        # --- User's question ---
        sections.append(f"USER QUERY: {query_text}")
        
        # --- Extracted entities (if available) ---
        if extracted_entities:
            entity_parts = []
            if extracted_entities.disease:
                entity_parts.append(f"Disease: {extracted_entities.disease}")
            if extracted_entities.cell_type_mentioned:
                entity_parts.append(f"Cell type: {extracted_entities.cell_type_mentioned}")
            if extracted_entities.genes_of_interest:
                entity_parts.append(
                    f"Genes of interest: {', '.join(extracted_entities.genes_of_interest)}"
                )
            if extracted_entities.pathways_mentioned:
                entity_parts.append(
                    f"Pathways: {', '.join(extracted_entities.pathways_mentioned)}"
                )
            if extracted_entities.query_intent:
                entity_parts.append(f"Intent: {extracted_entities.query_intent}")
            if entity_parts:
                sections.append("EXTRACTED ENTITIES: " + "; ".join(entity_parts))
        
        # --- Quality report ---
        qr = quality_report
        quality_lines = [
            f"Data type: {qr.data_type.value}",
            f"Dimensions: {qr.n_genes} genes × {qr.n_samples} samples",
            f"Noise level: {qr.noise_level.value}",
            f"Quality tier: {qr.quality_tier.value}",
            f"Mean library size: {qr.mean_library_size:,.0f}",
            f"Library size CV: {qr.cv_library_size:.2f}",
            f"Gene filtering: {qr.genes_before_filtering} → {qr.genes_after_filtering}",
            f"Gene detection rate: {qr.gene_detection_rate:.2f}",
        ]
        if qr.cv_housekeeping is not None:
            quality_lines.append(
                f"Housekeeping gene CV: {qr.cv_housekeeping:.3f} "
                f"(genes: {', '.join(qr.housekeeping_genes_found)})"
            )
        if qr.pseudobulk_info:
            pb = qr.pseudobulk_info
            quality_lines.append(
                f"Pseudobulk: {pb.n_cells_total} {pb.cell_type_analyzed} cells "
                f"across {pb.n_donors} donors"
            )
            if pb.donors_excluded:
                quality_lines.append(
                    f"Donors excluded (< {pb.min_cells_threshold} cells): "
                    f"{', '.join(pb.donors_excluded)}"
                )
        if qr.warnings:
            quality_lines.append(f"Warnings: {'; '.join(qr.warnings)}")
        
        sections.append("QUALITY REPORT:\n  " + "\n  ".join(quality_lines))
        
        # --- DEG results ---
        deg = deg_results
        deg_lines = [
            f"Source: {deg.source}",
            f"Comparison: {deg.comparison}",
            f"Thresholds: padj < {deg.padj_threshold}, |log2FC| > {deg.log2fc_threshold}",
            f"Total genes tested: {deg.total_genes_tested}",
            f"Significant DEGs: {deg.n_significant} ({deg.n_upregulated} up, {deg.n_downregulated} down)",
        ]
        if deg.top_upregulated:
            deg_lines.append(f"Top upregulated: {', '.join(deg.top_upregulated[:5])}")
        if deg.top_downregulated:
            deg_lines.append(f"Top downregulated: {', '.join(deg.top_downregulated[:5])}")
        if deg.warnings:
            deg_lines.append(f"Warnings: {'; '.join(deg.warnings)}")
        
        sections.append("DEG RESULTS:\n  " + "\n  ".join(deg_lines))
        
        return "\n\n".join(sections)
    
    def _call_llm(self, context: str, max_retries: int = 2) -> Tuple[dict, int]:
        """
        Make the Anthropic API call with tool use.
        Same retry pattern as entity_extraction.py.
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.config.llm_model,
                    max_tokens=800,
                    temperature=0.1,
                    system=EXPLANATION_SYSTEM_PROMPT,
                    tools=[EXPLANATION_TOOL],
                    tool_choice={
                        "type": "tool",
                        "name": "generate_explanation",
                    },
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Based on the following pipeline results, "
                                "generate an explanation:\n\n" + context
                            ),
                        }
                    ],
                )
                
                tool_block = response.content[0]
                
                if tool_block.type != "tool_use":
                    raise ValueError(
                        f"Expected tool_use block, got {tool_block.type}"
                    )
                
                tokens_used = (
                    response.usage.input_tokens + response.usage.output_tokens
                )
                
                logger.debug(
                    f"LLM call: {response.usage.input_tokens} input + "
                    f"{response.usage.output_tokens} output = {tokens_used} total"
                )
                
                return tool_block.input, tokens_used
                
            except APITimeoutError as e:
                last_error = e
                logger.warning(
                    f"API timeout (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
            except APIError as e:
                last_error = e
                if hasattr(e, "status_code") and 400 <= e.status_code < 500:
                    raise
                logger.warning(
                    f"API error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Unexpected error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
        
        raise RuntimeError(
            f"Explanation generation failed after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )
    
    # =========================================================================
    # FALLBACK: Template-based explanation without LLM
    # =========================================================================
    
    def generate_without_llm(
        self,
        query_text: str,
        quality_report: QualityReport,
        deg_results: DEGResults,
        extracted_entities: Optional[ExtractedEntities] = None,
    ) -> Agent1Explanation:
        """
        Fallback explanation using string templates.
        Used when API key is unavailable or for testing without LLM costs.
        
        Produces factually correct but less polished explanations
        compared to the LLM version.
        """
        logger.warning("Using template-based explanation (no LLM).")
        
        qr = quality_report
        deg = deg_results
        
        # --- Summary ---
        data_desc = f"{qr.data_type.value} data"
        if qr.pseudobulk_info:
            pb = qr.pseudobulk_info
            data_desc = (
                f"single-cell RNA-seq data, aggregated {pb.n_cells_total} "
                f"{pb.cell_type_analyzed} cells across {pb.n_donors} donors "
                f"into pseudobulk"
            )
        
        summary = (
            f"In response to your query about '{query_text[:100]}', "
            f"Agent 1 processed {data_desc} with "
            f"{qr.n_genes} genes × {qr.n_samples} samples. "
            f"Identified {deg.n_significant} significant DEGs "
            f"({deg.n_upregulated} up, {deg.n_downregulated} down) "
            f"comparing {deg.comparison}."
        )
        
        # --- Quality interpretation ---
        noise_desc = {
            "low": "low noise, indicating clean data suitable for sensitive analysis",
            "medium": "medium noise, suitable for standard analysis parameters",
            "high": "high noise, suggesting conservative analysis parameters are advisable",
        }
        cv_str = f" (CV={qr.cv_housekeeping:.3f})" if qr.cv_housekeeping else ""
        quality_interpretation = (
            f"The dataset shows {noise_desc.get(qr.noise_level.value, 'unknown noise')}"
            f"{cv_str}. "
            f"{qr.n_samples} samples provides "
            f"{'reasonable' if qr.n_samples >= 6 else 'limited'} statistical power."
        )
        
        # --- Context for Agent 2 (facts only) ---
        context_parts = [
            f"{deg.n_significant} DEGs ({deg.n_upregulated} up, {deg.n_downregulated} down)",
            f"noise_level={qr.noise_level.value}",
            f"quality_tier={qr.quality_tier.value}",
            f"n_samples={qr.n_samples}",
            f"data_type={qr.data_type.value}",
        ]
        if qr.cv_housekeeping is not None:
            context_parts.append(f"cv_housekeeping={qr.cv_housekeeping:.3f}")
        context_for_agent2 = ", ".join(context_parts)
        
        # --- Caveats ---
        caveats = list(qr.warnings) + list(deg.warnings)
        
        return Agent1Explanation(
            summary=summary,
            quality_interpretation=quality_interpretation,
            context_for_agent2=context_for_agent2,
            caveats=caveats,
        )
