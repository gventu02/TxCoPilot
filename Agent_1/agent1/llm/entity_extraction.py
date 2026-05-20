"""
Entity Extraction
==================
Extract biological entities from user's natural language query using LLM.

Uses Anthropic's tool use (function calling) for guaranteed structured output.
This is more robust than JSON prompting because:
    1. The API returns validated structured data, not free text to parse
    2. tool_choice="tool" forces the model to return a tool call (no free text)
    3. The schema is auto-generated from our Pydantic model (single source of truth)


Usage:
    from agent1.llm.entity_extraction import EntityExtractor
    
    extractor = EntityExtractor(config)
    entities = extractor.extract("What TFs drive fibrosis in IPF AT2 cells?")
    print(entities.disease)           # "IPF"
    print(entities.cell_type_mentioned)  # "AT2"
    print(entities.query_intent)      # "discovery"

Author: Giselle Nov 2025    
"""

import logging
import time
from typing import Optional, Tuple

from agent1.schemas_agent1 import Agent1Config, ExtractedEntities

logger = logging.getLogger(__name__)

# Lazy import: anthropic SDK is only needed for LLM extraction, not fallback.
# This lets the module load even if anthropic isn't installed (e.g., testing,
# or Docker images that only use extract_without_llm).
try:
    from anthropic import Anthropic, APIError, APITimeoutError
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


# =============================================================================
# HEURISTIC: Detect within-disease comparisons from labels
# =============================================================================

import re

# Common separators used in natural language comparisons
_SEP = r"(vs\.?|versus|against|compared\s+to|relative\s+to)"

_STAGING_PATTERNS = [
    # Letter+number staging: F0-F4 (fibrosis), T1-T4 (tumor),
    # N0-N3 (lymph node), M0-M1 (metastasis), G1-G4 (pathology grade)
    rf"[FfTtNnMmGg][0-4]\b.*{_SEP}.*[FfTtNnMmGg][0-4]\b",
    # Word-based staging (any disease)
    rf"stage\s*[IiVv0-9]+.*{_SEP}.*stage\s*[IiVv0-9]+",
    rf"stage\s*[A-Da-d]\b.*{_SEP}.*stage\s*[A-Da-d]\b",
    rf"grade\s*[0-9IiVv]+.*{_SEP}.*grade\s*[0-9IiVv]+",
    rf"gold\s*[0-9].*{_SEP}.*gold\s*[0-9]",           # COPD GOLD staging
    # Severity pairs — both directions
    rf"(low.?grade|mild|early|non.?progressive)\b.*{_SEP}.*\b(high.?grade|severe|late|advanced|progressive)",
    rf"(high.?grade|severe|late|advanced|progressive)\b.*{_SEP}.*\b(low.?grade|mild|early|non.?progressive)",
    # Disease state pairs (MS, IBD, autoimmune)
    rf"relaps\w*.*{_SEP}.*remitt\w*",
    rf"remitt\w*.*{_SEP}.*relaps\w*",
    rf"\bactive\b.*{_SEP}.*\b(inactive|stable|remission)\b",
    rf"\b(stable|remission)\b.*{_SEP}.*\bprogressive\b",
    # Treatment response
    rf"responder.*{_SEP}.*non.?responder",
    rf"non.?responder.*{_SEP}.*responder",
]

# Explicit stage/severity markers for independent label checking (Option B)
_STAGE_LABEL_RE = re.compile(
    r"_?[FfTtNnMmGg][0-9]\b"                           # F3, T2, N1, G2
    r"|stage[_\s]*[IiVv0-9]"                            # stage_I, stage2
    r"|grade[_\s]*[0-9IiVv]"                            # grade_2, gradeIII
    r"|gold[_\s]*[0-9]"                                 # GOLD1 (COPD)
    r"|_(early|late|mild|moderate|severe|advanced"
    r"|progressive|relapsing|remitting|active"
    r"|inactive|stable|unstable|refractory|recurrent)$",
    re.IGNORECASE,
)

# Healthy control synonyms — any study type
_CONTROL_TERMS = {
    "control", "healthy", "normal", "sham",
    "wt", "wild type", "wildtype", "baseline",
    "vehicle", "placebo", "untreated", "naive",
    "mock", "uninfected", "uninduced", "reference",
}


def detect_within_disease_comparison(
    query_text: str,
    disease_label: Optional[str] = None,
    control_label: Optional[str] = None,
) -> bool:
    """
    Heuristic: does this comparison look like within-disease staging/grading?

    Checks query text and metadata labels for patterns indicating a comparison
    between disease states rather than disease vs healthy control.

    Returns True if the comparison appears to be within the same disease.
    """
    # Build text to search: prompt + labels joined with a separator
    search_text = query_text
    if disease_label and control_label:
        search_text += f" {disease_label} vs {control_label}"

    for pattern in _STAGING_PATTERNS:
        if re.search(pattern, search_text, re.IGNORECASE):
            logger.info(
                f"Heuristic detected within-disease comparison: "
                f"matched pattern '{pattern}' in '{search_text[:80]}'"
            )
            return True

    # Label check: neither label looks like a healthy/reference control
    if disease_label and control_label:
        dl = disease_label.lower().replace("_", " ")
        cl = control_label.lower().replace("_", " ")
        neither_is_control = (
            not any(t in cl for t in _CONTROL_TERMS)
            and not any(t in dl for t in _CONTROL_TERMS)
        )
        if neither_is_control:
            has_stage = bool(
                _STAGE_LABEL_RE.search(disease_label)
                or _STAGE_LABEL_RE.search(control_label)
            )
            reason = (
                "explicit stage pattern in label"
                if has_stage
                else "neither label matches healthy-control terms"
            )
            logger.info(
                f"Heuristic: {reason} "
                f"(disease='{disease_label}', control='{control_label}')"
            )
            return True

    return False


# =============================================================================
# TOOL SCHEMA: What the LLM must return
# =============================================================================

# We define the tool schema explicitly rather than using
# ExtractedEntities.model_json_schema() because:
#   1. We only want the LLM to fill extraction fields, NOT validation fields
#      (validated_genes and invalid_genes are filled AFTER extraction by gene_validation.py)
#   2. We can add LLM-specific descriptions that guide extraction behavior
#      without polluting the Pydantic schema used for serialization

ENTITY_EXTRACTION_TOOL = {
    "name": "extract_biological_entities",
    "description": (
        "Extract biological entities from a user's natural language query "
        "about transcriptomic/gene expression analysis. "
        "Extract ONLY what is explicitly mentioned or clearly implied. "
        "Do NOT hallucinate entities that aren't in the query."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "disease": {
                "type": ["string", "null"],
                "description": (
                    "Disease or condition mentioned in the query. "
                    "Use standard abbreviations where common (e.g., 'IPF' not "
                    "'idiopathic pulmonary fibrosis' if user said 'IPF'). "
                    "null if no disease mentioned."
                ),
            },
            "genes_of_interest": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Gene symbols mentioned in the query. "
                    "Use official HGNC symbols in UPPERCASE (e.g., 'TGFB1' not 'TGF-beta 1'). "
                    "Empty list if no specific genes mentioned."
                ),
            },
            "pathways_mentioned": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Signaling pathways mentioned (e.g., 'TGF-beta signaling', "
                    "'Wnt pathway', 'NF-kB signaling'). "
                    "Use standard pathway names. Empty list if none mentioned."
                ),
            },
            "cell_type_mentioned": {
                "type": ["string", "null"],
                "description": (
                    "Cell type mentioned in the query. "
                    "Use common abbreviations (e.g., 'AT2', 'macrophages', "
                    "'hepatic stellate cells'). null if no cell type mentioned."
                ),
            },
            "query_intent": {
                "type": ["string", "null"],
                "enum": ["discovery", "validation", "comparison", "mechanism",
                         "out_of_scope", None],
                "description": (
                    "The user's analytical intent: "
                    "'discovery' = find new regulators/targets (most common), "
                    "'validation' = confirm known biology, "
                    "'comparison' = compare conditions/cell types, "
                    "'mechanism' = understand a specific pathway or mechanism, "
                    "'out_of_scope' = question cannot be answered with "
                    "transcriptomic/gene expression data (e.g., protein structure, "
                    "drug recommendations, clinical advice, non-biological questions)."
                ),
            },
            "comparison_type": {
                "type": ["string", "null"],
                "enum": ["disease_vs_control", "within_disease", None],
                "description": (
                    "Type of biological comparison: "
                    "'disease_vs_control' = comparing disease samples to healthy/normal controls "
                    "(e.g., IPF vs healthy, NASH vs normal liver, tumor vs adjacent normal). "
                    "'within_disease' = comparing stages, grades, or severities within the SAME "
                    "disease (e.g., early vs late fibrosis, F0-F1 vs F3-F4, stage I vs stage III, "
                    "low grade vs high grade, mild vs severe). These comparisons have subtler "
                    "fold changes because both groups are diseased. "
                    "null if comparison type cannot be determined from the query."
                ),
            },
        },
        "required": ["disease", "genes_of_interest", "pathways_mentioned",
                      "cell_type_mentioned", "query_intent", "comparison_type"],
    },
}

# System prompt: concise, focused on extraction behavior
EXTRACTION_SYSTEM_PROMPT = """\
You are a biological entity extractor for a transcriptomics analysis pipeline.

Your job: extract biological entities from user queries about gene expression,
differential expression, or network biology analysis.

Rules:
- Extract ONLY what is explicitly stated or clearly implied
- Use standard gene symbols (HGNC uppercase: TGFB1, TP53, SMAD3)
- Use standard pathway names (TGF-beta signaling, Wnt pathway, NF-kB signaling)
- For ambiguous cell types, use the most specific name given
- Do NOT infer genes from diseases (e.g., don't add TGFB1 just because IPF is mentioned)
- DISEASE EXTRACTION IS CRITICAL for downstream differential analysis. Extract the most
  specific disease term possible, even from indirect references (e.g., "kidney injury" →
  "kidney injury", "fibrotic lungs" → "pulmonary fibrosis", "tumor samples" → "cancer").
  Only return null if the query truly mentions no disease or condition.

SCOPE CHECK — set query_intent to 'out_of_scope' if the question:
- Cannot be answered with transcriptomic / gene expression data
- Asks about protein structures, drug dosages, clinical treatment, or non-biological topics
- Is completely unrelated to biology (e.g., weather, math, general knowledge)
Examples of out-of-scope: "What is the 3D structure of TP53?", "What drugs treat NASH?",
"How many patients are in the study?", "What is the capital of France?"
Examples that ARE in scope (even if unusual): "Does this NASH data show insulin resistance
signatures?" (cross-disease hypothesis), "Are IPF TFs active in COPD?" (shared pathway).
"""


# =============================================================================
# ENTITY EXTRACTOR
# =============================================================================

class EntityExtractor:
    """
    Extracts biological entities from user queries using Claude tool use.
    
    Design choice — tool use vs JSON prompting:
        Tool use guarantees structured output that matches our schema.
        JSON prompting requires defensive parsing (regex, try/except, retries).
        Tool use is faster to implement and more reliable in production.
    """
    
    def __init__(self, config: Agent1Config):
        self.config = config
        self._client: Optional[Anthropic] = None
    
    @property
    def client(self) -> "Anthropic":
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            if not _HAS_ANTHROPIC:
                raise ImportError(
                    "anthropic package required for LLM entity extraction. "
                    "Install with: pip install anthropic. "
                    "Or use extract_without_llm() for keyword-based fallback."
                )
            if not self.config.anthropic_api_key:
                raise ValueError(
                    "Anthropic API key required for entity extraction. "
                    "Set ANTHROPIC_API_KEY environment variable."
                )
            self._client = Anthropic(api_key=self.config.anthropic_api_key)
        return self._client
    
    def extract(self, query_text: str) -> Tuple[ExtractedEntities, int]:
        """
        Extract biological entities from a user query.
        
        Args:
            query_text: User's natural language question
            
        Returns:
            Tuple of:
                - ExtractedEntities populated with extracted fields
                  (validated_genes and invalid_genes will be empty — 
                   those are filled by gene_validation.py)
                - tokens_used: total tokens consumed
                
        Raises:
            ValueError: If API key is missing
            APIError: If Anthropic API call fails after retries
        """
        logger.info(f"Extracting entities from: '{query_text[:80]}...'")
        start_time = time.time()
        
        # Call the LLM with tool use
        raw_entities, tokens_used = self._call_llm(query_text)
        
        # Build ExtractedEntities from tool output
        # validated_genes and invalid_genes are left empty —
        # gene_validation.py fills those in a separate step
        entities = ExtractedEntities(
            disease=raw_entities.get("disease"),
            genes_of_interest=raw_entities.get("genes_of_interest", []),
            pathways_mentioned=raw_entities.get("pathways_mentioned", []),
            cell_type_mentioned=raw_entities.get("cell_type_mentioned"),
            query_intent=raw_entities.get("query_intent"),
            comparison_type=raw_entities.get("comparison_type"),
            # These stay empty until gene_validation.py runs:
            validated_genes=[],
            invalid_genes=[],
        )
        
        elapsed = time.time() - start_time
        logger.info(
            f"Extraction complete ({elapsed:.1f}s, {tokens_used} tokens): "
            f"disease={entities.disease}, "
            f"genes={entities.genes_of_interest}, "
            f"intent={entities.query_intent}"
        )
        
        return entities, tokens_used
    
    def _call_llm(self, query_text: str, max_retries: int = 2) -> Tuple[dict, int]:
        """
        Make the Anthropic API call with tool use.
        
        Returns:
            Tuple of (extracted_dict, total_tokens_used)
            
        Why tool_choice is set to {"type": "tool", "name": "extract_biological_entities"}:
            This FORCES the model to return a tool call. Without it, the model
            might respond with free text like "I'd be happy to help..." instead
            of structured output. This is the key difference from JSON prompting.
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.config.llm_model,
                    max_tokens=512,
                    temperature=0.1,
                    system=EXTRACTION_SYSTEM_PROMPT,
                    tools=[ENTITY_EXTRACTION_TOOL],
                    tool_choice={
                        "type": "tool",
                        "name": "extract_biological_entities",
                    },
                    messages=[
                        {
                            "role": "user",
                            "content": query_text,
                        }
                    ],
                )
                
                # Extract the tool call result
                # With tool_choice forcing a specific tool, response.content[0]
                # is guaranteed to be a ToolUseBlock
                tool_block = response.content[0]
                
                if tool_block.type != "tool_use":
                    raise ValueError(
                        f"Expected tool_use block, got {tool_block.type}. "
                        "This shouldn't happen with forced tool_choice."
                    )
                
                # token accounting
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
                # Don't retry on 4xx errors (bad request, auth failure)
                if hasattr(e, "status_code") and 400 <= e.status_code < 500:
                    raise
                logger.warning(
                    f"API error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
            except Exception as e:
                # Catch-all for unexpected errors (network issues, etc.)
                last_error = e
                logger.warning(
                    f"Unexpected error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
        
        # All retries exhausted
        raise RuntimeError(
            f"Entity extraction failed after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )
    
    def extract_without_llm(self, query_text: str) -> ExtractedEntities:
        """
        Fallback extraction using simple keyword matching.
        Used when API key is unavailable or for testing without LLM costs.
        
        This is NOT a replacement for LLM extraction — it catches only
        obvious patterns and misses nuance. Exists so the pipeline doesn't
        completely fail when the API is down.
        """
        logger.warning("Using fallback extraction (no LLM). Results will be limited.")
        
        query_lower = query_text.lower()
        
        # Simple disease keyword matching
        disease = None
        disease_keywords = {
            "ipf": "IPF",
            "idiopathic pulmonary fibrosis": "IPF",
            "liver fibrosis": "liver fibrosis",
            "nash": "NASH",
            "nafld": "NAFLD",
            "kidney fibrosis": "kidney fibrosis",
            "copd": "COPD",
            "cancer": "cancer",
        }
        for keyword, standard_name in disease_keywords.items():
            if keyword in query_lower:
                disease = standard_name
                break
        
        # Simple cell type matching
        cell_type = None
        celltype_keywords = {
            "at2": "AT2",
            "at1": "AT1",
            "macrophage": "Macrophages",
            "fibroblast": "Fibroblasts",
            "myofibroblast": "Myofibroblasts",
            "stellate": "Hepatic Stellate Cells",
            "epithelial": "Epithelial",
            "endothelial": "Endothelial",
        }
        for keyword, standard_name in celltype_keywords.items():
            if keyword in query_lower:
                cell_type = standard_name
                break
        
        # Simple intent detection
        intent = "discovery"  # default
        if any(w in query_lower for w in ["validate", "confirm", "verify"]):
            intent = "validation"
        elif any(w in query_lower for w in ["compare", "difference", "versus", "vs"]):
            intent = "comparison"
        elif any(w in query_lower for w in ["mechanism", "how does", "pathway"]):
            intent = "mechanism"

        # Basic out-of-scope detection for fallback mode
        bio_keywords = [
            "gene", "expression", "transcri", "deg", "pathway", "regulat",
            "activat", "inhibit", "signal", "tf", "transcription", "rna",
            "differenti", "upregulat", "downregulat", "fibrosis", "inflamm",
            "cancer", "tumor", "cell", "protein", "kinase", "receptor",
        ]
        has_bio_context = any(kw in query_lower for kw in bio_keywords)
        if not has_bio_context and not disease and not cell_type:
            intent = "out_of_scope"
        
        return ExtractedEntities(
            disease=disease,
            genes_of_interest=[],  # Can't reliably extract gene symbols with regex
            pathways_mentioned=[],
            cell_type_mentioned=cell_type,
            query_intent=intent,
            validated_genes=[],
            invalid_genes=[],
        )
