"""
Agent 2 — CARNIVAL Parameter Selection
================================================
Selects CARNIVAL pipeline parameters using a two-stage approach:
  1. Deterministic rules (fast, reproducible, no API cost)
  2. LangChain refinement (CoT reasoning, structured output, context-aware adjustments)

LangChain is used here specifically for:
  - Structured JSON output parsing with automatic retry on malformed responses
  - Chain composition (prompt | llm | parser) making the data flow explicit
  - CoT reasoning captured in the output for thesis documentation

If LangChain fails for any reason, falls back to direct Anthropic API call.
If that also fails, returns rule-based params unchanged (pipeline never blocked).

Architecture:
    rule_engine(agent1_result) → CARNIVALParams (defaults)
         ↓
    langchain_chain.invoke(context) → refined CARNIVALParams + rationale
         ↓ (on any failure)
    direct_anthropic_fallback() → refined CARNIVALParams
         ↓ (on any failure)  
    rule_based_params (unchanged)

Author: Giselle Ventura
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Safe parameter ranges — shared by initial selection and retry validation
_SAFE_RANGES = {
    "carnival_beta":    (0.05, 2.0),
    "top_n_tfs":        (10, 100),
    "solver_timeout":   (60, 3600),
}

# ---------------------------------------------------------------------------
# Lazy imports — don't fail at module load if LangChain not installed
# ---------------------------------------------------------------------------
def _try_import_langchain():
    """
    Attempt to import LangChain components without failing at module load time.

    Returns (ChatAnthropic, ChatPromptTemplate, StrOutputParser) if available,
    or (None, None, None) if LangChain is not installed — callers fall back to
    the direct Anthropic SDK path in that case.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        return ChatAnthropic, ChatPromptTemplate, StrOutputParser
    except ImportError:
        return None, None, None


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def select_parameters(
    noise_level: str,
    n_samples: int,
    n_degs: int,
    n_genes: int,
    original_query: str,
    extracted_entities: Optional[dict] = None,
    context_for_agent2: Optional[str] = None,
    use_llm: bool = True,
) -> dict:
    """
    Select CARNIVAL parameters for this dataset using rules + LangChain refinement.

    Args:
        noise_level: "low" | "medium" | "high" (from Agent1Result.quality_report)
        n_samples: Number of biological replicates
        n_degs: Number of significant DEGs
        n_genes: Total genes in expression matrix
        original_query: User's natural language query (passed through)
        extracted_entities: Dict from Agent1Result.extracted_entities (optional)
        context_for_agent2: Factual summary from Agent1Result.explanation (optional)
        use_llm: Set False to skip LLM refinement (testing/offline mode)

    Returns:
        Dict matching CARNIVALParams fields, plus:
            'rationale': LLM's chain-of-thought reasoning (str)
            'selected_by': 'rules' | 'rules+llm' | 'rules+llm_fallback'
    """
    # Stage 1: Deterministic rules
    params = _apply_rules(noise_level, n_samples, n_degs)
    logger.info(
        f"Rule-based params: beta={params['carnival_beta']}, "
        f"top_n_tfs={params['top_n_tfs']}, "
        f"solver_timeout={params['solver_timeout']}, "
        f"pkn_filter={params['pkn_filter']}"
    )

    if not use_llm:
        params['rationale'] = "LLM refinement skipped (use_llm=False)."
        params['selected_by'] = 'rules'
        return params

    # Stage 2: LangChain refinement
    context = _build_context(
        noise_level=noise_level,
        n_samples=n_samples,
        n_degs=n_degs,
        n_genes=n_genes,
        original_query=original_query,
        extracted_entities=extracted_entities,
        context_for_agent2=context_for_agent2,
        rule_params=params,
    )

    refined = _refine_with_langchain(params, context)
    return refined


# =============================================================================
# STAGE 1: DETERMINISTIC RULES
# =============================================================================

def _apply_rules(noise_level: str, n_samples: int, n_degs: int) -> dict:
    """
    Apply deterministic parameter selection rules.

    Rule table:
    The rules wer reached after reviewing the literature and empirical results on the IgAN, NASH, IPF dataset.
    Depending on the nature of the data, I think these would require further refinement. 
    ┌──────────┬──────────┬─────────┬──────────┬──────────────────────┐
    │ Noise    │ Beta     │ Top TFs │ Timeout  │ CollecTRI confidence │
    ├──────────┼──────────┼─────────┼──────────┼──────────────────────┤
    │ low      │ 0.2      │ 50-100  │ 600s     │ A, B, C              │
    │ medium   │ 0.5      │ 50      │ 1200s    │ A, B, C              │
    │ high     │ 1.0      │ 25-50   │ 1800s    │ A, B                 │
    └──────────┴──────────┴─────────┴──────────┴──────────────────────┘

    Additional rules:This rules were changing as I recalibrate beta with penalize="nodes" instead of "edges". I'm leaving thisnote here to document the rationale for the current beta values.
    - n_degs > 2000: beta → 1.0+ (many DEGs, need stricter network)
    - n_samples < 6: confidence → A, B only (low power)
    - solver: always "auto" (resolves to HIGHS at runtime)
    - pkn_filter: "full" by default (auto adjusts per solver at runtime)
    """
    noise = noise_level.lower() if noise_level else "medium"

    # Base params by noise level
    if noise == "low":
        params = {
            "carnival_beta": 0.1,
            "top_n_tfs": 50,
            "solver_timeout": 600,
            "progeny_top_genes": 200,
        }
    elif noise == "high":
        params = {
            "carnival_beta": 0.5,
            "top_n_tfs": 25,
            "solver_timeout": 1800,
            "progeny_top_genes": 200,
        }
    else:  # medium (default)
        params = {
            "carnival_beta": 0.2,
            "top_n_tfs": 50,
            "solver_timeout": 1200,
            "progeny_top_genes": 200,
        }

    # DEG count adjustments — COMMENTED OUT while calibrating beta with penalize="nodes"
    # These values were set for penalize="edges" and need recalibration.
    # Revisit after empirical beta experiments with IgAN dataset.
    # if n_degs < 50:
    #     params["carnival_beta"] = 0.1
    #     params["top_n_tfs"] = min(params["top_n_tfs"], 25)
    #     logger.info(f"Few DEGs ({n_degs}): beta → 0.1, top_n_tfs → {params['top_n_tfs']}")
    # elif n_degs > 2000:
    #     params["carnival_beta"] = max(params["carnival_beta"], 1.0)
    #     logger.info(f"Many DEGs ({n_degs}): beta → {params['carnival_beta']}")

    # Note: CollecTRI does not have A/B/C confidence levels (that was DoRothEA).
    # CollecTRI returns all interactions with signed weights. No filtering needed.

    # Fixed params (not adjusted by rules, may be overridden by LLM)
    # mip_gap/threads: reserved in schema but not yet forwarded to CORNETO solver
    params["solver"] = "auto"
    params["pkn_filter"] = "full"
    params["penalize"] = "edges"  # Edge penalty for comparison experiment (was "nodes" for Liu 2019)
    params["mip_gap"] = 0.05
    params["threads"] = 0
    params["tf_method"] = "viper"
    params["selected_by"] = "rules"
    params["rationale"] = ""

    return params


# =============================================================================
# STAGE 2: LANGCHAIN REFINEMENT
# =============================================================================
# The system prompt guides the LLM to critically evaluate the rule-based defaults
# and adjust them based on the biological context. It explicitly lists what the
# rules can and cannot see, and provides clear scenarios for when to adjust parameters.

_SYSTEM_PROMPT = """You are an expert computational biologist specializing in
causal network analysis using CARNIVAL. Your task is to critically evaluate
rule-based parameter selections and adjust them based on the biological context.

The rule-based system only considers noise level and DEG count. It CANNOT see:
- What disease this is (rare vs well-studied)
- Whether this is disease-vs-control or within-disease staging
- How many DEGs there are relative to what's biologically expected
- Whether the study is exploratory (cast a wide net) or confirmatory (focus on known biology)

YOU must fill that gap. Actively use the biological context to decide if the
defaults are appropriate. The rules are a starting point, not gospel.

WHEN TO ADJUST (you SHOULD adjust in these scenarios):
- Many DEGs (>2000) + well-studied disease: RAISE beta (0.3-0.8) to focus on strongest signals
  and avoid an uninterpretable hairball network.
- Few DEGs (<100) or rare/understudied disease: LOWER beta (0.05-0.15) to capture more
  network structure when data is sparse or biology is poorly characterized.
- Within-disease staging (e.g., early vs late, grade I vs III): LOWER beta — subtle fold
  changes mean the network needs to be more permissive to capture gradual signaling shifts.
- High noise + small sample size: RAISE beta to avoid fitting noise into the network.
- Very large pseudobulk dataset (>2000 DEGs): RAISE beta — pseudobulk inflates DEG counts.

WHEN TO ACCEPT DEFAULTS:
- Standard disease-vs-control with moderate DEGs (200-1000) and adequate samples (>10).
- The rule-based values already reflect the biological scenario above.

PARAMETER GUIDE:
- beta (0.05-2.0): Edge penalty controlling network size vs measurement fit.
  Low beta = larger network, captures more biology but may include noise.
  High beta = smaller, stricter network, higher confidence edges only.

- top_n_tfs (10-100): TFs passed to CARNIVAL as constraints.
  More TFs = more constrained but slower to solve.
  Reduce for noisy data (fewer confident TFs). Increase for clean, well-powered studies.

- solver_timeout (60-3600): Seconds before solver gives up.
  HIGHS is fast — 600s is usually sufficient. Increase only for very large networks.

- pkn_filter ("full"/"2hop"/"1hop"): Prior knowledge network size.
  IMPORTANT: Always return "full". Do NOT change pkn_filter — this is only adjusted
  after a confirmed solver timeout failure, which has not occurred at this stage.

Always respond in this exact format — no other text:

<reasoning>
Step-by-step analysis of the biological context and why parameters should or should
not be adjusted. Reference specific data points (DEG count, disease, noise, sample size).
2-4 sentences maximum.
</reasoning>
<parameters>
{{"carnival_beta": <float>, "top_n_tfs": <int>, "solver_timeout": <int>, "pkn_filter": "<string>"}}
</parameters>"""


_HUMAN_PROMPT = """DATASET CONTEXT:
{context}

RULE-BASED PARAMETER DEFAULTS:
- beta: {beta}
- top_n_tfs: {top_n_tfs}  
- solver_timeout: {solver_timeout}s
- pkn_filter: {pkn_filter}

Please review and refine these parameters for this specific dataset."""


def _refine_with_langchain(rule_params: dict, context: str) -> dict:
    """
    Refine rule-based parameters using a LangChain chain with CoT reasoning.

    Chain: ChatPromptTemplate | ChatAnthropic | StrOutputParser
    
    Uses StrOutputParser (not JsonOutputParser) because we want to capture
    the full <reasoning>...</reasoning><parameters>...</parameters> response
    and parse it ourselves. This gives us the CoT reasoning for the thesis.

    Falls back to direct Anthropic API on LangChain failure.
    Falls back to rule-based params on any LLM failure.
    """
    ChatAnthropic, ChatPromptTemplate, StrOutputParser = _try_import_langchain()

    if ChatAnthropic is None:
        logger.warning("LangChain not available. Trying direct Anthropic fallback.")
        return _refine_with_direct_api(rule_params, context)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set. Skipping LLM refinement.")
        rule_params['rationale'] = "LLM refinement skipped: no API key."
        rule_params['selected_by'] = 'rules'
        return rule_params

    try:
        # Build the LangChain chain
        # Pattern: prompt | llm | output_parser
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ])

        llm = ChatAnthropic(
            model="claude-sonnet-4-0",
            api_key=api_key,
            max_tokens=500,
            temperature=0.1,
        )

        output_parser = StrOutputParser()

        # Compose the chain using LangChain's pipe operator
        chain = prompt | llm | output_parser

        logger.info("Invoking LangChain parameter refinement chain...")
        response = chain.invoke({
            "context": context,
            "beta": rule_params["carnival_beta"],
            "top_n_tfs": rule_params["top_n_tfs"],
            "solver_timeout": rule_params["solver_timeout"],
            "pkn_filter": rule_params["pkn_filter"],
        })

        logger.info(f"LangChain response received ({len(response)} chars)")
        return _parse_llm_response(response, rule_params, source="rules+llm")

    except Exception as e:
        logger.warning(
            f"LangChain chain failed: {type(e).__name__}: {e}. "
            f"Trying direct Anthropic API fallback."
        )
        return _refine_with_direct_api(rule_params, context)


def _refine_with_direct_api(rule_params: dict, context: str) -> dict:
    """
    Fallback: direct Anthropic API call without LangChain.
    Same prompt, same parsing — just uses the SDK directly.
    """
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        human_message = _HUMAN_PROMPT.format(
            context=context,
            beta=rule_params["carnival_beta"],
            top_n_tfs=rule_params["top_n_tfs"],
            solver_timeout=rule_params["solver_timeout"],
            pkn_filter=rule_params["pkn_filter"],
        )

        response = client.messages.create(
            model="claude-sonnet-4-0",
            max_tokens=500,
            temperature=0.1,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": human_message}],
        )

        response_text = response.content[0].text
        logger.info("Direct Anthropic API fallback succeeded.")
        return _parse_llm_response(response_text, rule_params, source="rules+llm_fallback")

    except Exception as e:
        logger.warning(
            f"Direct API fallback also failed: {type(e).__name__}: {e}. "
            f"Using rule-based params unchanged."
        )
        rule_params['rationale'] = f"LLM refinement failed: {e}. Using rule-based defaults."
        rule_params['selected_by'] = 'rules'
        return rule_params


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def _parse_llm_response(
    response: str,
    rule_params: dict,
    source: str,
) -> dict:
    """
    Parse the LLM response and extract reasoning + parameter adjustments.

    Expected format:
        <reasoning>...</reasoning>
        <parameters>{"beta": ..., "top_n_tfs": ..., ...}</parameters>

    On any parse failure, returns rule_params unchanged with the raw
    response stored as rationale (so nothing is lost).
    """
    # Extract reasoning
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response[:200]

    # Extract parameters JSON
    params_match = re.search(r'<parameters>(.*?)</parameters>', response, re.DOTALL)

    if not params_match:
        logger.warning(
            "LLM response missing <parameters> block. "
            "Using rule-based params. Raw response stored as rationale."
        )
        rule_params['rationale'] = f"[LLM reasoning]: {reasoning}"
        rule_params['selected_by'] = 'rules'  # LLM responded but params not usable
        return rule_params

    try:
        llm_params = json.loads(params_match.group(1).strip())
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM parameters JSON: {e}. Using rule-based params.")
        rule_params['rationale'] = f"[LLM reasoning (params parse failed)]: {reasoning}"
        rule_params['selected_by'] = 'rules'  # JSON parse failed — params unchanged
        return rule_params

    # Merge: start with rule_params, apply LLM overrides
    # Only accept adjustments within safe bounds
    refined = rule_params.copy()
    adjustments = []

    for key, (low, high) in _SAFE_RANGES.items():
        if key in llm_params:
            val = llm_params[key]
            if isinstance(val, (int, float)) and low <= val <= high:
                if val != rule_params.get(key):
                    adjustments.append(f"{key}: {rule_params.get(key)} → {val}")
                    refined[key] = type(rule_params[key])(val)  # preserve int/float type
            else:
                logger.warning(
                    f"LLM suggested {key}={val} which is outside safe range "
                    f"[{low}, {high}]. Keeping rule-based value: {rule_params.get(key)}"
                )

    if "pkn_filter" in llm_params:
        val = llm_params["pkn_filter"]
        if val != rule_params.get("pkn_filter"):
            logger.warning(
                f"LLM attempted to change pkn_filter to '{val}'. "
                f"Rejected — pkn_filter is not LLM-adjustable at initial parameter selection."
            )
    # Hard guard: pkn_filter must always match rule-based value (never LLM-overridden)
    refined['pkn_filter'] = rule_params['pkn_filter']

    # Build rationale
    if adjustments:
        refined['rationale'] = (
            f"[LLM reasoning]: {reasoning}\n"
            f"[Adjustments made]: {', '.join(adjustments)}"
        )
        logger.info(f"LLM adjusted {len(adjustments)} parameter(s): {adjustments}")
    else:
        refined['rationale'] = (
            f"[LLM reasoning]: {reasoning}\n"
            f"[Adjustments made]: None — rule-based defaults accepted."
        )
        logger.info("LLM accepted all rule-based defaults.")

    refined['selected_by'] = source
    return refined


# =============================================================================
# CONTEXT BUILDER
# =============================================================================

def _build_context(
    noise_level: str,
    n_samples: int,
    n_degs: int,
    n_genes: int,
    original_query: str,
    extracted_entities: Optional[dict],
    context_for_agent2: Optional[str],
    rule_params: dict,
) -> str:
    """Build a concise context string for the LLM prompt."""
    lines = [
        f"User query: {original_query}",
        f"Dataset: {n_samples} samples, {n_genes} genes, {n_degs} DEGs",
        f"Noise level: {noise_level}",
    ]

    if context_for_agent2:
        lines.append(f"Pipeline summary: {context_for_agent2[:300]}")

    if extracted_entities:
        disease = extracted_entities.get("disease", "")
        cell_type = extracted_entities.get("cell_type", "")
        tfs = extracted_entities.get("transcription_factors", [])
        if disease:
            lines.append(f"Disease: {disease}")
        if cell_type:
            lines.append(f"Cell type: {cell_type}")
        if tfs:
            lines.append(f"TFs of interest: {', '.join(tfs[:5])}")

    return "\n".join(lines)


# =============================================================================
# FAILURE-GUIDED RETRY (called by r_bridge.py after CARNIVAL failure)
# =============================================================================

def refine_params_after_failure(
    current_params: dict,
    failure_type: str,
    attempt_number: int,
    n_tfs: int,
    n_edges: int,
) -> dict:
    """
    Ask LLM to diagnose a CARNIVAL failure and suggest parameter adjustments.
    Called by retry logic after a CARNIVAL solve failure.

    Args:
        current_params: The CARNIVALParams dict that just failed.
        failure_type: "empty_network" | "timeout" | "infeasible" | "solver_error"
        attempt_number: Which attempt just failed (1-indexed).
        n_tfs: Number of TF measurements used.
        n_edges: Number of PKN edges used.

    Returns:
        Updated params dict with adjustments for next attempt.
    """
    ChatAnthropic, ChatPromptTemplate, StrOutputParser = _try_import_langchain()

    failure_prompt = f"""CARNIVAL attempt {attempt_number} failed: {failure_type}

Current parameters:
- beta: {current_params.get('carnival_beta')}
- top_n_tfs: {current_params.get('top_n_tfs')}
- solver_timeout: {current_params.get('solver_timeout')}s
- pkn_filter: {current_params.get('pkn_filter')}
- solver: {current_params.get('solver')}

Network context:
- TF measurements: {n_tfs}
- PKN edges: {n_edges}

Failure type guidance:
- empty_network: beta likely too high, or too few TFs. Reduce beta, increase top_n_tfs.
- timeout: network too large for solver. Reduce top_n_tfs first (try 25, then 15).
  Only switch pkn_filter to '2hop' if top_n_tfs is already at minimum (10).
- infeasible: constraints too tight. Reduce top_n_tfs significantly, lower beta.
- solver_error: unknown solver error. Try minimal config: beta=0.1, top_n_tfs=15, pkn_filter='2hop'.

Suggest adjusted parameters for attempt {attempt_number + 1}.
Respond with JSON only — no other text:
{{"carnival_beta": <float>, "top_n_tfs": <int>, "solver_timeout": <int>, "pkn_filter": "<string>"}}"""

    try:
        if ChatAnthropic is not None:
            llm = ChatAnthropic(
                model="claude-sonnet-4-0",
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                max_tokens=150,
                temperature=0.1,
            )
            prompt = ChatPromptTemplate.from_messages([("human", failure_prompt)])
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({})
        else:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-0",
                max_tokens=150,
                temperature=0.1,
                messages=[{"role": "user", "content": failure_prompt}],
            )
            response = msg.content[0].text

        # Parse JSON response
        clean = re.sub(r'```json|```', '', response).strip()
        adjustments = json.loads(clean)

        updated = current_params.copy()
        for key in ["carnival_beta", "top_n_tfs", "solver_timeout"]:
            if key in adjustments:
                val = adjustments[key]
                if not isinstance(val, (int, float)):
                    logger.warning(f"LLM retry suggested non-numeric {key}={val!r}. Skipping.")
                    continue
                low, high = _SAFE_RANGES[key]
                clamped = max(low, min(high, val))
                if clamped != val:
                    logger.warning(f"LLM retry {key}={val} clamped to [{low}, {high}] → {clamped}")
                updated[key] = type(current_params.get(key, val))(clamped)
                logger.info(f"Retry adjustment: {key} → {updated[key]}")

        # Only allow pkn_filter change on timeout AND only if top_n_tfs is already at minimum
        if failure_type == "timeout" and updated.get("top_n_tfs", 50) <= 10:
            if "pkn_filter" in adjustments and adjustments["pkn_filter"] in {"2hop", "1hop"}:
                updated["pkn_filter"] = adjustments["pkn_filter"]
                logger.info(f"Retry adjustment: pkn_filter → {adjustments['pkn_filter']} (timeout, min TFs reached)")
        elif "pkn_filter" in adjustments and adjustments["pkn_filter"] != current_params.get("pkn_filter"):
            logger.warning(
                f"LLM suggested pkn_filter='{adjustments['pkn_filter']}' but "
                f"failure_type='{failure_type}' — rejected (only allowed on timeout + min TFs)."
            )

        updated['selected_by'] = 'rules+llm_retry'
        return updated

    except Exception as e:
        logger.warning(f"LLM retry guidance failed: {e}. Applying fixed fallback.")
        # Fixed fallback: each attempt gets progressively more permissive
        updated = current_params.copy()
        updated['carnival_beta'] = max(0.05, current_params.get('carnival_beta', 0.5) * 0.5)
        updated['top_n_tfs'] = max(10, current_params.get('top_n_tfs', 50) // 2)
        updated['solver_timeout'] = min(3600, current_params.get('solver_timeout', 600) * 2)
        if failure_type == "timeout" and updated.get('top_n_tfs', 50) <= 10:
            updated['pkn_filter'] = '2hop'
        updated['selected_by'] = 'rules_retry_fallback'
        return updated


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    print("=" * 60)
    print("Parameter Selector Test")
    print("=" * 60)

    result = select_parameters(
        noise_level="medium",
        n_samples=8,
        n_degs=369,
        n_genes=2000,
        original_query="What transcription factors drive fibrosis in IPF AT2 cells?",
        extracted_entities={"disease": "IPF", "cell_type": "AT2"},
        context_for_agent2="369 DEGs identified in AT2 cells comparing IPF vs Control.",
        use_llm=True,
    )

    print("\nFinal parameters:")
    for k, v in result.items():
        if k != 'rationale':
            print(f"  {k}: {v}")
    print(f"\nRationale:\n{result.get('rationale', 'None')}")
    print(f"\nSelected by: {result.get('selected_by')}")
