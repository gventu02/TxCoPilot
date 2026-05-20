"""
Agent 2 — Network Confidence Assessor
=======================================
After CARNIVAL produces a causal network, this module uses an LLM to evaluate
the result quality and produce a structured confidence assessment.

The assessment is:
  1. Shown to the user in the Streamlit results tab
  2. Passed to Agent 3 so it can caveat its interpretation appropriately

Metrics evaluated:
  - TF recovery: how many input TFs appear in the active network
  - Measured node recovery: how many DEGs are connected in the network
  - Disconnected components: fragmented network = low confidence
  - Network size relative to input DEGs
  - Edge/node ratio: reported but NOT penalized (ILP solver always produces ~1.0)

Author: Giselle Ventura
"""

import json
import logging
import os
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

try:
    from anthropic import Anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


ASSESSMENT_TOOL = {
    "name": "network_confidence_assessment",
    "description": (
        "Evaluate the quality and confidence of a CARNIVAL causal reconstructed network result. "
        "Provide a structured assessment based on network topology metrics."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Overall confidence in the network result.",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "A detailed explanation (3-5 sentences) of why this confidence "
                    "level was assigned. Reference specific metrics. Written for a "
                    "biologist, not a computer scientist."
                ),
            },
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "What the network does well (1-3 bullet points).",
            },
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Weaknesses or caveats (1-3 bullet points).",
            },
            "suggestions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Actionable suggestions for the user if confidence is medium/low. "
                    "Empty list if confidence is high."
                ),
            },
        },
        "required": ["confidence", "reasoning", "strengths", "limitations", "suggestions"],
    },
}

ASSESSMENT_SYSTEM_PROMPT = """\
You are a network biology expert evaluating the quality of a CARNIVAL causal network.

You will receive network metrics and must assess confidence. Use these guidelines:

IMPORTANT CONTEXT — ILP SOLVER CONSTRAINTS:
- This pipeline uses CARNIVAL's ILP solver, which enforces a TREE topology constraint
  (at most one incoming edge per node). This has two structural consequences:
  1. The edge/node ratio has an UPPER BOUND near 1.0 (a tree with N nodes has N-1 edges).
     A ratio close to 1.0 is the BEST CASE for ILP networks, not a problem.
     Ratios significantly below 1.0 indicate the solver dropped many edges:
       - >= 0.7: healthy (expected for ILP)
       - 0.5 to 0.7: moderate concern — some signaling paths may be missing
       - < 0.5: poor — the network is very sparse, missing many connections
     Do NOT expect or reward ratios above 1.0 — that is structurally not expected for ILP.
  2. DEG recovery will be inherently low — typically 1-5 DEGs in the network out
     of hundreds or thousands of input DEGs. This is EXPECTED AND NORMAL. The ILP
     tree can only include DEGs that lie directly on a signaling path between TFs
     and measured nodes. A network with 1-2 DEGs is functioning correctly.
     The deg_recovery_pct will often be < 1% — this is NOT a problem.
     Only flag DEG recovery if it is literally ZERO (no measured genes at all).

- The primary quality signals for an ILP network are:
  1. TF recovery (did the solver include the inferred regulators?)
  2. Edge/node ratio (closer to 1.0 = better; < 0.5 = sparse/poor)
  3. Network size (enough nodes for meaningful signaling context?)
  4. Connectedness (single component = coherent narrative)
  5. DEG recovery — only flag if ZERO. Having 1-2 DEGs is normal and expected but having more is better.

CONFIDENCE LEVELS:

HIGH confidence (all of):
- TF recovery > 60%
- Edge/node ratio >= 0.7
- Network has at least 20 active nodes
- At most 2 disconnected components
- At least 3 DEG in the network (n_measured_in_network >= 3)

MEDIUM confidence (any of):
- TF recovery 30-60%
- Edge/node ratio 0.5-0.7
- Network has 10-20 active nodes
- 3-5 disconnected components

LOW confidence (any of):
- TF recovery < 30%
- Edge/node ratio < 0.5
- Zero DEGs in network (n_measured_in_network = 0)
- More than 5 disconnected components
- Fewer than 10 active nodes

SPECIAL CASES to flag clearly:
- High TF recovery but very small network (< 15 nodes): the network captured
  regulators but not enough downstream signaling to be interpretable.
- Zero DEG recovery: the network is completely disconnected from the observed
  differential expression. This IS a serious problem.
- High TF recovery + decent size + any non-zero DEGs in network: the ILP tree
  structurally limits DEG inclusion to those on direct signaling paths. Having
  1-10 DEGs out of thousands is the expected outcome. Do NOT penalize confidence
  for low DEG counts. Do NOT call the DEG count "very low" or "poor" — note it
  as context and assess confidence based on TF recovery, ratio, size, and connectivity.

When writing reasoning, explain what the metrics mean biologically:
- Low TF recovery means the network may be missing important regulators
- Edge/node ratio near 1.0 is ideal for ILP; below 0.7 means the solver pruned
  many edges, potentially losing signaling paths; below 0.5 is very sparse
- Zero DEGs in the network means the causal model doesn't connect to any measured changes
- Having 1-5 DEGs is normal and expected for ILP — do NOT describe this negatively
- Disconnected components mean the network lacks a coherent signaling narrative

Suggestions should be actionable within this pipeline:

CRITICAL — beta_weight direction:
  beta_weight is an EDGE PENALTY. Higher beta = MORE penalty = FEWER edges = SMALLER network.
  To get a LARGER network with MORE edges, you must LOWER beta_weight (e.g., from 0.2 to 0.1).
  To get a SMALLER, stricter network, RAISE beta_weight.
  NEVER suggest "increasing beta" to expand a network, that is backwards.

Other actionable suggestions:
- Expanding the prior knowledge network (PKN) adds more signaling paths
- Lowering the CollecTRI confidence threshold includes more TF-target edges
- Increasing top_n_tfs allows more transcription factors as input
Do NOT suggest switching solvers.

Write for a biologist reading their experiment results. Be honest but constructive.
"""


def compute_network_metrics(
    G: nx.DiGraph,
    metadata: dict,
) -> dict:
    """Compute metrics the LLM needs for its assessment."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    active_nodes = [n for n, d in G.nodes(data=True) if d.get("activity", 0) != 0]
    n_active = len(active_nodes)

    # Edge/node ratio
    edge_node_ratio = round(n_edges / n_active, 2) if n_active > 0 else 0.0

    # TF recovery
    n_tfs_input = metadata.get("n_tfs_input", 0) or 0
    n_tfs_in_network = sum(
        1 for n, d in G.nodes(data=True)
        if d.get("node_type") == "TF" and d.get("activity", 0) != 0
    )
    tf_recovery_pct = round(100 * n_tfs_in_network / n_tfs_input, 1) if n_tfs_input > 0 else 0.0

    # Measured node (DEG) recovery
    n_degs_input = metadata.get("n_degs", 0) or 0
    n_measured_in_network = sum(
        1 for n, d in G.nodes(data=True)
        if d.get("node_type") == "measured" and d.get("activity", 0) != 0
    )
    deg_recovery_pct = round(100 * n_measured_in_network / n_degs_input, 1) if n_degs_input > 0 else 0.0

    # Disconnected components (on undirected view)
    n_components = nx.number_weakly_connected_components(G.subgraph(active_nodes))

    # Node type breakdown
    n_activated = sum(1 for n, d in G.nodes(data=True) if d.get("activity") == 1)
    n_repressed = sum(1 for n, d in G.nodes(data=True) if d.get("activity") == -1)
    n_intermediate = sum(
        1 for n, d in G.nodes(data=True)
        if d.get("node_type") == "intermediate" and d.get("activity", 0) != 0
    )

    return {
        "n_active_nodes": n_active,
        "n_edges": n_edges,
        "edge_node_ratio": edge_node_ratio,
        "n_tfs_input": n_tfs_input,
        "n_tfs_in_network": n_tfs_in_network,
        "tf_recovery_pct": tf_recovery_pct,
        "n_degs_input": n_degs_input,
        "n_measured_in_network": n_measured_in_network,
        "deg_recovery_pct": deg_recovery_pct,
        "n_disconnected_components": n_components,
        "n_activated": n_activated,
        "n_repressed": n_repressed,
        "n_intermediate_active": n_intermediate,
        "solver": metadata.get("solver", "unknown"),
        "beta_weight": metadata.get("beta_weight", "unknown"),
    }


def assess_network_confidence(
    G: nx.DiGraph,
    metadata: dict,
    api_key: Optional[str] = None,
) -> dict:
    """
    Assess network confidence using LLM + network metrics.

    Returns dict with keys:
        confidence: "high" | "medium" | "low"
        reasoning: str (3-5 sentences)
        strengths: list[str]
        limitations: list[str]
        suggestions: list[str]
        metrics: dict (raw metrics used for assessment)

    Falls back to rule-based assessment if LLM is unavailable.
    """
    metrics = compute_network_metrics(G, metadata)
    logger.info(
        f"Network metrics: {metrics['n_active_nodes']} active nodes, "
        f"ratio={metrics['edge_node_ratio']}, "
        f"TF recovery={metrics['tf_recovery_pct']}%, "
        f"DEG recovery={metrics['deg_recovery_pct']}%, "
        f"components={metrics['n_disconnected_components']}"
    )

    # Try LLM assessment
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if _HAS_ANTHROPIC and key:
        try:
            assessment = _llm_assess(metrics, key)
            assessment["metrics"] = metrics
            return assessment
        except Exception as e:
            logger.warning(f"LLM assessment failed ({e}), using rule-based fallback")

    # Fallback: deterministic rules
    assessment = _rule_based_assess(metrics)
    assessment["metrics"] = metrics
    return assessment


def _llm_assess(metrics: dict, api_key: str) -> dict:
    """Call Claude to assess network confidence."""
    client = Anthropic(api_key=api_key)

    # Remove deg_recovery_pct from LLM input — the tiny percentage (e.g. 0.3%)
    # misleads the LLM into treating normal ILP behavior as a failure.
    # n_measured_in_network (absolute count) and n_degs_input are still available.
    llm_metrics = {k: v for k, v in metrics.items() if k != "deg_recovery_pct"}

    user_msg = (
        "Assess the confidence of this CARNIVAL network result:\n\n"
        f"{json.dumps(llm_metrics, indent=2)}"
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        temperature=0.1,
        system=ASSESSMENT_SYSTEM_PROMPT,
        tools=[ASSESSMENT_TOOL],
        tool_choice={"type": "tool", "name": "network_confidence_assessment"},
        messages=[{"role": "user", "content": user_msg}],
    )

    tool_block = response.content[0]
    if tool_block.type != "tool_use":
        raise ValueError(f"Expected tool_use, got {tool_block.type}")

    tokens = response.usage.input_tokens + response.usage.output_tokens
    logger.info(f"Network assessment LLM call: {tokens} tokens")

    return tool_block.input


def _rule_based_assess(metrics: dict) -> dict:
    """
    Deterministic fallback when LLM is unavailable.

    Scoring (max 6 points):
      TF recovery:      0/1/2   (< 30% / 30-60% / > 60%)
      Edge/node ratio:  0/1     (< 0.7 / >= 0.7)
      Components:       0/1     (> 5 / <= 2)
      Network size:     0/1     (< 10 / >= 20)
      DEG present:      0/1     (zero DEGs / any DEGs in network)

    Edge/node ratio interpretation (ILP tree upper bound ≈ 1.0):
      >= 0.7: healthy — expected for ILP networks
      0.5 - 0.7: moderate concern — some signaling paths may be missing
      < 0.5: poor — very sparse network

    DEG recovery % is NOT scored — the ILP tree constraint structurally
    limits how many DEGs can appear. Only zero recovery is penalized.

    Thresholds: >= 4 high, >= 2 medium, else low
    """
    score = 0
    strengths = []
    limitations = []
    suggestions = []

    n_active = metrics["n_active_nodes"]
    tf_pct = metrics["tf_recovery_pct"]
    deg_pct = metrics["deg_recovery_pct"]
    n_measured = metrics["n_measured_in_network"]
    n_comp = metrics["n_disconnected_components"]
    ratio = metrics["edge_node_ratio"]

    # --- TF recovery (primary quality signal) ---
    if tf_pct > 60:
        score += 2
        strengths.append(
            f"Strong TF recovery ({metrics['n_tfs_in_network']}/{metrics['n_tfs_input']} "
            f"input TFs, {tf_pct}%) — most inferred regulators are connected "
            f"in the causal network."
        )
    elif tf_pct > 30:
        score += 1
        limitations.append(
            f"Moderate TF recovery ({tf_pct}%). Some important regulators may "
            f"be missing from the network."
        )
    else:
        limitations.append(
            f"Low TF recovery ({tf_pct}%). Most inferred transcription factors "
            f"are not connected in the network, suggesting the prior knowledge "
            f"network may lack relevant signaling paths."
        )
        suggestions.append(
            "Try expanding the prior knowledge network or lowering the "
            "CollecTRI confidence threshold to include more TF-target edges."
        )

    # --- Edge/node ratio (ILP upper bound ≈ 1.0) ---
    if ratio >= 0.7:
        score += 1
        strengths.append(
            f"Edge/node ratio ({ratio:.2f}) is healthy for an ILP network "
            f"(upper bound ≈ 1.0). The solver retained most signaling connections."
        )
    elif ratio >= 0.5:
        limitations.append(
            f"Edge/node ratio ({ratio:.2f}) is below the expected ILP range "
            f"(ideally >= 0.7). Some signaling paths may have been pruned by the solver."
        )
    else:
        limitations.append(
            f"Edge/node ratio ({ratio:.2f}) is very low — the network is unusually "
            f"sparse. Many signaling connections were dropped, which may leave gaps "
            f"in the causal narrative."
        )
        suggestions.append(
            "Try lowering beta_weight to allow more edges, or expanding the "
            "prior knowledge network to give the solver more path options."
        )

    # --- DEG presence (binary: any vs none) ---
    # ILP tree topology structurally limits DEG recovery to ~1-5%.
    # Only flag zero recovery as a problem.
    if n_measured > 0:
        score += 1
        if deg_pct > 5:
            strengths.append(
                f"DEG coverage ({metrics['n_measured_in_network']}/{metrics['n_degs_input']} "
                f"DEGs, {deg_pct}%) is above typical ILP range — the network connects "
                f"to a meaningful portion of the observed expression changes."
            )
        else:
            # Low but non-zero — note as ILP limitation, not a quality problem
            strengths.append(
                f"The network includes {n_measured} measured DEGs ({deg_pct}%). "
                f"Low DEG recovery is a known structural limitation of the ILP tree "
                f"topology, not a data quality issue."
            )
    else:
        limitations.append(
            "Zero DEG recovery — the causal network does not include any of the "
            "observed differentially expressed genes. The model is disconnected "
            "from what was actually measured in the expression data."
        )
        suggestions.append(
            "Try increasing top_n_tfs to allow more transcription factors, "
            "or lowering beta_weight to expand the network."
        )

    # --- Components ---
    if n_comp <= 2:
        score += 1
        if n_comp == 1:
            strengths.append(
                "The network forms a single connected component — the signaling "
                "narrative is coherent."
            )
    elif n_comp > 5:
        limitations.append(
            f"Fragmented network ({n_comp} disconnected components). "
            f"The signaling narrative lacks coherence."
        )

    # --- Network size ---
    if n_active >= 20:
        score += 1
    elif n_active < 10:
        limitations.append(
            f"Very small network ({n_active} active nodes). "
            f"May not capture the full complexity of disease signaling."
        )
        suggestions.append(
            "Try lowering the beta_weight parameter to allow a larger, "
            "more permissive network."
        )

    # --- Special case: high TF recovery but tiny network ---
    if tf_pct > 60 and n_active < 15:
        limitations.append(
            f"Although TF recovery is high ({tf_pct}%), the network is very small "
            f"({n_active} nodes). The regulators were captured but there is not "
            f"enough downstream signaling context to build a complete causal narrative."
        )
        # Don't let high TF recovery mask the size problem
        score = min(score, 2)

    # --- Special case: zero DEG recovery despite otherwise good network ---
    if n_measured == 0 and n_active > 0:
        # A network that explains none of the measured expression changes
        # is fundamentally disconnected from the data, regardless of other metrics.
        score = min(score, 1)

    # Determine confidence level (max 6 points)
    if score >= 4:
        confidence = "high"
    elif score >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    # Build reasoning
    reasoning = (
        f"The network has {n_active} active nodes with "
        f"{tf_pct}% TF recovery ({metrics['n_tfs_in_network']}/{metrics['n_tfs_input']}). "
        f"Edge/node ratio is {ratio:.2f} (ILP upper bound ≈ 1.0). "
        f"{n_measured} of {metrics['n_degs_input']} DEGs ({deg_pct}%) appear in the network. "
        f"The network has {n_comp} connected component(s). "
    )
    if deg_pct > 0 and deg_pct < 5:
        reasoning += (
            f"The low DEG recovery is a structural consequence of the ILP tree topology, "
            f"which limits the network to at most one incoming edge per node. "
        )
    if confidence == "high":
        reasoning += "Overall, the network provides strong evidence for causal interpretation."
    elif confidence == "medium":
        reasoning += "The network is usable but some results should be interpreted with caution."
    else:
        reasoning += (
            "The network has significant limitations that may affect the reliability "
            "of biological conclusions."
        )

    return {
        "confidence": confidence,
        "reasoning": reasoning,
        "strengths": strengths,
        "limitations": limitations,
        "suggestions": suggestions,
    }
