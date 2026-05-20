"""
Agent 2 Schemas
===============
Pydantic models documenting the real data contracts for the Agent 2 pipeline.

These mirror the actual @dataclass implementations:
  - CARNIVALResult  -> core/corneto_runner.py  (CARNIVAL solver output)
  - Agent2Result    -> core/output_packager.py  (final output to Agent 3)

Parameters are returned as plain dicts by llm/parameter_selector.py.
The parameter keys and their valid ranges are documented below.

INPUT:  Agent1Result (imported from Agent_1/agent1/schemas_agent1.py)
OUTPUT: Agent2Result (consumed by Agent 3 and Streamlit UI)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# =============================================================================
# CARNIVAL SOLVER OUTPUT (mirrors @dataclass in core/corneto_runner.py)
# =============================================================================

class CARNIVALResult(BaseModel):
    """
    Normalised CARNIVAL output from CORNETO ILP solver.

    Authoritative implementation: @dataclass in core/corneto_runner.py.

    This is the intermediate result between the solver (Step E) and the
    output packager (Step F). Agent 3 never sees this directly.
    """
    nodes_df: Any = Field(
        description=(
            "pd.DataFrame — all nodes. Index=node name, column='node_activity' (+1/-1/0)."
        )
    )
    edges_df: Any = Field(
        description=(
            "pd.DataFrame — all edges. Index=(frozenset, frozenset) tuples, "
            "column='edge_activity' (+1/-1/0)."
        )
    )
    stats: Dict[str, Any] = Field(
        description=(
            "Execution metadata. Keys: solver, beta_weight, build_time_seconds, "
            "solve_time_seconds, total_time_seconds, n_tfs_input, n_tfs_used, "
            "n_pkn_edges, n_processed_nodes, n_processed_edges, n_active_nodes, "
            "n_active_edges, n_activated_nodes, n_repressed_nodes, status, sample_name."
        )
    )
    sample_name: str = Field(
        default="sample",
        description="Label for this sample."
    )

    model_config = {"arbitrary_types_allowed": True}


# =============================================================================
# AGENT 2 RESULT — PRIMARY OUTPUT CONTRACT
# =============================================================================
# Authoritative implementation: @dataclass in core/output_packager.py.

class Agent2Result(BaseModel):
    """
    Complete output from Agent 2, consumed by Agent 3 and the Streamlit UI.

    The carnival_network is a signed directed nx.DiGraph where:
      - Node attributes:
          activity:       +1 (activated), -1 (repressed), 0 (inactive)
          activity_score: float (same value as activity)
          node_type:      'TF' | 'measured' | 'intermediate'
          is_active:      bool
      - Edge attributes:
          sign:           +1 (activation) or -1 (inhibition)
          interaction:    'ACTIVATES' | 'INHIBITS' | 'NEUTRAL'
          weight:         1.0 or 0.0 (abs of sign)
          is_active:      bool (True if CARNIVAL selected this edge)

    Agent 3 uses carnival_network for:
      1. Tool: get_active_regulators()   -> nodes where activity != 0
      2. Tool: get_causal_paths()        -> nx.shortest_path between nodes
      3. Tool: get_hub_nodes()           -> degree centrality
      4. RAG queries: node names as search terms

    Computed properties (on the @dataclass, not on this Pydantic mirror):
      - n_active_nodes:  int  (count of nodes with activity != 0)
      - n_active_edges:  int  (count of edges with is_active=True)
      - activated_nodes: list (nodes with activity == +1)
      - repressed_nodes: list (nodes with activity == -1)
    """
    carnival_network: Any = Field(
        description="nx.DiGraph — primary deliverable for Agent 3."
    )
    nodes_df: Any = Field(
        description="pd.DataFrame — raw CARNIVAL node activities (audit trail)."
    )
    edges_df: Any = Field(
        description="pd.DataFrame — raw CARNIVAL edge activities (audit trail)."
    )
    metadata: Dict[str, Any] = Field(
        description=(
            "Solver stats + Agent 1 context. Keys include: solver, beta_weight, "
            "solve_time_seconds, n_nodes_total, n_edges_total, n_active_nodes, "agent 2 has a schemas_agent2 file as well. is it not used?
            "n_active_edges, comparison, data_type, original_query, pipeline_stats, from_cache."
        )
    )
    summary: str = Field(
        description="LLM-ready text summary for Agent 3 context window."
    )
    sample_name: str = Field(
        default="sample",
        description="Label for this sample (e.g. 'igan_glomerular')."
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Pipeline warnings (e.g. low active nodes, TF coverage issues)."
    )
    edge_annotations: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Edge annotation lookup keyed by (src, tgt) tuples. "
            "Values: {'signor': [...], 'shared_pathways': [...], 'top3_pathways': [...]}."
        )
    )

    model_config = {"arbitrary_types_allowed": True}


# =============================================================================
# PARAMETER SELECTOR OUTPUT (plain dict — documented here for reference)
# =============================================================================
# llm/parameter_selector.py returns a plain dict with these keys:
#
#   carnival_beta:    float   (0.05 - 2.0)  Network sparsity penalty
#   top_n_tfs:        int     (10 - 100)    TFs passed to CARNIVAL
#   solver_timeout:   int     (60 - 3600)   Max solver runtime in seconds
#   solver:           str     "auto" | "HIGHS" | "SCIP" | "SCIPY"
#   pkn_filter:       str     "full" | "2hop" | "1hop"
#   penalize:         str     "nodes" | "edges" | "both"
#   progeny_top_genes: int    (100 - 200)   PROGENy genes per pathway
#   mip_gap:          float   MIP optimality gap (reserved, not yet forwarded)
#   threads:          int     Solver threads (reserved, not yet forwarded)
#   tf_method:        str     "viper" | "ulm"
#   rationale:        str     LLM chain-of-thought reasoning
#   selected_by:      str     "rules" | "rules+llm" | "rules+llm_fallback"
#   tokens_used:      int     LLM tokens consumed (0 if rules-only)
