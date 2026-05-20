"""
Agent 2 — CORNETO CARNIVAL Runner
==========================================
Runs Inverse CARNIVAL using CORNETO + HIGHS solver.

Replaces the R/lpSolve bridge entirely with a direct Python implementation using 
CORNETO's native ILP interface.

Architecture:
    pkn_df (DataFrame) + tf_scores (dict) + config
        ↓
    build_graph()     → corneto.Graph
    build_data()      → corneto.Data  (all source nodes as inputs, TFs as outputs)
    CarnivalILP.build() → ProblemDef
    ProblemDef.solve()  → optimal solution
        ↓
    extract_results() → CARNIVALResult (nodes_df, edges_df, stats)

Usage:
    from core.corneto_runner import run_carnival

    result = run_carnival(
        pkn_df=pkn_df,           # DataFrame: source | interaction | target
        tf_scores=tf_scores,     # dict: {TF_name: NES_score}
        beta_weight=0.1,         # edge penalty (lower = larger network)
        solver='HIGHS',          # HIGHS (default), SCIP, or SCIPY
        sample_name='ipf_at2',   # label for Data object
    )

    result.nodes_df          # DataFrame: node_name → node_activity (+1/-1/0)
    result.edges_df          # DataFrame: (source, target) → edge_activity (+1/-1/0)
    result.active_nodes_df   # Filtered: only non-zero activity nodes
    result.active_edges_df   # Filtered: only non-zero activity edges
    result.stats             # dict: solve_time, n_active_nodes, n_active_edges, ...
    result.summary           # str: human-readable summary for Agent 3
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import corneto as cn
import pandas as pd
from corneto.graph import Graph
from corneto.methods.future.carnival import CarnivalILP

logger = logging.getLogger(__name__)

# Default solver — confirmed working on 651-edge OmniPath PKN in 0.16s
DEFAULT_SOLVER = "HIGHS"
DEFAULT_BETA   = 0.5 # Edge penalty — controls sparsity of solution (lower = larger network, higher = smaller network)


# =============================================================================
# OUTPUT SCHEMA
# =============================================================================

@dataclass
class CARNIVALResult:
    """
    Normalised CARNIVAL output for Agent 3 consumption.

    Designed to be solver-agnostic — Agent 3 never knows whether
    R CARNIVAL or CORNETO ran. Same schema either way.
    """
    nodes_df: pd.DataFrame          # All nodes: node_activity column (+1/-1/0)
    edges_df: pd.DataFrame          # All edges: edge_activity column (+1/-1/0)
    stats: dict                     # Execution metadata
    sample_name: str = "sample"

    @property
    def active_nodes_df(self) -> pd.DataFrame:
        """Nodes with non-zero activity — the predicted regulators."""
        return self.nodes_df[self.nodes_df["node_activity"] != 0]

    @property
    def active_edges_df(self) -> pd.DataFrame:
        """Edges with non-zero activity — the causal paths."""
        return self.edges_df[self.edges_df["edge_activity"] != 0]

    @property
    def summary(self) -> str:
        """Human-readable summary for Agent 3 context window."""
        active_up   = (self.nodes_df["node_activity"] == 1).sum()
        active_down = (self.nodes_df["node_activity"] == -1).sum()
        n_edges     = (self.edges_df["edge_activity"] != 0).sum()

        top_up   = self.active_nodes_df[self.active_nodes_df["node_activity"] == 1].head(5)
        top_down = self.active_nodes_df[self.active_nodes_df["node_activity"] == -1].head(5)

        lines = [
            f"CARNIVAL Result ({self.sample_name})",
            f"Solver: {self.stats.get('solver')}  "
            f"Beta: {self.stats.get('beta_weight')}  "
            f"Solve time: {self.stats.get('solve_time_seconds', 0):.1f}s",
            f"Active nodes: {active_up} activated, {active_down} repressed "
            f"(of {len(self.nodes_df)} total)",
            f"Active edges: {n_edges} causal interactions",
            "",
            f"Top activated regulators: "
            f"{', '.join(str(n) for n in top_up.index.tolist())}",
            f"Top repressed regulators: "
            f"{', '.join(str(n) for n in top_down.index.tolist())}",
        ]
        return "\n".join(lines)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def run_carnival(
    pkn_df: pd.DataFrame,
    tf_scores: dict,
    beta_weight: float = DEFAULT_BETA,
    solver: str = DEFAULT_SOLVER,
    sample_name: str = "sample",
    verbosity: int = 0,
    deg_gene_set: Optional[set] = None,
    penalize: str = "edges",
    progeny_gene_scores: Optional[dict] = None,
) -> CARNIVALResult:
    """
    Run Inverse CARNIVAL via CORNETO.

    Inverse CARNIVAL: no known upstream perturbation.
    All source nodes in PKN are treated as potential perturbation inputs.
    TF NES scores constrain the output measurements.

    Args:
        pkn_df:      PKN DataFrame with columns: source | interaction | target
                     interaction values: +1 (activation) or -1 (inhibition)
        tf_scores:   Dict mapping TF gene symbols to NES scores from decoupler.
                     Example: {'SMAD2': 2.4, 'SP1': 1.8, 'EZH2': -1.2}
        beta_weight: Edge penalty — controls network sparsity.
                     Lower = larger network (more paths).
                     Higher = smaller, higher-confidence network.
                     Recommended range: 0.01 (permissive) to 0.5 (strict).
                     Default: 0.1
        solver:      Solver name. 'HIGHS' recommended (fast, open source, no license).
                     Also supported: 'SCIP', 'SCIPY' but have not had to default to it
        sample_name: Label for the sample (used in output summary).
        verbosity:   0 = silent, 1 = solver progress, 2 = full CVXPY output.
        deg_gene_set: Optional set of DEG gene symbols. If provided, only
                     source-only nodes in this set are used as _PERT inputs,
                     reducing the search space. If None or empty intersection,
                     all source-only nodes are used as inputs.
        penalize:    What to penalize in the ILP objective: "edges" (CORNETO
                     default, penalizes active edges), "nodes" (Liu 2019,
                     penalizes active nodes), or "both".

    Returns:
        CARNIVALResult with nodes_df, edges_df, stats, summary.

    Raises:
        ValueError: If PKN has wrong format or too few TFs found in network.
        RuntimeError: If solver fails to find optimal/feasible solution.
    """
    t_total = time.time()

    # Step 1: Build graph
    logger.info(f"Building CORNETO graph from {len(pkn_df)} PKN edges...")
    G = _build_graph(pkn_df, deg_gene_set)
    logger.info(f"Graph: {G.num_edges} edges, {G.num_vertices} vertices")

    # Step 2: Check TF overlap
    graph_nodes = set(G.V)
    tf_in_graph = {tf: score for tf, score in tf_scores.items() if tf in graph_nodes}
    tf_missing  = set(tf_scores.keys()) - graph_nodes

    if tf_missing:
        logger.warning(
            f"{len(tf_missing)} TFs not found in PKN (will be skipped): "
            f"{sorted(tf_missing)}"
        )
    if len(tf_in_graph) < 3:
        raise ValueError(
            f"Only {len(tf_in_graph)} TFs found in PKN — too few for CARNIVAL. "
            f"Missing: {sorted(tf_missing)[:10]}. "
            f"NEXT STEP: Check that pkn_builder.py used the same TF gene symbols "
            f"as tf_activity.py output."
        )

    logger.info(f"TF measurements: {len(tf_in_graph)}/{len(tf_scores)} found in PKN")

    # Step 2b: Prune graph — bidirectional BFS keeps only nodes reachable
    # from inputs (_PERT) AND that can reach outputs (TFs)
    input_nodes = ['_PERT']
    output_nodes = list(tf_in_graph.keys())
    pre_prune_v, pre_prune_e = G.num_vertices, G.num_edges
    G = G.prune(input_nodes, output_nodes)
    logger.info(
        f"After prune: {G.num_vertices} vertices, {G.num_edges} edges "
        f"(removed {pre_prune_v - G.num_vertices} vertices, "
        f"{pre_prune_e - G.num_edges} edges)"
    )

    # Recheck TF overlap after pruning (some TFs may have been pruned)
    graph_nodes_post = set(G.V)
    tf_pruned = {tf for tf in tf_in_graph if tf not in graph_nodes_post}
    if tf_pruned:
        logger.warning(
            f"{len(tf_pruned)} TFs pruned (unreachable from _PERT): "
            f"{sorted(tf_pruned)}"
        )
        tf_in_graph = {tf: s for tf, s in tf_in_graph.items() if tf in graph_nodes_post}
        if len(tf_in_graph) < 3:
            raise ValueError(
                f"Only {len(tf_in_graph)} TFs reachable after prune — too few for CARNIVAL. "
                f"Pruned TFs: {sorted(tf_pruned)[:10]}."
            )

    # Step 3: Build Data object
    data = _build_data(pkn_df, tf_in_graph, sample_name)

    # Step 4: Build ILP problem
    logger.info(f"Building CarnivalILP (beta={beta_weight}, penalize={penalize})...")
    t_build = time.time()
    c = CarnivalILP(beta_weight=beta_weight, penalize=penalize)
    P = c.build(G, data)
    build_time = time.time() - t_build

    # Step 4b: Add PROGENy soft objective — reduces effective activation cost
    # for pathway member genes with dual evidence (PROGENy weight × DE t-stat).
    # weight=-0.1 means a gene with score 1.0 gets 0.1 discount from β.
    n_progeny_scored = 0
    if progeny_gene_scores:
        vertex_selected = P.expr.vertex_activated + P.expr.vertex_inhibited
        scores = np.zeros(len(c.processed_graph.V))
        for i, v in enumerate(c.processed_graph.V):
            if v in progeny_gene_scores:
                scores[i] = progeny_gene_scores[v]
                n_progeny_scored += 1
        if n_progeny_scored > 0:
            # Normalize to [0, 1] so weight=-0.1 gives max 0.1 discount off β
            # (raw scores range 0–400, which would override β without normalization)
            max_score = scores.max()
            if max_score > 0:
                scores = scores / max_score
            obj = vertex_selected.T @ scores
            P.add_objective(obj, name="progeny_pathway_bias", weight=-0.1)
            logger.info(
                f"PROGENy soft objective: {n_progeny_scored} pathway genes scored "
                f"(of {len(progeny_gene_scores)} requested, weight=-0.1, "
                f"raw_max={max_score:.1f}, normalized to [0,1])"
            )
        else:
            logger.warning(
                f"PROGENy: none of {len(progeny_gene_scores)} gene scores "
                f"matched processed graph nodes — soft objective skipped"
            )

    n_vars        = P.expr.__len__() if hasattr(P.expr, '__len__') else 'unknown'
    proc_vertices = c.processed_graph.num_vertices
    proc_edges    = c.processed_graph.num_edges
    logger.info(
        f"ILP built in {build_time:.1f}s: "
        f"{proc_vertices} vertices, {proc_edges} edges in processed graph"
    )

    # --- Beta-scaling heuristic ---
    # Adjust beta based on graph density to maintain consistent sparsity
    # across datasets with different topologies. Keeping penalize="edges"
    # fixed, beta controls network size.
    #
    # avg_degree = proc_edges / proc_vertices if proc_vertices > 0 else 0
    # if avg_degree > 10:       # dense graph — raise beta to control size
    #     beta_weight *= 1.5
    #     logger.info(f"Beta-scaling: avg_degree={avg_degree:.1f} > 10, beta → {beta_weight:.3f}")
    # elif avg_degree < 3:      # sparse graph — lower beta to avoid empty network
    #     beta_weight *= 0.7
    #     logger.info(f"Beta-scaling: avg_degree={avg_degree:.1f} < 3, beta → {beta_weight:.3f}")
    # # To enable: uncomment above, rebuild CarnivalILP with adjusted beta_weight
    # # NOTE: if enabled, must move CarnivalILP construction AFTER this block

    if proc_vertices == 0:
        raise RuntimeError(
            "CORNETO processed_graph is empty after build(). "
            "TF nodes may not match graph nodes. "
            "NEXT STEP: Verify tf_scores keys match gene symbols in network.csv."
        )

    # Step 5: Solve — adapt solver options to processed graph size
    solver_opts = {"threads": 0}  # 0 = auto-detect (use all cores)
    if proc_edges > 40000:
        solver_opts["mip_rel_gap"] = 0.10
        solver_opts["time_limit"] = 600
        logger.info(f"Very large problem ({proc_edges} edges): mip_gap=10%, time_limit=600s, threads=auto")
    elif proc_edges > 15000:
        solver_opts["mip_rel_gap"] = 0.05
        solver_opts["time_limit"] = 600
        logger.info(f"Large problem ({proc_edges} edges): mip_gap=5%, time_limit=600s, threads=auto")
    elif proc_edges > 5000:
        solver_opts["mip_rel_gap"] = 0.05
        solver_opts["time_limit"] = 600
        logger.info(f"Medium problem ({proc_edges} edges): mip_gap=5%, time_limit=600s, threads=auto")
    else:
        solver_opts["time_limit"] = 300
    # < 5K edges: no mip_gap — HIGHS solves to optimality in seconds

    logger.info(f"Solving with {solver}...")
    t_solve = time.time()
    P.solve(verbosity=verbosity, solver=solver, **solver_opts)
    solve_time = time.time() - t_solve
    logger.info(f"Solved in {solve_time:.2f}s")

    # Step 6: Extract results
    result = _extract_results(
        P=P,
        c=c,
        sample_name=sample_name,
        beta_weight=beta_weight,
        solver=solver,
        build_time=build_time,
        solve_time=solve_time,
        total_time=time.time() - t_total,
        n_tfs_input=len(tf_scores),
        n_tfs_used=len(tf_in_graph),
        n_pkn_edges=len(pkn_df),
    )

    logger.info(
        f"CARNIVAL complete: "
        f"{len(result.active_nodes_df)} active nodes, "
        f"{len(result.active_edges_df)} active edges"
    )

    return result


# =============================================================================
# INTERNAL HELPERS - CARNIVAL paper has this in supplementary
# =============================================================================

def _build_graph(pkn_df: pd.DataFrame, deg_gene_set: Optional[set] = None) -> Graph:
    """
    Convert PKN DataFrame to corneto.Graph for Inverse CARNIVAL.

    Expects columns: source | interaction | target
    interaction: +1 (activation) or -1 (inhibition)

    Adds a virtual perturbation node (_PERT) connected to source-only
    nodes (nodes with only outgoing edges) with both +1 and -1 edges.
    This follows the inverse CARNIVAL formulation from Liu et al. (2019):
    _PERT is the single unpenalized input; actual input nodes are penalized
    normally, limiting the number of active inputs in the solution.

    If deg_gene_set is provided, only source-only nodes that are also DEGs
    are connected to _PERT. This reduces the search space by filtering out
    non-specific nodes (e.g. ubiquitin complexes) that are not differentially
    expressed. Falls back to all source-only nodes if the intersection is empty.
    """
    required = {'source', 'interaction', 'target'}
    if not required.issubset(pkn_df.columns):
        raise ValueError(
            f"PKN DataFrame missing required columns. "
            f"Got: {list(pkn_df.columns)}, need: {required}"
        )

    # Identify source-only nodes: nodes with outgoing edges but NO incoming edges
    sources = set(pkn_df['source'].astype(str))
    targets = set(pkn_df['target'].astype(str))
    all_source_only = sources - targets  # nodes that never appear as a target

    # Filter source-only nodes by DEG set if provided
    if deg_gene_set and len(deg_gene_set) > 0:
        filtered = all_source_only & deg_gene_set
        if len(filtered) > 0:
            logger.info(
                f"DEG-filtered inputs: {len(filtered)} of "
                f"{len(all_source_only)} source-only nodes are DEGs"
            )
            source_only = filtered
        else:
            logger.warning(
                f"No source-only nodes in DEG set ({len(all_source_only)} source-only, "
                f"{len(deg_gene_set)} DEGs). Using all source-only nodes."
            )
            source_only = all_source_only
    else:
        source_only = all_source_only

    tuples = [
        (str(r.source), int(r.interaction), str(r.target))
        for _, r in pkn_df.iterrows()
    ]

    # Add virtual _PERT node with +1 and -1 edges to each source-only node
    for node in source_only:
        tuples.append(('_PERT', 1, node))   # activating edge
        tuples.append(('_PERT', -1, node))  # inhibiting edge

    logger.info(
        f"Inverse CARNIVAL: {len(source_only)} source-only input nodes, "
        f"virtual _PERT node added with {len(source_only) * 2} edges"
    )

    return Graph.from_tuples(tuples)


def _build_data(
    pkn_df: pd.DataFrame,
    tf_scores: dict,
    sample_name: str,
) -> cn.Data:
    """
    Build corneto.Data for Inverse CARNIVAL.

    Follows Liu et al. (2019) inverse CARNIVAL formulation:
    - Single virtual _PERT node → role='input', value=0
      (the only unpenalized perturbation source)
    - TF measurement nodes → role='output', value=NES_score
    - Source-only nodes are NOT marked as inputs — they connect to _PERT
      via the graph and are penalized normally by the ILP.
    """
    graph_nodes = set(pkn_df['source'].tolist() + pkn_df['target'].tolist())

    sample_data = {}

    # Single virtual perturbation node as the only input (value=0 = unknown)
    sample_data['_PERT'] = {
        'value': 0,
        'mapping': 'vertex',
        'role': 'input'
    }

    # TF measurements as outputs
    for tf, score in tf_scores.items():
        if tf in graph_nodes:
            sample_data[tf] = {
                'value': float(score),
                'mapping': 'vertex',
                'role': 'output'
            }

    logger.debug(
        f"Data: 1 input (_PERT), "
        f"{sum(1 for v in sample_data.values() if v['role']=='output')} outputs"
    )

    return cn.Data.from_cdict({sample_name: sample_data})


def _extract_results(
    P,
    c: CarnivalILP,
    sample_name: str,
    beta_weight: float,
    solver: str,
    build_time: float,
    solve_time: float,
    total_time: float,
    n_tfs_input: int,
    n_tfs_used: int,
    n_pkn_edges: int,
) -> CARNIVALResult:
    """
    Extract node activities and edge activities from solved CORNETO problem.

    CarnivalILP output:
        P.expr.vertex_values  → node activation states (+1/-1/0)
        P.expr.edge_values    → edge usage (+1/-1/0)
        c.processed_graph.V   → node names (index for vertex_values)
        c.processed_graph.E   → edge tuples (index for edge_values)
    """
    try:
        nodes_df = pd.DataFrame(
            P.expr.vertex_values.value,
            index=c.processed_graph.V,
            columns=['node_activity']
        )
        nodes_df['node_activity'] = nodes_df['node_activity'].astype(int)
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract node activities: {e}. "
            f"Solver may have returned infeasible/unbounded solution. "
            f"NEXT STEP: Try lower beta_weight (current: {beta_weight}) "
            f"or check that TF signs are consistent with PKN topology."
        )

    try:
        edges_df = pd.DataFrame(
            P.expr.edge_values.value,
            index=c.processed_graph.E,
            columns=['edge_activity']
        )
        edges_df['edge_activity'] = edges_df['edge_activity'].astype(int)
    except Exception as e:
        raise RuntimeError(f"Failed to extract edge activities: {e}")

    # Remove virtual _PERT node and its edges from results
    # nodes_df has a regular Index (node name strings)
    nodes_df = nodes_df[~nodes_df.index.astype(str).str.startswith('_PERT')]
    nodes_df.index.name = 'node'

    # edges_df has a MultiIndex of tuples — can't call .astype() on MultiIndex directly.
    # Convert MultiIndex to flat string index, then filter, to avoid pandas MultiIndex errors.
    if isinstance(edges_df.index, pd.MultiIndex):
        edge_strs = edges_df.index.map(lambda t: str(t))
        pert_mask = edge_strs.str.contains('_PERT')
        edges_df = edges_df[~pert_mask]
        # Keep the original MultiIndex (downstream _parse_edge_index handles tuples)
    else:
        edges_df = edges_df[~edges_df.index.astype(str).str.contains('_PERT')]
    edges_df.index.name = 'edge'

    stats = {
        'solver': solver,
        'beta_weight': beta_weight,
        'build_time_seconds': round(build_time, 2),
        'solve_time_seconds': round(solve_time, 2),
        'total_time_seconds': round(total_time, 2),
        'n_tfs_input': n_tfs_input,
        'n_tfs_used': n_tfs_used,
        'n_pkn_edges': n_pkn_edges,
        'n_processed_nodes': len(nodes_df),
        'n_processed_edges': len(edges_df),
        'n_active_nodes': int((nodes_df['node_activity'] != 0).sum()),
        'n_active_edges': int((edges_df['edge_activity'] != 0).sum()),
        'n_activated_nodes': int((nodes_df['node_activity'] == 1).sum()),
        'n_repressed_nodes': int((nodes_df['node_activity'] == -1).sum()),
        'status': 'optimal',
        'sample_name': sample_name,
    }

    return CARNIVALResult(
        nodes_df=nodes_df,
        edges_df=edges_df,
        stats=stats,
        sample_name=sample_name,
    )


# =============================================================================
# QUICK Check :) 
# =============================================================================

def run_diagnostic(
    pkn_path: str,
    measurements_path: str,
    beta: float = 0.1,
    solver: str = DEFAULT_SOLVER,
) -> None:
    """
    Run a quick diagnostic to verify CORNETO is working correctly.

    Usage:
        from core.corneto_runner import run_diagnostic
        run_diagnostic(
            pkn_path='carnival_test/input/network.csv',
            measurements_path='carnival_test/input/measurements.csv',
        )
    """
    print("=" * 60)
    print("CORNETO Runner Diagnostic")
    print("=" * 60)

    pkn_df = pd.read_csv(pkn_path)
    meas_df = pd.read_csv(measurements_path)
    tf_scores = dict(zip(meas_df['TF'], meas_df['NES']))

    print(f"PKN: {len(pkn_df)} edges")
    print(f"TFs: {len(tf_scores)}")

    try:
        result = run_carnival(
            pkn_df=pkn_df,
            tf_scores=tf_scores,
            beta_weight=beta,
            solver=solver,
            verbosity=0,
        )
        print(f"\n CARNIVAL completed successfully")
        print(f"\n{result.summary}")
        print(f"\nFull stats: {result.stats}")

    except Exception as e:
        print(f"\n CARNIVAL failed: {e}")
        raise


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    run_diagnostic(
        pkn_path="carnival_test/input/network.csv",
        measurements_path="carnival_test/input/measurements.csv",
    )
