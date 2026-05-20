"""
Agent 2 — Output Packager
===================================
Converts CARNIVALResult (from corneto_runner.py) → Agent2Result.

Agent 3 expects:
  - carnival_network: nx.DiGraph  (signed directed graph, node/edge attributes)
  - metadata: dict                (solver stats, sample info, warnings)
  - summary: str                  (text description for LLM context window)

This file is the ONLY place that knows about both schemas.
Agent 3 imports Agent2Result and never touches CARNIVALResult directly.

Usage:
    from core.output_packager import package_results, Agent2Result

    carnival_result = run_carnival(...)          # from corneto_runner.py
    agent2_result   = package_results(
        carnival_result=carnival_result,
        agent1_result=agent1_result,             # for context (optional)
        output_dir='carnival_output',            # saves CSVs for audit trail
    )

    # Agent 3 consumes these:
    agent2_result.carnival_network               # nx.DiGraph
    agent2_result.summary                        # str for LLM context
    agent2_result.metadata                       # dict
    agent2_result.nodes_df                       # DataFrame (audit)
    agent2_result.edges_df                       # DataFrame (audit)
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import networkx as nx
import pandas as pd

try:
    from core.corneto_runner import CARNIVALResult
except ModuleNotFoundError:
    from corneto_runner import CARNIVALResult

logger = logging.getLogger(__name__)


# =============================================================================
# OUTPUT SCHEMA: What Agent 3 receives
# =============================================================================

@dataclass
class Agent2Result:
    """
    Complete output from Agent 2, consumed by Agent 3.

    This is the PRIMARY contract between Agent 2 and Agent 3.

    The carnival_network is a signed directed nx.DiGraph where:
      - Node attributes:
          activity:    +1 (activated), -1 (repressed), 0 (inactive)
          node_type:   'TF' | 'measured' | 'intermediate'
          activity_score: float  (same value as activity, for Neo4j compat)
          is_active:   bool
      - Edge attributes:
          sign:        +1 (activation) or -1 (inhibition)
          interaction: 'ACTIVATES' | 'INHIBITS' (string form of sign)
          weight:      1.0 or 0.0 (abs of sign)
          is_active:   bool  (True if CARNIVAL selected this edge)

    Agent 3 uses carnival_network for:
      1. Tool: get_active_regulators()   → nodes where activity != 0
      2. Tool: get_causal_paths()        → nx.shortest_path between nodes
      3. Tool: get_hub_nodes()           → degree centrality
      4. RAG queries: node names as search terms
    """
    carnival_network: nx.DiGraph          # Primary deliverable for Agent 3
    nodes_df: pd.DataFrame                # Raw node activities (audit trail)
    edges_df: pd.DataFrame                # Raw edge activities (audit trail)
    metadata: dict                        # Solver stats + sample info
    summary: str                          # LLM-ready text description
    sample_name: str = "sample"
    warnings: list = field(default_factory=list)
    edge_annotations: dict = field(default_factory=dict)
    network_confidence: dict = field(default_factory=dict)

    @property
    def n_active_nodes(self) -> int:
        return sum(1 for _, d in self.carnival_network.nodes(data=True)
                   if d.get('activity', 0) != 0)

    @property
    def n_active_edges(self) -> int:
        return sum(1 for _, _, d in self.carnival_network.edges(data=True)
                   if d.get('is_active', False))

    @property
    def activated_nodes(self) -> list:
        return [n for n, d in self.carnival_network.nodes(data=True)
                if d.get('activity', 0) == 1]

    @property
    def repressed_nodes(self) -> list:
        return [n for n, d in self.carnival_network.nodes(data=True)
                if d.get('activity', 0) == -1]

    def to_json_metadata(self) -> str:
        """Serialize metadata to JSON string for logging/handoff."""
        return json.dumps(self.metadata, indent=2)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def package_results(
    carnival_result: CARNIVALResult,
    agent1_context: Optional[dict] = None,
    output_dir: Optional[str] = None,
    tf_gene_set: Optional[set] = None,
    deg_gene_set: Optional[set] = None,
) -> Agent2Result:
    """
    Convert CARNIVALResult → Agent2Result for Agent 3 consumption.

    Args:
        carnival_result: Output from corneto_runner.run_carnival()
        agent1_context:  Optional dict with keys from Agent1Result:
                         'comparison', 'n_degs', 'data_type', 'original_query'
                         Used to enrich the summary text.
        output_dir:      If provided, saves nodes.csv, edges.csv, metadata.json
                         for audit trail and debugging.
        tf_gene_set:     Set of known TF gene symbols for node_type annotation.
                         If None, infers from node names only.
        deg_gene_set:    Set of DEG gene symbols for node_type='measured' annotation.
                         If None, DEG nodes are classified as 'intermediate'.

    Returns:
        Agent2Result ready for Agent 3.
    """
    logger.info("Packaging CARNIVAL results for Agent 3...")

    # Step 1: Build NetworkX graph from active CORNETO nodes/edges only
    G = _build_networkx_graph(
        nodes_df=carnival_result.active_nodes_df,
        edges_df=carnival_result.active_edges_df,
        tf_gene_set=tf_gene_set,
        deg_gene_set=deg_gene_set,
    )
    logger.info(
        f"NetworkX graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    # Step 1b: Build edge annotation lookup (for visualization)
    edge_annotations = _build_edge_annotations(G)

    # Step 2: Build metadata dict
    metadata = _build_metadata(
        carnival_result=carnival_result,
        agent1_context=agent1_context or {},
        graph=G,
    )

    # Step 3: Generate LLM-ready summary
    summary = _build_summary(
        carnival_result=carnival_result,
        agent1_context=agent1_context or {},
        graph=G,
    )

    # Step 4: Collect warnings
    warnings = _collect_warnings(carnival_result, G)
    for w in warnings:
        logger.warning(w)

    # Step 5: Optionally save output files for audit trail
    if output_dir:
        # Enrich nodes_df with node_type from the classified NetworkX graph
        nodes_out = carnival_result.nodes_df.copy()
        nodes_out['node_type'] = [
            G.nodes[str(n)].get('node_type', 'intermediate')
            if str(n) in G.nodes else 'intermediate'
            for n in nodes_out.index
        ]
        _save_outputs(
            output_dir=output_dir,
            nodes_df=nodes_out,
            edges_df=carnival_result.edges_df,
            metadata=metadata,
            sample_name=carnival_result.sample_name,
        )

    result = Agent2Result(
        carnival_network=G,
        nodes_df=carnival_result.nodes_df,
        edges_df=carnival_result.edges_df,
        metadata=metadata,
        summary=summary,
        sample_name=carnival_result.sample_name,
        warnings=warnings,
        edge_annotations=edge_annotations,
    )

    logger.info(
        f"Agent2Result ready: "
        f"{result.n_active_nodes} active nodes, "
        f"{result.n_active_edges} active edges"
    )
    return result


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _build_networkx_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    tf_gene_set: Optional[set] = None,
    deg_gene_set: Optional[set] = None,
) -> nx.DiGraph:
    """
    Build nx.DiGraph from CORNETO node/edge DataFrames.

    Node attributes:
        activity        (+1/-1/0)
        activity_score  (float, same value)
        is_active       (bool)
        node_type       ('TF'/'measured'/'intermediate')

    Edge attributes:
        sign        (+1/-1) — from CORNETO edge_activity
        interaction 'ACTIVATES' | 'INHIBITS' (string form)
        weight      1.0 or 0.0
        is_active   (bool)  — True if |edge_activity| > 0

    CORNETO edge index format: ('SOURCE', 'TARGET') tuples
    We extract source/target from the tuple index.
    """
    G = nx.DiGraph()

    # Add nodes
    for node_name, row in nodes_df.iterrows():
        activity = int(row['node_activity'])
        node_type = _classify_node(str(node_name), tf_gene_set, deg_gene_set)
        G.add_node(
            str(node_name),
            activity=activity,
            activity_score=float(activity),
            is_active=(activity != 0),
            node_type=node_type,
        )

    # Add edges — CORNETO stores edge index as (source, target) tuples
    for edge_idx, row in edges_df.iterrows():
        sign = int(row['edge_activity'])
        is_active = (sign != 0)

        # Parse edge index — could be tuple ('SRC', 'TGT') or string '(SRC)(TGT)'
        src, tgt = _parse_edge_index(edge_idx)

        if src and tgt and src in G.nodes and tgt in G.nodes:
            # Recover PKN biological sign from CORNETO's effective signal:
            # edge_activity = PKN_sign × source_activity, so
            # PKN_sign = edge_activity × source_activity (both ±1, so product inverts)
            src_activity = G.nodes[src].get('activity', sign)
            pkn_sign = sign * src_activity if src_activity != 0 else sign
            G.add_edge(
                src, tgt,
                sign=sign,
                interaction='ACTIVATES' if pkn_sign >= 1 else 'INHIBITS' if pkn_sign <= -1 else 'NEUTRAL',
                weight=float(abs(sign)) if sign != 0 else 0.0,
                is_active=is_active,
            )
        else:
            logger.debug(f"Skipping edge {edge_idx}: nodes not in graph")

    # Post-build: flag/fix orphan TFs (CARNIVAL ILP artifact)
    isolated_tfs = []
    for node, data in G.nodes(data=True):
        if data.get('node_type') != 'TF':
            continue
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

        if in_deg == 0 and out_deg == 0:
            # Isolated TF — no edges, ILP artifact → remove from graph
            isolated_tfs.append(node)
            logger.warning(f"Removing isolated TF '{node}': no edges in active network")
        elif in_deg == 0 and out_deg > 0:
            # Source TF — regulating downstream but no upstream signal → flag
            G.nodes[node]['orphan_tf'] = True
            logger.info(f"Orphan TF '{node}': no upstream regulation in active network")
    for node in isolated_tfs:
        G.remove_node(node)

    return G


def _parse_edge_index(edge_idx: Any) -> tuple[str, str]:
    """
    Parse CORNETO edge index to (source, target) string pair.

    CORNETO 1.0.0b7 returns edges indexed as:
        (frozenset({'SOURCE'}), frozenset({'TARGET'}))

    Falls back to plain tuple and string formats for forward compatibility.
    Returns ('', '') if parsing fails.
    """
    if isinstance(edge_idx, tuple) and len(edge_idx) == 2:
        src_raw, tgt_raw = edge_idx[0], edge_idx[1]

        # Primary format: (frozenset({'SRC'}), frozenset({'TGT'}))
        if isinstance(src_raw, frozenset) and isinstance(tgt_raw, frozenset):
            src = next(iter(src_raw), '')
            tgt = next(iter(tgt_raw), '')
            return str(src).strip(), str(tgt).strip()

        # Fallback: plain strings or other iterables
        return str(src_raw).strip(), str(tgt_raw).strip()

    if isinstance(edge_idx, str):
        import re
        parts = re.findall(r'\(([^)]+)\)', edge_idx)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        if '->' in edge_idx:
            parts = edge_idx.split('->')
            return parts[0].strip(), parts[1].strip()

    logger.debug(f"Could not parse edge index: {edge_idx!r} (type={type(edge_idx).__name__})")
    return '', ''


def _classify_node(
    node_name: str,
    tf_gene_set: Optional[set] = None,
    deg_gene_set: Optional[set] = None,
) -> str:
    """
    Classify node type for visualization and Agent 3 tool use.

    Priority:
    1. If in provided tf_gene_set → 'TF'
    2. If name matches known TF prefix patterns → 'TF'
    3. If in deg_gene_set → 'measured'  (DEG targets / observed genes)
    4. If name matches kinase/signaling patterns → 'intermediate'
    5. Default → 'intermediate'  (unclassified signaling nodes)
    """
    if tf_gene_set and node_name in tf_gene_set:
        return 'TF'

    # Heuristic TF patterns (covers CollecTRI TFs)
    tf_patterns = {
        'SMAD', 'STAT', 'NF', 'SP1', 'TP53', 'MYC', 'FOS', 'JUN',
        'GATA', 'SOX', 'PAX', 'HIF', 'NRF', 'YAP', 'TAZ', 'TEAD',
        'RARA', 'RARB', 'RARG', 'ESR', 'AR', 'GLI', 'TCF', 'LEF',
        'EPAS', 'HNF', 'CEBP', 'ETS', 'ELK', 'ELF',
    }
    for pattern in tf_patterns:
        if node_name.startswith(pattern):
            return 'TF'

    if deg_gene_set and node_name in deg_gene_set:
        return 'measured'

    # Kinase / signaling intermediates
    kinase_patterns = {'CDK', 'MAPK', 'AKT', 'PIK3', 'PRKC', 'AURK', 'PLK'}
    for pattern in kinase_patterns:
        if node_name.startswith(pattern):
            return 'intermediate'

    return 'intermediate'


# =============================================================================
# EDGE ANNOTATION HELPERS
# =============================================================================

_DATA_DIR = Path(__file__).parent.parent / "data"
MAX_PATHWAY_SIZE = 200


@functools.lru_cache(maxsize=1)
def _load_edge_annotation_tables() -> tuple:
    """Load and preprocess SIGNOR + pathway annotation CSVs (cached, one-time)."""
    # --- SIGNOR edge-level annotations ---
    signor_lookup = {}
    signor_path = _DATA_DIR / "signor_edge_annotations.csv"
    if signor_path.exists():
        signor_df = pd.read_csv(signor_path)
        for (src, tgt), grp in signor_df.groupby(["source", "target"]):
            signor_lookup[(src, tgt)] = grp.to_dict("records")
        logger.info(f"Loaded SIGNOR annotations: {len(signor_lookup)} edge pairs")
    else:
        logger.warning(f"SIGNOR annotations not found: {signor_path}")

    # --- Pathway membership annotations ---
    gene_pathways = {}  # gene -> set of (database, pathway)
    pathway_sizes = {}  # (database, pathway) -> gene count
    annot_path = _DATA_DIR / "omnipath_pathway_annotations.csv"
    if annot_path.exists():
        annot_df = pd.read_csv(annot_path)
        # Compute pathway sizes and filter by MAX_PATHWAY_SIZE
        size_df = annot_df.groupby(["database", "pathway"])["genesymbol"].nunique()
        for (db, pw), size in size_df.items():
            if size <= MAX_PATHWAY_SIZE:
                pathway_sizes[(db, pw)] = size
        # Build gene -> pathway lookup (only for pathways that passed size filter)
        valid_keys = set(pathway_sizes.keys())
        for _, row in annot_df.iterrows():
            key = (row["database"], row["pathway"])
            if key in valid_keys:
                gene_pathways.setdefault(row["genesymbol"], set()).add(key)
        logger.info(
            f"Loaded pathway annotations: {len(pathway_sizes)} pathways "
            f"(filtered to <={MAX_PATHWAY_SIZE} genes), {len(gene_pathways)} genes"
        )
    else:
        logger.warning(f"Pathway annotations not found: {annot_path}")

    return signor_lookup, pathway_sizes, gene_pathways


def _annotate_edge(
    src: str,
    tgt: str,
    signor_lookup: dict,
    pathway_sizes: dict,
    gene_pathways: dict,
) -> dict:
    """Build annotation dict for a single edge."""
    # SIGNOR mechanism evidence
    signor = signor_lookup.get((src, tgt), [])

    # Shared pathway context (intersection of src and tgt memberships)
    src_pw = gene_pathways.get(src, set())
    tgt_pw = gene_pathways.get(tgt, set())
    shared = src_pw & tgt_pw

    # Sort by specificity (fewest member genes first), cap at 50
    shared_list = sorted(
        [{"database": db, "pathway": pw, "size": pathway_sizes[(db, pw)]}
         for db, pw in shared],
        key=lambda x: x["size"],
    )[:50]

    return {
        "signor": signor,
        "shared_pathways": shared_list,
        "top3_pathways": shared_list[:3],
    }


def _build_edge_annotations(G: nx.DiGraph) -> dict:
    """Build annotation lookup for all edges in the graph."""
    signor_lookup, pathway_sizes, gene_pathways = _load_edge_annotation_tables()

    annotations = {}
    for src, tgt in G.edges():
        annotations[(src, tgt)] = _annotate_edge(
            src, tgt, signor_lookup, pathway_sizes, gene_pathways
        )

    n_signor = sum(1 for a in annotations.values() if a["signor"])
    n_pathway = sum(1 for a in annotations.values() if a["shared_pathways"])
    logger.info(
        f"Edge annotations: {len(annotations)} edges, "
        f"{n_signor} with SIGNOR, {n_pathway} with shared pathways"
    )
    return annotations


def _build_metadata(
    carnival_result: CARNIVALResult,
    agent1_context: dict,
    graph: nx.DiGraph,
) -> dict:
    """Build metadata dict combining CARNIVAL stats + Agent 1 context."""
    active_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_active')]
    active_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('is_active')]

    metadata = {
        # Solver info
        'solver': carnival_result.stats.get('solver'),
        'beta_weight': carnival_result.stats.get('beta_weight'),
        'solve_time_seconds': carnival_result.stats.get('solve_time_seconds'),
        'total_time_seconds': carnival_result.stats.get('total_time_seconds'),
        'status': carnival_result.stats.get('status', 'optimal'),

        # Network stats
        'n_nodes_total': graph.number_of_nodes(),
        'n_edges_total': graph.number_of_edges(),
        'n_active_nodes': len(active_nodes),
        'n_active_edges': len(active_edges),
        'n_activated_nodes': carnival_result.stats.get('n_activated_nodes', 0),
        'n_repressed_nodes': carnival_result.stats.get('n_repressed_nodes', 0),

        # Input info
        'sample_name': carnival_result.sample_name,
        'n_tfs_input': carnival_result.stats.get('n_tfs_input'),
        'n_tfs_used': carnival_result.stats.get('n_tfs_used'),
        'n_pkn_edges': carnival_result.stats.get('n_pkn_edges'),

        # Agent 1 context (if provided)
        'comparison': agent1_context.get('comparison', 'unknown'),
        'data_type': agent1_context.get('data_type', 'unknown'),
        'n_degs': agent1_context.get('n_degs'),
        'original_query': agent1_context.get('original_query'),

        # Parameter selection audit trail
        'parameter_method': agent1_context.get('parameter_method', 'unknown'),
        'parameter_rationale': agent1_context.get('parameter_rationale', ''),

        # Agent 3 compatible aliases (match _generate_network_summary keys)
        'n_nodes': graph.number_of_nodes(),
        'n_edges': graph.number_of_edges(),
        'n_tfs': carnival_result.stats.get('n_tfs_used'),
        'n_measured': sum(
            1 for _, d in graph.nodes(data=True)
            if d.get('node_type') == 'measured'
        ),
    }

    return metadata


def _build_summary(
    carnival_result: CARNIVALResult,
    agent1_context: dict,
    graph: nx.DiGraph,
) -> str:
    """
    Generate LLM-ready text summary for Agent 3 context window.

    This is what Agent 3's ReAct loop sees as the starting context.
    Keep it concise — it goes into the Claude Sonnet system prompt on every iteration.
    """
    stats = carnival_result.stats
    comparison = agent1_context.get('comparison', 'disease vs control')
    original_query = agent1_context.get('original_query', '')
    disease = agent1_context.get('disease') or ''
    data_type = agent1_context.get('data_type', 'unknown')

    activated = [n for n, d in graph.nodes(data=True) if d.get('activity') == 1]
    repressed  = [n for n, d in graph.nodes(data=True) if d.get('activity') == -1]
    active_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if d.get('is_active')]

    # Top regulators by degree centrality in active subgraph
    active_G = graph.subgraph(activated + repressed)
    try:
        centrality = nx.degree_centrality(active_G)
        top_hubs = sorted(centrality, key=centrality.get, reverse=True)[:5]
    except Exception:
        top_hubs = activated[:5]

    # Build comparison string with disease name when available
    comparison_str = comparison
    if disease and disease.lower() not in comparison.lower():
        comparison_str = f"{disease} vs Control ({comparison})"

    summary_lines = [
        f"=== CARNIVAL NETWORK ANALYSIS RESULT ===",
        f"Disease/Condition: {disease if disease else 'Not specified'}",
        f"Comparison: {comparison_str}",
        f"Data type: {data_type}",
        f"Sample: {carnival_result.sample_name}",
        f"",
        f"NETWORK SUMMARY:",
        f"  Total nodes in PKN subnetwork: {graph.number_of_nodes()}",
        f"  Causally active nodes: {len(activated) + len(repressed)} "
        f"({len(activated)} activated, {len(repressed)} repressed)",
        f"  Active causal edges: {len(active_edges)}",
        f"  Solver: {stats.get('solver')} | Beta: {stats.get('beta_weight')} "
        f"| Solve time: {stats.get('solve_time_seconds', 0):.1f}s",
        f"",
        f"TOP ACTIVATED REGULATORS: {', '.join(str(n) for n in activated[:8])}",
        f"TOP REPRESSED REGULATORS: {', '.join(str(n) for n in repressed[:8])}",
        f"",
        f"NETWORK HUBS (by degree centrality): {', '.join(str(n) for n in top_hubs)}",
        f"",
        f"TF COVERAGE: {stats.get('n_tfs_used')}/{stats.get('n_tfs_input')} TFs "
        f"from activity inference found in PKN",
    ]

    if original_query:
        summary_lines.insert(1, f"Original query: {original_query}")

    return "\n".join(summary_lines)


def _collect_warnings(
    carnival_result: CARNIVALResult,
    graph: nx.DiGraph,
) -> list:
    """Collect any warnings to surface to the user."""
    warnings = []
    stats = carnival_result.stats

    n_active = stats.get('n_active_nodes', 0)
    n_total  = stats.get('n_processed_nodes', 1)

    if n_active < 10:
        warnings.append(
            f"Only {n_active} active nodes found. "
            f"Consider lowering beta_weight (current: {stats.get('beta_weight')}) "
            f"to allow a larger network."
        )
    if n_active > 200:
        warnings.append(
            f"Large active network ({n_active} nodes). "
            f"Consider raising beta_weight to focus on high-confidence paths."
        )

    tfs_used  = stats.get('n_tfs_used', 0)
    tfs_input = stats.get('n_tfs_input', 1)
    if tfs_used / tfs_input < 0.7:
        warnings.append(
            f"Only {tfs_used}/{tfs_input} TFs found in PKN. "
            f"Some TF symbols may not match OmniPath gene names."
        )

    return warnings


def _save_outputs(
    output_dir: str,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    metadata: dict,
    sample_name: str,
) -> None:
    """Save output files for audit trail."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    nodes_path = out / f"{sample_name}_nodes.csv"
    edges_path = out / f"{sample_name}_edges.csv"
    meta_path  = out / f"{sample_name}_metadata.json"

    nodes_df.to_csv(nodes_path)

    # Convert frozenset MultiIndex to readable source/target columns
    if isinstance(edges_df.index, pd.MultiIndex):
        parsed = [_parse_edge_index(idx) for idx in edges_df.index]
        edges_flat = edges_df.copy()
        edges_flat.index = pd.RangeIndex(len(edges_flat))
        edges_flat.insert(0, 'source', [s for s, _ in parsed])
        edges_flat.insert(1, 'target', [t for _, t in parsed])
        edges_flat.to_csv(edges_path, index=False)
    else:
        edges_df.to_csv(edges_path)

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Output files saved to {out}/")
    logger.info(f"  {nodes_path.name}")
    logger.info(f"  {edges_path.name}")
    logger.info(f"  {meta_path.name}")


# =============================================================================
# QUICK DIAGNOSTIC
# =============================================================================

if __name__ == "__main__":
    import logging
    from pathlib import Path
    try:
        from core.corneto_runner import run_carnival
    except ModuleNotFoundError:
        from corneto_runner import run_carnival

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    _root = Path(__file__).parent.parent
    pkn_df  = pd.read_csv(_root / "carnival_test" / "input" / "network.csv")
    meas_df = pd.read_csv(_root / "carnival_test" / "input" / "measurements.csv")
    tf_scores = dict(zip(meas_df['TF'], meas_df['NES']))

    print("Running CARNIVAL...")
    carnival_result = run_carnival(
        pkn_df=pkn_df,
        tf_scores=tf_scores,
        beta_weight=0.1,
        solver='HIGHS',
        sample_name='ipf_at2',
        verbosity=0,
    )

    print("Packaging for Agent 3...")
    agent2_result = package_results(
        carnival_result=carnival_result,
        agent1_context={
            'comparison': 'IPF vs Control',
            'data_type': 'single_cell',
            'original_query': 'What TFs drive fibrosis in AT2 cells?',
        },
        output_dir=str(_root / "carnival_test" / "output"),
    )

    print("\n" + "="*60)
    print("AGENT 2 RESULT READY")
    print("="*60)
    print(f"carnival_network: {type(agent2_result.carnival_network).__name__} "
          f"({agent2_result.carnival_network.number_of_nodes()} nodes, "
          f"{agent2_result.carnival_network.number_of_edges()} edges)")
    print(f"Active nodes: {agent2_result.n_active_nodes}")
    print(f"Active edges: {agent2_result.n_active_edges}")
    print(f"Activated: {agent2_result.activated_nodes[:5]}")
    print(f"Repressed: {agent2_result.repressed_nodes[:5]}")
    print(f"\nWarnings: {agent2_result.warnings or 'None'}")
    print(f"\n--- SUMMARY FOR AGENT 3 ---")
    print(agent2_result.summary)
