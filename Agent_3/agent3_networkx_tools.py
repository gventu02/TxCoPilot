"""
Agent 3 NetworkX Analysis Tools
================================

Purpose: Provide structured tools for Agent 3 to interrogate CARNIVAL networks.

Design Principles (MCP-compliant):
1. Tool descriptions use markdown with explicit "Use Cases" and "Instructions"
2. Return values include "next steps" to guide the agent
3. Errors provide actionable suggestions
4. Tools are grouped by query type for dynamic loading

Author: Thesis Development
Date: November 2025
"""

import logging
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL REGISTRY (for dynamic loading)
# =============================================================================

# Keys match Agent 1's query_type values (entity_extraction.py → ExtractedEntities.query_type).
# Agent 3 calls get_tools_for_query_type(query_type) to load only the relevant subset.
TOOL_GROUPS = {
    'tf_activity': ['list_transcription_factors', 'calculate_node_centrality'],
    'network_query': ['find_shortest_path', 'get_node_neighbors'],
    'hub_identification': ['calculate_node_centrality', 'identify_network_modules'],
    'intervention_targets': ['calculate_node_centrality', 'get_edge_interactions'],
    'all': ['list_transcription_factors', 'calculate_node_centrality',
            'find_shortest_path', 'get_node_neighbors',
            'identify_network_modules', 'get_edge_interactions']
}


# =============================================================================
# TOOL 1: List Transcription Factors
# =============================================================================

def list_transcription_factors(
    G: nx.DiGraph,
    filter_by_activity: Optional[str] = None,
    top_n: Optional[int] = None
) -> Dict[str, Any]:
    """
    List transcription factors in the CARNIVAL network.
    
    ## Use Cases
    - "What TFs are in my network?"
    - "Which TFs are most activated/inhibited?"
    - "Show me top 10 TFs by activity"
    
    ## Instructions
    1. Identifies nodes with node_type='TF'
    2. Optionally filters by activity (positive/negative)
    3. Returns sorted list with activity scores
    
    ## Parameters
    - G: CARNIVAL NetworkX graph
    - filter_by_activity: 'positive', 'negative', or None (all)
    - top_n: Return only top N TFs by absolute activity
    
    ## Returns
    - tfs: List of TF names
    - activity_scores: Dict {gene: activity_score}
    - summary: Human-readable summary
    - next_steps: Suggested follow-up analyses
    
    ## Error Handling
    - If no TFs found: Returns empty list + suggestion to check node types
    - If activity attribute missing: Returns TFs without scores + warning
    """
    
    # Get all TF nodes
    node_types = nx.get_node_attributes(G, 'node_type')
    tfs = [node for node, ntype in node_types.items() if ntype == 'TF']
    
    if not tfs:
        return {
            'tfs': [],
            'activity_scores': {},
            'summary': "WARNING:No transcription factors found in network.",
            'next_steps': [
                "Check if node_type attribute is set correctly",
                "Verify CARNIVAL ran with DoRothEA TF priors"
            ],
            'error': 'NO_TFS_FOUND'
        }
    
    # Get activity scores
    activities = nx.get_node_attributes(G, 'activity')
    
    if not activities:
        return {
            'tfs': tfs,
            'activity_scores': {},
            'summary': f"OK:Found {len(tfs)} TFs (no activity scores available)",
            'next_steps': [
                "Activity scores may not be computed yet",
                "Run CARNIVAL with activity inference enabled"
            ],
            'warning': 'NO_ACTIVITY_SCORES'
        }
    
    # Filter TFs with activities
    tf_activities = {tf: activities.get(tf, 0.0) for tf in tfs}
    
    # Apply activity filter
    if filter_by_activity == 'positive':
        tf_activities = {tf: act for tf, act in tf_activities.items() if act > 0}
    elif filter_by_activity == 'negative':
        tf_activities = {tf: act for tf, act in tf_activities.items() if act < 0}
    
    # Sort by absolute activity
    sorted_tfs = sorted(tf_activities.items(), 
                       key=lambda x: abs(x[1]), 
                       reverse=True)
    
    # Apply top_n filter
    if top_n:
        sorted_tfs = sorted_tfs[:top_n]
    
    tfs_list = [tf for tf, _ in sorted_tfs]
    
    # Generate summary
    n_positive = sum(1 for _, act in sorted_tfs if act > 0)
    n_negative = sum(1 for _, act in sorted_tfs if act < 0)
    
    summary = f"""
 Found {len(tfs_list)} transcription factors
   • {n_positive} with positive activity (activated)
   • {n_negative} with negative activity (inhibited)
   • Top 3: {', '.join(tfs_list[:3])}
"""
    
    next_steps = [
        f"Investigate downstream targets of {tfs_list[0]} (top TF)",
        "Calculate centrality to identify hub TFs",
        "Find paths between top activated and inhibited TFs"
    ]
    
    return {
        'tfs': tfs_list,
        'activity_scores': dict(sorted_tfs),
        'summary': summary.strip(),
        'next_steps': next_steps,
        'metadata': {
            'n_total_tfs': len(tfs_list),
            'n_positive': n_positive,
            'n_negative': n_negative,
            'filter_applied': filter_by_activity,
            'top_n': top_n
        }
    }


# =============================================================================
# TOOL 2: Calculate Node Centrality
# =============================================================================

def calculate_node_centrality(
    G: nx.DiGraph,
    genes: Optional[List[str]] = None,
    metrics: List[str] = ['betweenness', 'degree'],
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Calculate centrality measures for nodes in network.
    
    ## Use Cases
    - "Which genes are most central in the network?"
    - "What's the betweenness centrality of TP53?"
    - "Find hub genes with high degree centrality"
    
    ## Instructions
    1. Calculates specified centrality metrics
    2. If genes specified, returns scores for those genes only
    3. Otherwise returns top N genes by each metric
    
    ## Parameters
    - G: CARNIVAL NetworkX graph
    - genes: Specific genes to analyze (optional)
    - metrics: List of centrality types (betweenness, degree, closeness, pagerank, eigenvector)
    - top_n: Number of top genes to return per metric
    
    ## Returns
    - centrality_scores: Dict {metric: {gene: score}}
    - top_genes: Dict {metric: [top_gene_list]}
    - summary: Interpretation of results
    - next_steps: Suggested analyses
    
    ## Error Handling
    - If metric not supported: Returns available metrics
    - If graph too small: Returns warning + partial results
    - If genes not in graph: Returns which genes are missing
    """
    
    SUPPORTED_METRICS = {
        'betweenness': nx.betweenness_centrality,
        'degree': nx.degree_centrality,
        'closeness': nx.closeness_centrality,
        'pagerank': nx.pagerank,
        'eigenvector': nx.eigenvector_centrality
    }
    
    # Validate metrics
    invalid_metrics = [m for m in metrics if m not in SUPPORTED_METRICS]
    if invalid_metrics:
        return {
            'error': 'INVALID_METRICS',
            'invalid_metrics': invalid_metrics,
            'supported_metrics': list(SUPPORTED_METRICS.keys()),
            'summary': f" Invalid metrics: {', '.join(invalid_metrics)}",
            'next_steps': [
                f"Use supported metrics: {', '.join(SUPPORTED_METRICS.keys())}"
            ]
        }
    
    # Validate genes if specified — calculate for available, warn on missing
    missing_genes = []
    if genes:
        missing_genes = [g for g in genes if g not in G.nodes()]
        genes = [g for g in genes if g in G.nodes()]
        if missing_genes:
            logger.warning(f"Centrality: {len(missing_genes)} genes not in network: {missing_genes}")
        if not genes:
            return {
                'error': 'ALL_GENES_NOT_FOUND',
                'missing_genes': missing_genes,
                'summary': f"None of the requested genes are in the network: {', '.join(missing_genes)}",
                'next_steps': [
                    "Check gene name spelling or aliases",
                    "Use calculate_node_centrality without specifying genes to see top hubs"
                ]
            }
    
    # Calculate centralities
    centrality_scores = {}
    errors = {}
    
    for metric in metrics:
        try:
            if metric == 'eigenvector':
                # Eigenvector can fail on directed graphs; use undirected with
                # higher max_iter to help convergence on dense CARNIVAL graphs
                try:
                    scores = SUPPORTED_METRICS[metric](G.to_undirected(), max_iter=1000)
                except nx.PowerIterationFailedConvergence:
                    logger.warning("Eigenvector centrality did not converge — falling back to PageRank")
                    scores = nx.pagerank(G.to_undirected())
                    errors['eigenvector'] = 'Did not converge, used PageRank as proxy'
            else:
                scores = SUPPORTED_METRICS[metric](G)
            centrality_scores[metric] = scores
        except Exception as e:
            errors[metric] = str(e)
            centrality_scores[metric] = {}
    
    if not centrality_scores or all(not v for v in centrality_scores.values()):
        return {
            'error': 'CALCULATION_FAILED',
            'errors': errors,
            'summary': " Failed to calculate any centrality metrics",
            'next_steps': [
                "Check if graph is connected",
                "Try different centrality metrics",
                "Verify graph has enough nodes (>5)"
            ]
        }
    
    # Filter to specific genes or get top N
    if genes:
        filtered_scores = {}
        for metric, scores in centrality_scores.items():
            filtered_scores[metric] = {g: scores.get(g, 0.0) for g in genes if g in G.nodes()}
        top_genes = {metric: list(scores.keys()) for metric, scores in filtered_scores.items()}
        centrality_scores = filtered_scores
    else:
        top_genes = {}
        for metric, scores in centrality_scores.items():
            sorted_genes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_genes[metric] = [g for g, _ in sorted_genes[:top_n]]
            # Keep only top N in centrality_scores for brevity
            centrality_scores[metric] = dict(sorted_genes[:top_n])
    
    # Generate summary
    summary_lines = [" Centrality analysis complete\n"]
    for metric in metrics:
        if metric in top_genes:
            top_3 = top_genes[metric][:3]
            summary_lines.append(f"   • {metric.capitalize()}: {', '.join(top_3)}")
    
    summary = "\n".join(summary_lines)
    
    # Suggest next steps
    top_hub = top_genes[metrics[0]][0] if top_genes else None
    next_steps = []
    if top_hub:
        next_steps = [
            f"Explore {top_hub} (top hub) - get neighbors and interactions",
            "Find paths involving these hub genes",
            "Check if hub genes are TFs or targets"
        ]
    else:
        next_steps = ["Specify genes of interest for targeted analysis"]
    
    result = {
        'centrality_scores': centrality_scores,
        'top_genes': top_genes,
        'summary': summary,
        'next_steps': next_steps,
        'metadata': {
            'metrics_calculated': list(centrality_scores.keys()),
            'metrics_failed': list(errors.keys()),
            'n_genes_analyzed': len(genes) if genes else G.number_of_nodes(),
            'top_n': top_n
        }
    }
    if missing_genes:
        result['warning'] = f"Genes not in network (skipped): {', '.join(missing_genes)}"
    return result


# =============================================================================
# TOOL 3: Find Shortest Path
# =============================================================================

def find_shortest_path(
    G: nx.DiGraph,
    source: str,
    target: str,
    max_length: Optional[int] = None,
    respect_edge_signs: bool = True
) -> Dict[str, Any]:
    """
    Find shortest path between two genes in network.
    
    ## Use Cases
    - "How does TP53 regulate COL1A1?"
    - "Find path from TGFB1 to ACTA2"
    - "What connects source X to target Y?"
    
    ## Instructions
    1. Uses Dijkstra's algorithm to find shortest path
    2. Optionally considers edge weights/signs
    3. Returns genes in path and interaction details
    
    ## Parameters
    - G: CARNIVAL NetworkX graph
    - source: Starting gene
    - target: Target gene
    - max_length: Maximum path length (None = no limit)
    - respect_edge_signs: Consider activating/inhibiting interactions
    
    ## Returns
    - path: List of genes from source to target
    - path_length: Number of edges
    - interactions: Details of each edge in path
    - summary: Biological interpretation
    - next_steps: Further analyses
    
    ## Error Handling
    - If nodes not in graph: Returns which gene is missing
    - If no path exists: Suggests checking connectivity or relaxing constraints
    - If path too long: Suggests max_length parameter
    """
    
    # Check if genes exist
    if source not in G.nodes():
        return {
            'error': 'SOURCE_NOT_FOUND',
            'source': source,
            'available_similar': _find_similar_genes(G, source),
            'summary': f"ERROR:Source gene '{source}' not in network",
            'next_steps': [
                "Check gene name spelling",
                "Try alternative gene symbols (e.g., TP53 vs TRP53)",
                f"Similar genes: {', '.join(_find_similar_genes(G, source)[:5])}"
            ]
        }
    
    if target not in G.nodes():
        return {
            'error': 'TARGET_NOT_FOUND',
            'target': target,
            'available_similar': _find_similar_genes(G, target),
            'summary': f"ERROR:Target gene '{target}' not in network",
            'next_steps': [
                "Check gene name spelling",
                f"Similar genes: {', '.join(_find_similar_genes(G, target)[:5])}"
            ]
        }
    
    # Find shortest path
    try:
        if respect_edge_signs:
            # Use edge weights (inversely)
            weights = nx.get_edge_attributes(G, 'weight')
            if weights:
                path = nx.shortest_path(G, source, target, weight='weight')
            else:
                path = nx.shortest_path(G, source, target)
        else:
            path = nx.shortest_path(G, source, target)
    except nx.NetworkXNoPath:
        return {
            'error': 'NO_PATH_FOUND',
            'source': source,
            'target': target,
            'summary': f"ERROR:No path exists from {source} to {target}",
            'next_steps': [
                "These genes may be in disconnected components",
                "Try reversing source/target",
                "Check if genes are connected indirectly through other nodes",
                f"Try: get_node_neighbors('{source}') to explore local connections"
            ]
        }
    except Exception as e:
        return {
            'error': 'CALCULATION_FAILED',
            'exception': str(e),
            'summary': f"ERROR:Error finding path: {str(e)}",
            'next_steps': ["Check graph structure", "Report this error if persistent"]
        }
    
    # Check max length
    path_length = len(path) - 1
    if max_length and path_length > max_length:
        return {
            'warning': 'PATH_TOO_LONG',
            'path': path,
            'path_length': path_length,
            'max_length': max_length,
            'summary': f"WARNING:Path found ({path_length} steps) exceeds max_length ({max_length})",
            'next_steps': [
                "Increase max_length parameter",
                "Look for alternative shorter paths",
                "Focus on intermediate hubs in this path"
            ]
        }
    
    # Extract interaction details
    interactions = []
    edge_signs = nx.get_edge_attributes(G, 'sign')
    edge_weights = nx.get_edge_attributes(G, 'weight')
    
    for i in range(len(path) - 1):
        src, tgt = path[i], path[i+1]
        edge_key = (src, tgt)
        sign = edge_signs.get(edge_key, 0)
        weight = edge_weights.get(edge_key, 1.0)
        
        interaction_type = "activates" if sign == 1 else "inhibits" if sign == -1 else "regulates"
        
        interactions.append({
            'source': src,
            'target': tgt,
            'sign': sign,
            'weight': weight,
            'interaction_type': interaction_type,
            'description': f"{src} {interaction_type} {tgt}"
        })
    
    # Generate biological summary
    activation_count = sum(1 for i in interactions if i['sign'] == 1)
    inhibition_count = sum(1 for i in interactions if i['sign'] == -1)
    
    summary = f"""
OK:Found path: {source} → {target}
   • Length: {path_length} steps
   • Nodes: {' → '.join(path)}
   • {activation_count} activations, {inhibition_count} inhibitions
"""
    
    # Determine overall effect
    if inhibition_count % 2 == 0:
        overall_effect = "activating"
    else:
        overall_effect = "inhibiting"
    
    summary += f"   • Overall effect: {overall_effect}"
    
    next_steps = [
        f"Examine intermediate nodes: {', '.join(path[1:-1])}",
        "Check if intermediate nodes are known regulators",
        "Look for alternative paths between same source/target"
    ]
    
    return {
        'path': path,
        'path_length': path_length,
        'interactions': interactions,
        'summary': summary.strip(),
        'next_steps': next_steps,
        'metadata': {
            'n_activations': activation_count,
            'n_inhibitions': inhibition_count,
            'overall_effect': overall_effect
        }
    }


# =============================================================================
# TOOL 4: Get Node Neighbors
# =============================================================================

def get_node_neighbors(
    G: nx.DiGraph,
    gene: str,
    direction: str = 'both',
    max_neighbors: int = 20
) -> Dict[str, Any]:
    """
    Get upstream and/or downstream neighbors of a gene.
    
    ## Use Cases
    - "What does TP53 regulate?"
    - "What regulates COL1A1?"
    - "Show me neighbors of TGFB1"
    
    ## Instructions
    1. Returns predecessors (upstream), successors (downstream), or both
    2. Includes interaction types (activation/inhibition)
    3. Sorted by edge weight/importance
    
    ## Parameters
    - G: CARNIVAL NetworkX graph
    - gene: Gene of interest
    - direction: 'upstream', 'downstream', or 'both'
    - max_neighbors: Maximum neighbors to return
    
    ## Returns
    - upstream: List of genes regulating target gene
    - downstream: List of genes regulated by target gene
    - interactions: Detailed interaction info
    - summary: Overview of regulatory relationships
    - next_steps: Suggested analyses
    
    ## Error Handling
    - If gene not in graph: Returns similar gene names
    - If no neighbors: Returns isolated node warning
    - If direction invalid: Returns valid options
    """
    
    if gene not in G.nodes():
        return {
            'error': 'GENE_NOT_FOUND',
            'gene': gene,
            'available_similar': _find_similar_genes(G, gene),
            'summary': f"ERROR:Gene '{gene}' not in network",
            'next_steps': [
                "Check gene name spelling",
                f"Similar genes: {', '.join(_find_similar_genes(G, gene)[:5])}"
            ]
        }
    
    if direction not in ['upstream', 'downstream', 'both']:
        return {
            'error': 'INVALID_DIRECTION',
            'direction': direction,
            'valid_options': ['upstream', 'downstream', 'both'],
            'summary': f"ERROR:Invalid direction: {direction}",
            'next_steps': ["Use 'upstream', 'downstream', or 'both'"]
        }
    
    # Get neighbors
    upstream = list(G.predecessors(gene)) if direction in ['upstream', 'both'] else []
    downstream = list(G.successors(gene)) if direction in ['downstream', 'both'] else []
    
    if not upstream and not downstream:
        return {
            'warning': 'NO_NEIGHBORS',
            'gene': gene,
            'upstream': [],
            'downstream': [],
            'summary': f"WARNING:{gene} has no neighbors in the network (isolated node)",
            'next_steps': [
                "This gene may not be connected in CARNIVAL output",
                "Check if gene is in DEG list or prior knowledge network",
                "Try expanding the network or relaxing CARNIVAL parameters"
            ]
        }
    
    # Get interaction details
    edge_signs = nx.get_edge_attributes(G, 'sign')
    edge_weights = nx.get_edge_attributes(G, 'weight')
    
    upstream_interactions = []
    for neighbor in upstream[:max_neighbors]:
        edge_key = (neighbor, gene)
        sign = edge_signs.get(edge_key, 0)
        weight = edge_weights.get(edge_key, 1.0)
        interaction = "activates" if sign == 1 else "inhibits" if sign == -1 else "regulates"
        
        upstream_interactions.append({
            'gene': neighbor,
            'sign': sign,
            'weight': weight,
            'interaction': f"{neighbor} {interaction} {gene}"
        })
    
    downstream_interactions = []
    for neighbor in downstream[:max_neighbors]:
        edge_key = (gene, neighbor)
        sign = edge_signs.get(edge_key, 0)
        weight = edge_weights.get(edge_key, 1.0)
        interaction = "activates" if sign == 1 else "inhibits" if sign == -1 else "regulates"
        
        downstream_interactions.append({
            'gene': neighbor,
            'sign': sign,
            'weight': weight,
            'interaction': f"{gene} {interaction} {neighbor}"
        })
    
    # Generate summary
    n_upstream_activations = sum(1 for i in upstream_interactions if i['sign'] == 1)
    n_upstream_inhibitions = sum(1 for i in upstream_interactions if i['sign'] == -1)
    n_downstream_activations = sum(1 for i in downstream_interactions if i['sign'] == 1)
    n_downstream_inhibitions = sum(1 for i in downstream_interactions if i['sign'] == -1)
    
    summary = f"OK:{gene} network neighborhood\n"
    if upstream:
        summary += f"   • Upstream: {len(upstream)} regulators ({n_upstream_activations} activate, {n_upstream_inhibitions} inhibit)\n"
    if downstream:
        summary += f"   • Downstream: {len(downstream)} targets ({n_downstream_activations} activated, {n_downstream_inhibitions} inhibited)"
    
    next_steps = []
    if upstream:
        top_regulator = upstream_interactions[0]['gene']
        next_steps.append(f"Investigate {top_regulator} (top upstream regulator)")
    if downstream:
        top_target = downstream_interactions[0]['gene']
        next_steps.append(f"Investigate {top_target} (top downstream target)")
    next_steps.append(f"Calculate centrality of {gene} to determine hub status")
    
    return {
        'gene': gene,
        'upstream': [i['gene'] for i in upstream_interactions],
        'downstream': [i['gene'] for i in downstream_interactions],
        'upstream_interactions': upstream_interactions,
        'downstream_interactions': downstream_interactions,
        'summary': summary.strip(),
        'next_steps': next_steps,
        'metadata': {
            'n_upstream': len(upstream),
            'n_downstream': len(downstream),
            'n_upstream_activations': n_upstream_activations,
            'n_upstream_inhibitions': n_upstream_inhibitions,
            'n_downstream_activations': n_downstream_activations,
            'n_downstream_inhibitions': n_downstream_inhibitions
        }
    }


# =============================================================================
# TOOL 5: Identify Network Modules
# =============================================================================

def identify_network_modules(
    G: nx.DiGraph,
    method: str = 'louvain',
    min_module_size: int = 3
) -> Dict[str, Any]:
    """
    Identify functional modules (communities) in network.
    
    ## Use Cases
    - "What are the main functional modules?"
    - "Group genes into pathway modules"
    - "Find gene clusters in the network"
    
    ## Instructions
    1. Uses community detection algorithms
    2. Returns genes grouped by module
    3. Provides module characterization (TFs, targets, etc.)
    
    ## Parameters
    - G: CARNIVAL NetworkX graph
    - method: 'louvain' or 'greedy' (modularity-based)
    - min_module_size: Minimum genes per module
    
    ## Returns
    - modules: Dict {module_id: [genes]}
    - module_stats: Size, connectivity, TF enrichment per module
    - summary: Overview of modular structure
    - next_steps: Deeper module analyses
    
    ## Error Handling
    - If too few nodes: Warning about module reliability
    - If graph disconnected: Analyzes largest component
    - If method fails: Falls back to alternative method
    """
    
    if G.number_of_nodes() < 10:
        return {
            'warning': 'TOO_FEW_NODES',
            'n_nodes': G.number_of_nodes(),
            'summary': "WARNING:Network too small for reliable module detection (need ≥10 nodes)",
            'next_steps': [
                "Expand network by relaxing CARNIVAL parameters",
                "Include more genes in analysis",
                "Module detection requires larger networks"
            ]
        }
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Check connectivity
    if not nx.is_connected(G_undirected):
        components = list(nx.connected_components(G_undirected))
        largest_component = max(components, key=len)
        G_undirected = G_undirected.subgraph(largest_component).copy()
        
        warning_msg = f"WARNING:Graph has {len(components)} disconnected components. Analyzing largest ({len(largest_component)} nodes)."
    else:
        warning_msg = None
    
    # Detect communities
    try:
        if method == 'louvain':
            import community as community_louvain  # python-louvain package
            partition = community_louvain.best_partition(G_undirected)
        elif method == 'greedy':
            communities = nx.community.greedy_modularity_communities(G_undirected)
            partition = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    partition[node] = idx
        else:
            return {
                'error': 'INVALID_METHOD',
                'method': method,
                'valid_methods': ['louvain', 'greedy'],
                'summary': f"ERROR:Invalid method: {method}",
                'next_steps': ["Use 'louvain' or 'greedy'"]
            }
    except ImportError:
        # Fallback if python-louvain not installed
        communities = nx.community.greedy_modularity_communities(G_undirected)
        partition = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                partition[node] = idx
    except Exception as e:
        return {
            'error': 'CALCULATION_FAILED',
            'exception': str(e),
            'summary': f"ERROR:Module detection failed: {str(e)}",
            'next_steps': [
                "Try alternative method",
                "Check graph structure",
                "Install python-louvain: pip install python-louvain"
            ]
        }
    
    # Organize modules
    modules = {}
    for node, module_id in partition.items():
        if module_id not in modules:
            modules[module_id] = []
        modules[module_id].append(node)
    
    # Filter by minimum size
    modules = {mid: genes for mid, genes in modules.items() 
              if len(genes) >= min_module_size}
    
    if not modules:
        return {
            'warning': 'NO_MODULES_FOUND',
            'summary': f"WARNING:No modules with ≥{min_module_size} genes found",
            'next_steps': [
                f"Reduce min_module_size (currently {min_module_size})",
                "Network may be too small or highly connected"
            ]
        }
    
    # Calculate module statistics
    node_types = nx.get_node_attributes(G, 'node_type')
    
    module_stats = {}
    for module_id, genes in modules.items():
        n_tfs = sum(1 for g in genes if node_types.get(g) == 'TF')
        n_measured = sum(1 for g in genes if node_types.get(g) == 'measured')
        n_signaling = sum(1 for g in genes if node_types.get(g) == 'intermediate')
        
        # Calculate internal connectivity
        subgraph = G.subgraph(genes)
        internal_edges = subgraph.number_of_edges()
        
        module_stats[module_id] = {
            'size': len(genes),
            'n_tfs': n_tfs,
            'n_measured': n_measured,
            'n_signaling': n_signaling,
            'internal_edges': internal_edges,
            'density': nx.density(subgraph),
            'top_genes': genes[:5]  # First 5 genes
        }
    
    # Generate summary
    summary = f"OK:Identified {len(modules)} functional modules\n"
    for mid in sorted(modules.keys(), key=lambda x: len(modules[x]), reverse=True)[:3]:
        stats = module_stats[mid]
        summary += f"   • Module {mid}: {stats['size']} genes ({stats['n_tfs']} TFs, {stats['internal_edges']} edges)\n"
    
    summary += f"\n   Method: {method}, Min size: {min_module_size}"
    
    if warning_msg:
        summary = warning_msg + "\n\n" + summary
    
    next_steps = [
        "Characterize each module with pathway enrichment",
        "Identify inter-module connections (bridges)",
        "Focus on TF-rich modules for regulatory analysis"
    ]
    
    return {
        'modules': modules,
        'module_stats': module_stats,
        'n_modules': len(modules),
        'summary': summary.strip(),
        'next_steps': next_steps,
        'metadata': {
            'method': method,
            'min_module_size': min_module_size,
            'largest_module': max(len(g) for g in modules.values()),
            'smallest_module': min(len(g) for g in modules.values())
        }
    }


# =============================================================================
# TOOL 6: Get Edge Interactions
# =============================================================================

def get_edge_interactions(
    G: nx.DiGraph,
    interaction_type: Optional[str] = None,
    genes: Optional[List[str]] = None,
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Get interaction details for edges in network.
    
    ## Use Cases
    - "Show me all inhibitory interactions"
    - "What interactions involve TP53?"
    - "List strongest regulatory connections"
    
    ## Instructions
    1. Filters edges by type (activation/inhibition)
    2. Optionally filters by genes involved
    3. Returns sorted by interaction strength
    
    ## Parameters
    - G: CARNIVAL NetworkX graph
    - interaction_type: 'activation', 'inhibition', or None (all)
    - genes: Filter to interactions involving these genes
    - top_n: Maximum interactions to return
    
    ## Returns
    - interactions: List of edge details
    - summary: Statistics about interaction types
    - next_steps: Further analyses
    
    ## Error Handling
    - If no interactions match: Returns suggestion to relax filters
    - If genes not found: Returns which genes are missing
    """
    
    edge_signs = nx.get_edge_attributes(G, 'sign')
    edge_weights = nx.get_edge_attributes(G, 'weight')
    
    # Get all edges
    interactions = []
    for source, target in G.edges():
        edge_key = (source, target)
        sign = edge_signs.get(edge_key, 0)
        weight = edge_weights.get(edge_key, 1.0)
        
        # Determine interaction type
        if sign == 1:
            itype = 'activation'
        elif sign == -1:
            itype = 'inhibition'
        else:
            itype = 'unknown'
        
        # Filter by interaction type
        if interaction_type and itype != interaction_type:
            continue
        
        # Filter by genes
        if genes and source not in genes and target not in genes:
            continue
        
        interactions.append({
            'source': source,
            'target': target,
            'sign': sign,
            'weight': weight,
            'interaction_type': itype,
            'description': f"{source} {itype}s {target}"
        })
    
    if not interactions:
        filter_desc = []
        if interaction_type:
            filter_desc.append(f"type={interaction_type}")
        if genes:
            filter_desc.append(f"genes={','.join(genes[:3])}")
        
        return {
            'warning': 'NO_INTERACTIONS_FOUND',
            'filters': filter_desc,
            'interactions': [],
            'summary': f"WARNING:No interactions found with filters: {', '.join(filter_desc)}",
            'next_steps': [
                "Remove filters to see all interactions",
                "Check gene names are correct",
                "Verify interaction_type is spelled correctly"
            ]
        }
    
    # Sort by weight
    interactions = sorted(interactions, key=lambda x: abs(x['weight']), reverse=True)
    
    # Limit to top_n
    interactions = interactions[:top_n]
    
    # Generate summary
    n_activations = sum(1 for i in interactions if i['sign'] == 1)
    n_inhibitions = sum(1 for i in interactions if i['sign'] == -1)
    n_unknown = sum(1 for i in interactions if i['sign'] == 0)
    
    summary = f"""
OK:Found {len(interactions)} interactions
   • {n_activations} activations
   • {n_inhibitions} inhibitions
   • {n_unknown} unknown
   • Top interaction: {interactions[0]['description']}
"""
    
    next_steps = [
        "Focus on high-weight interactions (strongest evidence)",
        "Trace paths through these interactions",
        "Validate key interactions with literature"
    ]
    
    return {
        'interactions': interactions,
        'n_interactions': len(interactions),
        'summary': summary.strip(),
        'next_steps': next_steps,
        'metadata': {
            'n_activations': n_activations,
            'n_inhibitions': n_inhibitions,
            'n_unknown': n_unknown,
            'filter_type': interaction_type,
            'filter_genes': genes,
            'top_n': top_n
        }
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _find_similar_genes(G: nx.DiGraph, query: str, n: int = 5) -> List[str]:
    """
    Find genes in graph with similar names (prefix + substring matching).
    """
    query_lower = query.lower()
    all_genes = list(G.nodes())

    # Full prefix match first (e.g., "TP53" matches "TP53")
    similar = [g for g in all_genes if g.lower().startswith(query_lower)]

    if len(similar) < n:
        # Try shorter prefix (min 2 chars) for partial matches
        prefix = query_lower[:max(2, len(query_lower))]
        prefix_matches = [g for g in all_genes
                          if g.lower().startswith(prefix) and g not in similar]
        similar.extend(prefix_matches)

    if len(similar) < n:
        # Fallback: genes containing the query as substring
        substring_matches = [g for g in all_genes
                             if query_lower in g.lower() and g not in similar]
        similar.extend(substring_matches)

    return similar[:n]


def get_tools_for_query_type(query_type: str) -> List[str]:
    """
    Dynamically load tools based on query type.
    
    Args:
        query_type: From Agent 1's classification
    
    Returns:
        List of tool names to expose to Agent 3
    """
    return TOOL_GROUPS.get(query_type, TOOL_GROUPS['all'])


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AGENT 3 NETWORKX TOOLS - DEMONSTRATION")
    print("=" * 70)
    
    # Generate mock network
    from mock_carnival_generator import quick_mock_network
    G = quick_mock_network(seed=42)
    
    print("\n🧪 Testing Tool 1: List Transcription Factors")
    print("=" * 70)
    result = list_transcription_factors(G, top_n=5)
    print(result['summary'])
    print(f"Next steps: {result['next_steps'][0]}")
    
    print("\n🧪 Testing Tool 2: Calculate Centrality")
    print("=" * 70)
    result = calculate_node_centrality(G, metrics=['betweenness', 'degree'], top_n=5)
    print(result['summary'])
    
    print("\n🧪 Testing Tool 3: Find Shortest Path")
    print("=" * 70)
    tfs = list_transcription_factors(G)['tfs']
    if len(tfs) >= 2:
        result = find_shortest_path(G, tfs[0], tfs[1])
        print(result['summary'])
    
    print("\n🧪 Testing Tool 4: Get Neighbors")
    print("=" * 70)
    if tfs:
        result = get_node_neighbors(G, tfs[0], direction='both', max_neighbors=5)
        print(result['summary'])
    
    print("\nOK:All tools functional!")