"""
Agent 3 Network Visualizer
==========================

Automatically visualizes subnetworks extracted during Agent 3 analysis.
Designed for post-ReAct loop rendering of disease gene candidates and their regulatory context.

Key Features:
- Extracts subnetworks around important nodes identified by Agent 3
- Biological encoding: node colors by type, edge colors by interaction
- Publication-quality figures with clear legends
- Automatic layout optimization for readability

Author: Giselle Ventura
Date: November 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Set, Optional, Tuple
import numpy as np
from pathlib import Path


class Agent3NetworkVisualizer:
    """
    Post-analysis network visualization for Agent 3.
    
    Renders focused subnetworks showing disease gene candidates
    and their immediate regulatory context from CARNIVAL networks.
    """
    
    # Biological color scheme
    NODE_COLORS = {
        'TF': '#E74C3C',                   # Red for TFs (CARNIVAL node_type)
        'transcriptionFactor': '#E74C3C',   # Alias for compatibility
        'measured': '#3498DB',              # Blue for measured genes
        'intermediate': '#95A5A6',          # Gray for signaling intermediates (CARNIVAL node_type)
        'signaling': '#95A5A6',             # Alias for compatibility
        'highlight': '#F39C12'              # Orange for key findings
    }
    
    EDGE_COLORS = {
        1: '#2ECC71',   # Green for activation
        -1: '#E74C3C',  # Red for inhibition
        0: '#95A5A6'    # Gray for unknown
    }
    
    def __init__(self, carnival_network: nx.DiGraph, output_dir: str = "./visualizations"):
        """
        Initialize visualizer with CARNIVAL network.
        
        Args:
            carnival_network: NetworkX DiGraph from mock CARNIVAL or real CARNIVAL
            output_dir: Directory to save visualization outputs
        """
        self.network = carnival_network
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def extract_subnetwork(
        self, 
        key_nodes: List[str], 
        k_hops: int = 1,
        max_nodes: int = 50
    ) -> nx.DiGraph:
        """
        Extract subnetwork around key nodes identified by Agent 3.
        
        Args:
            key_nodes: List of important node IDs (e.g., top TFs from analysis)
            k_hops: Number of hops to include neighbors (default: 1)
            max_nodes: Maximum nodes to include (prevents hairballs)
            
        Returns:
            Subnetwork as NetworkX DiGraph
        """
        # Start with key nodes
        subnetwork_nodes: Set[str] = set(key_nodes)
        
        # Add k-hop neighbors by expanding from the frontier each hop
        for node in key_nodes:
            if node not in self.network:
                continue

            frontier = {node}
            for _ in range(k_hops):
                next_frontier = set()
                for n in frontier:
                    next_frontier.update(self.network.predecessors(n))
                    next_frontier.update(self.network.successors(n))
                subnetwork_nodes.update(next_frontier)
                frontier = next_frontier
        
        # Limit size if too large
        if len(subnetwork_nodes) > max_nodes:
            # Prioritize nodes by degree centrality
            degree_scores = nx.degree_centrality(self.network)
            sorted_nodes = sorted(
                subnetwork_nodes, 
                key=lambda n: degree_scores.get(n, 0), 
                reverse=True
            )
            subnetwork_nodes = set(sorted_nodes[:max_nodes])
        
        # Extract subgraph
        subnetwork = self.network.subgraph(subnetwork_nodes).copy()
        
        # Mark key nodes for highlighting
        for node in key_nodes:
            if node in subnetwork:
                subnetwork.nodes[node]['highlight'] = True
        
        return subnetwork
    
    def _get_node_colors(self, subnetwork: nx.DiGraph) -> List[str]:
        """Assign colors to nodes based on type and highlighting."""
        colors = []
        for node in subnetwork.nodes():
            node_data = subnetwork.nodes[node]
            
            # Highlight key findings
            if node_data.get('highlight', False):
                colors.append(self.NODE_COLORS['highlight'])
            else:
                # Color by biological type
                node_type = node_data.get('node_type', 'intermediate')
                colors.append(self.NODE_COLORS.get(node_type, self.NODE_COLORS['intermediate']))
        
        return colors
    
    def _get_node_sizes(self, subnetwork: nx.DiGraph) -> List[float]:
        """Scale node sizes by importance (degree centrality or highlight status)."""
        sizes = []
        degree_centrality = nx.degree_centrality(subnetwork)
        
        for node in subnetwork.nodes():
            node_data = subnetwork.nodes[node]
            
            # Base size
            base_size = 300
            
            # Larger for highlighted nodes
            if node_data.get('highlight', False):
                sizes.append(base_size * 3)
            else:
                # Scale by centrality
                centrality = degree_centrality.get(node, 0)
                sizes.append(base_size * (1 + centrality * 2))
        
        return sizes
    
    def _get_edge_colors(self, subnetwork: nx.DiGraph) -> List[str]:
        """Assign colors to edges based on sign attribute (activation/inhibition)."""
        colors = []
        for u, v in subnetwork.edges():
            edge_data = subnetwork.edges[u, v]
            interaction = str(edge_data.get('interaction', ''))
            pkn_sign = 1 if interaction == 'ACTIVATES' else -1 if interaction == 'INHIBITS' else 0
            colors.append(self.EDGE_COLORS.get(pkn_sign, self.EDGE_COLORS[0]))

        return colors
    
    def _get_edge_widths(self, subnetwork: nx.DiGraph) -> List[float]:
        """Scale edge widths by confidence/weight if available."""
        widths = []
        for u, v in subnetwork.edges():
            edge_data = subnetwork.edges[u, v]
            weight = edge_data.get('weight', 1.0)
            widths.append(1.0 + weight * 2)  # Scale between 1-3
        
        return widths
    
    def visualize_subnetwork(
        self,
        key_nodes: List[str],
        title: str = "Network Analysis Results",
        k_hops: int = 1,
        layout: str = "spring",
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create publication-quality visualization of subnetwork.
        
        Args:
            key_nodes: Important nodes identified by Agent 3
            title: Figure title
            k_hops: Neighbor hops to include
            layout: NetworkX layout algorithm ('spring', 'kamada_kawai', 'circular')
            figsize: Figure size in inches
            save_path: Optional path to save figure (auto-generates if None)
            
        Returns:
            Matplotlib figure object
        """
        # Extract subnetwork
        subnetwork = self.extract_subnetwork(key_nodes, k_hops=k_hops)
        
        if len(subnetwork.nodes()) == 0:
            print(f"Warning: No nodes found for key_nodes={key_nodes}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute layout
        if layout == "spring":
            pos = nx.spring_layout(subnetwork, k=0.5, iterations=50, seed=42)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(subnetwork)
        elif layout == "circular":
            pos = nx.circular_layout(subnetwork)
        else:
            pos = nx.spring_layout(subnetwork, seed=42)
        
        # Get visual attributes
        node_colors = self._get_node_colors(subnetwork)
        node_sizes = self._get_node_sizes(subnetwork)
        edge_colors = self._get_edge_colors(subnetwork)
        edge_widths = self._get_edge_widths(subnetwork)
        
        # Draw network
        nx.draw_networkx_edges(
            subnetwork, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
        
        nx.draw_networkx_nodes(
            subnetwork, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            linewidths=2,
            edgecolors='white',
            ax=ax
        )
        
        nx.draw_networkx_labels(
            subnetwork, pos,
            font_size=8,
            font_weight='bold',
            font_color='black',
            ax=ax
        )
        
        # Create legend
        legend_elements = [
            mpatches.Patch(color=self.NODE_COLORS['TF'],
                          label='Transcription Factor'),
            mpatches.Patch(color=self.NODE_COLORS['measured'],
                          label='Measured Gene'),
            mpatches.Patch(color=self.NODE_COLORS['intermediate'],
                          label='Signaling Intermediate'),
            mpatches.Patch(color=self.NODE_COLORS['highlight'], 
                          label='Key Finding'),
            mpatches.Patch(color=self.EDGE_COLORS[1], 
                          label='Activation'),
            mpatches.Patch(color=self.EDGE_COLORS[-1], 
                          label='Inhibition')
        ]
        
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True
        )
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            # Auto-generate filename
            clean_title = "".join(c if c.isalnum() else "_" for c in title)
            save_path = self.output_dir / f"{clean_title}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Visualization saved: {save_path}")
        
        return fig
    
    def visualize_from_agent3_answer(
        self,
        agent3_answer: Dict,
        carnival_network: Optional[nx.DiGraph] = None
    ) -> Optional[plt.Figure]:
        """
        Automatically extract key nodes from Agent 3's final answer and visualize.
        
        This is the main entry point for post-analysis visualization.
        
        Args:
            agent3_answer: Dictionary containing Agent 3's final answer
                Expected keys: 'answer', 'key_nodes', 'question'
            carnival_network: Optional override for network (uses self.network if None)
            
        Returns:
            Matplotlib figure or None if visualization not possible
        """
        if carnival_network is not None:
            self.network = carnival_network
        
        # Extract key nodes from answer
        key_nodes = agent3_answer.get('key_nodes', [])
        question = agent3_answer.get('question', 'Network Analysis')
        
        if not key_nodes:
            print("⚠️  No key nodes found in Agent 3 answer. Skipping visualization.")
            return None
        
        print(f"🎨 Visualizing subnetwork around {len(key_nodes)} key nodes...")
        
        # Create visualization
        fig = self.visualize_subnetwork(
            key_nodes=key_nodes,
            title=f"Agent 3 Analysis: {question[:50]}...",
            k_hops=1
        )
        
        return fig


def integrate_with_agent3_pipeline(
    agent3_results: Dict,
    carnival_network: nx.DiGraph,
    output_dir: str = "./visualizations"
) -> Optional[str]:
    """
    Integration function to call after Agent 3 completes its ReAct loop.
    
    Args:
        agent3_results: Full results dictionary from Agent 3
        carnival_network: NetworkX graph from CARNIVAL
        output_dir: Where to save visualizations
        
    Returns:
        Path to saved visualization or None
        
    Example:
        >>> # In agent3_react_loop.py after ReAct completes:
        >>> viz_path = integrate_with_agent3_pipeline(
        ...     agent3_results=final_answer,
        ...     carnival_network=mock_carnival_network
        ... )
    """
    visualizer = Agent3NetworkVisualizer(carnival_network, output_dir=output_dir)
    
    fig = visualizer.visualize_from_agent3_answer(agent3_results)
    
    if fig is not None:
        # Return path to saved figure
        clean_title = "".join(c if c.isalnum() else "_" for c in agent3_results.get('question', 'analysis'))
        save_path = Path(output_dir) / f"{clean_title}.png"
        return str(save_path)
    
    return None


# Example usage for testing
if __name__ == "__main__":
    print("Agent 3 Network Visualizer - Test Mode")
    print("="*70)
    
    # This would be replaced with actual mock CARNIVAL network
    print("\n⚠️  Run this module from agent3_react_loop.py after analysis completes.")
    print("    See integrate_with_agent3_pipeline() for integration example.")
