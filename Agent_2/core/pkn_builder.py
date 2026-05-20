"""
Agent 2 — Prior Knowledge Network Builder
===================================================
Loads the OmniPath signed directed network and prepares it for CARNIVAL.

Data source: OmniPath via OmnipathR (R package), pre-extracted to CSV.
The PKN is static, I extracted once from R and cached locally. 
I would recommend updating the cache every 6-12 months to keep up with OmniPath updates, 
but it's not critical to do more often than that.

To refresh the PKN cache, run this in R:
    library(OmnipathR)
    pkn <- import_omnipath_interactions(datasets=c('omnipath'), organism=9606)
    pkn_signed <- pkn[!is.na(pkn$is_stimulation) & !is.na(pkn$is_inhibition), ]
    pkn_signed$sign <- ifelse(
        pkn_signed$is_stimulation==1 & pkn_signed$is_inhibition==0, 1,
        ifelse(pkn_signed$is_inhibition==1 & pkn_signed$is_stimulation==0, -1, NA)
    )
    pkn_signed <- pkn_signed[!is.na(pkn_signed$sign),
        c('source_genesymbol','sign','target_genesymbol')]
    colnames(pkn_signed) <- c('source','interaction','target')
    write.csv(pkn_signed, 'data/omnipath_pkn_raw.csv', row.names=FALSE)

Output file format (network.csv) — required by CORNETO CARNIVAL:
    source,interaction,target
    CALM3,-1,TRPC1
    TGFB1,1,SMAD3
    ...
    interaction: +1 = activation, -1 = inhibition
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

# Default cache location
_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data"
_PKN_CACHE_FILENAME = "omnipath_pkn_raw.csv"


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def build_pkn(
    output_dir: str,
    pkn_filter: str = "full",
    anchor_genes: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, dict]:
    """
    Load and prepare the OmniPath Prior Knowledge Network for CARNIVAL.

    Steps:
      1. Load PKN from local CSV cache (extracted via OmnipathR)
      2. Validate format (source | interaction | target, values +1/-1)
      3. Apply hop filter if requested
      4. Write network.csv for the CARNIVAL R bridge

    Args:
        output_dir: Directory to write network.csv.
        pkn_filter: "full" | "2hop" | "1hop".
            "full"  = all edges (~70K). Works with HIGHS/SCIP/SCIPY. (This will almost always be the case)
            "2hop"  = edges where both nodes are within 2 hops of anchor genes.
            "1hop"  = edges where both nodes are within 1 hop of anchor genes.
        anchor_genes: Gene symbols to anchor hop filtering.
            Typically: top TFs from Step B + DEG gene list.
            Required if pkn_filter != "full". Ignored if pkn_filter == "full".
        cache_dir: Directory containing omnipath_pkn_raw.csv.
            Defaults to ../data/ relative to agent2/.

    Returns:
        Tuple of:
          - pkn_df: Filtered PKN DataFrame (source | interaction | target)
          - network_path: Absolute path to network.csv for CARNIVAL R script
          - stats: Dict with diagnostic information for PKNResult schema

    Raises:
        FileNotFoundError: If PKN cache file not found.
        ValueError: If PKN has wrong format or too few edges after filtering.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load PKN
    logger.info("Step D.1: Loading OmniPath PKN from cache...")
    pkn_df = _load_pkn_cache(cache_dir or str(_DEFAULT_CACHE_DIR))
    full_size = len(pkn_df)
    logger.info(
        f"PKN loaded: {full_size} edges, "
        f"{pkn_df['source'].nunique()} sources, "
        f"{pkn_df['target'].nunique()} targets, "
        f"+1: {(pkn_df['interaction']==1).sum()}, "
        f"-1: {(pkn_df['interaction']==-1).sum()}"
    )

    # Step 2: Apply hop filter
    filtered_df, n_anchor_used = _apply_hop_filter(
        pkn_df=pkn_df,
        pkn_filter=pkn_filter,
        anchor_genes=anchor_genes or [],
    )

    # Step 3: Validate result
    if len(filtered_df) < 100:
        raise ValueError(
            f"PKN has only {len(filtered_df)} edges after '{pkn_filter}' filter. "
            f"CARNIVAL needs at least ~100 edges to find solutions. "
            f"NEXT STEP: Switch to pkn_filter='full', or add more anchor genes "
            f"(currently using {n_anchor_used} anchor genes found in PKN)."
        )

    # Step 4: Write network.csv for CORNETO CARNIVAL
    network_path = output_path / "network.csv"
    filtered_df.to_csv(network_path, index=False)
    logger.info(
        f"network.csv written: {network_path} "
        f"({len(filtered_df)} edges, filter='{pkn_filter}')"
    )

    # Build stats
    all_nodes = set(filtered_df['source']) | set(filtered_df['target'])
    stats = {
        "n_edges": len(filtered_df),
        "n_nodes": len(all_nodes),
        "n_edges_filtered_from": full_size if pkn_filter != "full" else None,
        "filter_applied": pkn_filter,
        "anchor_genes_used": anchor_genes[:10] if anchor_genes else [],
        "network_path": str(network_path.resolve()),
    }

    return filtered_df, str(network_path.resolve()), stats

# =============================================================================
# PKN LOADING
# =============================================================================

def _load_pkn_cache(cache_dir: str) -> pd.DataFrame:
    """
    Load the OmniPath PKN from the local CSV cache.

    Expected format: source | interaction | target
    interaction values: +1 (activation) or -1 (inhibition)

    Raises:
        FileNotFoundError: If cache file doesn't exist with instructions to regenerate.
    """
    cache_path = Path(cache_dir) / _PKN_CACHE_FILENAME

    if not cache_path.exists():
        raise FileNotFoundError(
            f"OmniPath PKN cache not found: {cache_path}\n"
            f"NEXT STEP: Extract PKN from R using OmnipathR:\n"
            f"  library(OmnipathR)\n"
            f"  pkn <- import_omnipath_interactions(datasets=c('omnipath'), organism=9606)\n"
            f"  pkn_signed <- pkn[!is.na(pkn$is_stimulation) & !is.na(pkn$is_inhibition),]\n"
            f"  pkn_signed$sign <- ifelse(\n"
            f"      pkn_signed$is_stimulation==1 & pkn_signed$is_inhibition==0, 1,\n"
            f"      ifelse(pkn_signed$is_inhibition==1 & pkn_signed$is_stimulation==0, -1, NA)\n"
            f"  )\n"
            f"  pkn_signed <- pkn_signed[!is.na(pkn_signed$sign),\n"
            f"      c('source_genesymbol','sign','target_genesymbol')]\n"
            f"  colnames(pkn_signed) <- c('source','interaction','target')\n"
            f"  write.csv(pkn_signed, '{cache_path}', row.names=FALSE)"
        )

    pkn = pd.read_csv(cache_path)
    pkn = _validate_pkn(pkn)
    return pkn


def _validate_pkn(pkn: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the PKN DataFrame.

    Checks:
    - Required columns: source, interaction, target
    - interaction values are only +1 or -1
    - No self-loops
    - No duplicate edges
    """
    required = {'source', 'interaction', 'target'}
    if not required.issubset(pkn.columns):
        raise ValueError(
            f"PKN missing required columns. "
            f"Got: {list(pkn.columns)}, expected: {required}. "
            f"NEXT STEP: Re-extract from R — column names must be exactly "
            f"'source', 'interaction', 'target'."
        )

    # Strip whitespace from gene names (common CSV artifact)
    pkn['source'] = pkn['source'].str.strip()
    pkn['target'] = pkn['target'].str.strip()
    pkn['interaction'] = pkn['interaction'].astype(int)

    # Validate interaction values
    invalid = pkn[~pkn['interaction'].isin([1, -1])]
    if len(invalid) > 0:
        logger.warning(
            f"Dropping {len(invalid)} edges with invalid interaction values "
            f"(expected +1/-1, got: {invalid['interaction'].unique()[:5]})"
        )
        pkn = pkn[pkn['interaction'].isin([1, -1])]

    # Remove self-loops (source == target)
    self_loops = pkn[pkn['source'] == pkn['target']]
    if len(self_loops) > 0:
        logger.info(f"Removing {len(self_loops)} self-loops from PKN.")
        pkn = pkn[pkn['source'] != pkn['target']]

    # Remove duplicate edges (keep first, which has highest OmniPath confidence)
    before = len(pkn)
    pkn = pkn.drop_duplicates(subset=['source', 'target'], keep='first')
    if len(pkn) < before:
        logger.info(f"Removed {before - len(pkn)} duplicate edges from PKN.")

    pkn = pkn.reset_index(drop=True)
    logger.info(
        f"PKN validated: {len(pkn)} edges, "
        f"{pkn['source'].nunique()} unique sources, "
        f"{pkn['target'].nunique()} unique targets"
    )
    return pkn
# # =============================================================================
# # Contrained - HOP FILTERING
# # =============================================================================

def _apply_hop_filter(
    pkn_df: pd.DataFrame,
    pkn_filter: str,
    anchor_genes: List[str],
) -> Tuple[pd.DataFrame, int]:
    """
    Filter PKN to the signaling neighborhood of anchor TF genes.

    KEY CHANGE from original: uses forward-only BFS seeded from TFs.
    Reverse (upstream) is limited to 1 hop max regardless of pkn_filter,
    because pulling full upstream neighborhoods of hub TFs explodes network size.

    For CARNIVAL specifically: we want paths THROUGH the TFs, not the entire
    bidirectional cloud around them.
    """
    if pkn_filter == "full" or not anchor_genes:
        if pkn_filter != "full":
            logger.warning(
                f"pkn_filter='{pkn_filter}' but no anchor genes provided. "
                f"Falling back to 'full' network."
            )
        logger.info(f"PKN filter: full — using all {len(pkn_df)} edges")
        return pkn_df, 0

    n_hops = 2 if pkn_filter == "2hop" else 1

    # Build directed graph
    G = nx.DiGraph()
    for _, row in pkn_df.iterrows():
        G.add_edge(row['source'], row['target'])

    pkn_genes = set(G.nodes())
    valid_anchors = [g for g in anchor_genes if g in pkn_genes]
    n_anchor_found = len(valid_anchors)

    if n_anchor_found == 0:
        logger.warning(
            f"None of the {len(anchor_genes)} anchor genes found in PKN. "
            f"Falling back to full network."
        )
        return pkn_df, 0

    logger.info(
        f"PKN filter: {pkn_filter} — "
        f"{n_anchor_found}/{len(anchor_genes)} anchor genes found in PKN"
    )

    # Forward BFS: what do TFs regulate? (n_hops forward)
    fwd_reachable = set(valid_anchors)
    for gene in valid_anchors:
        fwd = nx.single_source_shortest_path_length(G, gene, cutoff=n_hops)
        fwd_reachable.update(fwd.keys())

    # Reverse BFS: what directly regulates our TFs? (ALWAYS 1 hop, never more)
    # Capped at 1 hop regardless of pkn_filter — avoids hub TF explosion
    rev_reachable = set(valid_anchors)
    G_rev = G.reverse()
    for gene in valid_anchors:
        rev = nx.single_source_shortest_path_length(G_rev, gene, cutoff=1)
        rev_reachable.update(rev.keys())

    # Combined reachable = forward neighborhood + direct upstream regulators
    reachable = fwd_reachable | rev_reachable

    # Keep edges where source is in reachable AND target is in reachable
    # Additional constraint: at least one endpoint must be an anchor TF
    # This prevents "floating" edges between non-TF nodes that happen
    # to both be in the reachable set but don't connect to the TFs
    anchor_set = set(valid_anchors)
    filtered = pkn_df[
        pkn_df['source'].isin(reachable) &
        pkn_df['target'].isin(reachable) &
        (pkn_df['source'].isin(anchor_set) | pkn_df['target'].isin(anchor_set))
    ].copy().reset_index(drop=True)

    reduction_pct = (1 - len(filtered) / len(pkn_df)) * 100
    logger.info(
        f"PKN after {pkn_filter} filter: "
        f"{len(filtered)}/{len(pkn_df)} edges ({reduction_pct:.0f}% reduction), "
        f"forward reachable: {len(fwd_reachable)} nodes, "
        f"reverse reachable: {len(rev_reachable)} nodes"
    )

    return filtered, n_anchor_found


# =============================================================================
# CONVENIENCE: QUICK DIAGNOSTIC
# =============================================================================

def run_pkn_diagnostic(cache_dir: Optional[str] = None) -> None:
    """
    Print a diagnostic report for PKN loading.
    Run this to verify the cache is valid before running the full pipeline.

    Usage:
        from agent2.core.pkn_builder import run_pkn_diagnostic
        run_pkn_diagnostic()
    """
    print("=" * 60)
    print("OmniPath PKN Diagnostic")
    print("=" * 60)

    cache = cache_dir or str(_DEFAULT_CACHE_DIR)
    cache_path = Path(cache) / _PKN_CACHE_FILENAME

    print(f"Cache location: {cache_path}")
    print(f"Cache exists: {cache_path.exists()}")

    if not cache_path.exists():
        print(" PKN cache missing. See docstring for R extraction command.")
        return

    try:
        pkn = _load_pkn_cache(cache)
        print(f" PKN loaded: {len(pkn)} edges")
        print(f"   Nodes: {len(set(pkn['source']) | set(pkn['target']))}")
        print(f"   Activations (+1): {(pkn['interaction']==1).sum()}")
        print(f"   Inhibitions (-1): {(pkn['interaction']==-1).sum()}")
        print(f"   Sample rows:")
        print(pkn.head(3).to_string(index=False))

        # Test hop filter with example genes
        test_genes = ['TGFB1', 'SMAD3', 'TP53', 'MYC']
        filtered, n_found = _apply_hop_filter(pkn, "2hop", test_genes)
        print(f"\n2-hop filter test (anchors: {test_genes}):")
        print(f"   Found in PKN: {n_found}/{len(test_genes)}")
        print(f"   Edges after filter: {len(filtered)}")

    except Exception as e:
        print(f"cl PKN loading failed: {e}")


if __name__ == "__main__":
    run_pkn_diagnostic()
