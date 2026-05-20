"""
KEGG Over-Representation Analysis (ORA) on CARNIVAL network nodes.

Hypergeometric test: are CARNIVAL activated/repressed nodes enriched
in specific KEGG pathways relative to the OmniPath PKN background (universe) 

Used by Streamlit UI as additoinal informaiton but this is NOT used for building the network.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

_DEFAULT_GMT = Path(__file__).parent.parent / "data" / "c2.cp.kegg_legacy.v2026.1.Hs.symbols.gmt"
_DEFAULT_PKN = Path(__file__).parent.parent / "data" / "omnipath_pkn_raw.csv"


def parse_gmt(gmt_path: str | Path) -> dict[str, set[str]]:
    """Parse a GMT file into {pathway_name: set of gene symbols}."""
    pathways = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                pathways[parts[0]] = set(parts[2:])
    return pathways


def _hypergeom_test(node_set: set, pathway_genes: set, universe: set) -> Optional[dict]:
    """Single hypergeometric test for one pathway."""
    pathway_in_universe = pathway_genes & universe
    overlap = pathway_in_universe & node_set
    M = len(universe)
    n = len(pathway_in_universe)
    N = len(node_set & universe)
    k = len(overlap)
    if k == 0 or n == 0:
        return None
    pval = hypergeom.sf(k - 1, M, n, N)
    return {
        'overlap_count': k,
        'pathway_in_universe': n,
        'node_set_size': N,
        'universe_size': M,
        'pval': pval,
        'overlap_genes': ','.join(sorted(overlap)),
    }


def run_kegg_ora(
    carnival_network,
    pkn_csv: str | Path = _DEFAULT_PKN,
    gmt_file: str | Path = _DEFAULT_GMT,
) -> pd.DataFrame:
    """
    Run KEGG ORA on CARNIVAL activated and repressed nodes.

    Args:
        carnival_network: nx.DiGraph with node attribute 'activity' (+1/-1/0)
        pkn_csv: Path to OmniPath PKN CSV (source/target columns) for universe
        gmt_file: Path to KEGG GMT file

    Returns:
        DataFrame with columns: pathway, direction, overlap_count,
        pathway_in_universe, node_set_size, universe_size, pval, padj,
        overlap_genes. Sorted by padj ascending.
        Returns empty DataFrame if no overlaps found.
    """
    # Build universe from PKN
    pkn = pd.read_csv(pkn_csv)
    universe = set(pkn['source']) | set(pkn['target'])
    logger.info(f"KEGG ORA: PKN universe = {len(universe)} genes")

    # Separate activated and repressed nodes
    activated = {n for n, d in carnival_network.nodes(data=True) if d.get('activity', 0) == 1}
    repressed = {n for n, d in carnival_network.nodes(data=True) if d.get('activity', 0) == -1}
    logger.info(f"KEGG ORA: {len(activated)} activated, {len(repressed)} repressed nodes")

    # Load KEGG pathways
    pathways = parse_gmt(gmt_file)
    logger.info(f"KEGG ORA: {len(pathways)} pathways loaded")

    # Test both directions
    results = []
    for direction, node_set in [('activated', activated), ('repressed', repressed)]:
        for pathway_name, pathway_genes in pathways.items():
            result = _hypergeom_test(node_set, pathway_genes, universe)
            if result:
                results.append({'pathway': pathway_name, 'direction': direction, **result})

    if not results:
        logger.warning("KEGG ORA: no overlaps found")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df['padj'] = multipletests(df['pval'], method='fdr_bh')[1]
    df = df.sort_values('padj')

    n_sig = (df['padj'] < 0.05).sum()
    logger.info(f"KEGG ORA: {n_sig} significant pathways (padj < 0.05)")
    return df
