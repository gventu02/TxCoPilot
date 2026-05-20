"""
Gene Validation
================
Validate gene symbols extracted from user queries against OmniPath.

Why validate?
    - Users may misspell gene names ("TGFB" instead of "TGFB1")
    - LLM extraction may produce non-standard symbols
    - CARNIVAL needs valid gene symbols to build its network
    - Early validation prevents silent failures in Agent 2

Strategy:
    1. Load gene symbol set from cache (JSON file, fast, works offline/Docker)
    2. If cache missing, fetch from OmniPath API and build cache
    3. Validate extracted genes against the set
    4. Return validated + invalid lists for downstream use

Cache design (Docker-friendly):
    - Cache is a JSON file at config.gene_cache_path (default: data/gene_symbols_cache.json)
    - Built once from OmniPath, then reused across runs
    - Portable: ship the cache file inside the Docker image
    - Regenerate with: GeneValidator(config).rebuild_cache()

Usage:
    from agent1.validation.gene_validation import GeneValidator
    
    validator = GeneValidator(config)
    validated, invalid = validator.validate(["TGFB1", "SMAD3", "FAKEGENE"])
    # validated = ["TGFB1", "SMAD3"]
    # invalid = ["FAKEGENE"]
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

from agent1.schemas_agent1 import Agent1Config

logger = logging.getLogger(__name__)


# =============================================================================
# OMNIPATH API CONSTANTS
# =============================================================================

# OmniPath REST API endpoints
# The interactions endpoint returns source_genesymbol and target_genesymbol
# We collect unique gene symbols from all interactions to build our validation set
OMNIPATH_INTERACTIONS_URL = "https://omnipathdb.org/interactions"

# Minimum expected gene count — sanity check for API responses
# Human genome has ~20,000 protein-coding genes; OmniPath interactions cover ~8,000-15,000
MIN_EXPECTED_GENES = 5_000


# =============================================================================
# GENE VALIDATOR
# =============================================================================

class GeneValidator:
    """
    Validates gene symbols against OmniPath's protein database.
    
    Uses a two-tier strategy:
        1. JSON cache file (fast, works offline, Docker-portable)
        2. OmniPath API fallback (builds cache if missing)
    
    The cache contains a set of all gene symbols known to OmniPath.
    This is ~18,000 symbols, stored as a ~300KB JSON file.
    """
    
    def __init__(self, config: Agent1Config):
        self.config = config
        self._gene_set: Optional[Set[str]] = None
    
    def validate(self, gene_symbols: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate a list of gene symbols.
        
        Args:
            gene_symbols: Gene symbols to validate (e.g., ["TGFB1", "SMAD3"])
            
        Returns:
            Tuple of:
                - validated: Genes found in OmniPath (valid symbols)
                - invalid: Genes NOT found (possible typos or non-standard names)
                
        Note: Validation is case-insensitive. "tgfb1" matches "TGFB1".
        The returned validated list uses the canonical (uppercase) form.
        """
        if not gene_symbols:
            return [], []
        
        gene_set = self._get_gene_set()
        
        if not gene_set:
            # No cache and API unavailable — can't validate
            logger.warning(
                "Gene validation unavailable (no cache and API unreachable). "
                "Passing all genes through as unvalidated."
            )
            return gene_symbols, []
        
        # Build case-insensitive lookup: lowercase → canonical form
        canonical_lookup = {g.lower(): g for g in gene_set}
        
        validated = []
        invalid = []
        
        for gene in gene_symbols:
            canonical = canonical_lookup.get(gene.strip().lower())
            if canonical:
                validated.append(canonical)
            else:
                invalid.append(gene)
        
        logger.info(
            f"Gene validation: {len(validated)}/{len(gene_symbols)} valid, "
            f"{len(invalid)} invalid"
        )
        if invalid:
            logger.info(f"Invalid genes: {invalid}")
        
        return validated, invalid
    
    def _get_gene_set(self) -> Optional[Set[str]]:
        """
        Get the set of valid gene symbols. Tries cache first, then API.
        Cached in memory after first load.
        """
        if self._gene_set is not None:
            return self._gene_set
        
        # Try loading from file cache
        cache_path = Path(self.config.gene_cache_path)
        gene_set = self._load_cache(cache_path)
        
        if gene_set:
            self._gene_set = gene_set
            return self._gene_set
        
        # Cache miss — try to build from OmniPath API
        logger.info("Gene cache not found. Fetching from OmniPath API...")
        gene_set = self._fetch_from_omnipath()
        
        if gene_set:
            self._save_cache(gene_set, cache_path)
            self._gene_set = gene_set
            return self._gene_set
        
        # Both cache and API failed
        logger.warning("Could not load gene symbols from cache or API.")
        return None
    
    # =========================================================================
    # CACHE: JSON file read/write
    # =========================================================================
    
    @staticmethod
    def _load_cache(cache_path: Path) -> Optional[Set[str]]:
        """Load gene symbols from JSON cache file."""
        if not cache_path.exists():
            logger.debug(f"Cache file not found: {cache_path}")
            return None
        
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            
            gene_list = data.get("gene_symbols", [])
            
            if len(gene_list) < MIN_EXPECTED_GENES:
                logger.warning(
                    f"Cache has only {len(gene_list)} genes "
                    f"(expected >= {MIN_EXPECTED_GENES}). Cache may be corrupt."
                )
                return None
            
            logger.info(
                f"Loaded {len(gene_list)} gene symbols from cache: {cache_path}"
            )
            return set(gene_list)
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to read gene cache: {e}")
            return None
    
    @staticmethod
    def _save_cache(gene_set: Set[str], cache_path: Path) -> None:
        """Save gene symbols to JSON cache file."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "gene_symbols": sorted(gene_set),  # sorted for deterministic output
            "count": len(gene_set),
            "source": "OmniPath proteins endpoint",
            "note": "Regenerate with: GeneValidator(config).rebuild_cache()",
        }
        
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(gene_set)} gene symbols to cache: {cache_path}")
    
    # =========================================================================
    # OMNIPATH API: Fetch gene symbols
    # =========================================================================
    
    def _fetch_from_omnipath(self) -> Optional[Set[str]]:
        """
        Fetch all known gene symbols from OmniPath.
        
        Strategy 1: Use the `omnipath` Python client (pip install omnipath)
            - Cleanest approach, handles pagination and formats automatically
        Strategy 2: REST API fallback using requests
            - Fetches interactions as TSV (default format), extracts gene symbols
            
        Returns None if both strategies fail.
        """
        # --- Strategy 1: omnipath Python client ---
        gene_set = self._fetch_via_omnipath_client()
        if gene_set:
            return gene_set
        
        # --- Strategy 2: REST API with TSV parsing ---
        gene_set = self._fetch_via_rest_api()
        if gene_set:
            return gene_set
        
        return None
    
    def _fetch_via_omnipath_client(self) -> Optional[Set[str]]:
        """
        Fetch gene symbols using the omnipath Python package.
        This is the preferred method — it handles all API details.
        
        Install: pip install omnipath
        """
        try:
            from omnipath.interactions import AllInteractions
            
            logger.info("Fetching gene symbols via omnipath Python client...")
            start_time = time.time()
            
            # Get all interactions (returns DataFrame with source_genesymbol, target_genesymbol)
            interactions = AllInteractions.get(
                genesymbols=True,
                organisms=9606,
            )
            
            gene_symbols = set()
            if "source_genesymbol" in interactions.columns:
                gene_symbols.update(interactions["source_genesymbol"].dropna().unique())
            if "target_genesymbol" in interactions.columns:
                gene_symbols.update(interactions["target_genesymbol"].dropna().unique())
            
            elapsed = time.time() - start_time
            logger.info(
                f"Fetched {len(gene_symbols)} gene symbols via omnipath client "
                f"in {elapsed:.1f}s"
            )
            return gene_symbols if len(gene_symbols) >= MIN_EXPECTED_GENES else None
            
        except ImportError:
            logger.debug("omnipath package not installed. Trying REST API fallback.")
            return None
        except Exception as e:
            logger.warning(f"omnipath client failed: {e}. Trying REST API fallback.")
            return None
    
    def _fetch_via_rest_api(self) -> Optional[Set[str]]:
        """
        Fetch gene symbols from OmniPath REST API using requests.
        
        Uses the /interactions endpoint with TSV format (OmniPath's default).
        Fetches all datasets to maximize gene coverage.
        """
        try:
            import requests
            import io
        except ImportError:
            logger.warning(
                "requests package not available. Cannot fetch from OmniPath API. "
                f"Provide a pre-built gene cache at: {self.config.gene_cache_path}"
            )
            return None
        
        try:
            logger.info(f"Fetching gene symbols from OmniPath REST API...")
            start_time = time.time()
            
            # Request interactions as TSV (default format) with gene symbols
            # Using all_interactions dataset for maximum gene coverage
            response = requests.get(
                OMNIPATH_INTERACTIONS_URL,
                params={
                    "genesymbols": "yes",
                    "organisms": "9606",
                    "datasets": "omnipath,pathwayextra,kinaseextra,ligrecextra",
                    "fields": "sources",
                },
                timeout=self.config.omnipath_timeout,
            )
            response.raise_for_status()
            
            # Parse TSV response
            import pandas as pd
            df = pd.read_csv(io.StringIO(response.text), sep="\t")
            
            elapsed = time.time() - start_time
            
            gene_symbols = set()
            if "source_genesymbol" in df.columns:
                gene_symbols.update(df["source_genesymbol"].dropna().unique())
            if "target_genesymbol" in df.columns:
                gene_symbols.update(df["target_genesymbol"].dropna().unique())
            
            if len(gene_symbols) < MIN_EXPECTED_GENES:
                logger.warning(
                    f"OmniPath returned only {len(gene_symbols)} gene symbols "
                    f"(expected >= {MIN_EXPECTED_GENES}). Response may be incomplete."
                )
            
            logger.info(
                f"Fetched {len(gene_symbols)} gene symbols from OmniPath REST API "
                f"in {elapsed:.1f}s"
            )
            return gene_symbols if gene_symbols else None
            
        except requests.exceptions.Timeout:
            logger.warning(
                f"OmniPath API timeout ({self.config.omnipath_timeout}s). "
                "Provide a pre-built gene cache for offline use."
            )
            return None
        except Exception as e:
            logger.warning(f"OmniPath REST API error: {e}")
            return None
    
    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================
    
    def rebuild_cache(self) -> bool:
        """
        Force rebuild the gene symbol cache from OmniPath API.
        
        Call this once during setup or Docker image build to create
        the cache file. After that, validation works offline.
        
        Returns:
            True if cache was successfully rebuilt, False otherwise.
            
        Usage:
            # One-time setup (or in Dockerfile):
            python -c "
            from agent1.config import get_config
            from agent1.validation.gene_validation import GeneValidator
            GeneValidator(get_config()).rebuild_cache()
            "
        """
        logger.info("Rebuilding gene symbol cache from OmniPath...")
        gene_set = self._fetch_from_omnipath()
        
        if gene_set:
            cache_path = Path(self.config.gene_cache_path)
            self._save_cache(gene_set, cache_path)
            self._gene_set = gene_set
            return True
        
        logger.error("Failed to rebuild gene cache. OmniPath API may be down.")
        return False
    
    @property
    def cache_exists(self) -> bool:
        """Check if the gene cache file exists."""
        return Path(self.config.gene_cache_path).exists()
    
    @property
    def cache_size(self) -> Optional[int]:
        """Number of genes in cache, or None if cache doesn't exist."""
        gene_set = self._get_gene_set()
        return len(gene_set) if gene_set else None
