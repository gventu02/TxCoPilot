"""
gene_id_normalizer.py
======================
Detects gene identifier type and maps to HGNC symbols via MyGene.info.

Integrates into Agent 1's DataValidator pipeline.
Called automatically when expression matrix or DEG row IDs are not HGNC symbols.

Design principles:
    - Detect first, map second — never guess silently
    - Cache results to avoid redundant API calls
    - Graceful degradation: warn and continue if mapping fails partially
    - Always species='human' —  system is human-only

Usage:
    normalizer = GeneIDNormalizer()
    
    # Auto-detect and map
    result = normalizer.normalize(df, 'column')
    print(result.detected_type)   # 'entrez_id'
    print(result.mapping_rate)    # 0.94 (94% of IDs mapped)
    
    # Use mapped dataframe
    df_hgnc = result.mapped_df    # row IDs now HGNC symbols
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
import mygene 

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class GeneIDType:
    """Detected gene identifier types."""
    HGNC_SYMBOL     = "hgnc_symbol"      # TP53, ACTB — already correct
    ENTREZ_ID       = "entrez_id"         # 7157, 1282 — integers
    ENSEMBL_GENE    = "ensembl_gene"      # ENSG00000141510
    ENSEMBL_VERSION = "ensembl_versioned" # ENSG00000141510.11
    AFFYMETRIX      = "affymetrix_probe"  # 209696_at, 1255_g_at
    ILLUMINA        = "illumina_probe"    # ILMN_1343291
    REFSEQ          = "refseq"            # NM_000546, NR_024540
    UNIPROT         = "uniprot"           # P04637
    MIXED           = "mixed"             # Multiple types present
    UNKNOWN         = "unknown"           # Cannot determine


# Maps detected type → MyGene scopes string
MYGENE_SCOPES = {
    GeneIDType.ENTREZ_ID:       "entrezgene",
    GeneIDType.ENSEMBL_GENE:    "ensembl.gene",
    GeneIDType.ENSEMBL_VERSION: "ensembl.gene",
    GeneIDType.AFFYMETRIX:      "reporter",
    GeneIDType.ILLUMINA:        "reporter",
    GeneIDType.REFSEQ:          "refseq",
    GeneIDType.UNIPROT:         "uniprot",
    GeneIDType.HGNC_SYMBOL:     "symbol",
    GeneIDType.MIXED:           "entrezgene,ensembl.gene,symbol,reporter,refseq,uniprot",
}


# =============================================================================
# RESULT
# =============================================================================

@dataclass
class NormalizationResult:
    """Result from gene ID normalization."""
    
    detected_type: str
    original_ids: list
    mapped_df: Optional[pd.DataFrame] = None
    
    # Mapping statistics
    n_input: int = 0
    n_mapped: int = 0
    n_unmapped: int = 0
    n_duplicate_symbols: int = 0
    
    # Human-readable
    mapping_rate: float = 0.0
    warnings: list = field(default_factory=list)
    
    # Whether normalization actually changed anything
    normalization_needed: bool = True

    @property
    def summary(self) -> str:
        return (
            f"Detected: {self.detected_type} | "
            f"Mapped {self.n_mapped}/{self.n_input} "
            f"({self.mapping_rate:.1%}) to HGNC symbols"
        )


# =============================================================================
# DETECTOR
# =============================================================================

class GeneIDDetector:
    """
    Heuristic detection of gene identifier type from a sample of IDs.
    
    Uses format pattern matching — no network calls needed.
    Inspects first 100 IDs (enough to be confident, fast enough to not block).
    """

    def detect(self, ids: list[str]) -> str:
        """
        Detect gene ID type from a list of identifiers.
        
        Args:
            ids: List of gene identifiers (strings)
            
        Returns:
            GeneIDType constant string
        """
        # Sample for efficiency — 100 is enough
        sample = [str(s).strip() for s in ids[:100] if str(s).strip()]
        
        if not sample:
            return GeneIDType.UNKNOWN
        
        scores = {t: 0 for t in [
            GeneIDType.HGNC_SYMBOL, GeneIDType.ENTREZ_ID,
            GeneIDType.ENSEMBL_GENE, GeneIDType.ENSEMBL_VERSION,
            GeneIDType.AFFYMETRIX, GeneIDType.ILLUMINA,
            GeneIDType.REFSEQ, GeneIDType.UNIPROT,
        ]}
        
        for s in sample:
            if re.match(r'^ENSG\d{11}\.\d+$', s):
                scores[GeneIDType.ENSEMBL_VERSION] += 1
            elif re.match(r'^ENSG\d{11}$', s):
                scores[GeneIDType.ENSEMBL_GENE] += 1
            elif re.match(r'^\d+_[asxg]_?at$', s):
                scores[GeneIDType.AFFYMETRIX] += 1  # e.g. 209696_at, 1255_s_at
            elif re.match(r'^ILMN_\d+$', s):
                scores[GeneIDType.ILLUMINA] += 1
            elif re.match(r'^(NM|NR|XM|XR)_\d+', s):
                scores[GeneIDType.REFSEQ] += 1
            elif re.match(r'^[A-Z][0-9][A-Z0-9]{3}[0-9]$', s):
                scores[GeneIDType.UNIPROT] += 1  # e.g. P04637
            elif re.match(r'^\d+$', s):
                scores[GeneIDType.ENTREZ_ID] += 1
            elif re.match(r'^[A-Z][A-Z0-9-]{1,14}$', s) and len(s) <= 14:
                scores[GeneIDType.HGNC_SYMBOL] += 1
        
        # Find dominant type
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        if best_score == 0:
            return GeneIDType.UNKNOWN
        
        # If no clear majority (mixed data)
        coverage = best_score / len(sample)
        if coverage < 0.7:
            logger.warning(
                f"Mixed gene ID types detected. "
                f"Best match: {best_type} ({coverage:.0%} of sample). "
                f"Will attempt multi-scope mapping."
            )
            return GeneIDType.MIXED
        
        logger.info(f"Detected gene ID type: {best_type} ({coverage:.0%} match)")
        return best_type


# =============================================================================
# NORMALIZER
# =============================================================================

class GeneIDNormalizer:
    """
    Maps any supported gene identifier type to HGNC symbols via MyGene.info.
    
    Handles:
        - Entrez IDs (integers)       → common in microarrays, older databases
        - Ensembl gene IDs            → modern RNA-seq (STAR/featureCounts)
        - Ensembl versioned           → strips .version suffix before mapping
        - Affymetrix probe IDs (_at)  → classic microarray format
        - Illumina probe IDs          → Illumina BeadChip arrays
        - RefSeq accessions           → NCBI pipelines
        - UniProt IDs                 → protein-level databases
        - HGNC symbols                → already correct, validates only
    
    For duplicate probes mapping to same gene:
        Keeps the row with highest mean expression (standard practice).
    """
    
    def __init__(self, species: str = "human", cache: bool = True):
        self.species = species
        self.cache = cache
        self._cache: dict = {}
        self.detector = GeneIDDetector()
    
    def normalize(
        self,
        df: pd.DataFrame,
        id_type: Optional[str] = None,
        column: Optional[str] = None,
        sort_by: Optional[str] = None, 
    ) -> NormalizationResult:
        """
    Normalize gene IDs to HGNC symbols.
    
    Args:
        df:       Expression matrix (genes × samples) OR DEG DataFrame
        id_type:  Override auto-detection (use GeneIDType constants)
        column:   If provided, read/write gene IDs from this column
                  instead of df.index. Use for DEG files where gene
                  IDs are in a column (e.g. 'gene', 'gene_name').
        sort_by:  Column to use for duplicate tie-breaking.
                  Default: None → uses mean expression (index mode)
                                   or first occurrence (column mode)
                  For DEG files: pass 'padj' to keep most significant.
    
    Examples:
        # Expression matrix (genes × samples)
        result = normalizer.normalize(expr_df)
        
        # DEG file with gene IDs in 'gene' column
        result = normalizer.normalize(deg_df, column='gene', sort_by='padj')
    """
        if column is not None:
            if column not in df.columns:
                raise ValueError(
                    f"Column '{column}' not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )
            ids = [str(x) for x in df[column].tolist()]
        else:
            ids = [str(x) for x in df.index.tolist()]

        detected = id_type or self.detector.detect(ids)

        result = NormalizationResult(
            detected_type=detected,
            original_ids=ids,
            n_input=len(ids),
        )

        if detected == GeneIDType.HGNC_SYMBOL:
            result.normalization_needed = False
            result.mapped_df = df
            result.n_mapped = len(ids)
            result.mapping_rate = 1.0
            return result

        if detected == GeneIDType.UNKNOWN:
            result.warnings.append(
                "Could not determine gene ID type. "
                "Expected HGNC symbols, Entrez IDs, Ensembl IDs, "
                "Affymetrix probes (e.g. 209696_at), or RefSeq accessions."
            )
            result.mapped_df = df
            return result

        # ── FIX: strip versions AND sync df.index before mapping ──────────────
        if detected == GeneIDType.ENSEMBL_VERSION:
            ids = [s.split('.')[0] for s in ids]
            df = df.copy()
            if column is not None:
                df[column] = ids          # sync the column
            else:
                df.index = ids            # sync the index
            synced = df[column].tolist() if column else list(df.index)
            if synced != ids:
                raise ValueError(
                    f"ID drift after Ensembl version stripping: "
                    f"expected {len(ids)} IDs to match, got mismatch at "
                    f"position {next(i for i, (a, b) in enumerate(zip(synced, ids)) if a != b)}"
                )
            logger.info("Stripped Ensembl version suffixes.")

        # Now both ids and df.index are consistent — _apply_mapping is safe
        mapping = self._query_mygene(ids, detected)
        # mapped_df, stats = self._apply_mapping(df, mapping)
        mapped_df, stats = self._apply_mapping(df, mapping, column=column, sort_by=sort_by)


        result.mapped_df = mapped_df
        result.n_mapped = stats['n_mapped']
        result.n_unmapped = stats['n_unmapped']
        result.n_duplicate_symbols = stats['n_duplicates']
        result.mapping_rate = result.n_mapped / result.n_input if result.n_input > 0 else 0.0

        if result.mapping_rate < 0.5:
            result.warnings.append(
                f"Low mapping rate ({result.mapping_rate:.1%}). "
                f"Only {result.n_mapped}/{result.n_input} IDs mapped. "
                f"Detected type: {detected}. Verify species='{self.species}'."
            )
        elif result.mapping_rate < 0.8:
            result.warnings.append(
                f"Moderate mapping rate ({result.mapping_rate:.1%}). "
                f"{result.n_unmapped} unmapped IDs will be dropped."
            )

        if result.n_duplicate_symbols > 0:
            result.warnings.append(
                f"{result.n_duplicate_symbols} duplicate symbols resolved "
                "by keeping highest mean expression row."
            )

        logger.info(result.summary)
        return result
    
    def _query_mygene(self, ids: list[str], id_type: str) -> dict:
        """
        Query MyGene.info API and return {original_id: hgnc_symbol} mapping.
        
        Batches automatically for large lists (>1000 IDs).
        """
        # Check cache
        # cache_key = (tuple(ids[:10]), id_type)  # Lightweight cache key
        uncached = [i for i in ids if i not in self._cache]
        
        if not uncached:
            return {i: self._cache[i] for i in ids if i in self._cache}
        
        try:
            import mygene
        except ImportError:
            logger.warning(
                "mygene package not installed. "
                "Install with: pip install mygene\n"
                "Gene IDs will NOT be mapped to HGNC symbols — "
                "TF inference may fail downstream."
            )
            return {}
        
        scopes = MYGENE_SCOPES.get(id_type, "entrezgene,ensembl.gene,symbol")
        
        logger.info(
            f"Querying MyGene.info: {len(uncached)} IDs, "
            f"scopes='{scopes}', species='{self.species}'"
        )
        
        mg = mygene.MyGeneInfo()
        
        try:
            results = mg.querymany(
                uncached,
                scopes=scopes,
                fields='symbol',
                species=self.species,
                returnall=False,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"MyGene.info query failed: {e}")
            return {}
        
        # Parse results
        mapping = {}
        for hit in results:
            query_id = str(hit.get('query', ''))
            symbol = hit.get('symbol')
            not_found = hit.get('notfound', False)
            
            if not_found or not symbol:
                mapping[query_id] = None  # Unmapped
            else:
                mapping[query_id] = symbol
        
        # Cache results
        if self.cache:
            self._cache.update(mapping)
        
        return mapping
    
    def _apply_mapping(
        self,
        df: pd.DataFrame,
        mapping: dict,
        column: Optional[str] = None,    # ← NEW
        sort_by: Optional[str] = None,   # ← NEW
    ) -> tuple[pd.DataFrame, dict]:
        """
        Apply gene symbol mapping to DataFrame.
        
        Args:
            df:       DataFrame to map
            mapping:  {original_id: hgnc_symbol} from MyGene.info
            column:   If set, map this column (DEG mode).
                    If None, map the index (expression matrix mode).
            sort_by:  Column to use for duplicate resolution.
                    None → highest mean expression (expression mode)
                        → first occurrence (column mode)
                    'padj' → keep most significant DEG (recommended for DEGs)
        """
        df = df.copy()

        # ── Get original IDs and compute mapped symbols ────────────────────────
        if column is not None:
            original_ids = [str(x) for x in df[column].tolist()]
        else:
            original_ids = [str(x) for x in df.index.tolist()]

        new_symbols = [mapping.get(x) for x in original_ids]

        n_mapped = sum(1 for s in new_symbols if s is not None)
        n_unmapped = sum(1 for s in new_symbols if s is None)

        # ── Write mapped symbols back ──────────────────────────────────────────
        if column is not None:
            df[column] = new_symbols
            df = df[df[column].notna()].copy()   # drop unmapped rows
            df[column] = df[column].astype(str)
            id_series = df[column]
        else:
            df.index = new_symbols
            df = df[pd.Series(df.index).notna().values].copy()
            df.index = df.index.astype(str)
            id_series = pd.Series(df.index)

        # ── Resolve duplicates ─────────────────────────────────────────────────
        n_duplicates = 0
        has_duplicates = (
            df[column].duplicated().any() if column is not None
            else df.index.duplicated().any()
        )

        if has_duplicates:
            n_duplicates = (
                df[column].duplicated(keep=False).sum() if column is not None
                else df.index.duplicated(keep=False).sum()
            )

            if sort_by and sort_by in df.columns:
                # DEG mode: keep most significant (lowest padj)
                ascending = True   # lower padj = more significant
                df = df.sort_values(sort_by, ascending=ascending)
            elif column is None:
                # Expression mode: keep highest mean expression
                df['_mean_expr'] = df.select_dtypes(include='number').mean(axis=1)
                df = df.sort_values('_mean_expr', ascending=False)
                df = df.drop(columns=['_mean_expr'])
            # else: no sort_by specified in column mode → keep first occurrence

            if column is not None:
                df = df[~df[column].duplicated(keep='first')].copy()
            else:
                df = df[~df.index.duplicated(keep='first')].copy()

        return df, {
            'n_mapped': n_mapped,
            'n_unmapped': n_unmapped,
            'n_duplicates': n_duplicates,
        }