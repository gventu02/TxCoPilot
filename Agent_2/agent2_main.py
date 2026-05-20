"""
Agent 2 — CARNIVAL Pipeline Orchestrator
==========================================
Receives Agent1Result, runs the complete causal network analysis pipeline,
and returns Agent2Result for Agent 3 consumption.

Pipeline steps (actual execution order):
  A. Validate Agent1Result and extract inputs
  D. Select CARNIVAL parameters (rules + LLM refinement)
  C. Infer TF activities (decoupler VIPER + CollecTRI, ULM fallback)
  B. Build Prior Knowledge Network (OmniPath, TF-anchored, uses inferred TFs as anchors)
  E. Run CARNIVAL (CORNETO + HIGHS solver, with retry on empty network)
  F. Package results (CARNIVALResult → Agent2Result with nx.DiGraph)

Usage:
    # Standard: from Agent 1 output
    from agent2_main import Agent2Pipeline

    pipeline = Agent2Pipeline(api_key=os.environ['ANTHROPIC_API_KEY'])
    result = pipeline.run(agent1_result)

    # result.carnival_network  → nx.DiGraph for Agent 3
    # result.summary           → str for Agent 3 context window
    # result.metadata          → dict with solver stats

    # Standalone Used for testing Only: from CSV files (testing / bypass Agent 1)
    result = pipeline.run_from_files(
        deg_path='carnival_test/input/measurements.csv',
        pkn_path='carnival_test/input/network.csv',
        sample_name='ipf_at2',
    )

Author: Giselle Ventura
Date: January 2026
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from core.carnival_cache import CARNIVALCache
from core.corneto_runner import run_carnival, CARNIVALResult
from core.output_packager import package_results, Agent2Result
from core.pkn_builder import build_pkn          # build_pkn(tf_list, ...) → DataFrame
from core.tf_activity import infer_tf_activity  # infer_tf_activity(deg_path, ...) → dict

logger = logging.getLogger(__name__)

# Module-level cache singleton — shared across pipeline runs
_cache = CARNIVALCache()


# =============================================================================
# CONSTANTS — single place to update solver/beta defaults
# =============================================================================

DEFAULT_SOLVER    = "HIGHS"
DEFAULT_BETA      = 0.5   # Liu 2019 reference (penalize="edges" for comparison)
DEFAULT_TOP_N_TFS = 50
OUTPUT_DIR        = str(Path(__file__).parent / "agent2_output")

# Translate parameter_selector vocabulary → corneto_runner vocabulary
_SOLVER_MAP = {
    "auto":  "HIGHS",   # "auto" means "best available open-source solver"
    "HIGHS": "HIGHS",
    "SCIP":  "SCIP",
    "SCIPY": "SCIPY",
}


# =============================================================================
# PIPELINE RESULT WRAPPER
# =============================================================================

@dataclass
class PipelineStats:
    """Timing and token usage for the full Agent 2 pipeline."""
    pkn_build_seconds:    float = 0.0
    tf_infer_seconds:     float = 0.0
    param_select_seconds: float = 0.0
    carnival_seconds:     float = 0.0
    total_seconds:        float = 0.0
    tokens_used:          int   = 0    # Not yet extracted from LLM response metadata
    parameter_method:     str   = "unknown"  # 'rules' | 'rules+llm' | 'rules+llm_fallback'

    def log_summary(self) -> None:
        logger.info(
            f"Pipeline timing — "
            f"PKN: {self.pkn_build_seconds:.1f}s | "
            f"TF: {self.tf_infer_seconds:.1f}s | "
            f"Params: {self.param_select_seconds:.1f}s | "
            f"CARNIVAL: {self.carnival_seconds:.1f}s | "
            f"Total: {self.total_seconds:.1f}s"
        )


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class Agent2Pipeline:
    """
    Agent 2: CARNIVAL causal network analysis pipeline.

    Designed to be instantiated once and called multiple times
    (e.g., across multiple samples or cell types).

    Args:
        api_key:    Anthropic API key for LLM parameter selection.
                    Falls back to ANTHROPIC_API_KEY env var if None.
        use_llm:    If False, skips LLM parameter refinement (faster, deterministic).
                    Useful for testing or when API is unavailable.
        output_dir: Directory for intermediate and output files.
        verbose:    If True, enables DEBUG logging for this pipeline.
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        use_llm:    bool = True,
        output_dir: str  = OUTPUT_DIR,
        verbose:    bool = False,
    ) -> None:
        self.api_key    = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.use_llm    = use_llm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            logging.getLogger(__name__).setLevel(logging.DEBUG)

        if not self.api_key and use_llm:
            logger.warning(
                "No Anthropic API key found — LLM parameter refinement disabled. "
                "Set ANTHROPIC_API_KEY environment variable to enable it."
            )

    # -------------------------------------------------------------------------
    # PRIMARY ENTRY POINT: Agent 1 → Agent 2
    # -------------------------------------------------------------------------

    def run(self, agent1_result) -> Agent2Result:
        """
        Run the full pipeline from an Agent1Result object.

        This is the primary entry point when running within the full
        3-agent system.

        Args:
            agent1_result: Agent1Result Pydantic model from Agent 1.

        Returns:
            Agent2Result with carnival_network (nx.DiGraph) for Agent 3.

        Raises:
            ValueError: If agent1_result is missing required fields.
            RuntimeError: If CARNIVAL fails after parameter adjustment.
        """
        t_total = time.time()
        stats   = PipelineStats()

        logger.info("=" * 60)
        logger.info("Agent 2 pipeline starting")
        logger.info("=" * 60)

        # Validate and extract inputs from Agent 1
        inputs = self._extract_inputs(agent1_result)
        logger.info(
            f"Inputs: {inputs['n_degs']} DEGs | "
            f"{inputs['n_samples']} samples | "
            f"noise={inputs['noise_level']} | "
            f"query='{inputs['original_query'][:60]}...'"
        )

        # Select CARNIVAL parameters
        params, stats.param_select_seconds = self._select_parameters(inputs)
        stats.tokens_used        = params.get("tokens_used", 0)
        stats.parameter_method   = params.get("selected_by", "unknown")

        # Store selected params into inputs for downstream steps
        inputs["pkn_filter"] = params.get("pkn_filter", "full")
        inputs["top_n_tfs"]  = params.get("top_n_tfs", DEFAULT_TOP_N_TFS)

        # Infer TF activities
        tf_scores, stats.tf_infer_seconds = self._infer_tf_activities(
            inputs, top_n_tfs=inputs["top_n_tfs"]
        )

        # Score pathways (PROGENy + MLM) — soft objective for CARNIVAL
        progeny_gene_scores = None
        expr_path = inputs.get("expression_matrix_path")
        if expr_path and Path(expr_path).exists():
            try:
                from core.pathway_scoring import score_pathways, get_progeny_gene_scores
                meta_path = inputs.get("metadata_path")
                disease_hint = inputs.get("extracted_entities", {}).get("disease")
                condition_a, condition_b = self._detect_conditions(meta_path, disease_hint=disease_hint)

                # Enrich agent1_context with detected disease/condition names
                # Only fill in when Agent 1 left these as generic/empty
                if condition_a and not inputs["agent1_context"].get("disease"):
                    inputs["agent1_context"]["disease"] = condition_a
                current_comp = inputs["agent1_context"].get("comparison", "")
                if condition_a and condition_b and current_comp in ("", "disease vs control"):
                    inputs["agent1_context"]["comparison"] = f"{condition_a} vs {condition_b}"

                pw_result = score_pathways(
                    expression_matrix_path=expr_path,
                    output_dir=str(self.output_dir),
                    metadata_path=meta_path,
                    condition_a=condition_a,
                    condition_b=condition_b,
                )
                # Compute per-gene importance for CARNIVAL soft objective
                contrast_path = self.output_dir / "contrast_tstats.csv"
                progeny_gene_scores = get_progeny_gene_scores(
                    significant_pathways=pw_result["significant_pathways"],
                    contrast_tstats_path=str(contrast_path) if contrast_path.exists() else None,
                    tf_gene_set=set(tf_scores.keys()),
                )
                logger.info(
                    f"Step C2: {pw_result['n_pathways']} pathways scored, "
                    f"{pw_result['n_significant']} significant, "
                    f"{len(progeny_gene_scores)} genes for soft objective"
                )
            except Exception as e:
                logger.warning(f"Step C2: Pathway scoring failed (non-fatal): {e}")
                progeny_gene_scores = None
        # progeny_gene_scores = None  # TOGGLE: uncomment this line to disable PROGENy for testing
        # logger.info("PROGENy soft objective: DISABLED (progeny_gene_scores set to None)")

        # Build Prior Knowledge Network (uses inferred TFs as anchors)
        inputs["tf_scores"] = tf_scores
        pkn_df, stats.pkn_build_seconds = self._build_pkn(inputs)

        # Load significant DEG gene names for _PERT filtering + node classification
        deg_gene_set = None
        deg_path = inputs.get("deg_file_path")
        if deg_path and Path(deg_path).exists():
            try:
                deg_df = pd.read_csv(deg_path)
                gene_col = next(
                    (c for c in deg_df.columns if c.lower() in ("gene", "gene_name", "gene_symbol")),
                    None,
                )
                if gene_col:
                    if 'significant' in deg_df.columns:
                        sig_df = deg_df[deg_df['significant'] == True]
                        deg_gene_set = set(sig_df[gene_col].dropna().astype(str))
                        logger.info(
                            f"Loaded {len(deg_gene_set)} significant DEGs "
                            f"(of {len(deg_df)} total genes) for _PERT filtering"
                        )
                    else:
                        deg_gene_set = set(deg_df[gene_col].dropna().astype(str))
                        logger.info(f"Loaded {len(deg_gene_set)} DEG genes (no 'significant' column)")
            except Exception as e:
                logger.warning(f"Could not load DEG genes: {e}")

        # --- Cache check ---
        expr_path = inputs.get("expression_matrix_path", "")
        _cache_key = None
        if expr_path and Path(expr_path).exists():
            try:
                _cache_key = _cache.make_key(
                    expression_path=expr_path,
                    beta_weight=float(params.get("carnival_beta", DEFAULT_BETA)),
                    pkn_filter=params.get("pkn_filter", "full"),
                    penalize=params.get("penalize", "edges"),
                    has_progeny=progeny_gene_scores is not None,
                )
                cached_result = _cache.load(_cache_key)
                if cached_result is not None:
                    logger.info("[Cache] Returning cached CARNIVAL result — skipping ILP solve")
                    cached_result.metadata["from_cache"] = True
                    stats.total_seconds = time.time() - t_total
                    return cached_result
            except Exception as e:
                logger.warning(f"[Cache] Cache check failed (non-fatal): {e}")
                _cache_key = None
        # --- End cache check ---

        # Run CARNIVAL and package results
        agent2_result, stats.carnival_seconds = self._run_and_package(
            pkn_df=pkn_df,
            tf_scores=tf_scores,
            params=params,
            agent1_context=inputs["agent1_context"],
            sample_name=inputs["sample_name"],
            deg_gene_set=deg_gene_set,
            penalize=params.get("penalize", "edges"),
            progeny_gene_scores=progeny_gene_scores,
        )

        # Network confidence assessment (LLM + fallback)
        try:
            from llm.network_assessor import assess_network_confidence
            confidence = assess_network_confidence(
                G=agent2_result.carnival_network,
                metadata=agent2_result.metadata,
                api_key=self.api_key,
            )
            agent2_result.network_confidence = confidence
            logger.info(
                f"Network confidence: {confidence.get('confidence', 'unknown')} "
                f"(TF recovery={confidence.get('metrics', {}).get('tf_recovery_pct', '?')}%, "
                f"DEG recovery={confidence.get('metrics', {}).get('deg_recovery_pct', '?')}%)"
            )
        except Exception as e:
            logger.warning(f"Network confidence assessment failed (non-fatal): {e}")
            agent2_result.network_confidence = {}

        stats.total_seconds = time.time() - t_total
        stats.log_summary()

        # Attach pipeline stats to result metadata
        agent2_result.metadata["pipeline_stats"] = {
            "pkn_build_seconds":    round(stats.pkn_build_seconds, 2),
            "tf_infer_seconds":     round(stats.tf_infer_seconds, 2),
            "param_select_seconds": round(stats.param_select_seconds, 2),
            "carnival_seconds":     round(stats.carnival_seconds, 2),
            "total_seconds":        round(stats.total_seconds, 2),
            "parameter_method":     stats.parameter_method,
        }
        agent2_result.metadata["from_cache"] = False

        # --- Cache save ---
        if _cache_key is not None:
            try:
                _cache.save(
                    key=_cache_key,
                    result=agent2_result,
                    expression_path=expr_path,
                    beta_weight=float(params.get("carnival_beta", DEFAULT_BETA)),
                    pkn_filter=params.get("pkn_filter", "full"),
                    penalize=params.get("penalize", "edges"),
                    has_progeny=progeny_gene_scores is not None,
                )
            except Exception as e:
                logger.warning(f"[Cache] Cache save failed (non-fatal): {e}")
        # --- End cache save ---

        logger.info(
            f"Agent 2 complete in {stats.total_seconds:.1f}s — "
            f"{agent2_result.n_active_nodes} active nodes, "
            f"{agent2_result.n_active_edges} active edges"
        )

        return agent2_result

    # -------------------------------------------------------------------------
    # SECONDARY ENTRY POINT: standalone testing (bypass Agent 1) Used for testing only
    # -------------------------------------------------------------------------

    def run_from_files(
        self,
        deg_path:    str,
        pkn_path:    str,
        sample_name: str = "sample",
        beta:        float = DEFAULT_BETA,
        top_n_tfs:   int   = DEFAULT_TOP_N_TFS,
        penalize:    str   = "edges",
    ) -> Agent2Result:
        """
        Run pipeline directly from CSV files, bypassing Agent 1.

        Used for:
        - Testing corneto_runner + output_packager in isolation
        - Re-running CARNIVAL on pre-computed inputs
        - Debugging specific samples

        Args:
            deg_path:    Path to measurements CSV with columns: TF, NES
            pkn_path:    Path to network CSV with columns: source, interaction, target
            sample_name: Label for this sample (used in output filenames)
            beta:        CARNIVAL beta weight (overrides parameter selector)
            top_n_tfs:   Number of top TFs to use
            penalize:    ILP penalty target: 'nodes' (Liu 2019), 'edges', or 'both'

        Returns:
            Agent2Result ready for Agent 3.
        """
        logger.info(f"Running from files (sample={sample_name})")

        pkn_df    = pd.read_csv(pkn_path)
        meas_df   = pd.read_csv(deg_path)
        tf_scores = dict(zip(meas_df["TF"], meas_df["NES"]))

        # Apply top_n_tfs filter
        if len(tf_scores) > top_n_tfs:
            tf_scores = dict(
                sorted(tf_scores.items(), key=lambda x: abs(x[1]), reverse=True)
                [:top_n_tfs]
            )
            logger.info(f"Using top {top_n_tfs} TFs by |NES|")

        carnival_result = run_carnival(
            pkn_df=pkn_df,
            tf_scores=tf_scores,
            beta_weight=beta,
            solver=DEFAULT_SOLVER,
            sample_name=sample_name,
            verbosity=0,
            penalize=penalize,
        )

        return package_results(
            carnival_result=carnival_result,
            agent1_context={"sample_name": sample_name},
            output_dir=str(self.output_dir),
            tf_gene_set=set(tf_scores.keys()),
        )

    # -------------------------------------------------------------------------
    # PIPELINE STEPS (implemented as separate methods for clarity and testability)
    # -------------------------------------------------------------------------

    def _extract_inputs(self, agent1_result) -> dict:
        """
        Extract and validate all inputs from Agent1Result.

        Returns a flat dict so downstream steps have a clean interface
        and don't need to navigate the Pydantic model directly.
        """
        qr  = agent1_result.quality_report
        deg = agent1_result.deg_results
        ent = agent1_result.extracted_entities
        exp = agent1_result.explanation

        if qr is None or deg is None:
            raise ValueError(
                "Agent1Result is missing quality_report or deg_results. "
                "Ensure Agent 1 completed successfully before calling Agent 2. "
                f"Agent 1 status was: {agent1_result.status}"
            )

        # Extract entities safely — schema fields may be None, review
        disease   = getattr(ent, "disease", None)              if ent else None
        cell_type = getattr(ent, "cell_type_mentioned", None) if ent else None
        tf_list   = getattr(ent, "genes_of_interest", [])     if ent else []

        # Build context dict for parameter selector and output packager
        agent1_context = {
            "comparison":     getattr(deg, "comparison", "disease vs control"),
            "data_type":      str(getattr(qr,  "data_type",   "unknown")),
            "n_degs":         getattr(deg, "n_significant", 0),
            "original_query": agent1_result.original_query or "",
            "disease":        disease,
            "cell_type":      cell_type,
        }

        # Build sample name from disease + cell type when available
        sample_parts = [
            disease.lower().replace(" ", "_")   if disease   else None,
            cell_type.lower().replace(" ", "_") if cell_type else None,
        ]
        sample_name = "_".join(p for p in sample_parts if p) or "sample"

        # Sanitize for Windows filenames — remove chars invalid in NTFS paths
        import re as _re
        sample_name = _re.sub(r'[<>:"/\\|?*]', '', sample_name)

        return {
            # For PKN builder
            "tf_list":          tf_list,
            "expression_matrix_path": self._resolve_path(agent1_result.expression_matrix_path),
            # "deg_file_path":          self._resolve_path(agent1_result.deg_results.deg_file_path),  # we'll confirm this field name below
            "deg_file_path": self._resolve_path(agent1_result.deg_results.deg_file_path),
            "metadata_path": self._resolve_path(getattr(agent1_result, "metadata_path", None)),

            # For parameter selector
            "noise_level":      str(getattr(qr, "noise_level",   "medium")),
            "n_samples":        int(getattr(qr, "n_samples",      8)),
            "n_degs":           int(getattr(deg, "n_significant", 0)),
            "n_genes":          int(getattr(qr, "n_genes",        0)),
            "original_query":   agent1_result.original_query or "",
            "extracted_entities": {
                "disease":   disease,
                "cell_type": cell_type,
                "transcription_factors": tf_list,
            },
            "context_for_agent2": (
                getattr(exp, "recommendations_for_agent2", None) if exp else None
            ),

            # For output packager, did this cause issues ? we can remove if not needed
            "agent1_context":   agent1_context,
            "sample_name":      sample_name,
        }
    
    def _resolve_path(self, raw_path: str | None) -> str | None:
        """
        Resolve a potentially-relative path from Agent 1 to an absolute path.

        Agent 1 writes relative paths (e.g. 'agent1_output/file.csv').
        These are relative to Agent 1's working directory, not Agent 2's.
        We try three locations in order:
            1. Already absolute → use as-is
            2. Relative to project root (Agent_1/ and Agent_2/ are siblings)
            3. Relative to cwd (fallback)
        """
        if raw_path is None:
            return None

        p = Path(raw_path)

        # Already absolute — use as-is but warn if missing
        if p.is_absolute():
            if not p.exists():
                logger.warning(f"_resolve_path: absolute path does not exist: {p}")
            return str(p)

        # Try: project_root / relative_path
        # e.g. "agent1_output/file.csv" → "Thesis Code/Agent_1/agent1_output/file.csv"
        candidate = Path(__file__).parent.parent / "Agent_1" / p
        if candidate.exists():
            return str(candidate)

        # Fallback: relative to cwd
        candidate = Path.cwd() / p
        if candidate.exists():
            return str(candidate)

        # No candidate found — warn and return best guess so caller gets a clear error
        resolved = str(Path(__file__).parent.parent / "Agent_1" / p)
        logger.warning(f"_resolve_path: '{raw_path}' not found at any search location. "
                       f"Returning best guess: {resolved}")
        return resolved

    def _build_pkn(self, inputs: dict) -> tuple[pd.DataFrame, float]:
        """
        Build TF-anchored Prior Knowledge Network from OmniPath.

        Returns:
            (pkn_df, elapsed_seconds)
        """
        t0 = time.time()
        logger.info("Step B: Building Prior Knowledge Network...")

        # Merge query TFs + CollecTRI inferred TFs
        query_tfs = inputs.get("tf_list") or []
        inferred_tfs = list(inputs.get("tf_scores", {}).keys())
        anchor_genes = list(set(query_tfs + inferred_tfs)) or None

        pkn_df, _network_path, _pkn_stats = build_pkn(
            anchor_genes=anchor_genes,
            output_dir=str(self.output_dir),
            pkn_filter=inputs.get("pkn_filter", "full"),
        )

        elapsed = time.time() - t0
        logger.info(f"PKN built: {len(pkn_df)} edges in {elapsed:.1f}s")
        return pkn_df, elapsed

    def _infer_tf_activities(self, inputs: dict, top_n_tfs: int = DEFAULT_TOP_N_TFS) -> tuple[dict, float]:
        """
        Infer transcription factor activity scores from expression data.

        Prefers contrast-based VIPER (expression matrix + metadata), which uses the
        full per-sample expression profile and a disease-vs-control contrast for
        higher-confidence scores. Falls back to DEG weighted mean when only
        precomputed DEG results are available (less accurate — see warning in logs).

        Regulon selection:
            DoRothEA  → microarray (curated, fractional confidence weights)
            CollecTRI → RNA-seq / scRNA-seq (larger coverage, sign-based)

        Returns:
            (tf_scores dict {gene: mean_NES}, elapsed_seconds)

        Raises:
            FileNotFoundError: If neither expression matrix nor DEG file is found.
        """
        t0 = time.time()
        logger.info("Step C: Inferring TF activities...")

        deg_path  = inputs.get("deg_file_path")
        expr_path = inputs.get("expression_matrix_path")
        meta_path = inputs.get("metadata_path")

        # Detect condition labels from metadata (for contrast-based VIPER)
        disease_hint = inputs.get("extracted_entities", {}).get("disease")
        condition_a, condition_b = self._detect_conditions(meta_path, disease_hint=disease_hint)

        # Prefer contrast-based VIPER (expression matrix + metadata).
        # Fall back to DEG weighted mean when only precomputed DEGs are available.
        if expr_path and Path(expr_path).exists():
            logger.info("Using contrast-based VIPER for TF inference from expression matrix")
            from core.tf_activity import infer_tf_activity
            # DoRothEA for microarray (curated, fractional weights);
            # CollecTRI for RNA-seq/scRNA-seq (larger coverage, sign-based)
            _dt = inputs.get("agent1_context", {}).get("data_type", "")
            _regulon = "dorothea" if "MICROARRAY" in _dt.upper() else "collectri"
            logger.info(f"Regulon selection: data_type={_dt} → regulon_db={_regulon}")
            tf_scores_df, measurements_path, stats = infer_tf_activity(
                expression_matrix_path=expr_path,
                deg_file_path=deg_path,
                output_dir=str(self.output_dir),
                top_n_tfs=top_n_tfs,
                metadata_path=meta_path,
                condition_a=condition_a,
                condition_b=condition_b,
                regulon_db=_regulon,
            )
        elif deg_path and Path(deg_path).exists():
            logger.warning(
                "No expression matrix available — falling back to DEG weighted mean "
                "for TF inference. Results may be less accurate than ULM, which uses "
                "the full expression profile. For best results, provide an expression matrix."
            )
            from core.tf_activity import infer_tf_activity_from_degs
            tf_scores_df, measurements_path, stats = infer_tf_activity_from_degs(
                deg_file_path=deg_path,
                output_dir=str(self.output_dir),
                top_n_tfs=top_n_tfs,
            )
        else:
            raise FileNotFoundError(
                "Neither expression_matrix_path nor deg_file_path found. "
                "Agent 2 needs at least one to infer TF activities."
            )

        nes_means = tf_scores_df.mean(axis=0)
        tf_scores  = {tf: float(nes_means[tf]) for tf in nes_means.index}

        elapsed = time.time() - t0
        logger.info(
            f"TF activities: {len(tf_scores)} TFs in {elapsed:.1f}s | "
            f"method={stats['method_used']} | top={stats['n_tfs_selected']}"
        )
        return tf_scores, elapsed

    @staticmethod
    def _detect_conditions(
        metadata_path: str | None,
        disease_hint: str | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Read metadata CSV and detect disease vs control condition labels.

        Returns (condition_a, condition_b) where condition_a is the disease
        and condition_b is the control. Identifies control by case-insensitive
        match against common control labels.

        Returns (None, None) if conditions cannot be determined — caller
        should fall back to per-sample ULM (no contrast possible).

        Args:
            metadata_path: Path to metadata CSV with 'condition' column.
            disease_hint: Disease name from Agent 1 extracted_entities.
                Used as condition_a when metadata is missing/unparseable,
                paired with "Control" as condition_b.
        """
        CONTROL_LABELS = {"control", "ctrl", "normal", "healthy", "wt", "wildtype", "wild-type"}

        if not metadata_path or not Path(metadata_path).exists():
            if disease_hint:
                logger.warning(
                    f"No metadata file — using Agent 1 disease hint: "
                    f"'{disease_hint}' vs 'Control'"
                )
                return disease_hint, "Control"
            logger.warning(
                "No metadata file and no disease hint from Agent 1. "
                "Cannot determine conditions — will fall back to per-sample ULM."
            )
            return None, None

        meta = pd.read_csv(metadata_path)
        meta.columns = [c.strip().lower() for c in meta.columns]

        if "condition" not in meta.columns:
            if disease_hint:
                logger.warning(
                    f"Metadata has no 'condition' column — using Agent 1 disease hint: "
                    f"'{disease_hint}' vs 'Control'"
                )
                return disease_hint, "Control"
            logger.warning(
                "Metadata has no 'condition' column and no disease hint from Agent 1. "
                "Cannot determine conditions — will fall back to per-sample ULM."
            )
            return None, None

        conditions = meta["condition"].unique().tolist()
        if len(conditions) != 2:
            logger.warning(
                f"Expected 2 conditions, found {len(conditions)}: {conditions}. "
                f"Using first two."
            )
            conditions = conditions[:2]

        # Identify which is control
        ctrl = [c for c in conditions if c.lower() in CONTROL_LABELS]
        if ctrl:
            condition_b = ctrl[0]
            condition_a = [c for c in conditions if c != condition_b][0]
        else:
            # No obvious control — use order as-is, second is assumed control
            condition_a, condition_b = conditions[0], conditions[1]
            logger.warning(
                f"Could not identify control condition from {conditions}. "
                f"Using '{condition_a}' as disease, '{condition_b}' as control."
            )

        logger.info(f"Detected conditions: disease='{condition_a}', control='{condition_b}'")
        return condition_a, condition_b

#=============================================================================

    def _select_parameters(self, inputs: dict) -> tuple[dict, float]:
        """
        Step D: Select CARNIVAL parameters (deterministic rules + LLM refinement).

        Returns:
            (params dict, elapsed_seconds)
        """
        from llm.parameter_selector import select_parameters

        t0 = time.time()
        logger.info("Step D: Selecting CARNIVAL parameters...")

        params = select_parameters(
            noise_level=inputs["noise_level"],
            n_samples=inputs["n_samples"],
            n_degs=inputs["n_degs"],
            n_genes=inputs["n_genes"],
            original_query=inputs["original_query"],
            extracted_entities=inputs["extracted_entities"],
            context_for_agent2=inputs["context_for_agent2"],
            use_llm=self.use_llm and bool(self.api_key),
        )

        elapsed = time.time() - t0
        logger.info(
            f"Parameters selected ({params['selected_by']}): "
            f"beta={params['carnival_beta']}, "
            f"top_n_tfs={params['top_n_tfs']}, "
            f"solver={params.get('solver', 'auto')}"
        )
        return params, elapsed

    def _run_and_package(
        self,
        pkn_df:         pd.DataFrame,
        tf_scores:      dict,
        params:         dict,
        agent1_context: dict,
        sample_name:    str,
        deg_gene_set:   set | None = None,
        penalize:       str = "edges",
        progeny_gene_scores: dict | None = None,
    ) -> tuple[Agent2Result, float]:
        """
        Steps: Run CARNIVAL and package results for Agent 3.

        Handles vocabulary translation between parameter_selector and corneto_runner.

        Returns:
            (Agent2Result, elapsed_seconds)
        """
        t0 = time.time()

        # --- Translate parameter_selector → corneto_runner vocabulary ---
        beta   = float(params.get("carnival_beta", DEFAULT_BETA))
        solver = _SOLVER_MAP.get(
            str(params.get("solver", "auto")),
            DEFAULT_SOLVER,
        )
        top_n  = int(params.get("top_n_tfs", DEFAULT_TOP_N_TFS))

        # Save unfiltered scores for potential retry (may need more TFs)
        all_tf_scores = tf_scores

        # Apply top_n_tfs filter — keep highest |NES| TFs
        if len(tf_scores) > top_n:
            tf_scores = dict(
                sorted(tf_scores.items(), key=lambda x: abs(x[1]), reverse=True)
                [:top_n]
            )
            logger.info(f"Filtered to top {top_n} TFs by |NES|")

        logger.info(
            f"Step E: Running CARNIVAL "
            f"(beta={beta}, solver={solver}, penalize={penalize}, n_tfs={len(tf_scores)})..."
        )

        carnival_result: CARNIVALResult = run_carnival(
            pkn_df=pkn_df,
            tf_scores=tf_scores,
            beta_weight=beta,
            solver=solver,
            sample_name=sample_name,
            verbosity=0,
            deg_gene_set=deg_gene_set,
            penalize=penalize,
            progeny_gene_scores=progeny_gene_scores,
        )

        # Retry once if CARNIVAL returned an empty network.
        # Retry adjusts beta/top_n_tfs but reuses the same PKN. This is fine
        # because pkn_filter defaults to "full" (~70K edges) — the full search
        # space is already loaded. Lower beta / more TFs just means CARNIVAL
        # selects more edges from the existing PKN, no rebuild needed.
        # PKN rebuild would only matter if pkn_filter changed (e.g. "full" → "2hop"),
        # but lets make 2-hop not feasible since it proved unevetful in testing and adds complexity.
        if len(carnival_result.active_edges_df) == 0:
            logger.warning(
                f"CARNIVAL returned empty network (beta={beta}, n_tfs={len(tf_scores)}). "
                f"Attempting retry with adjusted parameters."
            )
            from llm.parameter_selector import refine_params_after_failure
            retry_params = refine_params_after_failure(
                current_params=params,
                failure_type="empty_network",
                attempt_number=1,
                n_tfs=len(tf_scores),
                n_edges=len(pkn_df),
            )
            retry_beta = float(retry_params.get("carnival_beta", DEFAULT_BETA))
            retry_top_n = int(retry_params.get("top_n_tfs", top_n))
            # Re-filter from ALL TFs (not the already-filtered set)
            retry_tf_scores = dict(
                sorted(all_tf_scores.items(), key=lambda x: abs(x[1]), reverse=True)
                [:retry_top_n]
            )
            logger.info(
                f"Retry: beta {beta} → {retry_beta}, "
                f"top_n_tfs {top_n} → {retry_top_n}"
            )
            carnival_result = run_carnival(
                pkn_df=pkn_df,
                tf_scores=retry_tf_scores,
                beta_weight=retry_beta,
                solver=solver,
                sample_name=sample_name,
                verbosity=0,
                deg_gene_set=deg_gene_set,
                penalize=penalize,
                progeny_gene_scores=progeny_gene_scores,
            )
            if len(carnival_result.active_edges_df) == 0:
                logger.warning("Retry also returned empty network. Returning result as-is.")
            # Update tf_scores to match what CARNIVAL actually used, so
            # _classify_node() labels all retry TFs correctly (not as intermediates)
            tf_scores = retry_tf_scores

        logger.info("Step F: Packaging results for Agent 3...")

        agent2_result: Agent2Result = package_results(
            carnival_result=carnival_result,
            # agent1_context carries both Agent 1 outputs (comparison, data_type, etc.)
            # and Agent 2's parameter selection audit trail (rationale, method)
            agent1_context={
                **agent1_context,
                "parameter_rationale": params.get("rationale", ""),
                "parameter_method":    params.get("selected_by", "unknown"),
            },
            output_dir=str(self.output_dir),
            tf_gene_set=set(tf_scores.keys()),
            deg_gene_set=deg_gene_set,
        )

        elapsed = time.time() - t0
        return agent2_result, elapsed


# =============================================================================
# CONVENIENCE FUNCTION — for callers that just run Agent 2, no need to manage pipeline object
# =============================================================================

def run_agent2(
    agent1_result,
    api_key:    Optional[str] = None,
    use_llm:    bool = True,
    output_dir: str  = OUTPUT_DIR,
) -> Agent2Result:
    """
    Single-call convenience wrapper for the Agent 2 pipeline.

    Equivalent to:
        pipeline = Agent2Pipeline(api_key, use_llm, output_dir)
        return pipeline.run(agent1_result)

    Use Agent2Pipeline directly when running multiple samples
    (avoids re-initializing on each call).
    """
    pipeline = Agent2Pipeline(
        api_key=api_key,
        use_llm=use_llm,
        output_dir=output_dir,
    )
    return pipeline.run(agent1_result)


# =============================================================================
# DIAGNOSTIC — run from command line to verify full pipeline
# =============================================================================

if __name__ == "__main__":
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    _root = Path(__file__).parent

    print("=" * 60)
    print("Agent 2 Diagnostic (standalone mode)")
    print("=" * 60)

    pipeline = Agent2Pipeline(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        use_llm=False,   # deterministic only for diagnostic
        output_dir=str(_root / "agent2_output"),
    )

    result = pipeline.run_from_files(
        deg_path=str(_root / "carnival_test" / "input" / "measurements.csv"),
        pkn_path=str(_root / "carnival_test" / "input" / "network.csv"),
        sample_name="ipf_at2_diagnostic",
    )

    print(f"\n{'=' * 60}")
    print("AGENT 2 RESULT")
    print("=" * 60)
    print(f"carnival_network : {type(result.carnival_network).__name__} "
          f"({result.carnival_network.number_of_nodes()} nodes, "
          f"{result.carnival_network.number_of_edges()} edges)")
    print(f"Active nodes     : {result.n_active_nodes}")
    print(f"Active edges     : {result.n_active_edges}")
    print(f"Activated        : {result.activated_nodes[:5]}")
    print(f"Repressed        : {result.repressed_nodes[:5]}")
    print(f"Warnings         : {result.warnings or 'None'}")
    print(f"\n--- Summary for Agent 3 ---")
    print(result.summary)
    print(f"\n--- Pipeline Stats ---")
    stats = result.metadata.get("pipeline_stats", {})
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n Agent 2 diagnostic complete")
