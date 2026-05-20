"""
carnival_cache.py
-----------------
Dataset-fingerprint-based cache for CARNIVAL ILP results.

Motivation
----------
CARNIVAL via CORNETO + HiGHS can take 3-15 minutes per run. Re-running
identical configurations (same dataset, same parameters) wastes time
during demos and iterative analysis. This module caches the full
Agent2Result to disk and restores it instantly on repeat runs.

Cache Key
---------
A SHA-256 hash of the following, concatenated:
  - expression_matrix MD5 content hash
  - beta_weight (float, rounded to 3dp)
  - pkn_filter ('full')
  - penalize ('nodes' | 'edges' | 'both')
  - has_progeny (bool — whether PROGENy soft objective was applied)

This means:
  Same dataset + params -> cache hit, instant result
  File contents changed -> cache miss, re-runs automatically
  Different beta -> cache miss, fresh solve
  Uses MD5 of file contents (~5MB, negligible vs 300s solve)

Cache Storage
-------------
Default location: Agent_2/cache/
  {hash[:32]}.pkl    -- pickled Agent2Result (networkx graph + dataframes)
  {hash[:32]}.json   -- human-readable cache manifest (for inspection)
"""

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default cache directory — relative to this file's location (Agent_2/core/)
# Resolves to: Agent_2/cache/
# ---------------------------------------------------------------------------
_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "cache"


class CARNIVALCache:
    """
    Disk-based cache for Agent2Result objects, keyed by dataset fingerprint.

    Thread safety: Not thread-safe. Streamlit 1.x is single-threaded
    per session so this is fine for thesis use.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Cache] Initialized. Cache dir: {self.cache_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def make_key(
        self,
        expression_path: str,
        beta_weight: float,
        pkn_filter: str = "full",
        penalize: str = "edges",
        has_progeny: bool = False,
    ) -> str:
        """
        Compute a SHA-256 cache key from dataset fingerprint + parameters.

        Uses file mtime + size rather than file contents for speed.
        Cache automatically invalidates if the file is modified on disk.

        Parameters
        ----------
        expression_path : str
            Absolute path to expression_matrix_clean.csv
        beta_weight : float
            CARNIVAL beta (regularization weight)
        pkn_filter : str
            PKN filter applied ('full' or '2hop')
        penalize : str
            CARNIVAL penalty mode ('nodes', 'edges', or 'both')
        has_progeny : bool
            Whether PROGENy soft objective was applied

        Returns
        -------
        str
            64-character hex SHA-256 hash
        """
        path = Path(expression_path).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"[Cache] Expression matrix not found: {path}\n"
                f"Cannot compute cache key without the file."
            )

        # Hash file contents (not mtime) so re-running Agent 1 on
        # identical data still hits the cache.
        content_hash = hashlib.md5(path.read_bytes()).hexdigest()
        fingerprint = "|".join([
            content_hash,
            f"{beta_weight:.3f}",
            pkn_filter.lower(),
            penalize.lower(),
            str(has_progeny).lower(),
        ])

        key = hashlib.sha256(fingerprint.encode()).hexdigest()
        logger.debug(f"[Cache] Key computed: {key[:16]}... for {path.name} "
                     f"(beta={beta_weight}, pkn={pkn_filter}, progeny={has_progeny})")
        return key

    def load(self, key: str) -> Optional[Any]:
        """
        Load a cached Agent2Result if it exists.

        Returns None on any failure (missing file, corrupt pickle,
        version mismatch) -- caller should re-run CARNIVAL.
        """
        pkl_path = self._pkl_path(key)

        if not pkl_path.exists():
            logger.info(f"[Cache] MISS -- no cached result for {key[:16]}")
            return None

        try:
            with open(pkl_path, "rb") as f:
                result = pickle.load(f)

            # Load manifest for logging
            manifest = self._load_manifest(key)
            if manifest:
                logger.info(
                    f"[Cache] HIT -- {key[:16]} | "
                    f"nodes={manifest.get('n_active_nodes')}, "
                    f"edges={manifest.get('n_active_edges')}, "
                    f"cached_at={manifest.get('cached_at')}"
                )
            else:
                logger.info(f"[Cache] HIT -- {key[:16]}")

            return result

        except (pickle.UnpicklingError, EOFError, AttributeError) as e:
            logger.warning(
                f"[Cache] Corrupt cache entry {key[:16]}: {e}. "
                f"Deleting and re-running CARNIVAL."
            )
            self._delete_entry(key)
            return None

        except Exception as e:
            logger.warning(
                f"[Cache] Failed to load {key[:16]}: {e}. "
                f"Re-running CARNIVAL."
            )
            return None

    def save(
        self,
        key: str,
        result: Any,
        expression_path: Optional[str] = None,
        beta_weight: Optional[float] = None,
        pkn_filter: Optional[str] = None,
        penalize: Optional[str] = None,
        has_progeny: Optional[bool] = None,
    ) -> bool:
        """
        Persist an Agent2Result to disk.

        Returns True if saved successfully, False on failure.
        Never raises — cache failure must not break the pipeline.
        """
        pkl_path = self._pkl_path(key)

        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Write human-readable manifest
            self._write_manifest(
                key=key,
                result=result,
                expression_path=expression_path,
                beta_weight=beta_weight,
                pkn_filter=pkn_filter,
                penalize=penalize,
                has_progeny=has_progeny,
            )

            size_mb = pkl_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"[Cache] Saved {key[:16]} -> {pkl_path.name} ({size_mb:.1f} MB)"
            )
            return True

        except Exception as e:
            logger.error(
                f"[Cache] Failed to save {key[:16]}: {e}. "
                f"Pipeline continues without caching."
            )
            return False

    def invalidate(self, key: str) -> bool:
        """Manually delete a cache entry. Returns True if existed."""
        if self._pkl_path(key).exists():
            self._delete_entry(key)
            logger.info(f"[Cache] Invalidated {key[:16]}")
            return True
        return False

    def clear_all(self) -> int:
        """Delete all cache entries. Returns count deleted."""
        count = 0
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
            count += 1
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        logger.info(f"[Cache] Cleared {count} entries from {self.cache_dir}")
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pkl_path(self, key: str) -> Path:
        return self.cache_dir / f"{key[:32]}.pkl"

    def _manifest_path(self, key: str) -> Path:
        return self.cache_dir / f"{key[:32]}.json"

    def _delete_entry(self, key: str) -> None:
        for path in [self._pkl_path(key), self._manifest_path(key)]:
            if path.exists():
                path.unlink()

    def _load_manifest(self, key: str) -> Optional[dict]:
        path = self._manifest_path(key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def _write_manifest(
        self,
        key: str,
        result: Any,
        expression_path: Optional[str],
        beta_weight: Optional[float],
        pkn_filter: Optional[str],
        penalize: Optional[str],
        has_progeny: Optional[bool],
    ) -> None:
        """Write a JSON manifest alongside the pickle for human inspection."""
        try:
            n_active_nodes = 0
            n_active_edges = 0
            solve_time = 0.0

            if hasattr(result, "metadata") and isinstance(result.metadata, dict):
                n_active_nodes = result.metadata.get("n_active_nodes", 0)
                n_active_edges = result.metadata.get("n_active_edges", 0)
                solve_time = result.metadata.get("solve_time_seconds", 0.0)
            if hasattr(result, "n_active_nodes"):
                n_active_nodes = result.n_active_nodes
            if hasattr(result, "n_active_edges"):
                n_active_edges = result.n_active_edges

            # File stats
            expr_path_str = str(expression_path) if expression_path else "unknown"
            expr_mtime = 0.0
            expr_size = 0
            if expression_path:
                try:
                    stat = Path(expression_path).stat()
                    expr_mtime = stat.st_mtime
                    expr_size = stat.st_size
                except Exception:
                    pass

            manifest = {
                "cache_key": key,
                "cache_key_short": key[:32],
                "expression_path": expr_path_str,
                "expression_mtime": expr_mtime,
                "expression_size_bytes": expr_size,
                "beta_weight": beta_weight,
                "pkn_filter": pkn_filter,
                "penalize": penalize,
                "has_progeny": has_progeny,
                "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_active_nodes": n_active_nodes,
                "n_active_edges": n_active_edges,
                "solve_time_s": solve_time,
            }

            with open(self._manifest_path(key), "w") as f:
                json.dump(manifest, f, indent=2)

        except Exception as e:
            logger.debug(f"[Cache] Manifest write failed (non-critical): {e}")
