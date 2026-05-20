"""
Agent 1 Configuration
======================
Centralized configuration using environment variables and relative paths.
Docker-ready: no hardcoded paths, all settings overridable via env vars.

Usage:
    from agent1.config import get_config
    
    config = get_config()
    print(config.output_dir)  # relative path
    print(config.anthropic_api_key)  # from env var
    
Environment Variables:
    ANTHROPIC_API_KEY       - Required for LLM calls (standard name)
    AGENT1_ANTHROPIC_API_KEY - Alternative (prefixed) API key
    AGENT1_OUTPUT_DIR       - Override default output directory
    AGENT1_PADJ_THRESHOLD   - Override default p-value threshold
    AGENT1_LOG2FC_THRESHOLD - Override default fold-change threshold
    AGENT1_LLM_MODEL        - Override default Claude model
    
    Any field in Agent1Config can be overridden with the AGENT1_ prefix.
    See schemas_agent1.py Agent1Config for the full list.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

from agent1.schemas_agent1 import Agent1Config

logger = logging.getLogger(__name__)


def _resolve_api_key() -> str | None:
    """
    Resolve Anthropic API key from environment.
    Checks ANTHROPIC_API_KEY (standard) and AGENT1_ANTHROPIC_API_KEY (prefixed).
    
    Why both? ANTHROPIC_API_KEY is the standard name used by the Anthropic SDK
    and other tools. AGENT1_ANTHROPIC_API_KEY follows our env_prefix convention.
    We check both so it Just Works regardless of which one the user sets.
    """
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("AGENT1_ANTHROPIC_API_KEY")
    if not key:
        logger.warning(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable. "
            "LLM features (entity extraction, explanation) will be disabled."
        )
    return key


@lru_cache(maxsize=1)
def get_config() -> Agent1Config:
    """
    Load Agent 1 configuration.
    
    How it works:
        1. BaseSettings auto-reads all AGENT1_* environment variables
        2. We only manually handle anthropic_api_key (dual env var fallback)
        3. Everything else comes from env vars or schema defaults automatically
    
    Priority (highest wins):
        1. Kwargs passed here (only anthropic_api_key)
        2. AGENT1_* environment variables (handled by BaseSettings)
        3. Default values defined in Agent1Config schema
    
    Cached after first call (singleton pattern).
    
    Examples:
        # Minimal — uses all defaults, reads env vars automatically:
        config = get_config()
        
        # With env vars set:
        # export AGENT1_PADJ_THRESHOLD=0.01
        # export ANTHROPIC_API_KEY=sk-ant-...
        config = get_config()  # padj_threshold=0.01, key loaded
    """
    config = Agent1Config(
        # Only manually pass anthropic_api_key because it has dual-name fallback.
        # All other fields are auto-loaded from AGENT1_* env vars by BaseSettings.
        anthropic_api_key=_resolve_api_key(),
    )
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Agent 1 config loaded. Output dir: {output_path.resolve()}")
    logger.debug(
        f"Config: model={config.llm_model}, "
        f"padj={config.padj_threshold}, "
        f"log2fc={config.log2fc_threshold}"
    )
    
    return config


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for Agent 1.
    Call once at startup.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
