"""
Clarification Generator
========================
Generate conversational clarification questions using LLM tool use.

Takes a structured ClarificationRequest (with issue + details) and asks the LLM
to produce a natural, user-friendly question and context explanation.

Used for biological decision points (scope guard, threshold choice, disease mismatch).
Data validation clarifications remain hardcoded — "file not found" doesn't need LLM polish.
"""

import logging
import time
from typing import Optional, Tuple

from agent1.schemas_agent1 import Agent1Config, ClarificationRequest

logger = logging.getLogger(__name__)

try:
    from anthropic import Anthropic, APIError, APITimeoutError
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


# =============================================================================
# TOOL SCHEMA
# =============================================================================

CLARIFICATION_TOOL = {
    "name": "generate_clarification_question",
    "description": (
        "Generate a clear, conversational question to ask the user when the "
        "pipeline needs their input on a biological decision."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "A conversational question for the user (2-4 sentences). "
                    "Explain what was detected and what decision is needed. "
                    "Write as a helpful collaborator, not a system error message."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Brief explanation (1-3 sentences) helping the user understand "
                    "why this matters for their analysis. Include practical implications."
                ),
            },
        },
        "required": ["question", "context"],
    },
}


CLARIFICATION_SYSTEM_PROMPT = """\
You are a helpful bioinformatics pipeline assistant. Your job is to generate
clear, conversational questions when the pipeline needs user input.

Write as a knowledgeable collaborator — friendly but precise. The user is a
biologist who understands their experiment but may not know the pipeline's
technical details.

Rules:
- Lead with what was detected, then ask what the user wants to do
- Reference specific values from the details (disease names, labels, thresholds)
- If options are provided, briefly explain the trade-off of each
- Keep the question under 4 sentences
- Keep the context under 3 sentences
- Do NOT use jargon without brief explanation
- Do NOT be alarmist — frame choices as decisions, not problems
"""


# =============================================================================
# GENERATOR
# =============================================================================

class ClarificationGenerator:
    """Generate LLM-phrased clarification questions from structured issue descriptions."""

    def __init__(self, config: Agent1Config):
        self.config = config
        self._client: Optional["Anthropic"] = None

    @property
    def client(self) -> "Anthropic":
        if self._client is None:
            if not _HAS_ANTHROPIC:
                raise ImportError("anthropic package required for clarification generation.")
            if not self.config.anthropic_api_key:
                raise ValueError("Anthropic API key required.")
            self._client = Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    def generate(self, request: ClarificationRequest) -> Tuple[ClarificationRequest, int]:
        """
        Fill in question + context on a ClarificationRequest using LLM.

        Args:
            request: ClarificationRequest with issue, details, options set.
                     question and context will be overwritten.

        Returns:
            Tuple of (updated ClarificationRequest, tokens_used)
        """
        if not request.issue:
            # No structured issue — this is a hardcoded clarification, skip LLM
            return request, 0

        user_message = self._build_user_message(request)

        try:
            raw_output, tokens = self._call_llm(user_message)
            request.question = raw_output.get("question", request.question)
            request.context = raw_output.get("context", request.context)
            logger.info(f"LLM clarification generated ({tokens} tokens): {request.question[:80]}...")
            return request, tokens
        except Exception as e:
            logger.warning(f"LLM clarification failed ({e}), using fallback")
            request = self._fallback(request)
            return request, 0

    def _build_user_message(self, request: ClarificationRequest) -> str:
        """Build the user message from structured issue data."""
        parts = [
            f"Issue type: {request.issue}",
            f"Field needing user input: {request.field}",
        ]
        if request.details:
            parts.append(f"Details: {request.details}")
        if request.options:
            parts.append(f"Options to present: {request.options}")
        if request.default:
            parts.append(f"Recommended default: {request.default}")
        return "\n".join(parts)

    def _call_llm(self, user_message: str, max_retries: int = 1) -> Tuple[dict, int]:
        """Make forced tool-use API call."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.config.llm_model,
                    max_tokens=300,
                    temperature=0.1,
                    system=CLARIFICATION_SYSTEM_PROMPT,
                    tools=[CLARIFICATION_TOOL],
                    tool_choice={"type": "tool", "name": "generate_clarification_question"},
                    messages=[{"role": "user", "content": user_message}],
                )
                tool_block = response.content[0]
                if tool_block.type != "tool_use":
                    raise ValueError(f"Expected tool_use, got {tool_block.type}")
                tokens = response.usage.input_tokens + response.usage.output_tokens
                return tool_block.input, tokens
            except (APITimeoutError, APIError) as e:
                last_error = e
                logger.warning(f"Clarification LLM attempt {attempt + 1} failed: {e}")
            except Exception as e:
                last_error = e
                break
        raise RuntimeError(f"Clarification generation failed: {last_error}")

    @staticmethod
    def _fallback(request: ClarificationRequest) -> ClarificationRequest:
        """Generate a reasonable hardcoded question if LLM fails."""
        fallbacks = {
            "out_of_scope_query": (
                "Your question doesn't appear to be answerable with transcriptomic data. "
                "Could you rephrase it to focus on gene regulation, pathway activity, "
                "or transcription factors?",
                "This pipeline analyzes gene expression and signaling networks."
            ),
            "within_disease_comparison": (
                "This looks like a within-disease comparison (e.g., staging or grading). "
                "Would you like to use a relaxed fold-change threshold to capture subtler signals?",
                "Within-disease comparisons typically have smaller fold changes than disease vs control."
            ),
            "disease_mismatch": (
                "The disease in your query doesn't match the dataset labels. "
                "Would you like to proceed with the dataset's actual groups?",
                "This can happen with cross-disease hypotheses."
            ),
        }
        question, context = fallbacks.get(
            request.issue,
            ("The pipeline needs your input to proceed.", "Please review and respond.")
        )
        if not request.question:
            request.question = question
        if not request.context:
            request.context = context
        return request
