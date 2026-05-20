"""
Agent 3 Input/Output Schemas
Pydantic contracts for the Agent 2 → Agent 3 interface.
"""
from __future__ import annotations
from typing import Optional, Dict, List, Any
from enum import Enum
import networkx as nx
from pydantic import BaseModel, Field


class Agent3Status(str, Enum):
    SUCCESS = "SUCCESS"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"
    FAILED = "FAILED"


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0


class Agent3Input(BaseModel):
    """
    Contract for Agent 2 → Agent 3 handoff.
    All fields required except network_summary.
    """
    user_query: str = Field(
        description="Original natural language question from the user"
    )
    carnival_network: Any = Field(
        description="nx.DiGraph from Agent 2. Nodes have activity, is_active, node_type attributes."
    )
    network_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pre-computed network stats from Agent 2 metadata. Passed as network_summary to Agent 3."
    )
    query_type: str = Field(
        default="all",
        description="Query classification from Agent 1 extracted_entities.query_intent"
    )
    user_genes_of_interest: Optional[List[str]] = Field(
        default=None,
        description="Validated gene symbols the user specifically asked about (from Agent 1)"
    )

    model_config = {"arbitrary_types_allowed": True}  # allows nx.DiGraph


class ReasoningStep(BaseModel):
    iteration: int
    thought: str = ""
    action: str
    input: Optional[Dict[str, Any]] = None
    observation: str


class Agent3Result(BaseModel):
    """
    Agent 3 output contract.
    Matches the Dict currently returned by answer_question().
    """
    answer: str = Field(description="Biological interpretation text")
    reasoning_trace: List[ReasoningStep] = Field(
        default_factory=list,
        description="ReAct reasoning steps"
    )
    iterations: int = Field(description="Number of ReAct iterations used")
    tokens_used: TokenUsage = Field(default_factory=TokenUsage)
    cost_estimate: float = Field(default=0.0)
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="RAG literature results"
    )
    status: Agent3Status = Field(default=Agent3Status.SUCCESS)
    warning: Optional[str] = Field(
        default=None,
        description="Warning message if answer may be incomplete or re-prompt failed"
    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Agent3Result":
        """Convert Agent3's existing Dict return to typed schema."""
        tokens = d.get("tokens_used", {})
        return cls(
            answer=d.get("answer", ""),
            reasoning_trace=d.get("reasoning_trace", []),
            iterations=d.get("iterations", 0),
            tokens_used=TokenUsage(
                input=tokens.get("input", 0),
                output=tokens.get("output", 0),
                total=tokens.get("total", 0)
            ),
            cost_estimate=d.get("cost_estimate", 0.0),
            citations=d.get("citations", []),
            status=Agent3Status(d.get("status", "SUCCESS")),
            warning=d.get("warning"),
        )