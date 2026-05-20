"""
Agent 1: Data Preparation & Quality Assessment
================================================
Prepares expression data for CARNIVAL causal network analysis.

Responsibilities:
    - Validate input data (bulk RNA-seq or scRNA-seq)
    - Pseudobulk aggregation (if scRNA)
    - Differential expression analysis (PyDESeq2)
    - Quality assessment (noise, library size, gene coverage)
    - Entity extraction from user query (LLM)
    - Gene symbol validation (OmniPath)
    - LLM explanation of results

Output:
    Agent1Result (Pydantic model, JSON-serializable)
    → Consumed by Agent 2 (CARNIVAL pipeline)
"""

__version__ = "0.1.0"
