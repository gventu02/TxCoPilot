"""
Agent 3: Network Interpretation Agent (ReAct Loop + RAG)
=========================================================

Purpose: Answer biological questions about CARNIVAL networks using:
1. NetworkX analysis tools (for quantitative network analysis)
2. RAG from project papers (for biological context and interpretation)
3. ReAct pattern (Thought → Action → Observation → repeat)

Architecture:
- Claude Sonnet 4 via Anthropic API
- Zero-shot prompting with structured system instructions
- Dynamic tool loading based on query type
- Citation-backed answers using RAG

Author: Giselle Ventura
Date: November 2025
"""

import anthropic
import logging
import networkx as nx
from typing import Dict, List, Any, Optional
import json
import os
from dataclasses import dataclass
from dotenv import load_dotenv
# agent3_visualizer_nx provides standalone matplotlib subnetwork figures
# (not used in Streamlit pipeline — Cytoscape handles viz there)

logger = logging.getLogger(__name__)

# Import NetworkX tools
from agent3_networkx_tools import (
    list_transcription_factors,
    calculate_node_centrality,
    find_shortest_path,
    get_node_neighbors,
    identify_network_modules,
    get_edge_interactions,
    get_tools_for_query_type
)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Agent3Config:
    """Configuration for Agent 3"""
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 10000 
    temperature: float = 0.5
    max_iterations: int = 10
    enable_rag: bool = True  # Toggle RAG on/off
    require_citations: bool = True  # Force citations for biological claims
    

# =============================================================================
# TOOL DEFINITIONS FOR CLAUDE API
# =============================================================================

def get_tool_definitions(query_type: str = 'all') -> List[Dict]:
    """
    Generate Anthropic API tool definitions with MCP-compliant descriptions.
    
    Args:
        query_type: Dynamically load tools based on query type
    
    Returns:
        List of tool definitions for Claude API
    """
    
    # Base tool definitions (always include these)
    tools = [
        {
            "name": "list_transcription_factors",
            "description": """
# List Transcription Factors

**Use Cases:**
- "What TFs are in my network?"
- "Which TFs are most activated/inhibited?"
- "Show me top 10 TFs by activity"

**Instructions:**
1. Identifies nodes with node_type='TF'
2. Optionally filters by activity (positive/negative)
3. Returns sorted list with activity scores

**When to Use:**
- User asks about transcription factors
- Need to identify key regulators
- Want to see TF activation patterns
""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filter_by_activity": {
                        "type": "string",
                        "enum": ["positive", "negative", "all"],
                        "description": "Filter TFs by activity direction (optional)"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Return only top N TFs by absolute activity (optional)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "calculate_node_centrality",
            "description": """
# Calculate Node Centrality

**Use Cases:**
- "Which genes are most central in the network?"
- "What's the betweenness centrality of a node?"
- "Find hub genes with high degree centrality"

**Instructions:**
1. Calculates specified centrality metrics
2. If genes specified, returns scores for those genes only
3. Otherwise returns top N genes by each metric

**Supported Metrics:**
- betweenness: Measures nodes on shortest paths (bottlenecks)
- degree: Number of connections
- closeness: Average distance to all other nodes
- pagerank: Google's algorithm (importance)
- eigenvector: Connected to important nodes

**When to Use:**
- Identify hub genes
- Determine gene importance
- Find regulatory bottlenecks
""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific genes to analyze (optional, leave empty for top N)"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["betweenness", "degree", "closeness", "pagerank", "eigenvector"]
                        },
                        "description": "Centrality metrics to calculate"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top genes to return per metric (default: 10)"
                    }
                },
                "required": ["metrics"]
            }
        },
        {
            "name": "find_shortest_path",
            "description": """
# Find Shortest Path

**Use Cases:**
- "How does TP53 regulate COL1A1?"
- "Find path from TGFB1 to ACTA2"
- "What connects source X to target Y?"

**Instructions:**
1. Uses Dijkstra's algorithm to find shortest path
2. Returns genes in path and interaction details
3. Interprets biological meaning (activation/inhibition cascade)

**When to Use:**
- Understand regulatory connections
- Trace signal propagation
- Validate hypothesized pathways

**Important:** Call this AFTER you know both genes exist in network
""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Starting gene symbol"
                    },
                    "target": {
                        "type": "string",
                        "description": "Target gene symbol"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum path length in edges (optional)"
                    }
                },
                "required": ["source", "target"]
            }
        },
        {
            "name": "get_node_neighbors",
            "description": """
# Get Node Neighbors

**Use Cases:**
- "What does TP53 regulate?"
- "What regulates COL1A1?"
- "Show me neighbors of TGFB1"

**Instructions:**
1. Returns predecessors (upstream), successors (downstream), or both
2. Includes interaction types (activation/inhibition)
3. Sorted by edge weight/importance

**Directions:**
- 'upstream': What regulates this gene
- 'downstream': What this gene regulates
- 'both': Complete neighborhood

**When to Use:**
- Explore gene's regulatory context
- Find direct regulators/targets
- Build local network view
""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol of interest"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["upstream", "downstream", "both"],
                        "description": "Which neighbors to return (default: both)"
                    },
                    "max_neighbors": {
                        "type": "integer",
                        "description": "Maximum neighbors to return (default: 20)"
                    }
                },
                "required": ["gene"]
            }
        },
        {
            "name": "identify_network_modules",
            "description": """
# Identify Network Modules

**Use Cases:**
- "What are the main functional modules?"
- "Group genes into pathway modules"
- "Find gene clusters in the network"

**Instructions:**
1. Uses community detection algorithms (Louvain or Greedy)
2. Returns genes grouped by module
3. Provides module characterization (TFs, targets, connectivity)

**When to Use:**
- Discover functional organization
- Find pathway-level patterns
- Group genes for enrichment analysis

**Note:** Requires network with ≥10 nodes for reliable results
""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["louvain", "greedy"],
                        "description": "Community detection method (default: louvain)"
                    },
                    "min_module_size": {
                        "type": "integer",
                        "description": "Minimum genes per module (default: 3)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "get_edge_interactions",
            "description": """
# Get Edge Interactions

**Use Cases:**
- "Show me all inhibitory interactions"
- "What interactions involve TP53?"
- "List strongest regulatory connections"

**Instructions:**
1. Filters edges by type (activation/inhibition)
2. Optionally filters by genes involved
3. Returns sorted by interaction strength

**Interaction Types:**
- 'activation': Positive regulation
- 'inhibition': Negative regulation
- None: All interactions

**When to Use:**
- Focus on specific interaction types
- Find strongest connections
- Analyze regulatory logic
""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "interaction_type": {
                        "type": "string",
                        "enum": ["activation", "inhibition"],
                        "description": "Filter by interaction type (optional)"
                    },
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter to interactions involving these genes (optional)"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Maximum interactions to return (default: 20)"
                    }
                },
                "required": []
            }
        }
    ]
    
    # Filter tools based on query type (dynamic loading)
    tool_names = get_tools_for_query_type(query_type)
    filtered_tools = [t for t in tools if t['name'] in tool_names]
    
    return filtered_tools if filtered_tools else tools  # Fallback to all tools


# =============================================================================
# TOOL EXECUTION
# =============================================================================

def execute_tool(tool_name: str, tool_input: Dict, carnival_graph: nx.DiGraph) -> Dict:
    """
    Execute a NetworkX tool and return results.
    
    Args:
        tool_name: Name of tool to execute
        tool_input: Parameters from Claude
        carnival_graph: The CARNIVAL network to analyze
    
    Returns:
        Tool execution results
    """
    
    # Map tool names to functions
    TOOL_MAP = {
        'list_transcription_factors': list_transcription_factors,
        'calculate_node_centrality': calculate_node_centrality,
        'find_shortest_path': find_shortest_path,
        'get_node_neighbors': get_node_neighbors,
        'identify_network_modules': identify_network_modules,
        'get_edge_interactions': get_edge_interactions
    }
    
    if tool_name not in TOOL_MAP:
        return {
            'error': 'UNKNOWN_TOOL',
            'tool_name': tool_name,
            'summary': f" Tool '{tool_name}' not found"
        }
    
    try:
        # Execute tool with graph as first argument
        func = TOOL_MAP[tool_name]
        result = func(carnival_graph, **tool_input)
        return result
    except Exception as e:
        return {
            'error': 'EXECUTION_FAILED',
            'tool_name': tool_name,
            'exception': str(e),
            'summary': f" Tool execution failed: {str(e)}"
        }


# =============================================================================
# AGENT 3 REACT LOOP
# =============================================================================

class Agent3NetworkInterpreter:
    """
    Agent 3: Network Interpretation Agent using ReAct pattern.
    """
    
    def __init__(self, 
                 api_key: str,
                 rag_system = None,
                 config: Agent3Config = None
                 ):
        """
        Initialize Agent 3.
        
        Args:
            api_key: Anthropic API key
            rag_system: RAG system instance (RAGSystem or compatible).
                       If None, will load default from ./rag_system/vector_db
            config: Configuration for Agent 3
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.config = config or Agent3Config()
        if rag_system is None:
            logger.info("Loading default RAG system...")
            self.rag = self._load_default_rag()
        else:
            logger.info("Using provided RAG system")
            self.rag = rag_system
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _load_default_rag(self):
        """
        Load default RAG system if none provided.
        
        Returns:
            RAGSystem instance or None if loading fails
        """
        try:
            from pathlib import Path
            from rag_system.rag_system import RAGSystem

            # Try to find vector_db
            vector_db_path = Path(__file__).parent / "rag_system" / "vector_db"

            if not vector_db_path.exists():
                logger.warning(f"Vector DB not found at {vector_db_path}, RAG disabled")
                return None

            # Load RAG
            rag = RAGSystem.load(str(vector_db_path))
            chunks_count = len(rag.chunks) if rag.chunks else 0
            logger.info(f"RAG loaded: {chunks_count} chunks")
            return rag

        except ImportError as ie:
            logger.warning(f"RAGSystem not found, RAG disabled: {ie}")
            return None
        except Exception as e:
            logger.warning(f"RAG load failed: {e}", exc_info=True)
            return None
        
    def _extract_key_nodes(self, answer_text: str, graph: nx.DiGraph) -> List[str]:
        """Extract gene names mentioned in answer that exist in graph."""
        import re
        
        # Get valid genes from graph
        valid_genes = set(graph.nodes())
        
        # Find gene-like words (all caps, 2+ letters)
        mentioned = re.findall(r'\b[A-Z][A-Z0-9]+\b', answer_text)
        
        # Keep only valid genes
        key_nodes = [g for g in mentioned if g in valid_genes]
        
        # Remove duplicates, preserve order
        seen = set()
        unique_nodes = []
        for node in key_nodes:
            if node not in seen:
                seen.add(node)
                unique_nodes.append(node)
        
        return unique_nodes

    # In-scope keywords: if NONE of these appear and no gene-like symbols detected,
    # the question is likely out of scope for network interpretation.
    _IN_SCOPE_SIGNALS = {
        # Network/graph
        "network", "node", "edge", "hub", "path", "module", "cluster",
        "upstream", "downstream", "neighbor", "centrality", "degree",
        "connected", "component", "subnetwork", "topology",
        # Biology
        "gene", "transcription factor", "tf", "pathway", "signaling",
        "expression", "regulation", "regulate", "activate", "repress",
        "inhibit", "kinase", "receptor", "ligand", "protein", "mutation",
        "phosphorylat", "acetylat", "methylat", "ubiquitin",
        # Omics
        "transcriptom", "deg", "fold change", "differential",
        "upregulat", "downregulat", "enrichment", "kegg", "ora",
        "rna-seq", "microarray", "scrna", "pseudobulk",
        # Disease/biology context
        "disease", "fibrosis", "cancer", "tumor", "inflammation",
        "immune", "cell", "tissue", "organ", "patient", "sample",
        "phenotype", "biomarker", "therapeutic", "target", "drug target",
        # CARNIVAL-specific
        "carnival", "causal", "ilp", "prior knowledge", "pkn",
        "measured", "intermediate", "signaling node",
    }

    def _check_scope(self, question: str) -> bool:
        """
        Check if the question is relevant to network biology interpretation.

        Uses an inverted approach: instead of listing out-of-scope topics (infinite),
        checks for the PRESENCE of in-scope signals. If none are found, returns True
        so the caller can inject a scope-aware prefix for the LLM.

        Returns:
            True  — question appears OUT of scope (no biological signals found)
            False — question appears IN scope (at least one signal matched)
        """
        import re
        q_lower = question.lower()

        # Check for in-scope keywords (substring match for partial words like "regulat")
        for signal in self._IN_SCOPE_SIGNALS:
            if signal in q_lower:
                return False  # In scope

        # Check for gene-like symbols (e.g., TP53, MAPK1, STAT3)
        gene_like = re.findall(r'\b[A-Z][A-Z0-9]{1,10}\b', question)
        # Filter out common English abbreviations
        noise_words = {
            "THE", "AND", "FOR", "NOT", "BUT", "ARE", "WAS", "HAS", "HAD",
            "THIS", "THAT", "WITH", "FROM", "INTO", "ALSO", "CAN", "MAY",
            "ALL", "HOW", "WHY", "WHO", "WHAT", "WHEN", "WHERE", "WHICH",
        }
        gene_candidates = [g for g in gene_like if g not in noise_words]
        if gene_candidates:
            return False  # Likely mentions a gene — in scope

        # No in-scope signals found
        logger.info(f"Scope check: no biological signals found in '{question[:80]}'")
        return True

    def _validate_answer(
        self,
        answer: str,
        carnival_graph: nx.DiGraph,
        rag_citations: List[Dict],
    ) -> Optional[str]:
        """
        Post-ReAct validation: check for hallucinated genes and fabricated citations.
        Returns a warning string if issues found, None if clean.
        """
        import re
        warnings = []

        # 1. Gene name validation — find gene-like tokens not in the network
        valid_nodes = set(carnival_graph.nodes())
        skip_words = {
            "THE", "AND", "FOR", "NOT", "BUT", "ARE", "WAS", "HAS", "HAD",
            "THIS", "THAT", "WITH", "FROM", "INTO", "ALSO", "BEEN", "WILL",
            "CAN", "MAY", "ALL", "TFS", "DEG", "KEGG", "ORA", "PKN", "ILP",
            "NES", "DNA", "RNA", "LLM", "RAG", "CARNIVAL", "HIGH", "LOW",
            "MEDIUM", "NASH", "IPF", "COPD", "SIGNOR", "OMNIPATH",
        }
        mentioned_genes = set(re.findall(r'\b[A-Z][A-Z0-9]{1,10}\b', answer))
        mentioned_genes -= skip_words
        hallucinated = mentioned_genes - valid_nodes
        # Only flag tokens that look like gene symbols
        likely_genes = {
            g for g in hallucinated
            if len(g) >= 3 and (
                any(c.isdigit() for c in g)
                or g.startswith(("TGF", "SMAD", "STAT", "JAK", "MAP", "CDK",
                                 "COL", "MMP", "TNF", "VEGF", "FGF",
                                 "WNT", "NOTCH", "HIF", "NFK", "IRF"))
            )
        }
        if likely_genes:
            gene_list = ", ".join(sorted(likely_genes)[:5])
            warnings.append(
                f"Gene references not found in network: {gene_list}. "
                f"These were mentioned but are not nodes in the CARNIVAL network."
            )

        # 2. Citation validation — flag author citations when no RAG literature was retrieved
        cited_refs = re.findall(r'([A-Z][a-z]+(?:\s+et\s+al\.?))', answer)
        if cited_refs and not rag_citations:
            warnings.append(
                "Author references appear in the answer but no literature was "
                "retrieved — these citations may not be grounded in the provided sources."
            )

        if warnings:
            return "Post-loop validation: " + " | ".join(warnings)
        return None

    def answer_question(self,
                       question: str,
                       carnival_graph: nx.DiGraph,
                       query_type: str = 'all',
                       network_summary: Optional[Dict] = None,
                       conversation_history: Optional[List[Dict]] = None,
                       user_genes_of_interest: Optional[List[str]] = None,
                       disease_summary: Optional[str] = None) -> Dict:
        """
        Answer a biological question about the CARNIVAL network.

        Args:
            question: User's question in natural language
            carnival_graph: CARNIVAL NetworkX graph to analyze
            query_type: Query classification from Agent 1 (for dynamic tool loading)
            network_summary: Optional pre-computed network statistics
            conversation_history: Optional list of prior Q&A exchanges
            user_genes_of_interest: Validated gene symbols the user asked about (from Agent 1)
            disease_summary: Optional text summary of the biological context from Agent 2

        Returns:
            Dict with answer, reasoning trace, citations, and metadata
        """
        
        logger.info(f"Agent 3: Answering question — {question[:80]}")
        logger.info(f"Query type: {query_type}")

        # ── Scope check: flag out-of-scope questions ────────
        _out_of_scope = self._check_scope(question)
        if _out_of_scope:
            logger.info("Agent 3: Out-of-scope query detected — LLM will explain")

        # Reset token counters for this question
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Merge provided metadata with computed topology stats
        topo_stats = self._generate_network_summary(carnival_graph)
        if network_summary is not None:
            topo_stats.update(network_summary)
        network_summary = topo_stats
        
        # Retrieve relevant literature (graceful if RAG unavailable)
        rag_results = []
        relevant_results = []
        literature_context = ""
        RELEVANCE_THRESHOLD = 0.35
        if self.rag is not None:
            try:
                rag_results = self.rag.retrieve(question, top_k=6)
                relevant_results = [r for r in rag_results if r.get("score", 0) >= RELEVANCE_THRESHOLD]
                if relevant_results:
                    literature_context = self.rag.format_for_context(relevant_results)
                    logger.info(f"RAG: {len(relevant_results)}/{len(rag_results)} chunks above threshold {RELEVANCE_THRESHOLD}")
                else:
                    best = rag_results[0].get('score', 0) if rag_results else 0
                    logger.info(f"RAG: no chunks above threshold {RELEVANCE_THRESHOLD} (best: {best:.3f})")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        if not literature_context:
            literature_context = (
                "## Literature Context\n"
                "No sufficiently relevant literature found for this query. "
                "Answer based on network analysis and general biological knowledge. "
                "Do not fabricate citations."
            )
        
        # Dynamic tool loading
        tools = get_tool_definitions(query_type)
        logger.info(f"Loaded {len(tools)} tools for query type '{query_type}'")
        
        # Format conversation history for system prompt
        history_context = ""
        if conversation_history:
            history_lines = ["Previous questions in this session:"]
            for i, exchange in enumerate(conversation_history, 1):
                history_lines.append(f"Q{i}: {exchange['question']}")
                history_lines.append(f"A{i}: {exchange['answer'][:2000]}")
            history_context = "\n".join(history_lines)

        system_prompt = self._build_system_prompt(
            network_summary=network_summary,
            literature_context=literature_context,
            conversation_history=history_context,
            user_genes_of_interest=user_genes_of_interest,
            disease_summary=disease_summary,
        )
        
        # Initialize conversation
        if _out_of_scope:
            scope_prefix = (
                "[SCOPE NOTE: The system's pre-check did not find clear biological "
                "or network-related terms in this question. Before answering, explain "
                "to the user why this question may fall outside your expertise as a "
                "CARNIVAL network interpreter. Be specific about what you CAN help "
                "with (transcription factors, signaling pathways, gene connections, "
                "network topology, pathway enrichment). If you can find ANY relevant "
                "angle to the network, briefly address it. Then suggest 2-3 rephrased "
                "questions the user could ask instead. Do NOT call any tools.]\n\n"
            )
            messages = [{
                "role": "user",
                "content": f"{scope_prefix}Question: {question}"
            }]
        else:
            messages = [{
                "role": "user",
                "content": f"Question: {question}"
            }]
        
        # ReAct Loop
        iteration = 0
        reasoning_trace = []
        
        while iteration < self.config.max_iterations:
            iteration += 1
            logger.debug(f"Iteration {iteration}/{self.config.max_iterations}")
            
            # Call Claude
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                tools=tools,
                messages=messages,
                timeout=120,
            )
            
            # Track tokens
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            
            # Check stop reason
            if response.stop_reason == "end_turn":
                # Agent has final answer
                final_text = self._extract_text(response.content)
                answer_status = 'SUCCESS'
                answer_warning = None

                # If end_turn produced no text or just a ReAct thought fragment,
                # re-prompt once for a proper written answer
                is_incomplete = (
                    not final_text.strip()
                    or final_text.strip().startswith("Action:")
                    or final_text.strip().startswith("Let me")
                    or len(final_text.strip()) < 100
                )
                if is_incomplete:
                    logger.warning(f"Incomplete final answer (len={len(final_text.strip())}, "
                                   f"starts='{final_text.strip()[:50]}') — re-prompting")
                    try:
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({"role": "user", "content": (
                            "Please provide your final answer to the original question "
                            "as a written summary. Do not call any tools."
                        )})
                        followup = self.client.messages.create(
                            model=self.config.model,
                            max_tokens=2048,
                            system=system_prompt,
                            messages=messages,
                            timeout=120,
                        )
                        self.total_input_tokens += followup.usage.input_tokens
                        self.total_output_tokens += followup.usage.output_tokens
                        final_text = self._extract_text(followup.content)
                    except Exception as e:
                        logger.error(f"Re-prompt after end_turn failed: {e}")
                        answer_status = 'FAILED'
                        answer_warning = f'Answer may be incomplete — re-prompt failed: {e}'

                logger.info("Agent reached conclusion")

                # Post-loop validation: check for hallucinated genes/citations
                validation_warning = self._validate_answer(
                    final_text, carnival_graph, relevant_results
                )
                if validation_warning:
                    logger.warning(f"Validation: {validation_warning}")
                    if answer_warning:
                        answer_warning += " | " + validation_warning
                    else:
                        answer_warning = validation_warning

                return {
                    'answer': final_text,
                    'reasoning_trace': reasoning_trace,
                    'iterations': iteration,
                    'tokens_used': {
                        'input': self.total_input_tokens,
                        'output': self.total_output_tokens,
                        'total': self.total_input_tokens + self.total_output_tokens
                    },
                    'cost_estimate': self._estimate_cost(),
                    'citations': relevant_results,
                    'status': answer_status,
                    'warning': answer_warning,
                }
            
            elif response.stop_reason == "tool_use":
                # Agent wants to use tools
                tool_calls = [block for block in response.content if block.type == "tool_use"]
                # Extract the ReAct "thought" — Claude's reasoning before tool calls
                thought_text = " ".join(
                    block.text for block in response.content
                    if hasattr(block, "text") and block.type == "text"
                ).strip()
                logger.debug(f"Agent calling {len(tool_calls)} tool(s)")

                # Execute tools
                tool_results = []
                for tool_call in tool_calls:
                    logger.debug(f"  Tool: {tool_call.name}({json.dumps(tool_call.input, indent=2)[:100]}...)")

                    result = execute_tool(
                        tool_name=tool_call.name,
                        tool_input=tool_call.input,
                        carnival_graph=carnival_graph
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": json.dumps(result, indent=2)
                    })

                    # Log reasoning — now includes the thought (ReAct: Think → Act → Observe)
                    reasoning_trace.append({
                        'iteration': iteration,
                        'thought': thought_text,
                        'action': tool_call.name,
                        'input': tool_call.input,
                        'observation': result.get('summary', 'No summary')
                    })
                
                # Update conversation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            
            else:
                # Unexpected stop reason
                logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                break
        
        # Max iterations reached — force a synthesis answer
        logger.warning(f"Max iterations ({self.config.max_iterations}) reached — forcing synthesis")
        partial_answer = self._extract_text(response.content) if hasattr(response, 'content') else "Analysis incomplete"
        max_iter_warning = None  # synthesis usually succeeds — only warn on failure

        # Only attempt synthesis if messages contain tool results to synthesize
        last_content = messages[-1].get("content") if messages else None
        if isinstance(last_content, list):
            try:
                synth_messages = [m.copy() for m in messages]
                last_msg = synth_messages[-1]
                synthesis_text = (
                    "You have used all available analysis iterations. "
                    "Please synthesize everything you have learned into a final written answer "
                    "to the original question. Summarize your findings — do not call any more tools."
                )
                last_msg["content"] = list(last_msg["content"]) + [{"type": "text", "text": synthesis_text}]

                followup = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=synth_messages,
                    timeout=120,
                    # No tools= parameter — prevents model from calling tools
                )
                self.total_input_tokens += followup.usage.input_tokens
                self.total_output_tokens += followup.usage.output_tokens
                partial_answer = self._extract_text(followup.content)
            except Exception as e:
                logger.error(f"Synthesis re-prompt failed: {e}")
                max_iter_warning = f'Max iterations reached and synthesis re-prompt failed: {e}'
        else:
            logger.warning("No tool results in message history — skipping synthesis re-prompt")

        # Post-loop validation on synthesis answer
        validation_warning = self._validate_answer(
            partial_answer, carnival_graph, relevant_results
        )
        if validation_warning:
            logger.warning(f"Validation: {validation_warning}")
            if max_iter_warning:
                max_iter_warning += " | " + validation_warning
            else:
                max_iter_warning = validation_warning

        return {
            'answer': partial_answer,
            'reasoning_trace': reasoning_trace,
            'iterations': iteration,
            'tokens_used': {
                'input': self.total_input_tokens,
                'output': self.total_output_tokens,
                'total': self.total_input_tokens + self.total_output_tokens
            },
            'cost_estimate': self._estimate_cost(),
            'citations': relevant_results,
            'status': 'MAX_ITERATIONS_REACHED',
            'warning': max_iter_warning,
        }
    
    
    def _build_system_prompt(self, network_summary: Dict, literature_context: str,
                            conversation_history: str = "",
                            user_genes_of_interest: Optional[List[str]] = None,
                            disease_summary: Optional[str] = None) -> str:
        """
        Build the system prompt sent to Claude on every ReAct iteration.

        Assembles six sections in order:
          1. Disease/biological context (from Agent 2 summary, if provided)
          2. Network confidence assessment (from Agent 2 assessor, if available)
          3. Network topology statistics (nodes, edges, TF count, etc.)
          4. Retrieved literature context (RAG chunks, or a no-literature fallback)
          5. ReAct reasoning instructions and final-answer format
          6. User genes of interest and conversation history (appended if present)

        Citation rules adapt based on whether literature was retrieved:
        if RAG returned results above threshold, the prompt instructs Claude to
        cite them; otherwise it explicitly prohibits fabricating references.
        """

        # Disease/biological context block
        disease_block = ""
        if disease_summary:
            disease_block = f"## Biological Context\n{disease_summary}\n"

        # Network confidence assessment block (from Agent 2)
        confidence_block = ""
        confidence_data = getattr(self, '_network_confidence', None) or {}
        if confidence_data.get("confidence"):
            conf = confidence_data
            conf_lines = [
                f"## Network Confidence Assessment",
                f"**Overall confidence: {conf['confidence'].upper()}**",
                f"",
                f"{conf.get('reasoning', '')}",
            ]
            if conf.get("limitations"):
                conf_lines.append("\n**Limitations:**")
                for lim in conf["limitations"]:
                    conf_lines.append(f"- {lim}")
            if conf.get("suggestions"):
                conf_lines.append("\n**Suggestions for improvement:**")
                for sug in conf["suggestions"]:
                    conf_lines.append(f"- {sug}")
            conf_lines.append("")
            conf_lines.append(
                "IMPORTANT: Factor this confidence level into your answers. "
                "If confidence is medium or low, caveat your biological interpretations accordingly."
            )
            confidence_block = "\n".join(conf_lines) + "\n"

        # Citation rules based on whether literature was found
        has_literature = "No sufficiently relevant literature" not in literature_context
        if has_literature:
            citation_rules = (
                "6. **Cite literature** when making biological claims using the provided context. "
                "Use format: [Author Year] or [Paper Title]."
            )
        else:
            citation_rules = (
                "6. **No relevant literature was found** for this query. Do NOT fabricate citations. "
                "If you reference general biological knowledge, state: "
                "\"Based on established signaling biology\" — do not invent paper references."
            )

        prompt = f"""You are a network biology expert analyzing a CARNIVAL causal network to answer biological questions.

## Your Role
You are a knowledgeable collaborator helping a biologist understand their CARNIVAL
network results. Write the way a senior postdoc would explain findings to a colleague —
conversational but precise, weaving numbers into narrative rather than listing them.
Its ok to use bullet-point dumps. Lead with the biological insight, then back it up with evidence.

Core responsibilities:
- Interpret network analysis results in biological context
- Use NetworkX tools to quantify network properties
- Ground interpretations in literature when available
- Provide clear, evidence-based answers to biological questions

{disease_block}{confidence_block}## Network Context
{json.dumps(network_summary, indent=2)}

{literature_context}

## ReAct Pattern
You MUST follow this reasoning pattern:

1. **Thought**: What do I need to know to answer this question?
2. **Action**: Call appropriate tool(s) or reference literature
3. **Observation**: Interpret the result
4. **Repeat** until you have enough information

Then provide your **Final Answer** as a coherent narrative that:
- Leads with the biological interpretation (what does this mean for the disease?)
- Weaves in quantitative evidence naturally (e.g., "STAT3 emerges as the top hub
  with 12 downstream targets, suggesting..." rather than "STAT3: degree=12")
- References literature where available (do not fabricate)
- Notes uncertainty honestly but without undermining every statement

## Critical Rules
1. **Always use tools first** before answering — don't guess network properties
2. **Embed numbers in narrative** — include actual values but explain what they mean
3. **Be honest about uncertainty** — but don't hedge everything into meaninglessness
4. **Don't hallucinate** — if you don't have information, say so
5. **Think biologically** — the user cares about disease mechanisms, not graph theory
{citation_rules}

## Tool Usage Strategy
- Start with overview tools (list_transcription_factors, calculate_centrality)
- Then drill down (get_neighbors, find_paths) for specific questions
- Use multiple complementary tools to build complete picture

Now, answer the user's question using this approach."""

        if user_genes_of_interest:
            genes_str = ", ".join(user_genes_of_interest)
            prompt += f"""

## User Genes of Interest
The user specifically asked about these genes: {genes_str}.
Check whether each gene appears in the CARNIVAL network and address them in your answer.
If a gene is absent from the network, explain that it was not differentially expressed or not connected in the causal model."""

        if conversation_history:
            prompt += f"\n\n{conversation_history}"

        return prompt
    
    def _generate_network_summary(self, G: nx.DiGraph) -> Dict:
        """Generate summary statistics about network."""
        node_types = nx.get_node_attributes(G, 'node_type')
        
        return {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'n_tfs': sum(1 for t in node_types.values() if t == 'TF'),
            'n_measured': sum(1 for t in node_types.values() if t == 'measured'),
            'disease': G.graph.get('disease', 'unknown'),
            'control': G.graph.get('control', 'unknown'),
            'density': nx.density(G),
            'is_dag': nx.is_directed_acyclic_graph(G)
        }
    
    def _extract_text(self, content) -> str:
        """Extract text from Claude's response content."""
        if isinstance(content, list):
            text_blocks = [block.text for block in content if hasattr(block, 'text')]
            return '\n\n'.join(text_blocks)
        return str(content)
    
    def _estimate_cost(self) -> float:
        """Estimate API cost based on token usage."""
        # Claude Sonnet 4 pricing (approximate) as of November 2026
        INPUT_PRICE_PER_1M = 3.00  # $3 per 1M input tokens
        OUTPUT_PRICE_PER_1M = 15.00  # $15 per 1M output tokens
        
        input_cost = (self.total_input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
        output_cost = (self.total_output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
        
        return round(input_cost + output_cost, 4)
    

# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def quick_agent3_test(api_key: str, 
                      question: str = "What are the top 3 transcription factors by centrality?") -> Dict:
    """
    Quick test of Agent 3 with mock CARNIVAL network.
    
    Args:
        api_key: Anthropic API key
        question: Question to answer
    
    Returns:
        Agent 3 response
    """
    from mock_carnival_generator import quick_mock_network
    
    # Generate mock network
    G = quick_mock_network(seed=42)
    
    # Initialize Agent 3
    agent = Agent3NetworkInterpreter(api_key=api_key)
    
    # Answer question
    result = agent.answer_question(question, G, query_type='hub_identification')
    
    return result


# =============================================================================
# TESTING & DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AGENT 3 NETWORK INTERPRETER - DEMONSTRATION")
    print("=" * 70)
    
    # Load API key from .env file
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("\n  API key not found in .env file")
        print("   Create a .env file with: anthropic_API_key=your_key_here")
        print("   Or set environment variable: ANTHROPIC_API_KEY=your_key_here")
    else:
        print("\n API key loaded from .env file")
    
    print("\n Usage:")
    print("   from agent3_react_loop import Agent3NetworkInterpreter")
    print("   from mock_carnival_generator import quick_mock_network")
    print("   from dotenv import load_dotenv")
    print("   import os")
    print("   ")
    print("   # Setup")
    print("   load_dotenv()")
    print("   api_key = os.getenv('anthropic_API_key')")
    print("   G = quick_mock_network()")
    print("   agent = Agent3NetworkInterpreter(api_key)")
    print("   ")
    print("   # Ask question")
    print("   result = agent.answer_question(")
    print("       question='What are the key transcription factors?',")
    print("       carnival_graph=G")
    print("   )")
    print("   ")
    print("   print(result['answer'])")
    print("   print(f\"Cost: ${result['cost_estimate']}\")")