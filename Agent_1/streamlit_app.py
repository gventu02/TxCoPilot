"""
Thesis Pipeline UI — Streamlit Application
=============================================
Full pipeline orchestrator: Agent 1 → Agent 2 → Agent 3

This is the SINGLE entry point for users. It:
1. Collects user query + data files
2. Runs Agent 1 (data preparation + quality + DEGs + entity extraction)
3. Handles clarification loops (re-prompts user if Agent 1 needs info)
4. Auto-forwards successful results to Agent 2 (CARNIVAL pipeline)
5. Displays Agent 3 interpretation UI when complete

Docker-ready: all paths relative, all config from environment variables.

Usage:
    streamlit run streamlit_app.py

    # With custom port (Docker):
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

Environment Variables:
    ANTHROPIC_API_KEY       - Required for LLM calls
    AGENT1_OUTPUT_DIR       - Agent 1 output directory (default: agent1_output)
    PIPELINE_UPLOAD_DIR     - Where uploaded files are saved (default: uploads)
Author: Giselle
Date: February 2026
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Agent_2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Agent_3'))
from agent2_main import Agent2Pipeline
from core.kegg_ora import run_kegg_ora
from agent3_react_loop import Agent3NetworkInterpreter
from schemas_agent3 import Agent3Result
from network_visualization_link_analysis import render_network_tab
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from generate_thesis_dag import render_from_nx, render_to_bytes, dag_layout

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Transcriptomics Exploratory Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom theme — warm cream + blue accent palette
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Palette ──
       Background:  #F5F2F2  (warm light grey)
       Accent 1:    #FEB05D  (orange)
       Accent 2:    #5A7ACD  (blue)
       Dark:        #2B2A2A  (near-black text / elements)
    */

    /* Main background */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #F5F2F2;
    }
    /* Sidebar */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background-color: #ECEAEA;
    }
    /* Header / toolbar area */
    [data-testid="stHeader"] {
        background-color: #F5F2F2;
    }
    /* Metric cards and expanders */
    [data-testid="stMetric"] {
        background-color: #ECEAEA;
        border-radius: 8px;
        padding: 4px;
    }
    [data-testid="stExpander"] {
        background-color: #ECEAEA;
        border-radius: 8px;
    }

    /* ── Buttons — orange accent ── */
    .stButton > button {
        background-color: #FEB05D;
        color: #2B2A2A;
        border: 1px solid #e89a45;
    }
    .stButton > button:hover {
        background-color: #e89a45;
        color: #2B2A2A;
        border: 1px solid #d08838;
    }
    /* Primary button variant — blue */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #5A7ACD;
        color: #FFFFFF;
        border: 1px solid #4a68b5;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #4a68b5;
        color: #FFFFFF;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {
        background-color: #ECEAEA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #5A7ACD;
        color: #FFFFFF;
    }

    /* ── Text inputs, text areas ── */
    .stTextArea textarea, .stTextInput input {
        background-color: #F5F2F2 !important;
        color: #2B2A2A !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        background-color: #F5F2F2 !important;
        color: #2B2A2A !important;
    }

    /* ── File uploaders ── */
    [data-testid="stFileUploader"] {
        background-color: #F5F2F2 !important;
        border-radius: 8px;
    }
    [data-testid="stFileUploader"] section {
        background-color: #F5F2F2 !important;
        border: 2px dashed #5A7ACD !important;
        border-radius: 8px;
        padding: 12px;
    }
    [data-testid="stFileUploader"] section > div {
        color: #2B2A2A !important;
    }
    /* File uploader button */
    [data-testid="stFileUploader"] button {
        background-color: #5A7ACD !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    /* File uploader label text */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] label p {
        color: #2B2A2A !important;
    }

    /* ── Selectbox / slider / checkbox ── */
    [data-testid="stSelectbox"] {
        background-color: transparent;
    }
    [data-testid="stSelectbox"] > div > div {
        background-color: #F5F2F2 !important;
    }

    /* ── Divider lines ── */
    hr {
        border-color: #5A7ACD !important;
        opacity: 0.25;
    }

    /* ── st-link-analysis graph background ── */
    iframe[title="st_link_analysis.st_link_analysis"],
    [data-testid="stCustomComponentV1"] iframe {
        background-color: #F5F2F2 !important;
        border-radius: 8px;
    }

    /* ── Global text color ── */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li {
        color: #2B2A2A;
    }
    /* Sidebar text */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #2B2A2A !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Quiet noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants — all from environment variables with sensible defaults
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path(os.environ.get("PIPELINE_UPLOAD_DIR", "uploads"))

# Preset queries for users who don't know what to ask
PRESET_QUERIES = {
    " Identify key transcription factors": (
        "Identify the key transcription factors driving disease versus control "
        "and map their regulatory targets in the causal network."
    ),
    " Discover dysregulated pathways": (
        "What signaling pathways are most dysregulated in disease compared to "
        "control, and which upstream regulators are driving the changes?"
    ),
}

# Accepted file extensions
EXPRESSION_EXTENSIONS = [".csv", ".tsv", ".h5ad"]
DEG_EXTENSIONS = [".csv", ".tsv"]
METADATA_EXTENSIONS = [".csv", ".tsv"]


# =============================================================================
# SESSION STATE: Manage pipeline state across Streamlit reruns
# =============================================================================

def init_session_state() -> None:
    """
    Initialize all session state keys.

    Streamlit reruns the entire script on every interaction. Session state
    persists data across these reruns. We initialize once and check before
    overwriting so user progress is never lost.
    """
    defaults = {
        # Pipeline stage tracking
        "pipeline_stage": "input",  # input → clarification → running → results
        # User inputs (preserved across reruns)
        "query_text": "",
        "uploaded_expression": None,
        "uploaded_metadata": None,
        "uploaded_degs": None,
        # Advanced options
        "cell_type": None,
        "condition_column": None,
        "disease_label": None,
        "control_label": None,
        "sample_column": None,
        # Agent 1 results
        "agent1_result": None,
        "clarification_responses": {},
        "disease_mismatch_confirmed": False,
        "threshold_confirmed": False,
        "log2fc_threshold_override": None,
        # Agent 2 results
        "agent2_result": None,
        # Error tracking
        "error_message": None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Agent 2/3 pipeline state
    if "carnival_network" not in st.session_state:
        st.session_state.carnival_network = None
    if "agent2_result_full" not in st.session_state:
        st.session_state.agent2_result_full = None
    if "agent3_interpreter" not in st.session_state:
        st.session_state.agent3_interpreter = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "network_confidence" not in st.session_state:
        st.session_state.network_confidence = {}


# =============================================================================
# FILE HANDLING: Save uploads to disk for Agent 1
# =============================================================================

def save_uploaded_file(uploaded_file) -> Optional[Path]:
    """
    Save a Streamlit UploadedFile to the upload directory.

    Returns the saved path, or None if no file was provided.
    Docker note: UPLOAD_DIR is relative and configurable via env var.
    """
    if uploaded_file is None:
        return None

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DIR / uploaded_file.name

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    logger.info(f"Saved upload: {save_path} ({uploaded_file.size:,} bytes)")
    return save_path


def validate_file_extension(filename: str, allowed: list[str]) -> bool:
    """Check that the uploaded file has an acceptable extension."""
    return Path(filename).suffix.lower() in allowed


# =============================================================================
# AGENT 1 INTEGRATION
# =============================================================================

def run_agent1(
    query_text: str,
    expression_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    precomputed_deg_path: Optional[str] = None,
    cell_type: Optional[str] = None,
    condition_column: Optional[str] = None,
    disease_label: Optional[str] = None,
    control_label: Optional[str] = None,
    sample_column: Optional[str] = None,
):
    """
    Execute Agent 1 pipeline.

    Returns Agent1Result (Pydantic model).
    Imports are inside the function to fail gracefully if agent1 isn't installed.
    """
    from agent1.agent1_main import Agent1Pipeline
    from agent1.schemas_agent1 import UserQuery

    query = UserQuery(
        query_text=query_text,
        expression_path=expression_path,
        metadata_path=metadata_path,
        precomputed_deg_path=precomputed_deg_path,
        cell_type=cell_type,
        condition_column=condition_column,
        disease_label=disease_label,
        control_label=control_label,
        sample_column=sample_column,
        disease_mismatch_confirmed=st.session_state.get("disease_mismatch_confirmed", False),
        threshold_confirmed=st.session_state.get("threshold_confirmed", False),
        log2fc_threshold_override=st.session_state.get("log2fc_threshold_override"),
    )

    pipeline = Agent1Pipeline()
    return pipeline.run(query)


# =============================================================================
# AGENT 2 INTEGRATION
# =============================================================================

def run_agent2(agent1_result) -> dict:
    """Real Agent 2 pipeline — replaces stub."""
    try:
        pipeline = Agent2Pipeline(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        agent2_result = pipeline.run(agent1_result)

        # Store full result for Agent 3
        st.session_state.agent2_result_full = agent2_result
        st.session_state.carnival_network = agent2_result.carnival_network
        st.session_state.agent2_metadata = agent2_result.metadata
        st.session_state.edge_annotations = agent2_result.edge_annotations
        st.session_state.network_confidence = agent2_result.network_confidence

        # Set disease/control on graph for Agent 3's _generate_network_summary
        G = agent2_result.carnival_network
        comparison = agent2_result.metadata.get("comparison", "disease vs control")
        parts = comparison.replace(" (limma)", "").split(" vs ")
        G.graph["disease"] = parts[0].strip() if len(parts) == 2 else "unknown"
        G.graph["control"] = parts[1].strip() if len(parts) == 2 else "unknown"

        # Auto-generate DAG figure using Graphviz dot layout
        try:
            output_dir = os.path.join(os.path.dirname(__file__), "agent1_output")
            dag_path = render_from_nx(
                G,
                disease=G.graph["disease"],
                control=G.graph["control"],
                output_dir=output_dir,
            )
            if dag_path:
                st.session_state["dag_figure_path"] = dag_path
                logger.info(f"DAG figure saved: {dag_path}")
        except Exception as dag_err:
            logger.warning(f"DAG figure generation failed: {dag_err}")

        return {
            "status": "success",
            "carnival_network_path": None,  # network passed in-memory via st.session_state.carnival_network
            "network_stats": {
                "n_nodes": agent2_result.metadata.get("n_nodes_total", 0),
                "n_edges": agent2_result.metadata.get("n_edges_total", 0),
                "n_tfs": agent2_result.metadata.get("n_tfs_used", 0),
            },
            "message": agent2_result.summary[:200] if agent2_result.summary else "CARNIVAL analysis complete",
        }
    except Exception as e:
        return {
            "status": "error",
            "carnival_network_path": None,
            "network_stats": {"n_nodes": 0, "n_edges": 0, "n_tfs": 0},
            "message": f"Agent 2 failed: {str(e)}",
        }


# =============================================================================
# AGENT 3 INTEGRATION
# =============================================================================

def get_agent3_interpreter() -> Agent3NetworkInterpreter:
    """Load Agent 3 once per session — reuse across questions."""
    if st.session_state.agent3_interpreter is None:
        st.session_state.agent3_interpreter = Agent3NetworkInterpreter(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
    return st.session_state.agent3_interpreter


def run_agent3(question: str) -> dict:
    """Run Agent 3 with structured conversation history."""
    if st.session_state.carnival_network is None:
        return {
            "status": "error",
            "answer": "No network available. Run analysis first."
        }
    try:
        interpreter = get_agent3_interpreter()
        agent2_full = st.session_state.agent2_result_full

        # Extract user's genes of interest + query intent from Agent 1 result
        user_genes = None
        query_type = "all"
        agent1_result = st.session_state.get("agent1_result")
        if agent1_result and hasattr(agent1_result, "extracted_entities"):
            ent = agent1_result.extracted_entities
            if ent:
                user_genes = getattr(ent, "validated_genes", None) or None
                query_type = getattr(ent, "query_intent", None) or "all"

        # Strip internal timing data — not useful for LLM interpretation
        network_summary = agent2_full.metadata if agent2_full else None
        clean_summary = {k: v for k, v in network_summary.items() if k != 'pipeline_stats'} if network_summary else None

        # Cap history to last 20 exchanges to avoid context overflow
        recent_history = st.session_state.conversation_history[-20:]

        # Pass biological context summary (disease, comparison, top regulators)
        disease_summary = agent2_full.summary if agent2_full and hasattr(agent2_full, 'summary') else None

        # Pass network confidence assessment to Agent 3 for caveating
        if agent2_full and hasattr(agent2_full, 'network_confidence'):
            interpreter._network_confidence = agent2_full.network_confidence

        raw_result = interpreter.answer_question(
            question=question,
            carnival_graph=st.session_state.carnival_network,
            query_type=query_type,
            network_summary=clean_summary,
            conversation_history=recent_history,
            user_genes_of_interest=user_genes,
            disease_summary=disease_summary,
        )

        result = Agent3Result.from_dict(raw_result)

        st.session_state.conversation_history.append({
            "question": question,
            "answer": result.answer,
            "citations": result.citations,
            "cost": result.cost_estimate,
        })

        return {"status": "success", "result": result}

    except Exception as e:
        return {"status": "error", "answer": f"Agent 3 failed: {str(e)}"}


# =============================================================================
# UI COMPONENTS: Reusable display functions
# =============================================================================

def render_header() -> None:
    """Render the application header."""
    st.title("🧬 Transcriptomics Exploratory Tool")


def render_sidebar() -> None:
    """Render the sidebar with pipeline status and info."""
    with st.sidebar:
        st.header("Agent Status")

        stage = st.session_state.pipeline_stage
        stages = {
            "input": ("Collecting Input", "info"),
            "clarification": ("Awaiting Clarification", "warning"),
            "running_agent1": ("Agent 1: Preparing Data...", "info"),
            "running_agent2": ("Agent 2: Running CARNIVAL...", "info"),
            "results": ("✅ Analysis Complete", "success"),
            "error": ("❌ Error", "error"),
        }

        label, status_type = stages.get(stage, ("Unknown", "info"))
        if status_type == "success":
            st.success(label)
        elif status_type == "warning":
            st.warning(label)
        elif status_type == "error":
            st.error(label)
        else:
            st.info(label)

        st.divider()

        # Show results summary if available
        result = st.session_state.agent1_result
        if result and hasattr(result, "status"):
            st.subheader("Agent 1 Summary")
            st.write(f"**Status:** {result.status.value}")

            if result.quality_report:
                qr = result.quality_report
                st.metric("Genes", f"{qr.n_genes:,}")
                st.metric("Samples", qr.n_samples)
                st.metric("Noise Level", qr.noise_level.value.title())

            if result.deg_results:
                dr = result.deg_results
                st.metric("DEGs", f"{dr.n_significant:,}")
                col1, col2 = st.columns(2)
                col1.metric("Up", dr.n_upregulated)
                col2.metric("Down", dr.n_downregulated)

            if result.tokens_used:
                st.caption(f"LLM tokens used: {result.tokens_used:,}")

        st.divider()

        # Load pre-built network from Agent 2 output CSVs
        with st.expander("Load Pre-built Network"):
            nodes_file = st.file_uploader("Nodes CSV", type=["csv"], key="load_nodes_csv")
            edges_file = st.file_uploader("Edges CSV", type=["csv"], key="load_edges_csv")
            if st.button("Load Network", use_container_width=True):
                if nodes_file and edges_file:
                    try:
                        nodes_df = pd.read_csv(nodes_file)
                        edges_df = pd.read_csv(edges_file)
                        G = nx.DiGraph()
                        for _, row in nodes_df.iterrows():
                            activity = int(row["node_activity"])
                            G.add_node(
                                str(row["node"]),
                                activity=activity,
                                activity_score=float(activity),
                                is_active=(activity != 0),
                                node_type=row.get("node_type", "intermediate"),
                            )
                        for _, row in edges_df.iterrows():
                            src, tgt = str(row["source"]), str(row["target"])
                            sign = int(row["edge_activity"])
                            if src in G.nodes and tgt in G.nodes:
                                src_activity = G.nodes[src].get("activity", sign)
                                pkn_sign = sign * src_activity if src_activity != 0 else sign
                                G.add_edge(
                                    src, tgt,
                                    sign=sign,
                                    interaction="ACTIVATES" if pkn_sign >= 1 else "INHIBITS" if pkn_sign <= -1 else "NEUTRAL",
                                    weight=float(abs(sign)) if sign != 0 else 0.0,
                                    is_active=(sign != 0),
                                )
                        # Filter to active-only subgraph
                        active_nodes = [n for n, d in G.nodes(data=True) if d.get("is_active")]
                        G = G.subgraph(active_nodes).copy()
                        st.session_state.carnival_network = G
                        st.session_state.pipeline_stage = "results"
                        st.success(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load: {e}")
                else:
                    st.warning("Upload both nodes and edges CSVs")

        st.divider()

        # Reset button
        if st.button(" Start New Analysis", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()


def render_input_form() -> None:
    """
    Render the data upload and query input form.

    This is the main landing page. Users upload files and enter a query,
    or select a preset query.
    """
    st.header("1. Enter Your Query")

    # Preset query buttons
    st.write("**Quick start** — select a preset or write your own:")
    preset_cols = st.columns(len(PRESET_QUERIES))
    for i, (label, query) in enumerate(PRESET_QUERIES.items()):
        with preset_cols[i]:
            if st.button(label, use_container_width=True, key=f"preset_{i}"):
                st.session_state.query_text = query
                st.session_state.query_input = query
                st.rerun()

    # Query text area
    query_text = st.text_area(
        "What question do you have about your data:",
        value=st.session_state.query_text,
        height=100,
        placeholder=(
            "e.g., What transcription factors drive fibrosis in IPF AT2 cells?"
        ),
        key="query_input",
    )

    st.header("2. Upload Data")

    col_expr, col_meta = st.columns(2)

    with col_expr:
        expression_file = st.file_uploader(
            "Expression matrix (genes × samples)",
            type=["csv", "tsv", "h5ad"],
            key="expr_upload",
            help="CSV/TSV with genes as rows and samples as columns, or AnnData H5AD.",
        )

    with col_meta:
        metadata_file = st.file_uploader(
            "Metadata (sample annotations)",
            type=["csv", "tsv"],
            key="meta_upload",
            help=(
                "CSV with sample IDs and condition labels. "
                "Not needed if H5AD contains obs annotations."
            ),
        )

    deg_file = st.file_uploader(
        "Pre-computed DEG results (optional — skips DEG analysis)",
        type=["csv", "tsv"],
        key="deg_upload",
        help=(
            "CSV with gene, log2FoldChange, and padj columns. "
            "If provided, Agent 1 skips DESeq2 and uses these directly."
        ),
    )

    # Advanced options (collapsible)
    with st.expander(" Advanced Options", expanded=False):
        st.caption(
            "Agent 1 auto-detects most settings. "
            "Use these fields only if auto-detection fails or you want to override."
        )

        adv_col1, adv_col2 = st.columns(2)

        with adv_col1:
            cell_type_input = st.text_input(
                "Cell type (scRNA only)",
                value=st.session_state.cell_type or "",
                placeholder="e.g., AT2, Myofibroblasts",
                help="Required for single-cell data. Leave empty for bulk.",
            )
            condition_col_input = st.text_input(
                "Condition column",
                value=st.session_state.condition_column or "",
                placeholder="e.g., Diagnosis",
                help="Metadata column with disease/control labels.",
            )
            sample_col_input = st.text_input(
                "Sample/Donor column",
                value=st.session_state.sample_column or "",
                placeholder="e.g., Sample_Name",
                help="Column identifying biological replicates (for pseudobulk).",
            )

        with adv_col2:
            disease_input = st.text_input(
                "Disease label",
                value=st.session_state.disease_label or "",
                placeholder="e.g., IPF",
                help="Value in condition column for disease group.",
            )
            control_input = st.text_input(
                "Control label",
                value=st.session_state.control_label or "",
                placeholder="e.g., Control",
                help="Value in condition column for control group.",
            )

    # Validation and submit
    st.divider()

    has_query = bool(query_text and query_text.strip())
    has_expression = expression_file is not None
    has_degs = deg_file is not None
    has_data = has_expression or has_degs

    if not has_query:
        st.info("Enter a query or select a preset above to begin.")
    elif not has_data:
        st.info("Upload expression data or pre-computed DEGs to continue.")

    submit_disabled = not (has_query and has_data)

    if st.button(
        " Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=submit_disabled,
    ):
        # Save files to disk
        expr_path = save_uploaded_file(expression_file) if expression_file else None
        meta_path = save_uploaded_file(metadata_file) if metadata_file else None
        deg_path = save_uploaded_file(deg_file) if deg_file else None

        # Persist to session state
        st.session_state.query_text = query_text
        st.session_state.uploaded_expression = str(expr_path) if expr_path else None
        st.session_state.uploaded_metadata = str(meta_path) if meta_path else None
        st.session_state.uploaded_degs = str(deg_path) if deg_path else None

        # Persist advanced options (empty string → None)
        st.session_state.cell_type = cell_type_input.strip() or None
        st.session_state.condition_column = condition_col_input.strip() or None
        st.session_state.disease_label = disease_input.strip() or None
        st.session_state.control_label = control_input.strip() or None
        st.session_state.sample_column = sample_col_input.strip() or None

        # Transition to running
        st.session_state.pipeline_stage = "running_agent1"
        st.rerun()


def render_clarification() -> None:
    """
    Render clarification questions from Agent 1.

    When Agent 1 returns NEEDS_CLARIFICATION, the UI shows each question
    with options (if available) or free-text input. User answers are
    injected back into the query for a re-run.
    """
    result = st.session_state.agent1_result

    st.header("❓ Agent 1 Needs More Information")
    st.write(
        "The agent couldn't proceed without your input on the following. "
        "Please answer and click **Continue**."
    )

    responses = {}

    for i, clarification in enumerate(result.clarifications_needed):
        st.subheader(f"Question {i + 1}")
        st.write(clarification.question)

        if clarification.context:
            st.caption(clarification.context)

        if clarification.options:
            # Dropdown with options
            default_index = 0
            if clarification.default and clarification.default in clarification.options:
                default_index = clarification.options.index(clarification.default)

            responses[clarification.field] = st.selectbox(
                f"Select {clarification.field}:",
                options=clarification.options,
                index=default_index,
                key=f"clarify_{clarification.field}",
            )
        else:
            # Free-text input
            responses[clarification.field] = st.text_input(
                f"Enter {clarification.field}:",
                value=clarification.default or "",
                key=f"clarify_{clarification.field}",
            )

    st.divider()

    if st.button("▶ Continue", type="primary", use_container_width=True):
        # Map clarification responses back to UserQuery fields
        field_to_session_key = {
            "cell_type": "cell_type",
            "condition_column": "condition_column",
            "disease_label": "disease_label",
            "control_label": "control_label",
            "sample_column": "sample_column",
            "cell_type_column": "cell_type",  # maps to same key
        }

        for field, value in responses.items():
            # Disease mismatch: if user confirmed, set flag so re-run skips check
            if field == "disease_mismatch_confirmed":
                if "Yes" in value:
                    st.session_state["disease_mismatch_confirmed"] = True
                else:
                    st.warning("Please upload the correct dataset and re-run.")
                    st.session_state.pipeline_stage = "input"
                    st.rerun()
                continue
            # Threshold choice: parse selection and set override
            if field == "log2fc_threshold_choice":
                st.session_state["threshold_confirmed"] = True
                if "0.5" in value:
                    st.session_state["log2fc_threshold_override"] = 0.5
                else:
                    st.session_state["log2fc_threshold_override"] = 1.0
                continue
            session_key = field_to_session_key.get(field, field)
            st.session_state[session_key] = value

        st.session_state.clarification_responses = responses
        st.session_state.pipeline_stage = "running_agent1"
        st.rerun()


# =============================================================================
# PYVIS NETWORK VISUALIZATION
# =============================================================================

def get_node_type(G: nx.DiGraph, node: str) -> str:
    """Classify node type from graph attributes."""
    data = G.nodes[node]
    node_type = data.get('type', data.get('node_type', '')).lower()

    if node_type in ('tf', 'transcription_factor') or data.get('is_tf', False):
        return 'TF'
    elif node_type in ('measured', 'target', 'deg'):
        return 'measured'
    else:
        return 'intermediate'


def _build_pyvis_network_DISABLED(G: nx.DiGraph, top_n: int = 50, focus_gene: str = None,
                        deg_gene_set: set = None):
    """
    Build a PyVis Network from a NetworkX DiGraph.

    Args:
        G: CARNIVAL output network
        top_n: Number of top nodes by degree centrality to display
        focus_gene: If set, show only this gene + 2-hop neighborhood
        deg_gene_set: Set of DEG gene symbols (uppercase) for measured coloring

    Returns:
        Configured PyVis Network object
    """
    # --- Node selection ---
    if focus_gene and focus_gene in G.nodes:
        # 2-hop neighborhood subgraph
        neighbors_1 = set(G.predecessors(focus_gene)) | set(G.successors(focus_gene))
        neighbors_2 = set()
        for n in neighbors_1:
            neighbors_2 |= set(G.predecessors(n)) | set(G.successors(n))
        subgraph_nodes = {focus_gene} | neighbors_1 | neighbors_2
        subgraph = G.subgraph(subgraph_nodes)
    else:
        # Top N by degree centrality
        centrality = nx.degree_centrality(G)
        top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:top_n]
        subgraph = G.subgraph(top_nodes)

    # --- Build PyVis ---
    net = Network(
        height="600px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white"
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=100)

    # Color/size scheme
    type_colors = {
        'TF':           '#e74c3c',   # Red
        'measured':     '#3498db',   # Blue
        'signaling':    '#f39c12',   # Orange (legacy)
        'intermediate': '#f39c12',   # Orange (current graph value)
    }

    centrality = nx.degree_centrality(subgraph)
    max_c = max(centrality.values()) if centrality else 1

    # --- Add nodes ---
    for node in subgraph.nodes:
        node_type = get_node_type(subgraph, node)
        # Override: if node is in DEG gene set AND not already a TF, mark as measured
        if deg_gene_set and node.upper() in deg_gene_set and node_type != 'TF':
            node_type = 'measured'
        color = type_colors.get(node_type, '#95a5a6')
        # Size: 15-50px based on centrality
        size = 15 + (centrality[node] / max_c) * 35

        # Highlight focus gene
        border_color = "#FFD700" if node == focus_gene else color

        net.add_node(
            node,
            label=node,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": "#FFD700", "border": "#FFD700"}
            },
            size=size,
            title=f"<b>{node}</b><br>Type: {node_type}<br>Degree centrality: {centrality[node]:.3f}",
            shape="dot"
        )

    # --- Add edges ---
    for u, v, data in subgraph.edges(data=True):
        sign = data.get('sign', 0)
        if sign > 0:
            edge_color = '#2ecc71'
            edge_label = 'ACT'
        elif sign < 0:
            edge_color = '#e74c3c'
            edge_label = 'INH'
        else:
            edge_color = '#95a5a6'
            edge_label = ''

        net.add_edge(u, v, color=edge_color, title=edge_label, arrows="to", width=2)

    # --- Click event JS ---
    net.set_options("""
    var options = {
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)

    return net


def _get_pyvis_html_DISABLED(net) -> str:
    """Generate PyVis HTML for rendering in Streamlit."""
    return net.generate_html()


def _render_network_tab_pyvis_DISABLED(G: nx.DiGraph):
    """
    Complete network visualization UI component.
    Call this inside your Streamlit tab/section.
    """
    st.subheader("CARNIVAL Network Visualization")

    if G is None or len(G.nodes) == 0:
        st.info("Run the analysis first to see the network visualization.")
        return

    # Load DEG gene set for node coloring
    deg_gene_set = set()
    agent1_result = st.session_state.get("agent1_result")
    if agent1_result and hasattr(agent1_result, 'deg_results') and agent1_result.deg_results:
        deg_path = getattr(agent1_result.deg_results, 'deg_file_path', None)
        if deg_path:
            try:
                deg_df = pd.read_csv(deg_path)
                gene_col = next(
                    (c for c in ['gene', 'Gene', 'gene_name', 'gene_symbol', 'GENE'] if c in deg_df.columns),
                    deg_df.columns[0]
                )
                deg_gene_set = set(deg_df[gene_col].dropna().str.upper())
            except Exception:
                deg_gene_set = set()

    total_nodes = len(G.nodes)

    # --- Controls row ---
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        top_n = st.slider(
            "Number of nodes to display",
            min_value=min(10, total_nodes),
            max_value=total_nodes,
            value=min(50, total_nodes),
            step=5,
            help="Filter to top N nodes by degree centrality"
        )

    with col2:
        focus_gene = st.text_input(
            "Focus on gene (2-hop neighborhood):",
            placeholder="e.g. STAT3, TP53",
            key="focus_gene_input"
        ).strip().upper()

        if focus_gene and focus_gene not in G.nodes:
            st.warning(f"'{focus_gene}' not found in network. Available: {', '.join(list(G.nodes)[:5])}...")
            focus_gene = None

    with col3:
        st.write("")
        st.write("")
        if st.button("Show Full Network", on_click=lambda: st.session_state.update(focus_gene_input="")):
            focus_gene = None

    # --- Legend ---
    legend_cols = st.columns(5)
    with legend_cols[0]: st.markdown("**TF** (red)")
    with legend_cols[1]: st.markdown("**DEG (measured)** (blue)")
    with legend_cols[2]: st.markdown("**Intermediate** (orange)")
    with legend_cols[3]: st.markdown("**Activates** (green edge)")
    with legend_cols[4]: st.markdown("**Inhibits** (red edge)")

    # --- Build and render network ---
    with st.spinner("Building network..."):
        net = build_pyvis_network(G, top_n=top_n, focus_gene=focus_gene or None,
                                  deg_gene_set=deg_gene_set)
        html_content = get_pyvis_html(net)

    components.html(html_content, height=620, scrolling=False)

    # --- Node selector → auto-submit to Agent 3 ---
    st.divider()
    st.caption("Select a node to ask Agent 3 about:")
    sorted_nodes = sorted(G.nodes, key=lambda n: nx.degree_centrality(G)[n], reverse=True)
    selected_node = st.selectbox(
        "Select node to ask about:",
        options=[""] + sorted_nodes[:100],
        format_func=lambda x: f"{x} ({get_node_type(G, x)})" if x else "-- choose a node --",
        key="manual_node_select",
        label_visibility="collapsed"
    )

    if selected_node:
        if st.button(f"Ask about {selected_node}"):
            st.session_state.pending_node_question = (
                f"Tell me about {selected_node} and its role in this network"
            )
            st.rerun()


def render_running_agent1() -> None:
    """
    Execute Agent 1 and show progress.

    This function runs the pipeline and transitions to the next stage
    based on the result status.
    """
    st.header("Agent 1: Reviewing Your Data")

    progress_bar = st.progress(0, text="Initializing agent review...")

    try:
        progress_bar.progress(10, text="Validating data format...")
        logger.info(f"DEBUG UI: disease_label={st.session_state.disease_label}, control_label={st.session_state.control_label}")
        result = run_agent1(
            query_text=st.session_state.query_text,
            expression_path=st.session_state.uploaded_expression,
            metadata_path=st.session_state.uploaded_metadata,
            precomputed_deg_path=st.session_state.uploaded_degs,
            cell_type=st.session_state.cell_type,
            condition_column=st.session_state.condition_column,
            disease_label=st.session_state.disease_label,
            control_label=st.session_state.control_label,
            sample_column=st.session_state.sample_column,
        )

        st.session_state.agent1_result = result

        progress_bar.progress(100, text="Agent 1 complete!")

        # Route based on result status
        from agent1.schemas_agent1 import Agent1Status

        if result.status == Agent1Status.NEEDS_CLARIFICATION:
            st.session_state.pipeline_stage = "clarification"
            time.sleep(0.5)  # Brief pause so user sees progress complete
            st.rerun()

        elif result.status in (Agent1Status.SUCCESS, Agent1Status.SUCCESS_WITH_WARNINGS):
            # Show brief summary before auto-forwarding to Agent 2
            if result.explanation:
                st.success(result.explanation.summary)
            if result.status == Agent1Status.SUCCESS_WITH_WARNINGS:
                for warning in (result.quality_report.warnings if result.quality_report else []):
                    st.warning(warning)
                for warning in (result.deg_results.warnings if result.deg_results else []):
                    st.warning(warning)

            # 0-DEG guard: do NOT proceed to CARNIVAL if no significant DEGs
            if result.deg_results and result.deg_results.n_significant == 0:
                st.session_state.error_message = (
                    "No significant differentially expressed genes were found "
                    f"(padj < {result.deg_results.padj_threshold}, "
                    f"|log2FC| > {result.deg_results.log2fc_threshold}).\n\n"
                    "CARNIVAL requires DEGs as input to build a causal network. "
                    "Possible reasons:\n"
                    "- The selected disease and control groups may not differ transcriptionally\n"
                    "- The sample size may be too small to detect differences\n"
                    "- The thresholds may be too stringent for this dataset\n\n"
                    "Please check your disease/control label selection and try again."
                )
                st.session_state.pipeline_stage = "error"
                st.rerun()

            st.session_state.pipeline_stage = "running_agent2"
            time.sleep(1)  # Let user see the summary briefly
            st.rerun()

        elif result.status == Agent1Status.FAILED:
            error_msg = getattr(result, "error_message", None) or "Unknown error"
            st.session_state.error_message = error_msg
            st.session_state.pipeline_stage = "error"
            st.rerun()

    except ImportError as e:
        st.session_state.error_message = (
            f"Agent 1 module not found: {e}. "
            "Make sure agent1/ package is in the Python path."
        )
        st.session_state.pipeline_stage = "error"
        st.rerun()

    except Exception as e:
        logger.error(f"Agent 1 failed: {e}", exc_info=True)
        st.session_state.error_message = f"Agent 1 error: {type(e).__name__}: {e}"
        st.session_state.pipeline_stage = "error"
        st.rerun()


def render_running_agent2() -> None:
    """
    Execute Agent 2 (CARNIVAL) and show progress.
    """
    st.header("Agent 2: Building Causal Network")
    st.write("Running CARNIVAL pipeline — this may take several minutes...")

    progress_bar = st.progress(0, text="Initializing CARNIVAL ...")

    try:
        progress_bar.progress(20, text="Computing TF activities (CollecTRI)...")
        progress_bar.progress(40, text="Scoring pathways (PROGENy)...")
        progress_bar.progress(60, text="Building prior knowledge network (OmniPath)...")
        progress_bar.progress(80, text="Running CARNIVAL optimization...")

        agent2_result = run_agent2(st.session_state.agent1_result)
        st.session_state.agent2_result = agent2_result

        progress_bar.progress(100, text="Network construction complete!")

        if agent2_result.get("status") == "success":
            st.session_state.pipeline_stage = "results"
            time.sleep(0.5)
            st.rerun()
        else:
            error_msg = agent2_result.get("message", "Agent 2 failed")
            st.session_state.error_message = error_msg
            st.session_state.pipeline_stage = "error"
            st.rerun()

    except Exception as e:
        logger.error(f"Agent 2 failed: {e}", exc_info=True)
        st.session_state.error_message = f"Agent 2 error: {type(e).__name__}: {e}"
        st.session_state.pipeline_stage = "error"
        st.rerun()


def render_citations(citations: list, max_show: int = 3) -> None:
    """Render RAG citations grouped by paper with full chunk text and position."""
    if not citations:
        return
    from collections import OrderedDict
    grouped: OrderedDict = OrderedDict()
    for cite in citations:
        if not isinstance(cite, dict):
            continue
        source = cite.get('source', 'Unknown source')
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(cite)

    n_papers = len(grouped)
    label = f"Literature Sources ({n_papers} paper{'s' if n_papers != 1 else ''})"
    max_chunks_per_paper = 3
    with st.expander(label):
        for paper_idx, (source, chunks) in enumerate(grouped.items(), 1):
            source_name = source.replace('.pdf', '').replace('_', ' ')
            best_score = max(c.get('score', 0) for c in chunks)
            st.markdown(f"**[{paper_idx}] {source_name}** -- relevance: {best_score:.2f}")
            for chunk in chunks[:max_chunks_per_paper]:
                chunk_id = chunk.get('chunk_id', '?')
                total = chunk.get('total_chunks', '?')
                st.caption(f"[chunk {chunk_id}/{total}]")
                st.text(chunk.get('text', ''))
            if len(chunks) > max_chunks_per_paper:
                st.caption(f"({len(chunks) - max_chunks_per_paper} additional chunks omitted)")
            if paper_idx < n_papers:
                st.divider()


def render_reasoning_trace(result_obj) -> None:
    """Render Agent 3's ReAct reasoning trace as an expandable section."""
    trace = result_obj.reasoning_trace
    if not trace:
        return

    n = len(trace)
    with st.expander(f"ReAct Reasoning Trace ({n} step{'s' if n != 1 else ''})"):
        for step in trace:
            st.markdown(f"**Iteration {step.iteration}**")
            if step.thought:
                st.markdown(f"*Thought:* {step.thought}")
            st.markdown(f"*Action:* `{step.action}`")
            if step.input:
                params = ", ".join(f"{k}={v}" for k, v in step.input.items())
                st.caption(f"Parameters: {params}")
            st.markdown(f"*Observation:* {step.observation}")
            st.divider()


def render_results() -> None:
    """
    Render the results page and link to Agent 3.

    Shows Agent 1 summary, Agent 2 network stats, and provides
    a link to Agent 3's interactive interpretation UI (Flask app).
    """
    st.header("Analysis Complete")

    result = st.session_state.agent1_result
    agent2 = st.session_state.agent2_result

    # Agent 1 summary
    if result and result.explanation:
        st.subheader(" Data Preparation Summary")
        st.write(result.explanation.summary)

        if result.explanation.quality_interpretation:
            with st.expander("Quality Assessment Details"):
                st.write(result.explanation.quality_interpretation)

        if result.explanation.caveats:
            with st.expander("⚠️ Caveats & Limitations"):
                for caveat in result.explanation.caveats:
                    st.warning(caveat)

    # Network Confidence Assessment (from Agent 2 LLM)
    confidence_data = st.session_state.get("network_confidence", {})
    if confidence_data.get("confidence"):
        conf_level = confidence_data["confidence"]
        conf_color = {"high": "green", "medium": "orange", "low": "red"}.get(conf_level, "gray")
        conf_icon = {"high": ":white_check_mark:", "medium": ":warning:", "low": ":x:"}.get(conf_level, "")

        with st.expander(f"Network Confidence Assessment — {conf_level.upper()}", expanded=True):
            # Confidence badge
            st.markdown(f"### {conf_icon} Confidence: **{conf_level.upper()}**")

            # Reasoning (ample text space)
            st.markdown("#### Assessment")
            st.markdown(confidence_data.get("reasoning", ""))

            # Metrics summary
            metrics = confidence_data.get("metrics", {})
            if metrics:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Active Nodes", metrics.get("n_active_nodes", "—"))
                m2.metric("Edge/Node Ratio", metrics.get("edge_node_ratio", "—"))
                m3.metric("TF Recovery", f"{metrics.get('tf_recovery_pct', '?')}%")
                m4.metric("DEGs in Network", f"{metrics.get('n_measured_in_network', '?')}")

            # Helper: LLM may return a string instead of list — normalise
            def _as_list(val):
                if isinstance(val, str):
                    return [val] if val.strip() else []
                if isinstance(val, list):
                    return val
                return []

            # Strengths
            strengths = _as_list(confidence_data.get("strengths", []))
            if strengths:
                st.markdown("#### Strengths")
                for s in strengths:
                    st.markdown(f"- {s}")

            # Limitations
            limitations = _as_list(confidence_data.get("limitations", []))
            if limitations:
                st.markdown("#### Limitations")
                for lim in limitations:
                    st.markdown(f"- {lim}")

            # Suggestions (only for medium/low)
            suggestions = _as_list(confidence_data.get("suggestions", []))
            if suggestions:
                st.markdown("#### Suggestions for Improvement")
                for sug in suggestions:
                    st.markdown(f"- {sug}")

    # KEGG ORA — pathway enrichment on CARNIVAL nodes
    carnival_net = st.session_state.get("carnival_network")
    if carnival_net is not None and carnival_net.number_of_nodes() > 0:
        with st.expander("KEGG Pathway Enrichment (ORA)", expanded=False):
            try:
                ora_df = run_kegg_ora(carnival_net)
                if ora_df.empty:
                    st.info("No pathway overlaps found.")
                else:
                    sig = ora_df[ora_df['padj'] < 0.05]
                    if len(sig) > 0:
                        st.success(f"{len(sig)} significant pathways (padj < 0.05)")
                        display_df = sig[['pathway', 'direction', 'overlap_count', 'pval', 'padj', 'overlap_genes']].copy()
                    else:
                        st.info("No pathways reached significance (padj < 0.05). Showing top 10 by p-value.")
                        display_df = ora_df[['pathway', 'direction', 'overlap_count', 'pval', 'padj', 'overlap_genes']].head(10).copy()
                    display_df['pathway'] = display_df['pathway'].str.replace('KEGG_', '').str.replace('_', ' ').str.title()
                    display_df = display_df.rename(columns={
                        'pathway': 'Pathway', 'direction': 'Direction',
                        'overlap_count': 'Overlap', 'pval': 'P-value',
                        'padj': 'Adj. P-value', 'overlap_genes': 'Genes',
                    })
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"KEGG ORA failed: {e}")
                logger.error(f"KEGG ORA error: {e}", exc_info=True)

    # Agent 3 — in-process interpretation
    st.divider()
    st.subheader("Explore Your Network with Agent 3")
    st.write(
        "Your causal network is ready for interactive exploration. "
        "Agent 3 uses RAG-powered interpretation to answer biological questions "
        "about your network with citation-backed evidence."
    )

    # --- Network visualization (st-link-analysis) ---
    carnival_net = st.session_state.get("carnival_network")
    if carnival_net is not None:
        render_network_tab(
            G=carnival_net,
            metadata=st.session_state.get("agent2_metadata"),
        )

    # --- Chat interface ---
    st.divider()
    st.subheader("Ask a Question")

    # Display conversation history
    for exchange in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.write(exchange["question"])
        with st.chat_message("assistant"):
            st.write(exchange["answer"])
            render_citations(exchange.get("citations", []))
            if exchange.get("cost"):
                st.caption(f"Cost: ${exchange['cost']:.4f}")

    # Auto-forward original question to Agent 3 on first results render
    if not st.session_state.conversation_history and st.session_state.get("query_text"):
        st.session_state["chat_prefill"] = st.session_state.query_text

    # Auto-submit from network info panel (chat_prefill) or legacy node selector
    pending_q = st.session_state.pop("chat_prefill", None) or st.session_state.pop("pending_node_question", None)
    if pending_q:
        with st.chat_message("user"):
            st.write(pending_q)
        with st.chat_message("assistant"):
            with st.spinner("Agent 3 is analyzing your network..."):
                response = run_agent3(pending_q)
            if response["status"] == "success":
                result_obj = response["result"]
                if result_obj.warning:
                    st.warning(result_obj.warning)
                st.write(result_obj.answer or "Agent 3 completed analysis. See citations below.")
                render_reasoning_trace(result_obj)
                render_citations(result_obj.citations)
                st.caption(f"Cost: ${result_obj.cost_estimate:.4f}")
            else:
                st.error(response.get("answer", "Unknown error"))

    # Manual chat input
    if question := st.chat_input("Ask about your causal network..."):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Agent 3 is analyzing your network..."):
                response = run_agent3(question)
            if response["status"] == "success":
                result_obj = response["result"]
                if result_obj.warning:
                    st.warning(result_obj.warning)
                st.write(result_obj.answer or "Agent 3 completed analysis. See citations below.")
                render_reasoning_trace(result_obj)
                render_citations(result_obj.citations)
                st.caption(f"Cost: ${result_obj.cost_estimate:.4f}")
            else:
                st.error(response.get("answer", "Unknown error"))

    # Download outputs
    st.divider()
    st.subheader(" Download Outputs")

    if result and result.expression_matrix_path:
        expr_path = Path(result.expression_matrix_path)
        if expr_path.exists():
            with open(expr_path, "rb") as f:
                st.download_button(
                    "Download Clean Expression Matrix",
                    data=f.read(),
                    file_name="expression_matrix_clean.csv",
                    mime="text/csv",
                )

    if result and result.deg_results and result.deg_results.deg_file_path:
        deg_path = Path(result.deg_results.deg_file_path)
        if deg_path.exists():
            with open(deg_path, "rb") as f:
                st.download_button(
                    "Download DEG Results",
                    data=f.read(),
                    file_name="deg_results.csv",
                    mime="text/csv",
                )

    # JSON result for debugging / Agent 2 handoff
    if result:
        with st.expander(" Raw Agent 1 Result (JSON)"):
            st.json(result.model_dump(mode="json"))


def render_error() -> None:
    """Render error state with recovery options."""
    st.header("Something went wrong")

    error_msg = st.session_state.error_message or "An unknown error occurred."

    # Split into first line (summary) and rest (details)
    lines = error_msg.strip().split("\n", 1)
    st.error(lines[0])
    if len(lines) > 1:
        st.markdown(lines[1])

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Retry with Same Inputs", use_container_width=True):
            st.session_state.pipeline_stage = "running_agent1"
            st.session_state.error_message = None
            st.rerun()

    with col2:
        if st.button("Start Over", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# =============================================================================
# MAIN APPLICATION: Route to the correct stage
# =============================================================================

def main() -> None:
    """
    Main application entry point.

    Routes to the appropriate UI based on the current pipeline stage.
    Each stage is a self-contained function that manages its own
    display and state transitions.
    """
    init_session_state()
    render_header()
    render_sidebar()

    # Route to current stage
    stage = st.session_state.pipeline_stage

    if stage == "input":
        render_input_form()
    elif stage == "clarification":
        render_clarification()
    elif stage == "running_agent1":
        render_running_agent1()
    elif stage == "running_agent2":
        render_running_agent2()
    elif stage == "results":
        render_results()
    elif stage == "error":
        render_error()
    else:
        st.error(f"Unknown pipeline stage: {stage}")
        st.session_state.pipeline_stage = "input"
        st.rerun()


if __name__ == "__main__":
    main()
