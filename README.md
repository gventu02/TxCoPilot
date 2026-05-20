# TxCoPilot: Transcriptomics CoPilot

A conversational AI pipeline for causal network inference from transcriptomics data.
Given an RNA-seq or microarray dataset and a plain-English biological question,
TxCoPilot identifies which upstream signaling proteins are most likely driving the
observed gene expression changes, then explains the results in biological language
with supporting literature citations.

Built as a master's thesis project at Tufts University (2026).

---
# Authors
Giselle Ventura, Soha Hassoun
Department of Computer Science, Tufts University

## How It Works

```
User Input: data files (CSV) + natural language query
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Agent 1 — Data Preparation & QC                    │
│  • Parses your query (disease/control labels,        │
│    analysis type, thresholds)                        │
│  • Validates and normalizes gene IDs                 │
│  • Runs differential expression analysis             │
│    (PyDESeq2 for RNA-seq / limma for microarray)     │
└──────────────────────┬──────────────────────────────┘
                       │ Agent1Result (JSON)
                       ▼
┌─────────────────────────────────────────────────────┐
│  Agent 2 — Causal Network Inference                 │
│  • Scores TF activity                               |
| (CollecTRI or Dorothea via VIPER)                   │
│  • Scores pathway activity (PROGENy)                │
│  • Builds prior knowledge network (OmniPath,        │
│    ~70K edges)                                      │
│  • Runs CARNIVAL ILP to find the minimal signaling  │
│    subnetwork consistent with observed TF activity  │
└──────────────────────┬──────────────────────────────┘
                       │ Agent2Result (NetworkX DiGraph)
                       ▼
┌─────────────────────────────────────────────────────┐
│  Agent 3 — Biological Interpretation                │
│  • ReAct reasoning loop over network tools          │
│  • RAG over domain papers (PubMedBERT + FAISS)     │
│    — build index from your own paper copies         │
│  • Answers original question with citations         │
│  Note: KEGG pathway enrichment is shown in the UI  │
│  as a supplementary reference, it is NOT used as  │
│  input to network construction.                     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
         Streamlit UI: interactive Cytoscape network
                       + follow-up chat
```

> **TF inference method:** Agent 1 detects the input data type and selects the regulon automatically.
> CollecTRI (decoupler ULM) is used for RNA-seq and scRNA-seq. The IgAN demo (GSE37460, microarray)
> uses DoRothEA confidence tiers A/B/C, which gives better TF recovery on microarray platforms.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 | Conda recommended |
| Anthropic API key | — | Required for all three agents |
| HiGHS MIP solver | bundled | Installed automatically via `highspy` |

No commercial solver license is required — HiGHS is open source.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/gventu02/TxCoPilot.git
cd TxCoPilot

# 2. Create conda environment
conda create -n txcopilot python=3.11
conda activate txcopilot

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Anthropic API key
#    Linux / macOS:
export ANTHROPIC_API_KEY=""
#    Windows CMD:
set ANTHROPIC_API_KEY=sk-ant-...
#    Or add it to a .env file (never commit this):
echo ANTHROPIC_API_KEY=sk-ant-... > .env

# 5. Protect your API key, do this before your first commit
echo ".env" >> .gitignore
```

---

## Quick Start IgAN Demo (GSE37460)

Sample data is provided in `data/`:

```
data/
├── GSE37460_expression.csv   # probes × samples, log2 microarray intensities (Entrez IDs)
├── GSE37460_metadata.csv     # sample → condition mapping (IgAN / Control)
└── GSE37460_DEGs.csv         # pre-computed limma DEGs (gene, log2FoldChange, padj)
```

**Launch the UI:**

```bash
cd Agent_1
streamlit run streamlit_app.py
```

**Run the pipeline:**

1. Upload `GSE37460_expression.csv` and `GSE37460_metadata.csv`
2. Enter query: *"What signaling pathways drive IgAN progression compared to healthy controls?"*
3. Disease label: `IgAN` · Control label: `Control`
4. Click **Run Pipeline**

**Expected output:** ~77 nodes · ~75 edges · May vary depending on floating point estimations of solver * (padj < 0.05)

---

## Project Structure

```
TxCoPilot/
├── Agent_1/
│   ├── streamlit_app.py          # UI entry point — run this
│   └── agent1/
│       ├── agent1_main.py        # Step orchestrator (A → F)
│       ├── schemas_agent1.py     # Agent1Result — the Agent 1 → 2 interface
│       ├── config.py             # Runtime configuration and defaults
│       ├── core/
│       │   ├── data_validator.py     # Input file validation and QC
│       │   ├── deg_analysis.py       # PyDESeq2 DEG wrapper
│       │   ├── gene_id_normalizer.py # Gene symbol normalization (MyGene.info)
│       │   ├── limma_deg.py          # limma/inmoose microarray DEG wrapper
│       │   ├── pseudobulk.py         # Pseudobulk aggregation for scRNA-seq
│       │   └── quality_assessment.py # Sample QC metrics
│       ├── llm/
│       │   ├── entity_extraction.py       # Query parsing: labels, thresholds, data type
│       │   ├── clarification_generator.py # LLM-generated clarification questions
│       │   └── output_explanation.py      # Natural-language result summaries
│       └── validation/
│           └── gene_validation.py    # Gene symbol validation against PKN
│
├── Agent_2/
│   ├── agent2_main.py            # Pipeline orchestrator
│   ├── schemas_agent2.py         # Agent2Result schema
│   ├── core/
│   │   ├── tf_activity.py        # TF activity inference (DoRothEA / CollecTRI via decoupler)
│   │   ├── pkn_builder.py        # OmniPath PKN construction & filtering
│   │   ├── corneto_runner.py     # CARNIVAL ILP via CORNETO + HiGHS
│   │   ├── pathway_scoring.py    # PROGENy pathway activity scoring
│   │   ├── kegg_ora.py           # KEGG pathway over-representation analysis
│   │   ├── carnival_cache.py     # CARNIVAL result caching
│   │   └── output_packager.py    # Builds Agent2Result
│   └── llm/
│       ├── parameter_selector.py # LLM-driven beta/penalty selection
│       └── network_assessor.py   # LLM network quality assessment
│
├── Agent_3/
│   ├── agent3_react_loop.py      # ReAct reasoning loop + tool dispatch
│   ├── agent3_networkx_tools.py  # NetworkX analysis tool implementations
│   ├── agent3_visualizer_nx.py   # Matplotlib subnetwork visualization
│   ├── schemas_agent3.py         # Agent 3 config and result schemas
│   └── rag_system/
│       ├── rag_system.py             # RAG build + query engine (PubMedBERT + FAISS)
│       ├── vectordb_explain.txt      # Index build guide and indexed paper list
│       └── BUILD_RAG_INDEX.md        # Step-by-step index rebuild instructions
│
├── data/                         # Sample datasets (GSE37460 IgAN)
├── notebooks/                    # Reproducibility notebooks
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Reproducing Thesis Results

See `notebooks/reproduce_igan.ipynb` for the full GSE37460 IgAN pipeline.

> **RAG index:** The FAISS vector index is not included in this repository — `chunks.pkl`
> contains text extracted from copyrighted papers and cannot be distributed publicly.
> To build the index, obtain the source papers and follow the instructions in
> `Agent_3/rag_system/vectordb_explain.txt`. Agent 3 runs without RAG if no index is found
> (answers will lack literature citations).

| Parameter | Value |
|---|---|
| Dataset | GSE37460 (microarray, GPL14663) |
| Comparison | IgAN vs Control |
| DEG method | limma |
| padj threshold | 0.05 |
| \|log2FC\| threshold | 1.0 |
| CARNIVAL beta | 0.2 |
| PKN | OmniPath full (~70,274 edges) |
| **Expected nodes/edges** | **~77 / ~75** |

---

## Limitations

- **Two-group comparisons only.** The pipeline compares exactly one disease label against one control label. Multi-stage comparisons (e.g., F3+F4 vs F0-F1) require pre-merged metadata.
- **CARNIVAL ILP tree constraint.** The CORNETO ILP formulation enforces ≤1 incoming edge per node, producing tree-structured (not DAG) networks. The edge/node ratio is structurally capped at ~0.99.
- **OmniPath PKN coverage.** The prior knowledge network covers ~6,350 human proteins. Genes absent from OmniPath cannot appear as network nodes.

---

## Citation

Ventura, G. (2026). *TxCoPilot: A multi-agent AI system for causal network inference
from transcriptomics data.* M.S. Thesis, Tufts University.
