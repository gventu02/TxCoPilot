# RAG Vector Database — Architecture and Build Guide

This file describes the retrieval-augmented generation (RAG) system used by Agent 3
to retrieve relevant literature when answering network biology questions.

---

## Pre-built index

A pre-built index covering 54 thesis-relevant papers is already included in this
repository at `Agent_3/rag_system/vector_db/`. User may need to rebuild it to populate with papers that pertain to their area of research.

Rebuild only if you are adding new papers to the `papers/` directory.

---

## Architecture

```
PDF files
   │
   ▼
Text Extraction (pypdf)
   │  one string per PDF page, concatenated
   ▼
Reference Stripping (_strip_references_section)
   │  truncates at "References" / "Bibliography" / "Literature Cited"
   │  header found in the last 40 % of the document
   ▼
Chunking (1000 chars, 200-char overlap, sentence-boundary-aware)
   │  min chunk size 100 chars
   ▼
Bibliography Filter (_is_bibliography_chunk)
   │  drops chunks dense with "et al.", numbered refs, DOIs, year-volume patterns
   ▼
Embedding  (pritamdeka/S-PubMedBert-MS-MARCO — biomedical, 768 dims)
   │  batch size 32, L2-normalised for cosine similarity
   ▼
FAISS IndexFlatL2  (exact nearest-neighbour search)
   │  scores converted: cosine_sim = 1 - squared_L2 / 2
   ▼
Saved to disk:
   index.faiss     FAISS binary index
   chunks.pkl      parallel list of text strings
   metadata.pkl    source filename + chunk_id per chunk
   config.json     model name, embedding dim, doc/chunk counts
```

---

## Prerequisites

All runtime packages (sentence-transformers, faiss-cpu, torch) are already in
`requirements.txt`. The PDF ingestion library is listed separately:

```
pip install pypdf==6.7.4
```

---

## Building the index

Run from the repository root with `CS_150` activated:

```bash
python Agent_3/rag_system/rag_system.py \
    --build \
    --pdf_dir  Agent_3/rag_system/papers \
    --output   Agent_3/rag_system/vector_db
```

Expected output (abbreviated):

```
============================================================
Building RAG Index
============================================================
Found 54 PDF files
✓ 2019Liu_From Expression Footprints...pdf: 14 pages, 54231 characters
  2019Liu_From Expression Footprints...pdf: stripped 8412 chars of references
...
Chunking documents...
Filtered 47 bibliography-like chunks
Created 3821 chunks from 54 documents
Generating embeddings (this may take 1-2 minutes)...
100%|████████████████████| 3821/3821
Building FAISS index...
Index built successfully: 3821 vectors

============================================================
✓ Index built successfully!
  Documents: 54
  Chunks:    3821
  Saved to:  Agent_3/rag_system/vector_db
============================================================
```

Time estimate: 2–5 minutes on CPU (one-time cost).

---

## Verifying the index

```bash
python Agent_3/rag_system/rag_system.py \
    --stats \
    --db_path Agent_3/rag_system/vector_db
```

---

## Test query

```bash
python Agent_3/rag_system/rag_system.py \
    --query "What transcription factors regulate TGF-beta fibrosis signaling?" \
    --db_path Agent_3/rag_system/vector_db \
    --top_k 5
```

---

## Using programmatically

```python
from Agent_3.rag_system.rag_system import RAGSystem

rag = RAGSystem.load("Agent_3/rag_system/vector_db")
results = rag.retrieve("What is CARNIVAL?", top_k=5)

for r in results:
    print(f"Source: {r['source']}  Score: {r['score']:.3f}")
    print(r['text'][:300])
```

---

## Papers indexed (54 PDFs)

### Network optimization / CARNIVAL
- Liu 2019 — From Expression Footprints to Causal Pathways (CARNIVAL, main paper + supplement)
- Gjerga 2021 — Contextualizing Large-Scale Signalling Networks (CARNIVAL Vignette)
- Terfve 2012 — CellNOptR: Flexible Toolkit to Train Protein Signaling Networks
- Melas 2015 — Identification of Drug-Specific Pathways Based on Gene Expression Data

### Transcription factor activity
- Garcia-Alonso 2019 — Benchmark and Integration of Resources for TF Activity Estimation (DoRothEA)
- Müller-Dott 2023 — Expanding Coverage of Regulons from High-Confidence Prior Knowledge (CollecTRI)
- Schubert 2018 — Perturbation-Response Genes Reveal Signaling Footprints in Cancer (PROGENy)
- Binenbaum 2020 — Bioinformatic Framework for Transcription Factor Changes (senescence/signaling)
- Bild 2006 — Oncogenic Pathway Signatures in Human Cancers as a Guide to Targeted Therapies

### Prior knowledge networks (OmniPath)
- Türei 2016 — OmniPath Guidelines and Gateway (+ supplement)
- Türei 2020 — Integrated Intra- and Intercellular Signaling Knowledge
- Türei 2021 — Integrated Intra- and Intercellular Signaling Knowledge (updated)

### IgA Nephropathy
- Kiryluk 2023 — Genome-Wide Association Analyses Define Pathogenic Signaling Pathways in IgAN
- Tang 2021 — A Partial Picture of Single-Cell Transcriptomics of Human IgAN
- Yeo 2018 — New Insights into the Pathogenesis of IgA Nephropathy
- Maillard 2015 — Current Understanding of Complement in IgA Nephropathy
- Cheung 2015 — Pathogenesis of IgA Nephropathy and Implications for Treatment
- Lai 2012 — Pathogenesis of IgA Nephropathy
- Wade 2026 — Complement and Kidney Diseases: Targeted Treatments for IgAN
- Medjeral-Thomas 2021 — Complement Activation in IgA Nephropathy
- Mizerska-Wasiak 2023 — IgA and C3 Biomarkers in IgA Nephropathy
- Wang 2025 — Causal Relationships of JAK2 with Chronic Kidney Diseases
- Cox 2010 — Altered WNT/β-catenin and PI3K/Akt Pathways in IgAN
- Tamouza 2012 — IgA1 Immune Complex Activation of MAPK/ERK in Mesangial Cells
- Leung 2008 — TNF-α Synthesis by Mesangial Cells: MAPK and NF-κB

### Kidney fibrosis
- Yamashita 2024 — Mechanisms of Kidney Fibrosis and Routes Towards Therapy
- Park 2018 — Single-Cell Transcriptomics of Mouse Kidney (Potential Targets of Kidney Disease)
- Buhl 2020 — Dysregulated Mesenchymal PDGFR-β Drives Kidney Fibrosis
- Rayego-Mateos 2018 — CTGF Induces Renal Fibrosis via EGFR Activation
- Zhou 2015 — Wnt/β-catenin Signalling and Podocyte Dysfunction in Proteinuric Kidney Disease
- Tang 2018 — TGF-β Signalling in Renal Fibrosis: From Smads to Non-coding RNAs

### Liver / NASH
- Ramachandran 2019 — Resolving the Fibrotic Niche of Human Liver Cirrhosis at Single-Cell Level
- MacParland 2018 — Single-Cell RNA Sequencing of Human Liver (Intrahepatic Macrophage Populations)
- Starmann 2012 — Gene Expression Profiling in Steatohepatitis vs. Steatosis
- McPherson 2015 — Evidence of NAFLD Progression from Steatosis to Fibrosing-Steatohepatitis
- Friedman 2008 — Mechanisms of Hepatic Fibrogenesis
- Sia 2016 — Liver Cancer: Cell of Origin, Molecular Class, and Patient Prognosis

### Pulmonary fibrosis / IPF
- Moss 2021 — Pathogenic Mechanisms Underlying Idiopathic Pulmonary Fibrosis
- Richeldi 2017 — Idiopathic Pulmonary Fibrosis (Lancet review)
- Günther 2012 — Unravelling the Progressive Pathophysiology of Idiopathic Pulmonary Fibrosis
- Razzaque 2003 — Pulmonary Fibrosis: Cellular and Molecular Events
- Gibson 2003 — Pathophysiology and Management of Pulmonary Infections in Cystic Fibrosis

### Signaling pathways
- Nagaraj 2009 — Targeting TGF-β Signaling Pathway in Human Cancer
- Xu 2021 — MicroRNAs in TGF-β Signaling Pathway Associated with Fibrosis
- Rustad 2013 — Role of Focal Adhesion Complexes in Fibroblast Mechanotransduction
- Meng 2009 — Adherens Junctions: Molecular Architecture and Regulation
- Harris 2010 — Adherens Junctions: From Molecules to Morphogenesis
- Cui 2018 — Is β-Catenin a Druggable Target for Cancer Therapy?

### Single-cell / tumor microenvironment
- Qian 2020 — A Pan-Cancer Blueprint of the Heterogeneous Tumor Microenvironment (scRNA-seq)
- Deprez 2020 — A Single-Cell Atlas of the Human Healthy Airways
- Triantafyllidis 2023 — Machine Learning and Directed Network Optimization for TP53 Regulatory Patterns
