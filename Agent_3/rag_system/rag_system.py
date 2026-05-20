"""
RAG System for Agent 3 - Network Biology Knowledge Base
========================================================

This module builds and queries a vector database from PDF papers

Usage:
    # One-time setup: Build index from PDFs
    python rag_system.py --build --pdf_dir /mnt/project --output ./vector_db
    
    # Query the index
    python rag_system.py --query "What is betweenness centrality?" --db_path ./vector_db
    
    # Or use programmatically:
    from rag_system import RAGSystem
    rag = RAGSystem.load("./vector_db")
    results = rag.retrieve("What transcription factors regulate fibrosis?", top_k=5)
"""

import os
import re
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Core dependencies
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Complete RAG system for document retrieval.
    
    Architecture:
        PDFs → Text Extraction → Chunking → Embedding → FAISS Index → Retrieval
    
    Design Decisions:
        - Chunk size: 1000 chars (captures full paragraphs)
        - Overlap: 200 chars (preserves context at boundaries)
        - Model: pritamdeka/S-PubMedBert-MS-MARCO (biomedical domain, 768 dims)
        - Index: FAISS FlatL2 (exact search, small dataset)
    """
    
    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        """
        Initialize RAG system.

        Args:
            model_name: Sentence transformer model to use
                       Options:
                       - pritamdeka/S-PubMedBert-MS-MARCO: Biomedical domain, 768 dims (recommended)
                       - all-MiniLM-L6-v2: Fast, 384 dims, general-purpose
                       - all-mpnet-base-v2: Better quality, 768 dims, slower
        """
        self.model_name = model_name
        self.model = None  # Lazy load
        self.index = None
        self.chunks = None
        self.metadata = None
        
    def _load_model(self):
        """Lazy load the embedding model (saves startup time)"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
    
    @staticmethod
    def _strip_references_section(text: str) -> str:
        """
        Remove bibliography/references section from extracted PDF text.

        Looks for common section headers like "References", "Bibliography",
        "Literature Cited", etc. and truncates at the earliest match that
        appears in the last 40% of the document (to avoid false positives
        from mid-text mentions like "see References section").
        """
        # Only search the last 40% of text for reference headers
        cutoff_start = int(len(text) * 0.6)
        tail = text[cutoff_start:]

        # Patterns for reference section headers (case-insensitive)
        # Must be at start of line or after newline, possibly with a number prefix
        patterns = [
            r'\n\s*(?:\d+\.?\s+)?References\s*\n',
            r'\n\s*(?:\d+\.?\s+)?REFERENCES\s*\n',
            r'\n\s*Bibliography\s*\n',
            r'\n\s*BIBLIOGRAPHY\s*\n',
            r'\n\s*Literature\s+Cited\s*\n',
            r'\n\s*LITERATURE\s+CITED\s*\n',
            r'\n\s*Works\s+Cited\s*\n',
        ]

        earliest_pos = None
        for pattern in patterns:
            match = re.search(pattern, tail)
            if match:
                pos = cutoff_start + match.start()
                if earliest_pos is None or pos < earliest_pos:
                    earliest_pos = pos

        if earliest_pos is not None:
            return text[:earliest_pos].rstrip()
        return text

    @staticmethod
    def _is_bibliography_chunk(text: str) -> bool:
        """
        Heuristic check: does this chunk look like a bibliography/reference list?

        Catches chunks that slip through _strip_references_section (e.g., when
        the header format is unusual or references span multiple pages).

        Signals: dense "et al." usage, numbered citation patterns, DOI/URL density.
        """
        # Count bibliography signals
        et_al_count = len(re.findall(r'et\s+al\.', text))
        # Numbered refs like "1." "2." "23." at line starts
        numbered_refs = len(re.findall(r'(?:^|\n)\s*\d{1,3}\.\s+[A-Z]', text))
        # DOIs and URLs
        doi_count = len(re.findall(r'doi[:/]|https?://', text, re.IGNORECASE))
        # Year-volume-page patterns like "(2019)" or "2019;45:123-130"
        year_patterns = len(re.findall(r'\(\d{4}\)|\d{4};\d+:', text))

        total_signals = et_al_count + numbered_refs + doi_count + year_patterns

        # If chunk has 5+ bibliography signals, it's likely a reference list
        # (normal prose rarely has more than 1-2 "et al." per 1000 chars)
        return total_signals >= 5

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> Dict[str, any]:
        """
        Extract all text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with:
                - source: filename
                - text: extracted text
                - num_pages: page count
                
        Raises:
            Exception: If PDF cannot be read
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:  # Some PDFs have empty pages
                    text += page_text + "\n"
            
            return {
                "source": Path(pdf_path).name,
                "text": text,
                "num_pages": len(reader.pages),
                "path": pdf_path
            }
        except Exception as e:
            raise Exception(f"Failed to extract text from {pdf_path}: {str(e)}")
    
    @staticmethod
    def chunk_text(text: str, 
                   chunk_size: int = 1000, 
                   overlap: int = 200,
                   min_chunk_size: int = 100) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Strategy:
            - Target chunk_size characters per chunk
            - Overlap to preserve context at boundaries
            - Try to break at sentence boundaries (periods)
            - Filter out very small chunks (likely noise)
        
        Args:
            text: Text to chunk
            chunk_size: Target characters per chunk
            overlap: Characters to overlap between chunks
            min_chunk_size: Discard chunks smaller than this
            
        Returns:
            List of text chunks
            
        Example:
            >>> text = "First sentence. Second sentence. Third sentence."
            >>> chunks = chunk_text(text, chunk_size=20, overlap=5)
            >>> # Returns overlapping chunks that preserve sentence boundaries
        """
        if not text or len(text.strip()) == 0:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + chunk_size
            
            # Don't go past text end
            if end >= text_length:
                chunk = text[start:].strip()
                if len(chunk) >= min_chunk_size:
                    chunks.append(chunk)
                break
            
            # Try to break at sentence boundary (period followed by space)
            chunk = text[start:end]
            
            # Look for last period in the second half of chunk
            # (ensures we don't break too early)
            search_start = len(chunk) // 2
            last_period = chunk.rfind('. ', search_start)
            
            if last_period != -1:
                # Found a good break point
                end = start + last_period + 1  # Include the period
                chunk = text[start:end].strip()
            
            # Add chunk if it's substantial enough
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)
            
            # Move start position (with overlap)
            prev_start = start
            start = end - overlap
            
            # Ensure we're making progress (don't go backwards)
            if start <= prev_start:
                start = end  # Skip overlap if it causes us to go backwards
        
        return chunks
    
    def build_index(self, 
                    pdf_directory: str,
                    chunk_size: int = 1000,
                    overlap: int = 200) -> Tuple[int, int]:
        """
        Build vector index from all PDFs in a directory.
        
        Process:
            1. Extract text from all PDFs
            2. Chunk each document
            3. Generate embeddings for all chunks
            4. Build FAISS index
            5. Store chunks and metadata
        
        Args:
            pdf_directory: Directory containing PDF files
            chunk_size: Characters per chunk
            overlap: Character overlap between chunks
            
        Returns:
            (num_documents, num_chunks) tuple
            
        Raises:
            ValueError: If no PDFs found or all PDFs fail to process
        """
        self._load_model()
        
        # Find all PDFs
        pdf_paths = list(Path(pdf_directory).glob("*.pdf"))
        if not pdf_paths:
            raise ValueError(f"No PDF files found in {pdf_directory}")
        
        logger.info(f"Found {len(pdf_paths)} PDF files")
        
        # Extract text from all PDFs
        documents = []
        failed_pdfs = []
        
        for pdf_path in pdf_paths:
            try:
                doc = self.extract_text_from_pdf(str(pdf_path))
                
                # Skip empty documents
                if len(doc["text"].strip()) < 100:
                    logger.warning(f"Skipping {doc['source']}: too little text")
                    continue
                
                documents.append(doc)
                logger.info(f"✓ {doc['source']}: {doc['num_pages']} pages, "
                          f"{len(doc['text'])} characters")
                
            except Exception as e:
                failed_pdfs.append((pdf_path.name, str(e)))
                logger.error(f"✗ {pdf_path.name}: {e}")
        
        if not documents:
            raise ValueError("No documents successfully processed")
        
        if failed_pdfs:
            logger.warning(f"Failed to process {len(failed_pdfs)} PDFs")
        
        # Strip reference/bibliography sections before chunking
        for doc in documents:
            original_len = len(doc["text"])
            doc["text"] = self._strip_references_section(doc["text"])
            stripped = original_len - len(doc["text"])
            if stripped > 0:
                logger.info(f"  {doc['source']}: stripped {stripped} chars of references")

        # Chunk all documents
        all_chunks = []
        all_metadata = []
        bib_filtered = 0

        logger.info("Chunking documents...")
        for doc in documents:
            chunks = self.chunk_text(doc["text"], chunk_size, overlap)

            for i, chunk in enumerate(chunks):
                # Skip chunks that look like bibliography entries
                if self._is_bibliography_chunk(chunk):
                    bib_filtered += 1
                    continue
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": doc["source"],
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "source_path": doc["path"]
                })

        if bib_filtered > 0:
            logger.info(f"Filtered {bib_filtered} bibliography-like chunks")
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Generate embeddings
        logger.info("Generating embeddings (this may take 1-2 minutes)...")
        embeddings = self.model.encode(
            all_chunks, 
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32  # Process in batches for efficiency
        )
        embeddings = embeddings.astype('float32')
        
        logger.info(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        
        # Use FlatL2 for exact search (good for datasets < 1M vectors)
        # For larger datasets, could use IndexIVFFlat or IndexHNSWFlat
        index = faiss.IndexFlatL2(dimension)
        
        # Normalize embeddings for cosine similarity
        # (L2 distance on normalized vectors = cosine similarity)
        faiss.normalize_L2(embeddings)
        
        # Add to index
        index.add(embeddings)
        
        logger.info(f"Index built successfully: {index.ntotal} vectors")
        
        # Store everything
        self.index = index
        self.chunks = all_chunks
        self.metadata = all_metadata
        
        return len(documents), len(all_chunks)
    
    def retrieve(self, 
                 query: str, 
                 top_k: int = 5,
                 return_scores: bool = True) -> List[Dict]:
        """
        Retrieve most relevant chunks for a query.
        
        Process:
            1. Embed the query
            2. Search FAISS index for nearest neighbors
            3. Return chunks with metadata and similarity scores
        
        Args:
            query: User question or search query
            top_k: Number of chunks to retrieve
            return_scores: Include similarity scores in results
            
        Returns:
            List of dicts with:
                - text: chunk content
                - source: PDF filename
                - chunk_id: position in document
                - score: similarity score (0-1, higher = more similar)
                
        Raises:
            ValueError: If index not built/loaded
            
        Example:
            >>> rag = RAGSystem.load("./vector_db")
            >>> results = rag.retrieve("What is CARNIVAL?", top_k=3)
            >>> for r in results:
            >>>     print(f"Source: {r['source']}, Score: {r['score']:.3f}")
            >>>     print(f"Text: {r['text'][:200]}...")
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() or load() first")
        
        self._load_model()
        
        # Embed query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            result = {
                "text": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "chunk_id": self.metadata[idx]["chunk_id"],
                "total_chunks": self.metadata[idx]["total_chunks"]
            }
            
            if return_scores:
                # Convert squared L2 distance to cosine similarity.
                # FAISS IndexFlatL2 returns ||x-y||² (squared L2).
                # For normalized vectors: ||x-y||² = 2 - 2·cos_sim
                # Therefore: cos_sim = 1 - ||x-y||² / 2 = 1 - distance / 2
                similarity = 1 - (distance / 2)
                result["score"] = float(np.clip(similarity, 0, 1))
            
            results.append(result)
        
        return results
    
    def format_for_context(self, results: List[Dict]) -> str:
        """
        Format retrieval results for LLM context.

        Groups chunks by source paper, shows full chunk text with
        position indicators (chunk N/total).

        Args:
            results: List of result dicts from retrieve()

        Returns:
            Formatted string for LLM prompt
        """
        if not results:
            return "No relevant literature found."

        from collections import OrderedDict
        grouped: OrderedDict = OrderedDict()
        for result in results:
            source = result['source']
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(result)

        formatted = "## Relevant Literature:\n\n"
        max_chunks_per_paper = 3

        for paper_idx, (source, chunks) in enumerate(grouped.items(), 1):
            source_name = source.replace('.pdf', '').replace('_', ' ')
            best_score = max(c.get('score', 0) for c in chunks)
            formatted += f"### [{paper_idx}] {source_name} (relevance: {best_score:.2f})\n\n"

            for chunk in chunks[:max_chunks_per_paper]:
                chunk_id = chunk.get('chunk_id', '?')
                total = chunk.get('total_chunks', '?')
                formatted += f"*[chunk {chunk_id}/{total}]*\n"
                formatted += f"{chunk['text']}\n\n"

            if len(chunks) > max_chunks_per_paper:
                formatted += f"*({len(chunks) - max_chunks_per_paper} additional chunks omitted)*\n\n"

        return formatted

    def save(self, save_dir: str):
        """
        Save index, chunks, and metadata to disk.
        
        Saves:
            - index.faiss: FAISS index file
            - chunks.pkl: List of text chunks
            - metadata.pkl: List of metadata dicts
            - config.json: Model name and configuration
        
        Args:
            save_dir: Directory to save files
            
        Example:
            >>> rag = RAGSystem()
            >>> rag.build_index("/mnt/project")
            >>> rag.save("./vector_db")
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(save_dir, "index.faiss")
        faiss.write_index(self.index, index_path)
        logger.info(f"Saved index to {index_path}")
        
        # Save chunks
        chunks_path = os.path.join(save_dir, "chunks.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Saved {len(self.chunks)} chunks to {chunks_path}")
        
        # Save metadata
        metadata_path = os.path.join(save_dir, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save config
        config = {
            "model_name": self.model_name,
            "num_chunks": len(self.chunks),
            "embedding_dim": self.index.d,
            "num_documents": len(set(m["source"] for m in self.metadata))
        }
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved config to {config_path}")
        
        # Calculate and log storage size
        total_size = sum(
            os.path.getsize(os.path.join(save_dir, f))
            for f in ["index.faiss", "chunks.pkl", "metadata.pkl", "config.json"]
        )
        logger.info(f"Total storage: {total_size / 1024 / 1024:.1f} MB")
    
    @classmethod
    def load(cls, load_dir: str) -> 'RAGSystem':
        """
        Load a previously saved index.
        
        Args:
            load_dir: Directory containing saved files
            
        Returns:
            Loaded RAGSystem instance
            
        Example:
            >>> rag = RAGSystem.load("./vector_db")
            >>> results = rag.retrieve("What is DoRothEA?")
        """
        # Load config
        config_path = os.path.join(load_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create instance
        rag = cls(model_name=config["model_name"])
        
        # Load index
        index_path = os.path.join(load_dir, "index.faiss")
        rag.index = faiss.read_index(index_path)
        logger.info(f"Loaded index with {rag.index.ntotal} vectors")
        
        # Load chunks
        chunks_path = os.path.join(load_dir, "chunks.pkl")
        with open(chunks_path, "rb") as f:
            rag.chunks = pickle.load(f)
        logger.info(f"Loaded {len(rag.chunks)} chunks")
        
        # Load metadata
        metadata_path = os.path.join(load_dir, "metadata.pkl")
        with open(metadata_path, "rb") as f:
            rag.metadata = pickle.load(f)
        logger.info(f"Loaded metadata for {config['num_documents']} documents")
        
        return rag
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the indexed corpus.
        
        Returns:
            Dict with corpus statistics
        """
        if self.index is None:
            return {"status": "No index built"}
        
        sources = [m["source"] for m in self.metadata]
        unique_sources = set(sources)
        
        chunk_lengths = [len(chunk) for chunk in self.chunks]
        
        return {
            "num_documents": len(unique_sources),
            "num_chunks": len(self.chunks),
            "embedding_dimension": self.index.d,
            "model": self.model_name,
            "avg_chunk_length": np.mean(chunk_lengths),
            "min_chunk_length": np.min(chunk_lengths),
            "max_chunk_length": np.max(chunk_lengths),
            "documents": sorted(list(unique_sources))
        }


def main():
    """Command-line interface for RAG system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System for Network Biology Papers")
    parser.add_argument("--build", action="store_true", help="Build index from PDFs")
    parser.add_argument("--pdf_dir", type=str, help="Directory containing PDFs")
    parser.add_argument("--output", type=str, default="./vector_db", help="Output directory for index")
    parser.add_argument("--query", type=str, help="Query to search")
    parser.add_argument("--db_path", type=str, default="./vector_db", help="Path to saved index")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    
    args = parser.parse_args()
    
    if args.build:
        # Build new index
        if not args.pdf_dir:
            print("Error: --pdf_dir required when building index")
            return
        
        print("=" * 60)
        print("Building RAG Index")
        print("=" * 60)
        
        rag = RAGSystem()
        num_docs, num_chunks = rag.build_index(args.pdf_dir)
        rag.save(args.output)
        
        print("\n" + "=" * 60)
        print(f"✓ Index built successfully!")
        print(f"  Documents: {num_docs}")
        print(f"  Chunks: {num_chunks}")
        print(f"  Saved to: {args.output}")
        print("=" * 60)
        
    elif args.query:
        # Query existing index
        print("=" * 60)
        print(f"Querying: {args.query}")
        print("=" * 60)
        
        rag = RAGSystem.load(args.db_path)
        results = rag.retrieve(args.query, top_k=args.top_k)
        
        print(f"\nTop {len(results)} Results:\n")
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {result['source']}")
            print(f"Score: {result['score']:.3f}")
            print(f"Chunk {result['chunk_id'] + 1}/{result['total_chunks']}")
            print(f"\nText:\n{result['text'][:300]}...")
            print("-" * 60)
    
    elif args.stats:
        # Show statistics
        rag = RAGSystem.load(args.db_path)
        stats = rag.get_statistics()
        
        print("\n" + "=" * 60)
        print("Index Statistics")
        print("=" * 60)
        print(f"Documents: {stats['num_documents']}")
        print(f"Chunks: {stats['num_chunks']}")
        print(f"Model: {stats['model']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Avg chunk length: {stats['avg_chunk_length']:.0f} chars")
        print(f"\nDocuments indexed:")
        for doc in stats['documents']:
            print(f"  - {doc}")
        print("=" * 60)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()










