"""
build_index.py — Build FAISS Vector Index from Sensor Anomaly CSV Dataset

This script reads the sensor anomaly CSV dataset (data/sensor_anomaly_dataset.csv),
converts each row into a rich natural-language document suitable for RAG retrieval,
chunks the text into overlapping segments, embeds them using a SentenceTransformer
model, and builds a FAISS index for semantic search.

Output files (consumed by backend/rag_engine.py):
  - knowledge_base/faiss_index.bin  — serialized FAISS IndexFlatL2 index
  - knowledge_base/chunks.json     — JSON array of chunk objects with text + metadata

Dependencies: faiss-cpu, sentence-transformers, numpy
"""

import csv
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "sensor_anomaly_dataset.csv"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
FAISS_INDEX_PATH = KNOWLEDGE_BASE_DIR / "faiss_index.bin"
CHUNKS_OUTPUT_PATH = KNOWLEDGE_BASE_DIR / "chunks.json"

# ── Chunking parameters ───────────────────────────────────────────────────────
CHUNK_SIZE = 500       # Target chunk size in characters
CHUNK_OVERLAP = 100    # Overlap between consecutive chunks in characters


def row_to_document(row: dict) -> dict:
    """
    Convert a single CSV row into a document dict with keys: text, sensor_type, category.

    The text field is a rich natural-language description combining all relevant
    fields from the row so that a retrieval model can match diverse queries
    (e.g. by sensor type, location, anomaly name, cause, resolution steps, etc.).

    Args:
        row: A dict representing one CSV row (keys are column headers).

    Returns:
        A dict with 'text', 'sensor_type', and 'category' keys.
    """
    # Build a multi-line natural-language description of this anomaly record
    text_parts = [
        f"Sensor Type: {row.get('sensor_type_name', 'N/A')} ({row.get('sensor_type_code', 'N/A')})",
        f"Location: {row.get('location', 'N/A')}",
        f"Anomaly: {row.get('anomaly_name', 'N/A')}",
        (
            f"Anomaly Value: {row.get('anomaly_value', 'N/A')} {row.get('measurement_unit', '')}"
            f" (Normal Range: {row.get('normal_range_min', 'N/A')} - {row.get('normal_range_max', 'N/A')}"
            f" {row.get('measurement_unit', '')},"
            f" Baseline: {row.get('baseline_value', 'N/A')} {row.get('measurement_unit', '')})"
        ),
        f"Severity: {row.get('severity_level', 'N/A')}",
        "",  # blank line before cause
        f"Cause: {row.get('cause_of_anomaly', 'N/A')}",
        "",
        f"Resolution Steps: {row.get('resolution_steps_taken', 'N/A')}",
        "",
        f"Estimated Downtime: {row.get('estimated_downtime_minutes', 'N/A')} minutes",
        f"Prevention: {row.get('prevention_recommendation', 'N/A')}",
    ]

    text = "\n".join(text_parts)

    return {
        "text": text,
        "sensor_type": row.get("sensor_type_code", "UNKNOWN"),
        "category": row.get("anomaly_name", "Unknown"),
    }


def load_csv_documents(path: Path) -> list[dict]:
    """
    Read the CSV dataset and convert each row into a document dict.

    Each document has: text, sensor_type, category.

    Args:
        path: Path to the CSV file.

    Returns:
        A list of document dicts.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"CSV dataset not found at {path}. "
            "Make sure data/sensor_anomaly_dataset.csv exists."
        )

    documents = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = row_to_document(row)
            documents.append(doc)

    print(f"[build_index] Loaded {len(documents)} records from {path.name}")
    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a text string into overlapping chunks using a sliding window.

    Each chunk is approximately `chunk_size` characters long, and consecutive
    chunks overlap by `overlap` characters. The function tries to break at
    sentence or word boundaries to avoid cutting words in half.

    Args:
        text: The full document text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters that overlap between consecutive chunks.

    Returns:
        A list of text chunk strings.
    """
    # If the text fits in one chunk, return it as-is
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Extract a chunk of up to chunk_size characters
        end = start + chunk_size

        # If this is not the last chunk, try to break at a natural boundary
        if end < len(text):
            # Look for a sentence-ending period near the end of the chunk
            break_point = text.rfind(". ", start + chunk_size // 2, end)
            if break_point == -1:
                # No sentence boundary found; try a space instead
                break_point = text.rfind(" ", start + chunk_size // 2, end)
            if break_point != -1:
                end = break_point + 1  # Include the period/space

        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        # Advance the window, stepping back by `overlap` characters
        start = end - overlap

        # Safety: if we're barely advancing, force progress to avoid infinite loops
        if start <= (end - chunk_size):
            start = end

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk all documents and preserve metadata on each chunk.

    Each input document has: text, sensor_type, category.
    Each output chunk has: text, sensor_type, category.

    Args:
        documents: List of document dicts.

    Returns:
        A list of chunk dicts, each containing text, sensor_type, and category.
    """
    all_chunks = []

    for doc in documents:
        text = doc.get("text", "")
        sensor_type = doc.get("sensor_type", "UNKNOWN")
        category = doc.get("category", "Unknown")

        # Split the text into overlapping chunks
        text_chunks = chunk_text(text)

        # Create chunk objects with metadata
        for chunk_text_str in text_chunks:
            all_chunks.append({
                "text": chunk_text_str,
                "sensor_type": sensor_type,
                "category": category,
            })

    print(f"[build_index] Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Embed all chunk texts using the SentenceTransformer model.

    Embeddings are L2-normalized so that L2 distance and cosine similarity
    produce equivalent ranking. This matches how rag_engine.py queries the
    index (it calls model.encode with normalize_embeddings=True).

    Args:
        chunks: List of chunk dicts (each must have a 'text' field).
        model: A loaded SentenceTransformer model.

    Returns:
        A numpy array of shape (num_chunks, embedding_dim), dtype float32.
    """
    texts = [chunk["text"] for chunk in chunks]

    print(f"[build_index] Embedding {len(texts)} chunks (this may take a moment)...")

    # normalize_embeddings=True ensures unit-length vectors, so that
    # IndexFlatL2 distance is equivalent to (2 - 2*cosine_similarity),
    # giving the same ranking as inner-product search on normalized vectors.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"[build_index] Embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a FAISS IndexFlatL2 (exact L2 / Euclidean distance search).

    With normalized embeddings, smaller L2 distance = higher cosine similarity.

    Args:
        embeddings: Normalized embedding matrix of shape (n, d).

    Returns:
        A populated FAISS IndexFlatL2 index.
    """
    dimension = embeddings.shape[1]

    # Create a flat (brute-force) L2 index — exact search, no approximation
    index = faiss.IndexFlatL2(dimension)

    # Add all embedding vectors to the index
    index.add(embeddings)

    print(f"[build_index] Built FAISS IndexFlatL2 with {index.ntotal} vectors, dimension={dimension}")
    return index


def save_outputs(index: faiss.IndexFlatL2, chunks: list[dict]) -> None:
    """
    Save the FAISS index and chunk metadata to disk.

    Files saved:
      - knowledge_base/faiss_index.bin  — binary FAISS index (loaded via faiss.read_index)
      - knowledge_base/chunks.json      — JSON array of chunk objects
    """
    # Ensure the output directory exists
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Save the FAISS index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"[build_index] Saved FAISS index to {FAISS_INDEX_PATH}")

    # Save chunk metadata as JSON
    with open(CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"[build_index] Saved {len(chunks)} chunks to {CHUNKS_OUTPUT_PATH}")


def print_statistics(documents: list[dict], chunks: list[dict], embeddings: np.ndarray) -> None:
    """Print summary statistics about the built index."""
    # Gather unique sensor types and anomaly categories for insight
    sensor_types = set(doc["sensor_type"] for doc in documents)
    categories = set(doc["category"] for doc in documents)

    print("\n" + "=" * 60)
    print("  FAISS Index Build — Summary")
    print("=" * 60)
    print(f"  CSV source:                {CSV_PATH}")
    print(f"  Total records loaded:      {len(documents)}")
    print(f"  Unique sensor types:       {len(sensor_types)}")
    print(f"  Unique anomaly categories: {len(categories)}")
    print(f"  Total chunks created:      {len(chunks)}")
    print(f"  Embedding dimensions:      {embeddings.shape[1]}")
    print(f"  Chunk size (chars):        ~{CHUNK_SIZE}")
    print(f"  Chunk overlap (chars):     {CHUNK_OVERLAP}")
    print(f"  Index type:                IndexFlatL2")
    print(f"  Embeddings normalized:     True")
    print(f"  Index file:                {FAISS_INDEX_PATH}")
    print(f"  Chunks file:               {CHUNKS_OUTPUT_PATH}")
    print("=" * 60)

    # Print per-sensor-type breakdown
    from collections import Counter
    type_counts = Counter(doc["sensor_type"] for doc in documents)
    print("\n  Records per sensor type:")
    for stype, count in sorted(type_counts.items()):
        print(f"    {stype:12s} — {count}")
    print()


def main() -> None:
    """
    Main pipeline: load CSV, convert to documents, chunk, embed, build index, save.
    """
    # Step 1: Load records from the CSV dataset and convert to documents
    documents = load_csv_documents(CSV_PATH)

    # Step 2: Chunk documents into overlapping segments with metadata
    chunks = chunk_documents(documents)

    # Step 3: Load the embedding model
    print("[build_index] Loading SentenceTransformer model: all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 4: Embed all chunks (normalized for cosine/L2 compatibility)
    embeddings = embed_chunks(chunks, model)

    # Step 5: Build the FAISS index
    index = build_faiss_index(embeddings)

    # Step 6: Save the index and chunk metadata to disk
    save_outputs(index, chunks)

    # Step 7: Print summary statistics
    print_statistics(documents, chunks, embeddings)

    print("[build_index] Done! The knowledge base is ready for RAG retrieval.")


if __name__ == "__main__":
    main()
