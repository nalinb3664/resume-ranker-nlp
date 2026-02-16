import faiss
import numpy as np
import os
import pickle
import logging

INDEX_FILE = "models/faiss.index"
META_FILE = "models/meta.pkl"

dimension = 384  # SBERT dimension


# -----------------------------
# Create FAISS Index
# -----------------------------
def create_index():
    """
    Create cosine similarity optimized FAISS index.
    Using Inner Product for semantic search.
    """
    index = faiss.IndexFlatIP(dimension)
    return index


# -----------------------------
# Save Index + Metadata
# -----------------------------
def save_index(index, metadata):
    try:
        os.makedirs("models", exist_ok=True)

        # Safety check
        if index.d != dimension:
            raise ValueError("Index dimension mismatch.")

        faiss.write_index(index, INDEX_FILE)

        with open(META_FILE, "wb") as f:
            pickle.dump(metadata, f)

    except Exception as e:
        logging.error(f"Error saving FAISS index: {e}")
        raise


# -----------------------------
# Load Index + Metadata
# -----------------------------
def load_index():
    try:
        if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
            return create_index(), []

        index = faiss.read_index(INDEX_FILE)

        # Dimension validation
        if index.d != dimension:
            logging.warning("Dimension mismatch. Recreating index.")
            return create_index(), []

        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)

        return index, metadata

    except Exception as e:
        logging.error(f"Corrupted index detected: {e}")
        # Fallback safe reset
        return create_index(), []