"""Embedding generation for memory system."""

from typing import List, Optional

# Lazy-loaded model
_embedding_model = None
_model_name: Optional[str] = None


def get_embedding(text: str, model_name: str = "BAAI/bge-small-en-v1.5") -> List[float]:
    """Generate embedding for text using fastembed.

    Args:
        text: Text to embed
        model_name: Model name (e.g., BAAI/bge-small-en-v1.5)

    Returns:
        List of floats (embedding vector)
    """
    global _embedding_model, _model_name

    # Lazy load model, reload if model name changed
    if _embedding_model is None or _model_name != model_name:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError("fastembed is required for memory embeddings. Install with: pip install fastembed") from e

        _embedding_model = TextEmbedding(model_name=model_name)
        _model_name = model_name

    return list(_embedding_model.embed([text]))[0].tolist()


def get_embedding_dimension(model_name: str = "BAAI/bge-small-en-v1.5") -> int:
    """Get the embedding dimension for a model.

    Common dimensions:
    - BAAI/bge-small-en-v1.5: 384
    - all-MiniLM-L6-v2: 384
    - text-embedding-3-small: 1536
    """
    # Known dimensions to avoid loading model just for dimension
    known_dimensions = {
        "BAAI/bge-small-en-v1.5": 384,
        "all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
    }

    if model_name in known_dimensions:
        return known_dimensions[model_name]

    # Fall back to getting dimension from model
    embedding = get_embedding("test", model_name)
    return len(embedding)
