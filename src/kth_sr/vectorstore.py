from __future__ import annotations
import faiss
from pathlib import Path
import json
import numpy as np


class FAISS:
    """Vector store using FAISS library."""

    _vstore: faiss.IndexFlatL2
    """Vector store using FAISS library."""
    _metadata: list
    """Metadata for the vectors stored"""

    def __init__(self, dimension):
        self._vstore = faiss.IndexFlatL2(dimension)
        self._metadata = []

    def add(self, vectors: list, metadata: list | None = None):
        """Add a vectors to the store.

        Args:
            vectors (list): List of vectors to be added.
            metadata (list, optional): List of metadata for the vectors. Defaults to None.

        Raises:
            ValueError: Length of metadata should be same as length of vectors.
        """

        # validate input
        if metadata and len(vectors) != len(metadata):
            raise ValueError("Length of metadata should be same as length of vectors.")

        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)

        # Add vectors to the store
        self._vstore.add(vectors)

        # Add metadata to the store
        if metadata is not None:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([None] * len(vectors))

    def search(self, embeddings: list, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for the k nearest vectors to the given embedding.

        Args:
            embeddings (list): List of embeddings to search for.
            k (int): Number of nearest vectors to return.

        Returns:
            tuple: Tuple of distances and metadata.
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        distances, indices = self._vstore.search(embeddings, k)
        # indices are 2 dimensional array. Each row for each query.
        metadata = [[self._metadata[i] for i in row] for row in indices]
        return distances, np.array(metadata)

    def save(self, path: str):
        """Save the vector store to a file.

        Args:
            path (str): Path to save the vector store.
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._vstore, f"{path}/vector_store.index")
        with open(dir_path / "metadata.json", "w") as f:
            json.dump(self._metadata, f)

    @classmethod
    def load(cls, path: str) -> FAISS:
        """Load the vector store from a file.

        Args:
            path (str): Path to load the vector store.

        Returns:
            FAISS: Vector store object
        """
        dir_path = Path(path)

        vector_store = FAISS(64)
        vector_store._vstore = faiss.read_index(f"{path}/vector_store.index")
        with open(dir_path / "metadata.json", "r") as f:
            vector_store._metadata = json.load(f)
        return vector_store
