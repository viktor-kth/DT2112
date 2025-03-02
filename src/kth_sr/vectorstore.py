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

    def search(
        self, embeddings: list, k: int, threshold: float | None = None
    ) -> tuple[list[np.ndarray], list]:
        """Search for the k nearest vectors to the given embedding.

        Args:
            embeddings (list): List of embeddings to search for.
            k (int): Number of nearest vectors to return.
            threshold (float, optional): Threshold distance to filter the search results. Defaults to None.

        Returns:
            tuple: Tuple of `distances` and `metadata`.
            - distances (list[np.ndarray]): Distances of the search results.
            Distances are list of numpy arrays, because number of distances in each row can differ.
            - metadata (list): Metadata of the search results.

        Examples:
            Create template vector store
            >>> from kth_sr.vectorstore import FAISS
            >>> vstore = FAISS(2)
            >>> vstore.add([[1, 2], [3, 4]], ["a", "b"])

            Search for the nearest vector to `[1, 2]`
            >>> vstore.search([[1, 2]], 1)
            ([array([0.]], [['a']])  # distance, metadata

            Search for the 2 nearest vectors to vectors `[1, 2]` and `[3, 4]`
            >>> vstore.search([[1, 2], [3, 4]], 2)
            ([array([0., 8.]), array([0., 8.])], [['a', 'b'], ['b', 'a']])  # distance, metadata
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        distances, indices = self._vstore.search(embeddings, k)
        # indices are 2 dimensional array. Each row for each query.
        metadata = [[self._metadata[i] for i in row] for row in indices]

        if threshold is not None:
            distances, metadata = self.apply_threshold(distances, metadata, threshold)

        return list(distances), metadata

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

    @classmethod
    def apply_threshold(
        cls, distances: np.ndarray, metadata: list, threshold: float
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Apply threshold to the search results.

        Args:
            distances (np.ndarray): Distances of the search results.
            metadata (list): Metadata of the search results.
            threshold (float): Threshold to filter the search results.

        Returns:
            tuple: Tuple of `distances` and `metadata` after applying threshold.
            - distances (list[np.ndarray]): Filtered distances.
            Distances are list of numpy arrays, because number of distances in each row can differ.
            - metadata (list): Filtered metadata.

        Examples:
            >>> from kth_sr.vectorstore import FAISS
            >>> distances = [[0, 4], [0, 8]]
            >>> metadata = [['a', 'b'], ['c', 'd']]
            >>> FAISS.apply_threshold(distances, metadata, 5)
            ([array([0., 4.]), array([0.])], [['a', 'b'], ['c']])  # distance, metadata
        """
        if len(distances) != len(metadata):
            raise ValueError(
                "Length of metadata should be same as length of distances."
            )

        if not isinstance(distances, np.ndarray):
            distances = np.array(distances)

        # remove the distances greater than threshold
        distances_masked = np.where(distances <= threshold, distances, np.nan)
        distances_filtered = [i[~np.isnan(i)] for i in distances_masked]

        # remove the metadata corresponding to the distances greater than threshold
        metadata_filtered = [
            metadata[i][: len(row)] for i, row in enumerate(distances_filtered)
        ]

        return distances_filtered, metadata_filtered
