import faiss
from pathlib import Path
import json


class FAISS:
    """Vector store using FAISS library."""

    vstore: faiss.IndexFlatL2
    """Vector store using FAISS library."""
    metadata: list
    """Metadata for the vectors stored"""

    def __init__(self, dimension):
        self.vstore = faiss.IndexFlatL2(dimension)
        self.metadata = []

    pass

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

        # Add vectors to the store
        self.vstore.add(vectors)

        # Add metadata to the store
        if metadata is not None:
            for i, meta in enumerate(metadata):
                self.metadata.append(meta)

    def search(self, embedding: list, k: int):
        """Search for the k nearest vectors to the given embedding.

        Args:
            embedding (list): Embedding to search for.
            k (int): Number of nearest vectors to return.
        """
        distances, indices = self.vstore.search(embedding, k)
        metadata = [self.metadata[i] for i in indices[0]]
        return distances, metadata

    def save(self, path: str):
        """Save the vector store to a file.

        Args:
            path (str): Path to save the vector store.
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.vstore, f"{path}/vector_store.index")
        with open(dir_path / "metadata.json", "w") as f:
            json.dump(self.metadata, f)

    @classmethod
    def load(cls, path: str):
        """Load the vector store from a file.

        Args:
            path (str): Path to load the vector store.
        """
        dir_path = Path(path)

        vector_store = FAISS(64)
        vector_store.vstore = faiss.read_index(f"{path}/vector_store.index")
        with open(dir_path / "metadata.json", "r") as f:
            vector_store.metadata = json.load(f)
        return vector_store
