from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingStrategy(ABC):
    """Base class for embedding strategies.
    Each audio has multiple windows of embeddings. This class defines how to combine them into a single embedding.

    BaseEmbeddingStrategy has one abstract method `_apply()`
    which is implemented by the concrete strategies.
    """

    def apply(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply the strategy and normalize the result.
        Args:
            embeddings (np.ndarray): Array of embeddings.
        Returns:
            np.ndarray: Combined embedding.
        """
        return self.normalize(self._apply(embeddings))

    @abstractmethod
    def _apply(self, embeddings: np.ndarray) -> np.ndarray:
        """Abstract method to implement the strategy.

        Args:
            embeddings (np.ndarray): Array of embeddings.
        Returns:
            np.ndarray: Final embedding after applying the strategy.
        """

    def normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize the embedding.
        Args:
            embedding (np.ndarray): Embedding to normalize.
        Returns:
            np.ndarray: Normalized embedding.
        """
        return embedding / np.linalg.norm(embedding)


class FirstWindowStrategy(BaseEmbeddingStrategy):
    """First window strategy for embeddings.
    This strategy selects the first window of embeddings.
    """

    def _apply(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings[0]


class MeanStrategy(BaseEmbeddingStrategy):
    """Mean strategy for embeddings.
    This strategy calculates the mean of the embeddings.
    """

    def _apply(self, embeddings: np.ndarray) -> np.ndarray:
        return np.mean(embeddings, axis=0)
