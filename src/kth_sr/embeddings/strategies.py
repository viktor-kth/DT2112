from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingStrategy(ABC):
    """Base class for embedding strategies.
    Each audio has multiple windows of embeddings. This class defines how to combine them into a single embedding.
    """

    @abstractmethod
    def apply(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply the strategy.
        Args:
            embeddings (np.ndarray): Array of embeddings.
        Returns:
            np.ndarray: Combined embedding.
        """


class FirstWindowStrategy(BaseEmbeddingStrategy):
    """First window strategy for embeddings.
    This strategy selects the first window of embeddings.
    """

    def apply(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings[0]


class MeanStrategy(BaseEmbeddingStrategy):
    """Mean strategy for embeddings.
    This strategy calculates the mean of the embeddings.
    """

    def apply(self, embeddings: np.ndarray) -> np.ndarray:
        return np.mean(embeddings, axis=0)
