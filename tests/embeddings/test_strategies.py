from kth_sr.embeddings import FirstWindowStrategy, MeanStrategy
import pytest
import numpy as np


@pytest.mark.parametrize(
    "embeddings, expected",
    [
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([1, 2])),
        (np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([1, 2])),
    ],
)
def test_first_window_strategy(embeddings, expected):
    strategy = FirstWindowStrategy()
    result = strategy.apply(embeddings)
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "embeddings, expected",
    [
        (np.array([[1, 0], [0, 1], [0, -1]]), np.array([0, 1])),
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([3, 4])),
    ],
)
def test_mean_strategy(embeddings, expected):
    strategy = MeanStrategy()
    result = strategy.apply(embeddings)
    assert np.all(result == expected)
