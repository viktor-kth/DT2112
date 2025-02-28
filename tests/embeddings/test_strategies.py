from kth_sr.embeddings import FirstWindowStrategy, MeanStrategy
import pytest
import numpy as np


@pytest.mark.parametrize(
    "embeddings, expected",
    [
        (np.array([[1, 0], [3, 4], [5, 6]]), np.array([1, 0])),
        (np.array([[0, 2], [3, 4], [5, 6], [7, 8]]), np.array([0, 1])),
    ],
)
def test_first_window_strategy(embeddings, expected):
    strategy = FirstWindowStrategy()
    result = strategy.apply(embeddings)
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "embeddings, expected",
    # expected values are normalized embeddings after applying the mean strategy
    [
        (np.array([[1, 0], [0, 1], [0, -1]]), np.array([1, 0])),
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0.6, 0.8])),
    ],
)
def test_mean_strategy(embeddings, expected):
    strategy = MeanStrategy()
    result = strategy.apply(embeddings)
    assert np.all(result == expected)
