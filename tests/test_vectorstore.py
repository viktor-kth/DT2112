from kth_sr.vectorstore import FAISS
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dimension, array, metadata",
    [(2, [[1, 2], [3, 4]], ["a", "b"]), (3, [[1, 2, 3], [4, 5, 6]], ["a", "b"])],
)
def test_add_vectors(dimension, array, metadata):
    vstore = FAISS(dimension)
    vectors = np.array(array)
    vstore.add(vectors, metadata)
    assert vstore.metadata == metadata
    assert vstore.vstore.ntotal == len(array)
