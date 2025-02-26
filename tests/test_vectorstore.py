from kth_sr.vectorstore import FAISS
import numpy as np
import pytest
import shutil


@pytest.mark.parametrize(
    "dimension, array, metadata",
    [
        (2, [[1, 2], [3, 4]], ["a", "b"]),
        (3, [[1, 2, 3], [4, 5, 6]], ["a", "b"]),
        (2, [[1, 2], [3, 4]], None),
    ],
)
def test_add_vectors(dimension, array, metadata):
    vstore = FAISS(dimension)
    vectors = np.array(array)
    vstore.add(vectors, metadata)
    if metadata is None:
        metadata = [None] * len(array)
    assert vstore.metadata == metadata
    assert vstore.vstore.ntotal == len(array)


def test_save_load():
    tmp_path = "./tmp_faiss"
    vectors = np.array([[1, 2], [3, 4]])
    metadata = ["a", "b"]

    vstore = FAISS(2)
    vstore.add(vectors, metadata)
    vstore.save(tmp_path)

    vstore2 = FAISS.load(tmp_path)

    assert vstore2.metadata == metadata
    assert vstore2.vstore.ntotal == vectors.shape[0]
    assert vstore2.vstore.d == 2

    shutil.rmtree(tmp_path)
