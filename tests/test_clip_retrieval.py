import math

import numpy as np
import pytest

from clip_retrieval import StyleIndex, _scan_library, _fingerprint


def _build(root, cache, embedder):
    return StyleIndex(root=str(root), cache_path=str(cache), embedder=embedder).load_or_build()


def test_scan_library_layout(style_library):
    items = _scan_library(str(style_library))
    assert len(items) == 8
    shapes = {it.face_shape for it in items}
    lengths = {it.hair_length for it in items}
    assert shapes == {"oval", "square"}
    assert lengths == {"Short", "Long"}


def test_index_builds_and_embeddings_are_unit_norm(style_library, stub_embedder, tmp_path):
    idx = _build(style_library, tmp_path / "idx.npz", stub_embedder)
    assert idx.embeddings.shape == (8, 16)
    norms = np.linalg.norm(idx.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_cache_roundtrip_skips_reembed(style_library, stub_embedder, tmp_path):
    cache = tmp_path / "idx.npz"
    _build(style_library, cache, stub_embedder)
    assert cache.exists()

    calls = {"n": 0}
    def counting_embedder(imgs):
        calls["n"] += 1
        return stub_embedder(imgs)

    idx2 = StyleIndex(root=str(style_library), cache_path=str(cache),
                      embedder=counting_embedder).load_or_build()
    assert len(idx2.items) == 8
    # Only the query path should ever call the embedder after a warm cache.
    idx2.embed_image(str(next(style_library.rglob("*.jpg"))))
    assert calls["n"] == 1


def test_query_ranks_visually_similar_first(style_library, stub_embedder, tmp_path, query_image):
    idx = _build(style_library, tmp_path / "idx.npz", stub_embedder)
    res = idx.query(str(query_image), k=4)
    assert len(res) == 4
    # query image is red → 'oval' library is red → top hit should be oval
    assert res[0]["face_shape"] == "oval"
    # scores monotonically non-increasing
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)


def test_hair_length_filter(style_library, stub_embedder, tmp_path, query_image):
    idx = _build(style_library, tmp_path / "idx.npz", stub_embedder)
    res = idx.query(str(query_image), k=10, hair_length="Short")
    assert res and all(r["hair_length"] == "Short" for r in res)


def test_shape_weights_shift_ranking(style_library, stub_embedder, tmp_path, query_image):
    idx = _build(style_library, tmp_path / "idx.npz", stub_embedder)
    # Heavy prior towards 'square' should be able to overtake clip similarity.
    res = idx.query(str(query_image), k=1,
                    shape_weights={"square": 1.0, "oval": 0.0},
                    shape_bonus=5.0)
    assert res[0]["face_shape"] == "square"


def test_fingerprint_changes_when_library_changes(style_library):
    items = _scan_library(str(style_library))
    fp1 = _fingerprint(items)
    fp2 = _fingerprint(items[:-1])
    assert fp1 != fp2


def test_query_on_empty_index_raises(tmp_path, stub_embedder):
    idx = StyleIndex(root=str(tmp_path / "empty"), cache_path=str(tmp_path / "x.npz"),
                     embedder=stub_embedder)
    with pytest.raises(RuntimeError):
        idx.query(str(tmp_path / "whatever.jpg"))
