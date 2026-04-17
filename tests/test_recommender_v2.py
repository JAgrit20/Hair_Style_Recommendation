import sys
import types

import numpy as np
import pytest


def test_recommend_end_to_end_with_stubs(monkeypatch, style_library, stub_embedder,
                                         tmp_path, query_image):
    """Wire stubbed face_analysis + stubbed CLIP through recommender_v2.recommend."""
    # Stub mediapipe/cv2 if absent (same trick as test_face_analysis)
    for name in ("cv2", "mediapipe"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                m = types.ModuleType(name)
                if name == "mediapipe":
                    fm = types.SimpleNamespace(FaceMesh=lambda **kw: types.SimpleNamespace(
                        process=lambda *a, **k: types.SimpleNamespace(multi_face_landmarks=None)))
                    m.solutions = types.SimpleNamespace(face_mesh=fm)
                sys.modules[name] = m

    import face_analysis
    import clip_retrieval
    import recommender_v2

    fake = face_analysis.FaceAnalysis(
        success=True,
        features={"face_length_to_width": 1.4},
        shape_probabilities={"oval": 0.6, "square": 0.4},
        primary_shape="oval",
        secondary_shape="square",
        confidence=0.2,
        landmarks_2d=np.zeros((468, 2)),
    )
    monkeypatch.setattr(recommender_v2, "analyze_face", lambda p: fake)
    monkeypatch.setattr(recommender_v2, "annotate", lambda *a, **k: None)

    idx = clip_retrieval.StyleIndex(
        root=str(style_library),
        cache_path=str(tmp_path / "idx.npz"),
        embedder=stub_embedder,
    ).load_or_build()

    out = recommender_v2.recommend(
        image_path=str(query_image),
        hair_length="short",
        k=3,
        index=idx,
        write_annotated=False,
    )

    assert out["success"] is True
    assert out["face"]["primary_shape"] == "oval"
    assert out["face"]["advice"]
    assert len(out["recommendations"]) == 3
    for r in out["recommendations"]:
        assert r["hair_length"] == "Short"
        assert "reason" in r
        assert r["score"] >= r["clip_similarity"]  # bonus is non-negative here


def test_recommend_propagates_face_failure(monkeypatch, tmp_path):
    for name in ("cv2", "mediapipe"):
        sys.modules.setdefault(name, types.ModuleType(name))
        if name == "mediapipe" and not hasattr(sys.modules[name], "solutions"):
            fm = types.SimpleNamespace(FaceMesh=lambda **kw: types.SimpleNamespace(
                process=lambda *a, **k: types.SimpleNamespace(multi_face_landmarks=None)))
            sys.modules[name].solutions = types.SimpleNamespace(face_mesh=fm)

    import face_analysis
    import recommender_v2

    monkeypatch.setattr(recommender_v2, "analyze_face",
                        lambda p: face_analysis.FaceAnalysis(success=False, error="no face"))
    out = recommender_v2.recommend(image_path="x.jpg", index=object())
    assert out == {"success": False, "error": "no face"}
