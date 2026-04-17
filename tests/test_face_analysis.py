"""Unit tests for face_analysis that do NOT require mediapipe at runtime.

We test the pure-math layers (_compute_features, _score_shapes, FaceAnalysis
serialisation) by feeding synthetic landmark arrays. The mediapipe import is
stubbed if unavailable so CI without native wheels still passes.
"""
import sys
import math
import types

import numpy as np
import pytest


def _ensure_importable():
    """Stub cv2/mediapipe if missing so `import face_analysis` works on CI."""
    for name in ("cv2", "mediapipe"):
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            mod = types.ModuleType(name)
            if name == "cv2":
                mod.imread = lambda *a, **k: None
                mod.cvtColor = lambda x, *a, **k: x
                mod.COLOR_BGR2RGB = 0
            if name == "mediapipe":
                # face_analysis accesses mp.solutions.face_mesh.FaceMesh(...)
                fm = types.SimpleNamespace(FaceMesh=lambda **kw: types.SimpleNamespace(
                    process=lambda *a, **k: types.SimpleNamespace(multi_face_landmarks=None)
                ))
                mod.solutions = types.SimpleNamespace(face_mesh=fm)
            sys.modules[name] = mod


_ensure_importable()
import face_analysis as fa  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic landmark helpers
# ---------------------------------------------------------------------------

def _blank_landmarks():
    return np.zeros((468, 2), dtype=np.float64)


def _set(p, idx, xy):
    p[idx] = xy


def _make_face(length_to_width: float, jaw_to_forehead: float):
    """Construct a minimal landmark set that yields the requested key ratios."""
    p = _blank_landmarks()
    cheek_w = 100.0
    face_len = length_to_width * cheek_w
    forehead_w = 90.0
    jaw_w = jaw_to_forehead * forehead_w

    cx = 200.0
    top_y, chin_y = 50.0, 50.0 + face_len
    mid_y = 50.0 + 0.45 * face_len
    nose_y = 50.0 + 0.62 * face_len

    _set(p, fa.FOREHEAD_TOP, (cx, top_y))
    _set(p, fa.CHIN,         (cx, chin_y))
    _set(p, fa.NOSE_BRIDGE,  (cx, mid_y))
    _set(p, fa.NOSE_TIP,     (cx, nose_y))
    _set(p, fa.CHEEK_L,      (cx - cheek_w / 2, mid_y))
    _set(p, fa.CHEEK_R,      (cx + cheek_w / 2, mid_y))
    _set(p, fa.TEMPLE_L,     (cx - forehead_w / 2, top_y + 20))
    _set(p, fa.TEMPLE_R,     (cx + forehead_w / 2, top_y + 20))
    _set(p, fa.JAW_L,        (cx - jaw_w / 2, chin_y - 30))
    _set(p, fa.JAW_R,        (cx + jaw_w / 2, chin_y - 30))
    _set(p, fa.MIDJAW_L,     (cx - jaw_w * 0.35, chin_y - 12))
    _set(p, fa.MIDJAW_R,     (cx + jaw_w * 0.35, chin_y - 12))
    _set(p, fa.L_EYE_OUTER,  (cx - 35, mid_y - 20))
    _set(p, fa.R_EYE_OUTER,  (cx + 35, mid_y - 20))
    _set(p, fa.L_EYE_INNER,  (cx - 12, mid_y - 20))
    _set(p, fa.R_EYE_INNER,  (cx + 12, mid_y - 20))
    return p


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_features_are_scale_invariant():
    p = _make_face(1.45, 0.90)
    f1 = fa._compute_features(p)
    f2 = fa._compute_features(p * 3.7)  # uniform scale
    for k in f1:
        assert math.isclose(f1[k], f2[k], rel_tol=1e-6), k


def test_shape_probabilities_sum_to_one():
    p = _make_face(1.45, 0.90)
    probs = fa._score_shapes(fa._compute_features(p))
    assert set(probs) == set(fa.FACE_SHAPES)
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-6)
    for v in probs.values():
        assert 0.0 <= v <= 1.0


@pytest.mark.parametrize("lw,jf,expected", [
    (1.70, 0.92, "long"),
    (1.15, 0.98, "round"),
    (1.40, 0.72, "heart"),
])
def test_prototype_scoring_prefers_expected_shape(lw, jf, expected):
    feats = fa._compute_features(_make_face(lw, jf))
    probs = fa._score_shapes(feats)
    top = max(probs, key=probs.get)
    assert top == expected, f"expected {expected}, got {top} (probs={probs})"


def test_face_analysis_to_dict_drops_landmarks():
    res = fa.FaceAnalysis(
        success=True,
        features={"a": 1.0},
        shape_probabilities={"oval": 1.0},
        primary_shape="oval",
        secondary_shape="round",
        confidence=0.5,
        landmarks_2d=np.zeros((468, 2)),
    )
    d = res.to_dict()
    assert "landmarks_2d" not in d
    assert d["primary_shape"] == "oval"


def test_analyze_face_missing_file_is_graceful(tmp_path):
    res = fa.analyze_face(str(tmp_path / "nope.jpg"))
    assert res.success is False
    assert res.error
