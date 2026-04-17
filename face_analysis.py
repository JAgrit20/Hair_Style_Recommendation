"""
Mediapipe-based face analysis.

Replaces the dlib/face_recognition pipeline with Mediapipe FaceMesh (468 3D
landmarks) and produces:

  1. A named, scale-invariant geometric feature vector (ratios & angles only,
     so it's independent of camera distance / image resolution — no eye-
     alignment cropping pass needed).
  2. Soft probabilities over the 5 face-shape classes instead of a single hard
     label, plus a confidence score.
  3. An optional annotated debug image.

The output is a plain dict so it can be returned directly from a Flask endpoint
or fed into the downstream recommender.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


# ---------------------------------------------------------------------------
# Landmark index groups (subset of the 468 FaceMesh indices we actually use).
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ---------------------------------------------------------------------------

# Face oval, ordered clockwise starting top-center of forehead.
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

L_EYE_OUTER, L_EYE_INNER = 33, 133
R_EYE_OUTER, R_EYE_INNER = 263, 362
FOREHEAD_TOP = 10          # hairline-ish (top of mesh, not true hairline)
CHIN = 152
NOSE_BRIDGE = 6
NOSE_TIP = 1
MOUTH_L, MOUTH_R = 61, 291
CHEEK_L, CHEEK_R = 234, 454           # widest point at cheekbones
TEMPLE_L, TEMPLE_R = 127, 356         # forehead width just above brows
JAW_L, JAW_R = 172, 397               # jaw corner / gonion
MIDJAW_L, MIDJAW_R = 150, 379         # halfway between gonion and chin

FACE_SHAPES = ["oval", "round", "square", "heart", "long"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FaceAnalysis:
    success: bool
    error: Optional[str] = None

    features: Dict[str, float] = field(default_factory=dict)
    shape_probabilities: Dict[str, float] = field(default_factory=dict)
    primary_shape: Optional[str] = None
    secondary_shape: Optional[str] = None
    confidence: float = 0.0            # margin between top-1 and top-2 prob
    landmarks_2d: Optional[np.ndarray] = None  # (468, 2) pixel coords

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("landmarks_2d", None)    # don't serialise the raw array
        return d


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------

# FaceMesh is expensive to construct; keep one module-level instance.
_mp_face_mesh = mp.solutions.face_mesh
_detector = _mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


def _dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def _angle_deg(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Interior angle (degrees) at `vertex` in triangle a-vertex-b."""
    v1 = a - vertex
    v2 = b - vertex
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _extract_landmarks(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    h, w = image_bgr.shape[:2]
    results = _detector.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)
    return pts  # (468, 2)


def _compute_features(p: np.ndarray) -> Dict[str, float]:
    """All features are ratios or angles → scale & translation invariant."""
    face_len = _dist(p[FOREHEAD_TOP], p[CHIN])
    cheek_w = _dist(p[CHEEK_L], p[CHEEK_R])
    forehead_w = _dist(p[TEMPLE_L], p[TEMPLE_R])
    jaw_w = _dist(p[JAW_L], p[JAW_R])
    midjaw_w = _dist(p[MIDJAW_L], p[MIDJAW_R])
    interocular = _dist(p[L_EYE_OUTER], p[R_EYE_OUTER])

    # Vertical thirds (forehead / midface / lower face) — stylists use these.
    brow_mid = (p[L_EYE_OUTER] + p[R_EYE_OUTER]) / 2.0
    upper = _dist(p[FOREHEAD_TOP], brow_mid)
    middle = _dist(brow_mid, p[NOSE_TIP])
    lower = _dist(p[NOSE_TIP], p[CHIN])

    jaw_angle = _angle_deg(p[JAW_L], p[CHIN], p[JAW_R])
    chin_angle = _angle_deg(p[MIDJAW_L], p[CHIN], p[MIDJAW_R])
    cheekbone_prom = cheek_w / max(forehead_w, jaw_w)

    eps = 1e-9
    return {
        "face_length_to_width":   face_len / (cheek_w + eps),
        "forehead_to_cheek":      forehead_w / (cheek_w + eps),
        "jaw_to_cheek":           jaw_w / (cheek_w + eps),
        "jaw_to_forehead":        jaw_w / (forehead_w + eps),
        "midjaw_to_jaw":          midjaw_w / (jaw_w + eps),
        "interocular_to_width":   interocular / (cheek_w + eps),
        "upper_third_ratio":      upper / (face_len + eps),
        "middle_third_ratio":     middle / (face_len + eps),
        "lower_third_ratio":      lower / (face_len + eps),
        "jaw_angle_deg":          jaw_angle,
        "chin_angle_deg":         chin_angle,
        "cheekbone_prominence":   cheekbone_prom,
    }


def _gauss(x: float, mu: float, sigma: float) -> float:
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


# Prototype feature centres per shape. These are hand-tuned priors; once you
# regenerate `all_features.csv` with this extractor you can replace this with a
# learned classifier (see `train_shape_classifier` stub in this module).
_SHAPE_PROTOTYPES: Dict[str, Dict[str, Tuple[float, float]]] = {
    # feature_name: (expected_value, sigma)
    "oval": {
        "face_length_to_width": (1.45, 0.12),
        "jaw_to_forehead":      (0.88, 0.10),
        "jaw_angle_deg":        (122.0, 10.0),
        "cheekbone_prominence": (1.05, 0.06),
    },
    "round": {
        "face_length_to_width": (1.15, 0.10),
        "jaw_to_forehead":      (0.98, 0.08),
        "jaw_angle_deg":        (135.0, 10.0),
        "cheekbone_prominence": (1.04, 0.06),
    },
    "square": {
        "face_length_to_width": (1.20, 0.12),
        "jaw_to_forehead":      (1.02, 0.08),
        "jaw_angle_deg":        (108.0, 10.0),
        "midjaw_to_jaw":        (0.92, 0.06),
    },
    "heart": {
        "face_length_to_width": (1.40, 0.12),
        "jaw_to_forehead":      (0.72, 0.10),
        "chin_angle_deg":       (95.0, 12.0),
        "forehead_to_cheek":    (1.00, 0.06),
    },
    "long": {
        "face_length_to_width": (1.70, 0.15),
        "jaw_to_forehead":      (0.92, 0.10),
        "lower_third_ratio":    (0.40, 0.05),
    },
}


def _score_shapes(features: Dict[str, float]) -> Dict[str, float]:
    raw: Dict[str, float] = {}
    for shape, proto in _SHAPE_PROTOTYPES.items():
        s = 1.0
        for feat, (mu, sigma) in proto.items():
            s *= _gauss(features[feat], mu, sigma)
        # geometric mean so shapes with different #features are comparable
        raw[shape] = s ** (1.0 / len(proto))
    total = sum(raw.values()) + 1e-12
    return {k: v / total for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_face(image_path: str) -> FaceAnalysis:
    """Run the full pipeline on a single image file."""
    img = cv2.imread(image_path)
    if img is None:
        return FaceAnalysis(success=False, error=f"could not read image: {image_path}")

    pts = _extract_landmarks(img)
    if pts is None:
        return FaceAnalysis(success=False, error="no face detected")

    feats = _compute_features(pts)
    probs = _score_shapes(feats)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    primary, p1 = ranked[0]
    secondary, p2 = ranked[1]

    return FaceAnalysis(
        success=True,
        features={k: round(v, 4) for k, v in feats.items()},
        shape_probabilities={k: round(v, 4) for k, v in probs.items()},
        primary_shape=primary,
        secondary_shape=secondary,
        confidence=round(p1 - p2, 4),
        landmarks_2d=pts,
    )


def annotate(image_path: str, out_path: str) -> Optional[str]:
    """Write a debug image with the face oval + key measurement lines drawn."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    pts = _extract_landmarks(img)
    if pts is None:
        return None

    oval = pts[FACE_OVAL].astype(np.int32)
    cv2.polylines(img, [oval], isClosed=True, color=(0, 255, 0), thickness=2)

    pairs = [
        (CHEEK_L, CHEEK_R, (255, 200, 0)),
        (TEMPLE_L, TEMPLE_R, (255, 120, 0)),
        (JAW_L, JAW_R, (0, 120, 255)),
        (FOREHEAD_TOP, CHIN, (200, 0, 200)),
    ]
    for a, b, col in pairs:
        pa, pb = tuple(pts[a].astype(int)), tuple(pts[b].astype(int))
        cv2.line(img, pa, pb, col, 2)
        cv2.circle(img, pa, 3, col, -1)
        cv2.circle(img, pb, 3, col, -1)

    cv2.imwrite(out_path, img)
    return out_path


def feature_vector(image_path: str) -> Optional[np.ndarray]:
    """Convenience: just the numeric feature vector, ordered by sorted key."""
    res = analyze_face(image_path)
    if not res.success:
        return None
    keys = sorted(res.features.keys())
    return np.array([res.features[k] for k in keys], dtype=np.float64)


# ---------------------------------------------------------------------------
# Stub: regenerate training data with the new extractor.
# Run once over data/pics/<shape>/* to build a CSV compatible with a learned
# classifier that replaces _SHAPE_PROTOTYPES.
# ---------------------------------------------------------------------------

def build_feature_dataset(image_root: str, out_csv: str) -> None:
    import csv
    import pathlib

    rows: List[dict] = []
    for shape_dir in pathlib.Path(image_root).iterdir():
        if not shape_dir.is_dir():
            continue
        label = shape_dir.name
        for img in shape_dir.glob("*"):
            res = analyze_face(str(img))
            if not res.success:
                continue
            row = {"label": label, "file": str(img), **res.features}
            rows.append(row)

    if not rows:
        raise RuntimeError(f"no faces extracted under {image_root}")

    keys = ["label", "file"] + sorted(rows[0].keys() - {"label", "file"})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("usage: python face_analysis.py <image> [annotated_out.jpg]")
        sys.exit(1)

    result = analyze_face(sys.argv[1])
    print(json.dumps(result.to_dict(), indent=2))

    if len(sys.argv) >= 3 and result.success:
        annotate(sys.argv[1], sys.argv[2])
        print(f"annotated image written to {sys.argv[2]}")
