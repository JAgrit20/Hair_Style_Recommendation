"""
V2 recommender: fuses Mediapipe face geometry with CLIP visual similarity.

This module is the single orchestration point the Flask layer calls. It owns
no Flask types — pure functions in, dicts out — so it's trivially testable.
"""

from __future__ import annotations

import os
import uuid
from typing import Dict, List, Optional

from face_analysis import analyze_face, annotate, FaceAnalysis
from clip_retrieval import StyleIndex, get_index


# Human-readable rationale per shape. Shown in the UI under each result.
_SHAPE_ADVICE: Dict[str, str] = {
    "oval":   "Balanced proportions — most styles work; avoid heavy fringes that hide the forehead.",
    "round":  "Add height and angles on top; avoid width at the cheeks.",
    "square": "Soften the jawline with layers or waves; avoid blunt jaw-length cuts.",
    "heart":  "Add volume around the jaw to balance a wider forehead; side-swept fringes work well.",
    "long":   "Add width at the sides; avoid extra height on top or very long straight styles.",
}


def _normalise_hair_length(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    r = raw.strip().lower()
    if r in ("short", "s"):
        return "Short"
    if r in ("long", "l", "longer"):
        return "Long"
    if r in ("updo", "u", "up"):
        return "Updo"
    return None


def recommend(
    image_path: str,
    hair_length: Optional[str] = None,
    k: int = 6,
    index: Optional[StyleIndex] = None,
    write_annotated: bool = True,
    output_dir: str = "output",
) -> dict:
    """
    Full V2 pipeline.

    Returns a JSON-serialisable dict:
        {
          success, error?,
          face: { primary_shape, secondary_shape, confidence,
                  shape_probabilities, features, advice },
          annotated_img?,
          recommendations: [ {path, face_shape, hair_length,
                              clip_similarity, shape_bonus, score, reason}, ... ]
        }
    """
    fa: FaceAnalysis = analyze_face(image_path)
    if not fa.success:
        return {"success": False, "error": fa.error}

    idx = index or get_index()
    hl = _normalise_hair_length(hair_length)

    recs: List[dict] = idx.query(
        image_path,
        k=k,
        hair_length=hl,
        shape_weights=fa.shape_probabilities,
    )
    for r in recs:
        r["reason"] = _SHAPE_ADVICE.get(r["face_shape"], "")

    annotated_path: Optional[str] = None
    if write_annotated:
        os.makedirs(output_dir, exist_ok=True)
        annotated_path = os.path.join(output_dir, f"annotated_{uuid.uuid4().hex[:8]}.jpg")
        if annotate(image_path, annotated_path) is None:
            annotated_path = None

    return {
        "success": True,
        "face": {
            "primary_shape": fa.primary_shape,
            "secondary_shape": fa.secondary_shape,
            "confidence": fa.confidence,
            "shape_probabilities": fa.shape_probabilities,
            "features": fa.features,
            "advice": _SHAPE_ADVICE.get(fa.primary_shape, ""),
        },
        "annotated_img": annotated_path,
        "recommendations": recs,
    }
