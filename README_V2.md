# Hairstyle Recommender — V2

V2 replaces the dlib → 5-bucket classifier with a two-signal pipeline:

1. **Mediapipe FaceMesh** (468 landmarks) → 12 named, scale-invariant geometric
   features → **soft probabilities** over `oval / round / square / heart / long`
   with an explicit confidence margin.
2. **CLIP (open_clip ViT-B/32)** visual similarity between the user photo and
   every reference style in `data/rec_pics/`, fused with the shape prior:
   `score = cosine(clip) + 0.15 · P(shape)`.

The result is rendered in a new single-page UI at `/` (legacy UI lives at `/legacy`).

## Run

```bash
pip install -r requirements_v2.txt
python -m flask --app app run            # or: python app.py
# first /api/v2/analyze call builds data/clip_index.npz (one-time, ~30s CPU)
```

## Layout

```
face_analysis.py     mediapipe geometry + shape probabilities + overlay
clip_retrieval.py    CLIP embedder, persistent .npz index, similarity query
recommender_v2.py    fuses the two, returns JSON-ready dict
app.py               flask: /, /api/v2/upload, /api/v2/analyze, legacy routes
templates/v2.html    UI shell
static/v2.js|css     UI logic / styling
tests/               pytest, models stubbed (no GPU/mediapipe needed on CI)
```

## API

`POST /api/v2/upload` — multipart `file` → `{filename, url}`

`POST /api/v2/analyze` — `{filename, hair_length?, k?}` →
```json
{
  "success": true,
  "face": {
    "primary_shape": "oval", "secondary_shape": "heart", "confidence": 0.18,
    "shape_probabilities": {"oval": 0.41, ...},
    "features": {"face_length_to_width": 1.43, ...},
    "advice": "Balanced proportions — most styles work; ..."
  },
  "annotated_img": "/output/annotated_xxxx.jpg",
  "recommendations": [
    {"url": "/styles/oval/Long/12.jpg", "face_shape": "oval",
     "hair_length": "Long", "clip_similarity": 0.612,
     "shape_bonus": 0.061, "score": 0.673, "reason": "..."}
  ]
}
```

## Tests

```bash
pytest -q
```

Tests stub `mediapipe`, `cv2` and the CLIP embedder, so they run on a plain
Python env with only `numpy` + `pillow` + `pytest`.

## Tuning knobs

- `face_analysis._SHAPE_PROTOTYPES` — replace with a learned classifier after
  running `face_analysis.build_feature_dataset("data/pics", "mediapipe_features.csv")`.
- `clip_retrieval.StyleIndex.query(shape_bonus=...)` — weight of geometry prior
  vs pure visual similarity.
- Swap `_load_open_clip` for a face-specific encoder (ArcFace) if you want
  identity-aware similarity instead of general visual similarity.
