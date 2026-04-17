"""
CLIP-based visual similarity retrieval for hairstyle recommendations.

Builds a persistent embedding index over the reference style library
(`data/rec_pics/<shape>/<length>/*.jpg`) and ranks styles by cosine similarity
to the uploaded user photo.

Design notes
------------
* Backend is pluggable. Default is `open_clip` (ViT-B-32, laion2b weights).
  Tests inject a deterministic stub via `StyleIndex(embedder=...)`.
* The index is cached to a single `.npz` so cold start after the first build
  is ~instant and doesn't require a GPU.
* Everything is L2-normalised so similarity is a plain dot product.
"""

from __future__ import annotations

import os
import json
import pathlib
import hashlib
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Embedder backends
# ---------------------------------------------------------------------------

EmbedFn = Callable[[Sequence[Image.Image]], np.ndarray]


def _load_open_clip(model_name: str = "ViT-B-32",
                    pretrained: str = "laion2b_s34b_b79k",
                    device: Optional[str] = None) -> EmbedFn:
    """Return an embed(images) -> (N, D) float32 function backed by open_clip."""
    import torch
    import open_clip

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()

    @torch.no_grad()
    def _embed(images: Sequence[Image.Image]) -> np.ndarray:
        batch = torch.stack([preprocess(im.convert("RGB")) for im in images]).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    return _embed


def get_default_embedder() -> EmbedFn:
    """Lazily construct the default CLIP embedder (heavy — call once)."""
    return _load_open_clip()


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

@dataclass
class StyleItem:
    path: str
    face_shape: str
    hair_length: str

    def to_dict(self) -> dict:
        return asdict(self)


def _scan_library(root: str) -> List[StyleItem]:
    """data/rec_pics/<shape>/<length>/<file>"""
    items: List[StyleItem] = []
    rootp = pathlib.Path(root)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for shape_dir in sorted(p for p in rootp.iterdir() if p.is_dir()):
        for len_dir in sorted(p for p in shape_dir.iterdir() if p.is_dir()):
            for f in sorted(len_dir.iterdir()):
                if f.suffix.lower() in exts:
                    items.append(StyleItem(
                        path=str(f),
                        face_shape=shape_dir.name,
                        hair_length=len_dir.name,
                    ))
    return items


def _fingerprint(items: List[StyleItem]) -> str:
    h = hashlib.sha1()
    for it in items:
        h.update(it.path.encode())
    return h.hexdigest()[:16]


class StyleIndex:
    """In-memory index of (item metadata, embedding matrix)."""

    def __init__(
        self,
        root: str = "data/rec_pics",
        cache_path: str = "data/clip_index.npz",
        embedder: Optional[EmbedFn] = None,
        batch_size: int = 32,
    ):
        self.root = root
        self.cache_path = cache_path
        self.batch_size = batch_size
        self._embedder = embedder  # lazy if None
        self.items: List[StyleItem] = []
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    # -- lifecycle ---------------------------------------------------------

    @property
    def embedder(self) -> EmbedFn:
        if self._embedder is None:
            self._embedder = get_default_embedder()
        return self._embedder

    def load_or_build(self) -> "StyleIndex":
        items = _scan_library(self.root)
        if not items:
            raise RuntimeError(f"no reference images found under {self.root}")
        fp = _fingerprint(items)

        if os.path.exists(self.cache_path):
            data = np.load(self.cache_path, allow_pickle=False)
            if str(data["fingerprint"]) == fp:
                meta = json.loads(str(data["meta"]))
                self.items = [StyleItem(**m) for m in meta]
                self.embeddings = data["embeddings"].astype(np.float32)
                return self

        self.items = items
        self.embeddings = self._embed_paths([it.path for it in items])
        self._save(fp)
        return self

    def _embed_paths(self, paths: List[str]) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for i in range(0, len(paths), self.batch_size):
            batch_paths = paths[i:i + self.batch_size]
            imgs = [Image.open(p) for p in batch_paths]
            chunks.append(self.embedder(imgs))
            for im in imgs:
                im.close()
        embs = np.concatenate(chunks, axis=0)
        # ensure normalised even if a custom embedder forgot
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        return (embs / norms).astype(np.float32)

    def _save(self, fingerprint: str) -> None:
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        np.savez(
            self.cache_path,
            embeddings=self.embeddings,
            meta=json.dumps([it.to_dict() for it in self.items]),
            fingerprint=fingerprint,
        )

    # -- query -------------------------------------------------------------

    def embed_image(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as im:
            v = self.embedder([im])[0]
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype(np.float32)

    def query(
        self,
        image_path: str,
        k: int = 6,
        hair_length: Optional[str] = None,
        shape_weights: Optional[Dict[str, float]] = None,
        shape_bonus: float = 0.15,
    ) -> List[dict]:
        """
        Rank reference styles for `image_path`.

        score = clip_cosine + shape_bonus * P(shape_of_reference | user_face)

        `shape_weights` is the soft probability dict from face_analysis; if
        omitted, ranking is pure visual similarity.
        """
        if self.embeddings.shape[0] == 0:
            raise RuntimeError("index is empty — call load_or_build() first")

        q = self.embed_image(image_path)
        sims = self.embeddings @ q  # (N,)

        results = []
        for i, it in enumerate(self.items):
            if hair_length and it.hair_length.lower() != hair_length.lower():
                continue
            bonus = 0.0
            if shape_weights:
                bonus = shape_bonus * float(shape_weights.get(it.face_shape, 0.0))
            results.append({
                "path": it.path,
                "face_shape": it.face_shape,
                "hair_length": it.hair_length,
                "clip_similarity": round(float(sims[i]), 4),
                "shape_bonus": round(bonus, 4),
                "score": round(float(sims[i]) + bonus, 4),
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:k]


# Module-level singleton helpers ------------------------------------------------

_INDEX: Optional[StyleIndex] = None


def get_index(root: str = "data/rec_pics",
              cache_path: str = "data/clip_index.npz") -> StyleIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = StyleIndex(root=root, cache_path=cache_path).load_or_build()
    return _INDEX


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build or query the CLIP style index")
    ap.add_argument("cmd", choices=["build", "query"])
    ap.add_argument("--image", help="query image path")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--hair", default=None)
    args = ap.parse_args()

    idx = get_index()
    if args.cmd == "build":
        print(f"index built: {len(idx.items)} items, dim={idx.embeddings.shape[1]}")
    else:
        if not args.image:
            ap.error("--image required for query")
        for r in idx.query(args.image, k=args.k, hair_length=args.hair):
            print(f"{r['score']:.3f}  {r['face_shape']:<7} {r['hair_length']:<6} {r['path']}")
