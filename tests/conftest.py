import sys
import hashlib
import pathlib

import numpy as np
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _stub_embed(images):
    """Deterministic 16-dim 'embedding' derived from average pixel colour.

    Cheap, no torch/CLIP dependency, and stable across runs so tests can make
    real assertions about ranking order.
    """
    out = []
    for im in images:
        arr = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32)
        v = arr.mean(axis=(0, 1))                       # (3,)
        seed = int(hashlib.sha1(v.tobytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        extra = rng.standard_normal(13).astype(np.float32)
        vec = np.concatenate([v / 255.0, extra])        # (16,)
        vec /= (np.linalg.norm(vec) + 1e-9)
        out.append(vec)
    return np.stack(out)


@pytest.fixture
def stub_embedder():
    return _stub_embed


@pytest.fixture
def style_library(tmp_path):
    """Create a tiny rec_pics tree: 2 shapes × 2 lengths × 2 images."""
    root = tmp_path / "rec_pics"
    colours = {
        "oval":   {"Short": (200, 50, 50), "Long": (200, 80, 80)},
        "square": {"Short": (50, 50, 200), "Long": (80, 80, 200)},
    }
    for shape, lengths in colours.items():
        for length, col in lengths.items():
            d = root / shape / length
            d.mkdir(parents=True)
            for i in range(2):
                Image.new("RGB", (32, 32), col).save(d / f"{i}.jpg")
    return root


@pytest.fixture
def query_image(tmp_path):
    """A reddish query image — should rank closer to the 'oval' (red) library."""
    p = tmp_path / "user.jpg"
    Image.new("RGB", (32, 32), (210, 55, 55)).save(p)
    return p
