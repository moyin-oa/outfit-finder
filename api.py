import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
from io import BytesIO

import faiss
import numpy as np
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


INDEX_PATH = "index_artifacts/image.index"
META_PATH = "index_artifacts/meta.json"

DEVICE = "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
RETRIEVAL_K = 100
IMAGE_WEIGHT = 0.80
TEXT_WEIGHT = 0.20

app = FastAPI()

# allow your React app / extension to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

# load once at startup
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r") as f:
    meta_raw = json.load(f)
meta = meta_raw["items"] if isinstance(meta_raw, dict) and "items" in meta_raw else meta_raw

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
)
model.eval()
tokenizer = open_clip.get_tokenizer(MODEL_NAME)


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def build_title_embeddings(items, batch_size: int = 256) -> np.ndarray:
    d = index.d
    out = np.zeros((len(items), d), dtype=np.float32)
    texts = []
    positions = []

    for i, item in enumerate(items):
        title = (item.get("title") or "").strip()
        if not title:
            continue
        texts.append(title[:256])
        positions.append(i)

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        idxs = positions[start : start + batch_size]
        with torch.no_grad():
            tok = tokenizer(chunk).to(DEVICE)
            feats = model.encode_text(tok)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        out[idxs] = feats.cpu().numpy().astype("float32")

    return _normalize_rows(out)


title_embeddings = build_title_embeddings(meta)


def embed_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    with torch.no_grad():
        image_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        feats = model.encode_image(image_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


@app.get("/health")
def health():
    return {
        "ok": True,
        "index_ntotal": int(index.ntotal),
        "meta_items": int(len(meta)),
        "retrieval_k": RETRIEVAL_K,
        "weights": {
            "image": IMAGE_WEIGHT,
            "text": TEXT_WEIGHT,
        },
    }


@app.post("/search")
async def search(file: UploadFile = File(...), k: int = 5, category: str | None = None):
    content = await file.read()
    img = Image.open(BytesIO(content))
    q = embed_pil(img)

    SEARCH_K = max(k * 20, RETRIEVAL_K)  # search wider, then rerank
    distances, ids = index.search(q, SEARCH_K)
    qv = q[0]

    target_category = (category or "").strip().lower() or None

    all_results = []
    filtered_results = []
    for idx, score in zip(ids[0].tolist(), distances[0].tolist()):
        if idx < 0 or idx >= len(meta):
            continue
        item = meta[idx]

        category = str(item.get("category_hint") or "").strip().lower()
        raw_score = float(score)
        text_score = float(np.dot(qv, title_embeddings[idx]))
        final_score = (IMAGE_WEIGHT * raw_score) + (TEXT_WEIGHT * text_score)
        row = {
            "idx": idx,
            "raw_score": raw_score,
            "text_score": text_score,
            "final_score": float(final_score),
            "title": item.get("title"),
            "url": item.get("product_url"),
            "image": item.get("image_url"),
            "price_usd": item.get("price_usd"),
            "category": category,
        }
        all_results.append(row)

        if (target_category is None) or (category == target_category):
            filtered_results.append(row)

    using_filter = target_category is not None
    filtered_results.sort(key=lambda x: x["final_score"], reverse=True)
    all_results.sort(key=lambda x: x["final_score"], reverse=True)
    results = filtered_results[:k]
    if using_filter and not results:
        results = all_results[:k]

    return {
        "results": results,
        "count": len(results),
        "k": k,
        "category": target_category,
        "filtered": using_filter,
        "filter_fallback_used": using_filter and not filtered_results,
    }
