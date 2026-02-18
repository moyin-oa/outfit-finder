import json
import logging
import os
import re
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

import torch
import open_clip
import faiss

CATALOG_PATH = "catalog_amazon.json"
OUT_DIR = "index_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

INDEX_PATH = os.path.join(OUT_DIR, "image.index")
META_PATH = os.path.join(OUT_DIR, "meta.json")
CATEGORY_DIR = os.path.join(OUT_DIR, "by_category")
CATEGORY_MANIFEST_PATH = os.path.join(OUT_DIR, "category_manifest.json")

DEVICE = "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_image(url: str, timeout=12):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def normalize_category(value: str | None) -> str | None:
    if not value:
        return None
    out = value.strip().lower()
    return out or None


def category_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def infer_gender(title: str | None, query_seed: str | None) -> str | None:
    text = f"{title or ''} {query_seed or ''}".lower()
    if any(tok in text for tok in ("women", "woman", "womens", "ladies", "female")):
        return "women"
    if any(tok in text for tok in ("men", "man", "mens", "male")):
        return "men"
    if any(tok in text for tok in ("girl", "boy", "kid", "kids", "children", "child")):
        return "kids"
    if "unisex" in text:
        return "unisex"
    return None


def main():
    with open(CATALOG_PATH, "r") as f:
        catalog = json.load(f)  # list of items

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    model = model.to(DEVICE).eval()

    vectors = []
    meta = []

    for item in tqdm(catalog, desc="Embedding catalog images"):
        url = item.get("image_url")
        if not url:
            continue
        try:
            img = load_image(url)
            x = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                v = model.encode_image(x)
                v = v / v.norm(dim=-1, keepdim=True)
            vectors.append(v.cpu().numpy()[0].astype("float32"))
            title = item.get("title")
            query_seed = item.get("query_seed")
            meta.append(
                {
                    "id": item.get("id"),
                    "source": item.get("source"),
                    "asin": item.get("asin"),
                    "title": title,
                    "price_usd": item.get("price_usd"),
                    "rating": item.get("rating"),
                    "num_ratings": item.get("num_ratings"),
                    "product_url": item.get("product_url"),
                    "image_url": url,
                    "category_hint": normalize_category(item.get("category_hint")),
                    "query_seed": query_seed,
                    "gender_hint": infer_gender(title, query_seed),
                }
            )
        except Exception as exc:
            logging.warning("Skipping item due to image/embed failure: %s (%s)", url, exc)
            continue

    if not vectors:
        raise RuntimeError("No vectors created. Check catalog/image URLs.")

    emb = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])  # cosine similarity since vectors are normalized
    index.add(emb)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    os.makedirs(CATEGORY_DIR, exist_ok=True)
    category_to_global_ids = {}
    for i, item in enumerate(meta):
        cat = item.get("category_hint")
        if not cat:
            continue
        category_to_global_ids.setdefault(cat, []).append(i)

    manifest = {}
    for cat, global_ids in sorted(category_to_global_ids.items()):
        if not global_ids:
            continue
        cat_emb = emb[np.array(global_ids, dtype=np.int32)]
        cat_index = faiss.IndexFlatIP(cat_emb.shape[1])
        cat_index.add(cat_emb)
        slug = category_slug(cat)
        index_filename = f"{slug}.index"
        ids_filename = f"{slug}.ids.json"
        faiss.write_index(cat_index, os.path.join(CATEGORY_DIR, index_filename))
        with open(os.path.join(CATEGORY_DIR, ids_filename), "w") as f:
            json.dump(global_ids, f)
        manifest[cat] = {
            "slug": slug,
            "index_path": os.path.join(CATEGORY_DIR, index_filename),
            "ids_path": os.path.join(CATEGORY_DIR, ids_filename),
            "size": len(global_ids),
        }

    with open(CATEGORY_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved index with {len(meta)} items to {INDEX_PATH}")
    print(f"Saved {len(manifest)} category indexes to {CATEGORY_DIR}")
    print(f"Model: {MODEL_NAME} ({PRETRAINED})")

if __name__ == "__main__":
    main()
