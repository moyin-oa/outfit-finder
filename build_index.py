import json
import os
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

DEVICE = "cpu"

def load_image(url: str, timeout=12):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def main():
    with open(CATALOG_PATH, "r") as f:
        catalog = json.load(f)  # list of items

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
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
            meta.append({
                "title": item.get("title"),
                "price_usd": item.get("price_usd"),
                "product_url": item.get("product_url"),
                "image_url": url,
                "category_hint": item.get("category_hint"),
            })
        except Exception:
            continue

    if not vectors:
        raise RuntimeError("No vectors created. Check catalog/image URLs.")

    emb = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])  # cosine similarity since vectors are normalized
    index.add(emb)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved index with {len(meta)} items to {INDEX_PATH}")

if __name__ == "__main__":
    main()
