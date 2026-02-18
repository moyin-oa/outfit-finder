import argparse
import json
from io import BytesIO

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import faiss
import numpy as np
import requests
import torch
import open_clip
from PIL import Image

torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

INDEX_PATH = "index_artifacts/image.index"
META_PATH = "index_artifacts/meta.json"

DEVICE = "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
    )
    return model.eval(), preprocess


def embed_image_from_url(url: str, model, preprocess) -> np.ndarray:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")

    with torch.no_grad():
        image_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        feats = model.encode_image(image_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    vec = feats.cpu().numpy().astype("float32")  # shape (1, d)
    return vec

def load_meta(path: str):
    with open(path, "r") as f:
        meta = json.load(f)
    # normalize to a list of items
    if isinstance(meta, list):
        return meta
    if isinstance(meta, dict) and "items" in meta and isinstance(meta["items"], list):
        return meta["items"]
    raise ValueError("meta.json format not recognized")


def main():
    parser = argparse.ArgumentParser(description="Search nearest fashion items by image URL.")
    parser.add_argument(
        "--query-url",
        help="Image URL to use as the search query. Defaults to the first item image in meta.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors to return.")
    args = parser.parse_args()

    index = faiss.read_index(INDEX_PATH)
    meta = load_meta(META_PATH)
    model, preprocess = load_model()

    print("FAISS index size:", index.ntotal)
    print("Meta items:", len(meta))

    default_url = meta[0].get("image_url") or meta[0].get("image") or meta[0].get("product_photo")
    query_url = args.query_url or default_url
    if not query_url:
        raise ValueError("No query image URL provided and no usable URL found in meta[0].")
    print("Query image URL:", query_url)

    q = embed_image_from_url(query_url, model, preprocess)

    k = args.k
    distances, ids = index.search(q, k)

    print("\nTop matches:")
    for rank, (idx, dist) in enumerate(zip(ids[0].tolist(), distances[0].tolist()), start=1):
        item = meta[idx]
        title = item.get("title") or item.get("product_title") or item.get("name")
        url = item.get("url") or item.get("product_url")
        img = item.get("image_url") or item.get("product_photo") or item.get("image")
        print(f"{rank}. id={idx}  score={dist:.4f}")
        print(f"   {title}")
        print(f"   {url}")
        print(f"   {img}")


if __name__ == "__main__":
    main()
