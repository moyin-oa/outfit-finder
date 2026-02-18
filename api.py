import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import logging
from io import BytesIO
from typing import Any

import faiss
import numpy as np
import requests
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


INDEX_PATH = "index_artifacts/image.index"
META_PATH = "index_artifacts/meta.json"
CATEGORY_MANIFEST_PATH = "index_artifacts/category_manifest.json"

DEVICE = "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
RETRIEVAL_K = 100
AUTO_CATEGORY_K = 80
IMAGE_WEIGHT = 0.80
TEXT_WEIGHT = 0.20
OPENAI_MODEL = "gpt-4.1-mini"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
global_index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r") as f:
    meta_raw = json.load(f)
meta = meta_raw["items"] if isinstance(meta_raw, dict) and "items" in meta_raw else meta_raw

category_artifacts = {}
if os.path.exists(CATEGORY_MANIFEST_PATH):
    with open(CATEGORY_MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    if isinstance(manifest, dict):
        for cat, cfg in manifest.items():
            if not isinstance(cfg, dict):
                continue
            index_path = cfg.get("index_path")
            ids_path = cfg.get("ids_path")
            if not index_path or not ids_path:
                continue
            if not (os.path.exists(index_path) and os.path.exists(ids_path)):
                continue
            try:
                cat_index = faiss.read_index(index_path)
                with open(ids_path, "r") as fid:
                    cat_ids = json.load(fid)
                if isinstance(cat_ids, list):
                    category_artifacts[cat] = {"index": cat_index, "global_ids": cat_ids}
            except Exception as exc:
                logging.warning("Failed loading category index for '%s': %s", cat, exc)

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
    d = global_index.d
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


def normalize_category(value: str | None) -> str | None:
    if value is None:
        return None
    out = value.strip().lower()
    return out or None


def infer_gender(item: dict) -> str | None:
    existing = item.get("gender_hint")
    if existing:
        return str(existing).strip().lower()
    text = f"{item.get('title') or ''} {item.get('query_seed') or ''}".lower()
    if any(tok in text for tok in ("women", "woman", "womens", "ladies", "female")):
        return "women"
    if any(tok in text for tok in ("men", "man", "mens", "male")):
        return "men"
    if any(tok in text for tok in ("girl", "boy", "kid", "kids", "children", "child")):
        return "kids"
    if "unisex" in text:
        return "unisex"
    return None


def coerce_price(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def passes_filters(
    item: dict,
    target_category: str | None,
    price_min: float | None,
    price_max: float | None,
    target_gender: str | None,
) -> bool:
    item_category = normalize_category(item.get("category_hint"))
    if target_category is not None and item_category != target_category:
        return False

    item_gender = infer_gender(item)
    if target_gender is not None and item_gender != target_gender:
        return False

    p = coerce_price(item.get("price_usd"))
    if price_min is not None and (p is None or p < price_min):
        return False
    if price_max is not None and (p is None or p > price_max):
        return False

    return True


def predict_category_from_query(q: np.ndarray, top_k: int = AUTO_CATEGORY_K) -> str | None:
    scores_by_cat = predict_category_scores_from_query(q, top_k=top_k)
    if not scores_by_cat:
        return None
    return max(scores_by_cat.items(), key=lambda x: x[1])[0]


def predict_category_scores_from_query(q: np.ndarray, top_k: int = AUTO_CATEGORY_K) -> dict[str, float]:
    if global_index.ntotal <= 0:
        return {}
    k = min(top_k, global_index.ntotal)
    distances, ids = global_index.search(q, k)
    scores_by_cat = {}
    for idx, score in zip(ids[0].tolist(), distances[0].tolist()):
        if idx < 0 or idx >= len(meta):
            continue
        cat = normalize_category(meta[idx].get("category_hint"))
        if not cat:
            continue
        # Keep positive similarity contribution only.
        scores_by_cat[cat] = scores_by_cat.get(cat, 0.0) + max(0.0, float(score))
    return scores_by_cat


def parse_uploaded_image(file: UploadFile) -> Image.Image:
    if file is None:
        raise HTTPException(status_code=400, detail="Missing file upload")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        img = Image.open(BytesIO(content))
        img.verify()
        return Image.open(BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file")


def fallback_description_from_breakdown(
    breakdown: dict[str, Any],
    tone: str = "editorial",
    max_tips: int = 3,
    llm_error: str | None = None,
) -> dict[str, Any]:
    groups = breakdown.get("groups") or []
    detected = [g.get("category") for g in groups if g.get("category")]
    detected = [str(x) for x in detected][:5]
    if not detected:
        detected = ["item"]

    piece_summaries = []
    for g in groups[:5]:
        cat = g.get("category") or "item"
        top = (g.get("results") or [{}])[0] or {}
        piece_summaries.append(
            {
                "piece": str(cat),
                "summary": f"Top match: {top.get('title') or 'catalog item'}",
                "why_it_matches": "Visual similarity score and category alignment from the retrieval engine.",
            }
        )

    tips = [
        "Keep one focal piece and let other items stay neutral.",
        "Match texture family across pieces for a cohesive look.",
        "Use accessories in the same tone range to tie the outfit together.",
        "Balance proportions: structured pieces pair well with softer silhouettes.",
    ][:max_tips]

    confidence_notes = []
    for g in groups[:5]:
        conf = float(g.get("confidence") or 0.0)
        confidence_notes.append(
            f"{g.get('category')}: {round(conf * 100)}% detection confidence from visual retrieval."
        )

    return {
        "source": "fallback",
        "model": None,
        "llm_attempted": bool(llm_error),
        "llm_error": llm_error,
        "description": (
            f"A {tone} look centered around {', '.join(detected)} with visually similar catalog matches."
        ),
        "piece_summaries": piece_summaries,
        "styling_tips": tips,
        "confidence_notes": confidence_notes,
    }


def llm_description_from_breakdown(
    breakdown: dict[str, Any],
    tone: str = "editorial",
    max_tips: int = 3,
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    schema = {
        "name": "outfit_description",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["description", "piece_summaries", "styling_tips", "confidence_notes"],
            "properties": {
                "description": {"type": "string"},
                "piece_summaries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["piece", "summary", "why_it_matches"],
                        "properties": {
                            "piece": {"type": "string"},
                            "summary": {"type": "string"},
                            "why_it_matches": {"type": "string"},
                        },
                    },
                },
                "styling_tips": {"type": "array", "items": {"type": "string"}},
                "confidence_notes": {"type": "array", "items": {"type": "string"}},
            },
        },
        "strict": True,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a fashion stylist. Use only the provided retrieval output. "
                "Do not invent products or categories. Return strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Tone: {tone}. Max tips: {max_tips}. "
                f"Breakdown JSON: {json.dumps(breakdown, ensure_ascii=True)}"
            ),
        },
    ]

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": schema},
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    parsed = json.loads(raw)
    parsed["source"] = "llm"
    parsed["model"] = OPENAI_MODEL
    return parsed


@app.get("/health")
def health():
    return {
        "ok": True,
        "index_ntotal": int(global_index.ntotal),
        "meta_items": int(len(meta)),
        "category_indexes": sorted(category_artifacts.keys()),
        "retrieval_k": RETRIEVAL_K,
        "auto_category_k": AUTO_CATEGORY_K,
        "weights": {
            "image": IMAGE_WEIGHT,
            "text": TEXT_WEIGHT,
        },
    }


@app.post("/search")
async def search(
    file: UploadFile = File(...),
    k: int = 5,
    category: str | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    gender: str | None = None,
):
    if k <= 0:
        k = 5
    img = parse_uploaded_image(file)

    q = embed_pil(img)
    qv = q[0]

    requested_category = normalize_category(category)
    predicted_category = None
    category_source = "none"
    if requested_category is not None:
        target_category = requested_category
        category_source = "user"
    else:
        predicted_category = predict_category_from_query(q)
        target_category = predicted_category
        category_source = "predicted" if predicted_category else "none"

    target_gender = normalize_category(gender)
    if price_min is not None and price_max is not None and price_min > price_max:
        return {
            "results": [],
            "count": 0,
            "k": k,
            "error": "price_min cannot be greater than price_max",
        }

    search_index = global_index
    index_used = "global"
    cat_map = None
    if target_category and target_category in category_artifacts:
        search_index = category_artifacts[target_category]["index"]
        cat_map = category_artifacts[target_category]["global_ids"]
        index_used = f"category:{target_category}"

    search_k = max(k * 20, RETRIEVAL_K)
    search_k = min(search_k, search_index.ntotal) if search_index.ntotal > 0 else 0
    if search_k == 0:
        return {
            "results": [],
            "count": 0,
            "k": k,
            "requested_category": requested_category,
            "predicted_category": predicted_category,
            "category": target_category,
            "category_source": category_source,
            "gender": target_gender,
            "price_min": price_min,
            "price_max": price_max,
            "index_used": index_used,
        }

    distances, ids = search_index.search(q, search_k)

    all_results = []
    for local_idx, score in zip(ids[0].tolist(), distances[0].tolist()):
        if local_idx < 0:
            continue
        global_idx = int(cat_map[local_idx]) if cat_map is not None else int(local_idx)
        if global_idx < 0 or global_idx >= len(meta):
            continue
        item = meta[global_idx]
        if not passes_filters(item, target_category, price_min, price_max, target_gender):
            continue

        item_category = normalize_category(item.get("category_hint"))
        item_gender = infer_gender(item)
        raw_score = float(score)
        text_score = float(np.dot(qv, title_embeddings[global_idx]))
        final_score = (IMAGE_WEIGHT * raw_score) + (TEXT_WEIGHT * text_score)
        row = {
            "idx": global_idx,
            "id": item.get("id"),
            "asin": item.get("asin"),
            "raw_score": raw_score,
            "text_score": text_score,
            "final_score": float(final_score),
            "title": item.get("title"),
            "url": item.get("product_url"),
            "image": item.get("image_url"),
            "price_usd": item.get("price_usd"),
            "rating": item.get("rating"),
            "num_ratings": item.get("num_ratings"),
            "category": item_category,
            "gender": item_gender,
        }
        all_results.append(row)

    all_results.sort(key=lambda x: x["final_score"], reverse=True)
    results = all_results[:k]

    return {
        "results": results,
        "count": len(results),
        "k": k,
        "requested_category": requested_category,
        "predicted_category": predicted_category,
        "category": target_category,
        "category_source": category_source,
        "gender": target_gender,
        "price_min": price_min,
        "price_max": price_max,
        "index_used": index_used,
    }


@app.post("/breakdown")
async def breakdown(
    file: UploadFile = File(...),
    max_items: int = 3,
    k_per_item: int = 6,
    category: str | None = None,
):
    if max_items <= 0:
        max_items = 3
    if k_per_item <= 0:
        k_per_item = 6

    img = parse_uploaded_image(file)
    q = embed_pil(img)
    qv = q[0]

    requested_category = normalize_category(category)
    scores_by_cat = predict_category_scores_from_query(q)
    if requested_category is not None:
        if requested_category in scores_by_cat:
            ordered_categories = [requested_category]
        else:
            ordered_categories = [requested_category]
    else:
        ordered_categories = [
            cat for cat, _ in sorted(scores_by_cat.items(), key=lambda x: x[1], reverse=True)
        ]

    if not ordered_categories:
        ordered_categories = sorted(category_artifacts.keys())

    selected_categories = ordered_categories[: max_items]
    total_score = sum(max(0.0, scores_by_cat.get(c, 0.0)) for c in selected_categories)

    groups = []
    for cat in selected_categories:
        search_index = global_index
        cat_map = None
        index_used = "global"
        if cat in category_artifacts:
            search_index = category_artifacts[cat]["index"]
            cat_map = category_artifacts[cat]["global_ids"]
            index_used = f"category:{cat}"

        search_k = min(max(k_per_item * 20, RETRIEVAL_K), search_index.ntotal)
        if search_k <= 0:
            continue

        distances, ids = search_index.search(q, search_k)
        rows = []
        for local_idx, score in zip(ids[0].tolist(), distances[0].tolist()):
            if local_idx < 0:
                continue
            global_idx = int(cat_map[local_idx]) if cat_map is not None else int(local_idx)
            if global_idx < 0 or global_idx >= len(meta):
                continue
            item = meta[global_idx]
            item_category = normalize_category(item.get("category_hint"))
            if item_category != cat:
                continue
            raw_score = float(score)
            text_score = float(np.dot(qv, title_embeddings[global_idx]))
            final_score = (IMAGE_WEIGHT * raw_score) + (TEXT_WEIGHT * text_score)
            rows.append(
                {
                    "idx": global_idx,
                    "id": item.get("id"),
                    "asin": item.get("asin"),
                    "raw_score": raw_score,
                    "text_score": text_score,
                    "final_score": float(final_score),
                    "title": item.get("title"),
                    "url": item.get("product_url"),
                    "image": item.get("image_url"),
                    "price_usd": item.get("price_usd"),
                    "rating": item.get("rating"),
                    "num_ratings": item.get("num_ratings"),
                    "category": item_category,
                    "gender": infer_gender(item),
                }
            )

        rows.sort(key=lambda x: x["final_score"], reverse=True)
        rows = rows[:k_per_item]
        if not rows:
            continue

        cat_score = max(0.0, scores_by_cat.get(cat, 0.0))
        confidence = (cat_score / total_score) if total_score > 0 else 0.0
        groups.append(
            {
                "category": cat,
                "confidence": confidence,
                "index_used": index_used,
                "count": len(rows),
                "results": rows,
            }
        )

    groups.sort(key=lambda g: g["confidence"], reverse=True)
    return {
        "requested_category": requested_category,
        "detected_categories": [
            {
                "category": g["category"],
                "confidence": g["confidence"],
                "count": g["count"],
            }
            for g in groups
        ],
        "groups": groups,
        "max_items": max_items,
        "k_per_item": k_per_item,
    }


@app.post("/describe")
async def describe(payload: dict[str, Any]):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    # Accept either:
    # 1) {"breakdown": {...}, "tone": "...", "max_tips": 3}
    # 2) raw breakdown JSON directly.
    if isinstance(payload.get("breakdown"), dict):
        breakdown_payload = payload.get("breakdown") or {}
        tone = str(payload.get("tone") or "editorial")
        max_tips = int(payload.get("max_tips") or 3)
    else:
        breakdown_payload = payload
        tone = "editorial"
        max_tips = 3

    if not isinstance(breakdown_payload, dict):
        raise HTTPException(status_code=400, detail="breakdown must be a JSON object")
    if "groups" not in breakdown_payload:
        raise HTTPException(
            status_code=400,
            detail="breakdown payload missing 'groups'. Pass output from /breakdown.",
        )

    max_tips = max(1, min(max_tips, 8))

    force_llm = bool(payload.get("force_llm")) if isinstance(payload, dict) else False

    try:
        return llm_description_from_breakdown(
            breakdown=breakdown_payload,
            tone=tone,
            max_tips=max_tips,
        )
    except Exception as exc:
        err = str(exc)
        logging.warning("LLM describe failed, using fallback: %s", exc)
        if force_llm:
            raise HTTPException(status_code=502, detail=f"LLM call failed: {err}")
        return fallback_description_from_breakdown(
            breakdown=breakdown_payload,
            tone=tone,
            max_tips=max_tips,
            llm_error=err,
        )
