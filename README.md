# Outfit Finder

Find shoppable fashion items that look like something you saw. Upload a photo (or drag-select an outfit on any web page with the Chrome extension) and Outfit Finder returns visually similar products from an Amazon catalog.

## How it works

Product images are embedded with OpenCLIP (`ViT-B-32`, `laion2b_s34b_b79k`) and stored in FAISS indexes, one global and one per category (dresses, shoes, bags). At query time the API embeds your image, predicts its category, and ranks results by a blend of image similarity (80%) and title similarity (20%), with optional price and gender filters. The `/breakdown` endpoint splits a look into its likely pieces and returns matches for each one.

## Project layout

| Path | Purpose |
| --- | --- |
| `api.py` | FastAPI backend (`/search`, `/breakdown`, `/upload-capture`, `/capture/{id}`, `/describe`, `/health`) |
| `build_index.py` | Embeds `catalog_amazon.json` and writes indexes to `index_artifacts/` |
| `search_index.py` | CLI for quick nearest-neighbor checks against the index |
| `eval.py` | Scores `/search` against labeled queries |
| `frontend/` | Static web UI (deployed via Netlify) |
| `extension/` | Chrome extension for capturing outfits on any page |
| `queries/` | Sample query images |

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt tqdm

python build_index.py          # embed the catalog (downloads product images)
uvicorn api:app --reload       # API at http://127.0.0.1:8000
```

Then open `frontend/index.html` in a browser, or load the extension (see [`extension/README.md`](extension/README.md)).

Search from the command line:

```bash
curl -F "file=@queries/test.jpg" "http://127.0.0.1:8000/search?k=5&price_max=60"
python search_index.py --query-url <image-url> --k 5
```

Set `OPENAI_API_KEY` to enable LLM-written outfit descriptions in `/describe`; without it, a built-in fallback description is used.

## Evaluation

Copy `eval_labels.sample.json` to `eval_labels.json`, add your labels, start the API, and run:

```bash
python eval.py --labels eval_labels.json --k 5
```

Results are written to `eval_report.json`.