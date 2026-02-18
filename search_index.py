import json
import faiss

INDEX_PATH = "index_artifacts/image.index"
META_PATH = "index_artifacts/meta.json"

def main():
    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)

    # Load metadata (maps FAISS row -> product info)
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    # meta can be either a list or a dict depending on how you saved it
    if isinstance(meta, list):
        n_meta = len(meta)
    elif isinstance(meta, dict) and "items" in meta and isinstance(meta["items"], list):
        n_meta = len(meta["items"])
    else:
        n_meta = None

    print("FAISS index size:", index.ntotal)
    print("Meta items:", n_meta)
    if n_meta is not None and n_meta != index.ntotal:
        print("⚠️ Warning: meta count != index size (this will break lookup later).")

if __name__ == "__main__":
    main()
