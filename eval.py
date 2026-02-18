import argparse
import json
from pathlib import Path
from typing import Any

import requests


def load_labels(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError("Label file must be a JSON list")
    return data


def as_set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(v) for v in value if str(v).strip()}
    return set()


def evaluate_query(
    base_url: str,
    label: dict[str, Any],
    default_k: int,
    cwd: Path,
) -> dict[str, Any]:
    rel_image = label.get("query_image")
    if not rel_image:
        raise RuntimeError("Missing query_image in label")
    img_path = (cwd / str(rel_image)).resolve()
    if not img_path.exists():
        raise RuntimeError(f"Image not found: {img_path}")

    k = int(label.get("k") or default_k)
    params = {"k": k}
    for key in ("category", "gender", "price_min", "price_max"):
        if label.get(key) is not None:
            params[key] = label[key]

    with img_path.open("rb") as f:
        resp = requests.post(
            f"{base_url.rstrip('/')}/search",
            params=params,
            files={"file": (img_path.name, f, "image/jpeg")},
            timeout=90,
        )
    resp.raise_for_status()
    payload = resp.json()
    results = payload.get("results") or []

    expected_asins = as_set(label.get("expected_asins"))
    expected_ids = as_set(label.get("expected_ids"))
    expected_idxs = {int(v) for v in (label.get("expected_idxs") or [])}

    flags = []
    for row in results:
        is_rel = False
        if expected_asins and str(row.get("asin")) in expected_asins:
            is_rel = True
        if expected_ids and str(row.get("id")) in expected_ids:
            is_rel = True
        if expected_idxs and int(row.get("idx", -1)) in expected_idxs:
            is_rel = True
        flags.append(1 if is_rel else 0)

    p1 = float(flags[0]) if flags else 0.0
    top5 = flags[:5]
    p5 = (sum(top5) / 5.0) if top5 else 0.0

    expected_total = max(len(expected_asins), len(expected_ids), len(expected_idxs))
    if expected_total > 0:
        recall5 = min(1.0, sum(top5) / float(expected_total))
    else:
        recall5 = None

    return {
        "query_image": str(rel_image),
        "p1": p1,
        "p5": p5,
        "recall5": recall5,
        "num_results": len(results),
        "index_used": payload.get("index_used"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate /search using labeled queries.")
    parser.add_argument("--labels", default="eval_labels.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="eval_report.json")
    args = parser.parse_args()

    cwd = Path.cwd()
    labels = load_labels(Path(args.labels))
    rows = []
    for i, label in enumerate(labels, start=1):
        try:
            row = evaluate_query(args.base_url, label, args.k, cwd)
            rows.append(row)
            print(
                f"[{i}/{len(labels)}] {row['query_image']} "
                f"p1={row['p1']:.2f} p5={row['p5']:.2f} "
                f"recall5={row['recall5'] if row['recall5'] is not None else 'n/a'}"
            )
        except Exception as exc:
            print(f"[{i}/{len(labels)}] failed: {exc}")

    if not rows:
        raise RuntimeError("No successful evaluations")

    macro_p1 = sum(r["p1"] for r in rows) / len(rows)
    macro_p5 = sum(r["p5"] for r in rows) / len(rows)
    recall_rows = [r["recall5"] for r in rows if r["recall5"] is not None]
    macro_recall5 = (sum(recall_rows) / len(recall_rows)) if recall_rows else None

    report = {
        "num_queries": len(rows),
        "macro_p1": macro_p1,
        "macro_p5": macro_p5,
        "macro_recall5": macro_recall5,
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\nSummary")
    print(f"queries: {report['num_queries']}")
    print(f"macro_p1: {report['macro_p1']:.4f}")
    print(f"macro_p5: {report['macro_p5']:.4f}")
    if report["macro_recall5"] is not None:
        print(f"macro_recall5: {report['macro_recall5']:.4f}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
