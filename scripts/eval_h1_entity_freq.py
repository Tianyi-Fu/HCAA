#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
MU_DIR = ROOT / "experiments" / "multi_user_dataset"
RESULTS_ROOT = ROOT / "experiments" / "results"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True, help="result folder name under experiments/results")
    ap.add_argument("--entity-id", default="B4", help="baseline id for object-level history")
    ap.add_argument("--full-id", default="FULL", help="baseline id for full concept method")
    ap.add_argument("--out", default="", help="optional output csv path")
    return ap.parse_args()


def _target_of_correct(correct_asp: str) -> str:
    s = (correct_asp or "").strip()
    if "(" not in s or ")" not in s:
        return ""
    pred = s.split("(", 1)[0].strip()
    args = s.split("(", 1)[1].rsplit(")", 1)[0].split(",")
    args = [a.strip() for a in args if a.strip()]
    if pred in {"inside", "on", "has"}:
        return args[1] if len(args) >= 2 else ""
    if pred == "heated":
        return args[0] if args else ""
    return args[1] if len(args) >= 2 else (args[0] if args else "")


def _hist_counts_for_user(user_id: str) -> Dict[str, int]:
    p = MU_DIR / f"{user_id}_history.txt"
    out: Dict[str, int] = {}
    if not p.exists():
        return out
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or "|" not in s:
            continue
        asp = s.split("|", 1)[0].strip()
        t = _target_of_correct(asp)
        if t:
            out[t] = out.get(t, 0) + 1
    return out


def _bucket(cnt: int) -> str:
    if cnt <= 0:
        return "Zero-shot"
    if cnt <= 2:
        return "Sparse(1-2)"
    return "Frequent(>=3)"


def _load_method_cases(run_dir: Path, bid: str) -> Dict[Tuple[str, str, int, int], Dict]:
    out: Dict[Tuple[str, str, int, int], Dict] = {}
    for d in sorted(run_dir.glob(f"*__{bid}")):
        tj = d / "test_results.json"
        if not tj.exists():
            continue
        arr = json.loads(tj.read_text(encoding="utf-8"))
        left = d.name.split("__", 1)[0]
        level = ""
        parts = left.split("_")
        if parts and parts[-1] in {"l1", "l2", "l3", "l4"}:
            level = parts[-1]
        for r in arr:
            user_id = d.name.split("__", 1)[0]
            if user_id.endswith("_l1") or user_id.endswith("_l2") or user_id.endswith("_l3") or user_id.endswith("_l4"):
                user_id = "_".join(user_id.split("_")[:-1])
            key = (user_id, level, int(r.get("group", 0)), int(r.get("line", 0)))
            out[key] = r
    return out


def main() -> int:
    args = parse_args()
    run_dir = RESULTS_ROOT / args.run_tag
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    entity_cases = _load_method_cases(run_dir, args.entity_id)
    full_cases = _load_method_cases(run_dir, args.full_id)
    keys = sorted(set(entity_cases.keys()) & set(full_cases.keys()))
    if not keys:
        raise SystemExit("no overlapping cases between methods")

    user_hist: Dict[str, Dict[str, int]] = {}
    rows: List[Dict[str, object]] = []

    agg = {
        "Zero-shot": {"n": 0, "entity_ok": 0, "full_ok": 0},
        "Sparse(1-2)": {"n": 0, "entity_ok": 0, "full_ok": 0},
        "Frequent(>=3)": {"n": 0, "entity_ok": 0, "full_ok": 0},
    }

    for key in keys:
        user_id, level, g, line = key
        if user_id not in user_hist:
            user_hist[user_id] = _hist_counts_for_user(user_id)
        e = entity_cases[key]
        f = full_cases[key]
        target = _target_of_correct(str(f.get("correct", "")))
        cnt = int(user_hist[user_id].get(target, 0))
        b = _bucket(cnt)
        eok = bool(e.get("success", False))
        fok = bool(f.get("success", False))
        agg[b]["n"] += 1
        agg[b]["entity_ok"] += int(eok)
        agg[b]["full_ok"] += int(fok)
        rows.append(
            {
                "user_id": user_id,
                "level": level,
                "group": g,
                "line": line,
                "target": target,
                "hist_count": cnt,
                "bucket": b,
                "entity_success": int(eok),
                "full_success": int(fok),
            }
        )

    out_csv = Path(args.out) if args.out else (run_dir / "h1_entity_freq_cases.csv")
    out_summary = run_dir / "h1_entity_freq_summary.csv"
    out_md = run_dir / "h1_entity_freq_summary.md"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "user_id",
                "level",
                "group",
                "line",
                "target",
                "hist_count",
                "bucket",
                "entity_success",
                "full_success",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    srows: List[Dict[str, object]] = []
    for b in ("Zero-shot", "Sparse(1-2)", "Frequent(>=3)"):
        n = int(agg[b]["n"])
        eok = int(agg[b]["entity_ok"])
        fok = int(agg[b]["full_ok"])
        eacc = (eok / n) if n else 0.0
        facc = (fok / n) if n else 0.0
        srows.append(
            {
                "bucket": b,
                "n": n,
                "entity_acc": round(eacc, 6),
                "full_acc": round(facc, 6),
                "delta_full_minus_entity": round(facc - eacc, 6),
            }
        )

    with out_summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["bucket", "n", "entity_acc", "full_acc", "delta_full_minus_entity"],
        )
        w.writeheader()
        w.writerows(srows)

    lines = ["# H1 Entity-Frequency Buckets", ""]
    lines.append("| Bucket | N | Entity-name Acc | Full(concept) Acc | Delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in srows:
        lines.append(
            f"| {r['bucket']} | {r['n']} | {100*float(r['entity_acc']):.2f}% | "
            f"{100*float(r['full_acc']):.2f}% | {100*float(r['delta_full_minus_entity']):+.2f}% |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[H1] cases: {out_csv}")
    print(f"[H1] summary_csv: {out_summary}")
    print(f"[H1] summary_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
