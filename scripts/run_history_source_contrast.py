#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "experiments" / "results"
MU_DIR = ROOT / "experiments" / "multi_user_dataset"

USERS = [
    "user1",
    "user2",
    "user3",
    "user4",
    "user5",
]


def _read_history(user: str) -> str:
    p = MU_DIR / f"{user}_history.txt"
    if not p.exists():
        raise FileNotFoundError(f"missing history file: {p}")
    return p.read_text(encoding="utf-8")


def _write_histories(dst_dir: Path, user_to_text: Dict[str, str]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for u, txt in user_to_text.items():
        (dst_dir / f"{u}_history.txt").write_text(txt, encoding="utf-8")


def _build_wrong_dir_shift(dst_dir: Path, shift: int) -> Path:
    if shift <= 0 or shift >= len(USERS):
        raise ValueError(f"shift must be in [1,{len(USERS)-1}], got {shift}")
    src_by_user = {u: USERS[(i + shift) % len(USERS)] for i, u in enumerate(USERS)}
    user_to_text = {u: _read_history(src_by_user[u]) for u in USERS}
    _write_histories(dst_dir, user_to_text)
    return dst_dir


def _build_pooled_dir(dst_dir: Path) -> Path:
    pooled = "\n\n".join(_read_history(u).rstrip() for u in USERS).rstrip() + "\n"
    user_to_text = {u: pooled for u in USERS}
    _write_histories(dst_dir, user_to_text)
    return dst_dir


def _load_overall(csv_path: Path) -> Tuple[int, int, float]:
    total = 0
    success = 0
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = int(row.get("total", 0))
            s = int(row.get("success", 0))
            total += t
            success += s
    acc = (success / total) if total else 0.0
    return total, success, acc


def _load_by_level(csv_path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            lv = (row.get("level") or "").strip().lower()
            acc = float(row.get("overall_acc", row.get("acc", 0.0)))
            out[lv] = acc
    return out


def _run_one(mode_name: str, history_dir: Path, force_no_ask: bool, run_tag: str) -> Dict[str, object]:
    env = os.environ.copy()
    env["BASELINE8_RUN_TAG"] = run_tag
    env["BASELINE8_ONLY"] = "FULL"
    env["BASELINE8_FORCE_NO_ASK"] = "1" if force_no_ask else "0"
    env["BASELINE8_DISABLE_TIMEOUT"] = env.get("BASELINE8_DISABLE_TIMEOUT", "1")
    env["BASELINE8_HISTORY_DIR"] = str(history_dir)
    env["ALLOW_STALE_CACHE"] = "1"

    cmd = [sys.executable, str(SCRIPTS / "run_baseline8_matrix.py")]
    print(f"[RUN] mode={mode_name} ask={'noask' if force_no_ask else 'ask'} tag={run_tag}")
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"run failed for {mode_name} rc={proc.returncode}")

    run_root = RESULTS / run_tag
    csv_main = run_root / "baseline8_results.csv"
    csv_lv = run_root / "baseline8_results_by_level.csv"
    total, success, acc = _load_overall(csv_main)
    by_level = _load_by_level(csv_lv)
    return {
        "history_source": mode_name,
        "ask_mode": "noask" if force_no_ask else "ask",
        "run_tag": run_tag,
        "total": total,
        "success": success,
        "overall_acc": acc,
        "l1": by_level.get("l1", 0.0),
        "l2": by_level.get("l2", 0.0),
        "l3": by_level.get("l3", 0.0),
        "l4": by_level.get("l4", 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run same-user vs wrong-user vs pooled history contrast (FULL only).")
    ap.add_argument("--include-ask", action="store_true", help="also run ask mode")
    ap.add_argument(
        "--include-pooled",
        action="store_true",
        help="also run pooled (all-user concatenated history; not size-matched)",
    )
    ap.add_argument("--run-tag-prefix", default="history_source_contrast", help="result run tag prefix")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_root = ROOT / "experiments" / "tmp" / f"{args.run_tag_prefix}_{ts}"
    pooled_dir = _build_pooled_dir(tmp_root / "pooled")

    rows: List[Dict[str, object]] = []
    for force_no_ask in [True, False] if args.include_ask else [True]:
        tag = f"{args.run_tag_prefix}_same_user_{'noask' if force_no_ask else 'ask'}_{ts}"
        rows.append(_run_one("same_user", MU_DIR, force_no_ask, tag))

        wrong_shift_rows: List[Dict[str, object]] = []
        for shift in range(1, len(USERS)):
            wrong_dir = _build_wrong_dir_shift(tmp_root / f"wrong_shift{shift}", shift)
            tag = f"{args.run_tag_prefix}_wrong_shift{shift}_{'noask' if force_no_ask else 'ask'}_{ts}"
            r = _run_one(f"wrong_user_shift{shift}", wrong_dir, force_no_ask, tag)
            wrong_shift_rows.append(r)
            rows.append(r)

        avg = {
            "history_source": "wrong_user_avg4",
            "ask_mode": "noask" if force_no_ask else "ask",
            "run_tag": "AVG(wrong_shift1..4)",
            "total": int(sum(int(r["total"]) for r in wrong_shift_rows) / len(wrong_shift_rows)),
            "success": int(sum(int(r["success"]) for r in wrong_shift_rows) / len(wrong_shift_rows)),
            "overall_acc": sum(float(r["overall_acc"]) for r in wrong_shift_rows) / len(wrong_shift_rows),
            "l1": sum(float(r["l1"]) for r in wrong_shift_rows) / len(wrong_shift_rows),
            "l2": sum(float(r["l2"]) for r in wrong_shift_rows) / len(wrong_shift_rows),
            "l3": sum(float(r["l3"]) for r in wrong_shift_rows) / len(wrong_shift_rows),
            "l4": sum(float(r["l4"]) for r in wrong_shift_rows) / len(wrong_shift_rows),
        }
        rows.append(avg)

        if args.include_pooled:
            tag = f"{args.run_tag_prefix}_pooled_{'noask' if force_no_ask else 'ask'}_{ts}"
            rows.append(_run_one("pooled", pooled_dir, force_no_ask, tag))

    out_csv = RESULTS / f"{args.run_tag_prefix}_summary_{ts}.csv"
    out_txt = RESULTS / f"{args.run_tag_prefix}_summary_{ts}.txt"
    fieldnames = [
        "history_source",
        "ask_mode",
        "overall_acc",
        "l1",
        "l2",
        "l3",
        "l4",
        "success",
        "total",
        "run_tag",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with out_txt.open("w", encoding="utf-8") as f:
        f.write("History source contrast (FULL only)\n")
        f.write(f"tmp_histories: {tmp_root}\n\n")
        for r in rows:
            f.write(
                f"{r['history_source']:10s} {r['ask_mode']:5s} "
                f"overall={float(r['overall_acc']):.6f} "
                f"L1={float(r['l1']):.6f} L2={float(r['l2']):.6f} "
                f"L3={float(r['l3']):.6f} L4={float(r['l4']):.6f} "
                f"({int(r['success'])}/{int(r['total'])}) "
                f"run={r['run_tag']}\n"
            )

    print(f"[DONE] summary_csv={out_csv}")
    print(f"[DONE] summary_txt={out_txt}")
    print(f"[DONE] tmp_histories={tmp_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
