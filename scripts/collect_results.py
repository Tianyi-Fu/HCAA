#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "experiments" / "results"
MU_DIR = ROOT / "experiments" / "multi_user_dataset"
OUT_ROOT = ROOT / "result"

LEVELS = ["l1", "l2", "l3", "l4"]

PAPER_LABEL = {
    "B0": "B0_random",
    "B1": "B1_llm",
    "B1fhca": "B2_llm_asp",
    "B2": "B2_llm_asp",
    "B3": "B3_sem_sal",
    "B4": "B4_object",
    "L0F": "B5_l0attr",
    "B5": "B6_concept",
    "B6": "B7_no_asp",
    "FULL": "Proposed",
}

TABLE_ORDER = [
    "B0_random",
    "B1_llm",
    "B2_llm_asp",
    "B3_sem_sal",
    "B4_object",
    "B5_l0attr",
    "B6_concept",
    "B7_no_asp",
    "Proposed",
]

USERS = [
    "user1",
    "user2",
    "user3",
    "user4",
    "user5",
]


def _asked(rec: dict) -> bool:
    pred = str(rec.get("predicted") or "")
    if "__OBJ__" in pred or "__FURN__" in pred:
        return True
    return str(rec.get("decision") or "").upper() == "ASK_HUMAN"


def _pct(num: int, den: int) -> float | None:
    return round(100.0 * num / den, 1) if den else None


def _collect(run_dir: Path):
    cells = defaultdict(list)
    direct = run_dir / "test_results.json"
    if direct.is_file():
        name = run_dir.name
        level = name.rsplit("_", 1)[-1]
        user = name[: -(len(level) + 1)]
        if level in LEVELS:
            recs = json.loads(direct.read_text(encoding="utf-8"))
            for r in recs:
                r.setdefault("user", user)
                r.setdefault("level", level)
            cells[("Proposed", level)].extend(recs)
        return cells

    for cell in sorted(run_dir.iterdir()):
        tr = cell / "test_results.json"
        if not cell.is_dir() or not tr.is_file() or "__" not in cell.name:
            continue
        target, code_id = cell.name.rsplit("__", 1)
        label = PAPER_LABEL.get(code_id)
        if label is None:
            continue
        level = target.rsplit("_", 1)[-1]
        user = target[: -(len(level) + 1)]
        if level not in LEVELS:
            continue
        recs = json.loads(tr.read_text(encoding="utf-8"))
        for r in recs:
            r.setdefault("user", user)
            r.setdefault("level", level)
        cells[(label, level)].extend(recs)
    return cells


def _main_table(cells) -> dict:
    modes = {"noask_overall": {}, "ask_overall": {},
             "ask_answered": {}, "ask_clarification": {}}
    for label in TABLE_ORDER:
        rows = {m: [] for m in modes}
        for level in LEVELS:
            recs = cells.get((label, level), [])
            n = len(recs)
            correct = sum(1 for r in recs if r.get("success"))
            asked = sum(1 for r in recs if _asked(r))
            answered_correct = sum(
                1 for r in recs if r.get("success") and not _asked(r))
            rows["noask_overall"].append(_pct(correct, n))
            rows["ask_overall"].append(_pct(answered_correct, n))
            rows["ask_answered"].append(_pct(answered_correct, n - asked))
            rows["ask_clarification"].append(_pct(asked, n))
        for m, vals in rows.items():
            got = [v for v in vals if v is not None]
            if not got:
                continue
            if m == "ask_clarification":
                modes[m][label] = vals
            else:
                modes[m][label] = vals + [round(sum(got) / len(got), 1)]
    return modes


def _by_user(cells) -> dict:
    out = defaultdict(dict)
    for (label, level), recs in cells.items():
        per = defaultdict(lambda: [0, 0])
        for r in recs:
            u = r.get("user", "unknown")
            per[u][1] += 1
            per[u][0] += 1 if r.get("success") else 0
        for u, (ok, n) in per.items():
            out[label].setdefault(u, {})[level] = _pct(ok, n)
    for users in out.values():
        for lv in users.values():
            got = [lv[k] for k in LEVELS if lv.get(k) is not None]
            lv["avg"] = round(sum(got) / len(got), 1) if got else None
    return dict(out)


def _export_records(cells) -> int:
    dest = OUT_ROOT / "run_records"
    written = 0
    for label in TABLE_ORDER:
        recs = []
        for level in LEVELS:
            recs.extend(cells.get((label, level), []))
        if not recs:
            continue
        d = dest / label
        d.mkdir(parents=True, exist_ok=True)
        with (d / "test_results.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        written += 1
    return written


def _export_histories() -> int:
    asp_dir = OUT_ROOT / "histories" / "asp"
    nl_dir = OUT_ROOT / "histories" / "natural_language"
    asp_dir.mkdir(parents=True, exist_ok=True)
    nl_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for user in USERS:
        src = MU_DIR / f"{user}_history.txt"
        if not src.is_file():
            continue
        asp_lines, nl_lines = [], []
        for raw in src.read_text(encoding="utf-8").splitlines():
            if "|" not in raw:
                asp_lines.append(raw)
                nl_lines.append(raw)
                continue
            left, right = raw.split("|", 1)
            asp_lines.append(f"{left.strip()} |")
            nl_lines.append(right.strip())
        (asp_dir / f"{user}.txt").write_text("\n".join(asp_lines) + "\n", encoding="utf-8")
        (nl_dir / f"{user}.txt").write_text("\n".join(nl_lines) + "\n", encoding="utf-8")
        count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_tag", nargs="?")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if args.run_tag:
        run_dir = RESULTS_ROOT / args.run_tag
    else:
        dirs = [d for d in RESULTS_ROOT.iterdir() if d.is_dir()] if RESULTS_ROOT.is_dir() else []
        if not dirs:
            print(f"[ERROR] no run directories under {RESULTS_ROOT}")
            return 2
        run_dir = max(dirs, key=lambda d: d.stat().st_mtime)
        print(f"[INFO] latest run: {run_dir.name}")

    if not run_dir.is_dir():
        print(f"[ERROR] not found: {run_dir}")
        return 2

    if args.clean and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    cells = _collect(run_dir)
    if not cells:
        print(f"[ERROR] no scored cells in {run_dir}")
        return 3

    tables = OUT_ROOT / "results"
    tables.mkdir(parents=True, exist_ok=True)
    main_table = _main_table(cells)
    by_user = _by_user(cells)
    (tables / "main_table.json").write_text(
        json.dumps(main_table, indent=2) + "\n", encoding="utf-8")
    (tables / "by_user.json").write_text(
        json.dumps(by_user, indent=2) + "\n", encoding="utf-8")
    (tables / "all_results.json").write_text(
        json.dumps({"main_table": main_table, "by_user": by_user}, indent=2) + "\n",
        encoding="utf-8")

    lines = ["| Method | A1 | A2 | A3 | A4 | Avg |", "|---|---|---|---|---|---|"]
    for label in TABLE_ORDER:
        v = main_table["noask_overall"].get(label)
        if v:
            lines.append(f"| {label} | " +
                         " | ".join("-" if x is None else f"{x:.1f}" for x in v) + " |")
    (tables / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_rec = _export_records(cells)
    n_hist = _export_histories()
    print(f"[OK] tables      -> result/results")
    print(f"[OK] run_records -> result/run_records ({n_rec} baselines)")
    print(f"[OK] histories   -> result/histories ({n_hist} users)")
    print()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
