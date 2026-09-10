#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
import csv
import json

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "experiments" / "results"
HIST_DIR = ROOT / "experiments" / "multi_user_dataset"


MODE_MAP = {
    "pure_llm": "B1_llm_only",
    "llm_only": "B1_llm_only",
    "three_factor_llm": "B2_llm_replace_history",
    "three_factor_history": "Proposed_three_factor_history",
}
MODE_ORDER = ["pure_llm", "three_factor_llm", "three_factor_history"]
ASK_MODES = ["ask", "noask"]
LEVELS = ["a1", "a2", "a3", "a4"]


def _latest_summary_dir() -> Path:
    fixed = RESULTS_DIR / "_summary_current"
    if fixed.exists():
        return fixed
    summary_dirs = [p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name.startswith("_summary_")]
    if not summary_dirs:
        raise FileNotFoundError("No _summary_ directory found.")
    return max(summary_dirs, key=lambda p: p.stat().st_mtime)


def _parse_asp_cmd(cmd: str):
    m = re.match(r"^\s*([a-zA-Z_][\w]*)\(([^)]*)\)\s*$", cmd.strip())
    if not m:
        return None, []
    pred = m.group(1).strip()
    args = [a.strip() for a in m.group(2).split(",") if a.strip()]
    return pred, args


def _majority_choice_from_histories():
    majority = {}
    counts = defaultdict(Counter)
    for path in sorted(HIST_DIR.glob("*_history.txt")):
        if "old" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("[") or line.startswith("]"):
                continue
            left = line.split("|", 1)[0].strip()
            pred, args = _parse_asp_cmd(left)
            if not pred or not args:
                continue
            furniture = args[0]
            obj = args[1] if len(args) >= 2 else args[0]
            counts[(pred, furniture)][obj] += 1
    for key, ctr in counts.items():
        majority[key] = ctr.most_common(1)[0][0]
    return majority


def _iter_test_results():
    for res_dir in RESULTS_DIR.iterdir():
        if not res_dir.is_dir():
            continue
        if res_dir.name.startswith("_summary_"):
            continue
        tr = res_dir / "test_results.json"
        if tr.exists():
            yield res_dir.name, res_dir, tr


def _parse_run_name(run_name: str):
    parts = run_name.split("__")
    ask_mode = "ask"
    mode = None
    base = parts[0]
    if len(parts) >= 2:
        if parts[-1] in ("ask", "noask"):
            ask_mode = parts[-1]
            if len(parts) >= 3:
                mode = "__".join(parts[1:-1])
        else:
            mode = "__".join(parts[1:])
    target = f"{base}.txt"
    return target, mode, ask_mode


def _parse_user_level(target_name: str):
    m_user = re.match(r"^(user\d+_[a-zA-Z]+)", target_name)
    user_id = m_user.group(1) if m_user else ""
    m_lvl = re.search(r"_a(\d+)\.txt$", target_name)
    level = f"a{m_lvl.group(1)}" if m_lvl else ""
    return user_id, level


def _weighted_metrics(df: pd.DataFrame):
    total = df["total"].sum()
    success = df["success_total"].sum()
    answered_total = df["answered_total"].sum()
    answered_success = df["answered_success"].sum()
    ask_total = df["ask_total"].sum()
    hard_total = df["hard_error_total"].sum()
    return pd.Series(
        {
            "overall_acc": success / total if total else np.nan,
            "answer_acc": answered_success / answered_total if answered_total else np.nan,
            "abstain_rate": ask_total / total if total else np.nan,
            "hard_error_rate": hard_total / total if total else np.nan,
            "total": total,
        }
    )


def _md_table(df: pd.DataFrame, float_cols=None):
    if df is None or df.empty:
        return "N/A\n"
    out = df.copy()
    if float_cols is None:
        float_cols = [c for c in out.columns if out[c].dtype.kind in "fc"]
    for c in float_cols:
        out[c] = out[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    return out.to_markdown(index=False)


def main():
    summary_dir = _latest_summary_dir()
    agg_path = summary_dir / "summary_agg.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"missing {agg_path}")

    agg = pd.read_csv(agg_path)
    agg = agg[agg["mode"].isin(MODE_MAP.keys())].copy()
    if "derived_noask_overall_acc" in agg.columns and not (agg["ask_mode"] == "noask").any():
        derived = agg.copy()
        derived["ask_mode"] = "noask"
        derived["overall_acc"] = derived["derived_noask_overall_acc"]
        derived["success_total"] = derived["derived_noask_success"]
        derived["answered_total"] = derived["derived_noask_total"]
        derived["answered_success"] = derived["derived_noask_success"]
        derived["answer_acc"] = derived["derived_noask_overall_acc"]
        derived["ask_total"] = 0
        derived["abstain_rate"] = 0.0
        derived["hard_error_total"] = derived["derived_noask_hard_error_total"]
        derived["hard_error_rate"] = derived["derived_noask_hard_error_rate"]
        derived["total"] = derived["derived_noask_total"]
        agg = pd.concat([agg, derived], ignore_index=True)
    agg["level_mapped"] = agg["level"].replace({"a5": "a4"})
    agg = agg[agg["level_mapped"].isin(LEVELS)]
    agg["mode_label"] = agg["mode"].map(MODE_MAP)

    main_rows = []
    for ask_mode in ASK_MODES:
        sub = agg[agg["ask_mode"] == ask_mode]
        for mode in MODE_ORDER:
            g = sub[sub["mode"] == mode]
            if g.empty:
                continue
            s = _weighted_metrics(g)
            main_rows.append(
                {"ask_mode": ask_mode, "baseline": MODE_MAP[mode], **s.to_dict()}
            )
    main_df = pd.DataFrame(main_rows)

    level_rows = []
    for ask_mode in ASK_MODES:
        sub = agg[agg["ask_mode"] == ask_mode]
        for lvl in LEVELS:
            sub2 = sub[sub["level_mapped"] == lvl]
            for mode in MODE_ORDER:
                g = sub2[sub2["mode"] == mode]
                if g.empty:
                    continue
                s = _weighted_metrics(g)
                level_rows.append(
                    {
                        "ask_mode": ask_mode,
                        "level": lvl.upper(),
                        "baseline": MODE_MAP[mode],
                        **s.to_dict(),
                    }
                )
    level_df = pd.DataFrame(level_rows)

    trend_rows = []
    for ask_mode in ASK_MODES:
        sub = level_df[level_df["ask_mode"] == ask_mode]
        for baseline in MODE_MAP.values():
            row = {"ask_mode": ask_mode, "baseline": baseline}
            for lvl in LEVELS:
                v = sub[(sub["baseline"] == baseline) & (sub["level"] == lvl.upper())][
                    "overall_acc"
                ]
                row[lvl.upper()] = float(v.iloc[0]) if len(v) else np.nan
            row["L1->L4_drop"] = (
                row.get("L1", np.nan) - row.get("L4", np.nan)
                if pd.notna(row.get("L1")) and pd.notna(row.get("L4"))
                else np.nan
            )
            trend_rows.append(row)
    trend_df = pd.DataFrame(trend_rows)

    gap_rows = []
    for ask_mode in ASK_MODES:
        sub = level_df[level_df["ask_mode"] == ask_mode]
        for lvl in LEVELS:
            def get_val(baseline):
                v = sub[(sub["baseline"] == baseline) & (sub["level"] == lvl.upper())]["overall_acc"]
                return float(v.iloc[0]) if len(v) else np.nan
            b1 = get_val("B1_llm_only")
            b2 = get_val("B2_llm_replace_history")
            prop = get_val("Proposed_three_factor_history")
            gap_rows.append(
                {
                    "ask_mode": ask_mode,
                    "level": lvl.upper(),
                    "Proposed_minus_B2": (prop - b2) if pd.notna(prop) and pd.notna(b2) else np.nan,
                    "Proposed_minus_B1": (prop - b1) if pd.notna(prop) and pd.notna(b1) else np.nan,
                    "B2_minus_B1": (b2 - b1) if pd.notna(b2) and pd.notna(b1) else np.nan,
                }
            )
    gap_df = pd.DataFrame(gap_rows)

    user_rows = []
    for ask_mode in ASK_MODES:
        sub = agg[agg["ask_mode"] == ask_mode]
        for user in sorted(sub["user_id"].unique()):
            sub_u = sub[sub["user_id"] == user]
            for mode in MODE_ORDER:
                g = sub_u[sub_u["mode"] == mode]
                if g.empty:
                    continue
                s = _weighted_metrics(g)
                user_rows.append(
                    {
                        "ask_mode": ask_mode,
                        "user": user,
                        "baseline": MODE_MAP[mode],
                        "overall_acc": s["overall_acc"],
                        "total": s["total"],
                    }
                )
    user_df = pd.DataFrame(user_rows)
    sigma_df = (
        user_df.groupby(["ask_mode", "baseline"])["overall_acc"]
        .agg(["mean", "std"])
        .reset_index()
        if not user_df.empty
        else pd.DataFrame()
    )

    comp = pd.read_csv(agg_path)
    comp = comp[comp["mode"] == "three_factor_compare"]
    compare_df = None
    if not comp.empty:
        total_failed = comp["compare_total_stage1_failed"].sum()
        hist_ans = comp["compare_history_answered"].sum()
        hist_succ = comp["compare_history_success"].sum()
        llm_ans = comp["compare_llm_answered"].sum()
        llm_succ = comp["compare_llm_success"].sum()
        compare_df = pd.DataFrame(
            [
                {
                    "Stage1_failed_total": total_failed,
                    "History_answered": hist_ans,
                    "History_success": hist_succ,
                    "History_acc": (hist_succ / hist_ans) if hist_ans else np.nan,
                    "LLM_answered": llm_ans,
                    "LLM_success": llm_succ,
                    "LLM_acc": (llm_succ / llm_ans) if llm_ans else np.nan,
                }
            ]
        )

    majority = _majority_choice_from_histories()
    pers_rows = []
    user_acc_rows = []
    counts = defaultdict(lambda: {"total": 0, "success": 0})
    user_counts = defaultdict(lambda: {"total": 0, "success": 0})

    for run_name, res_dir, tr_path in _iter_test_results():
        target_name, mode, ask_mode = _parse_run_name(run_name)
        if mode not in MODE_MAP.keys():
            continue
        user_id, level = _parse_user_level(target_name)
        level_mapped = level.replace("a5", "a4") if level else ""
        if level_mapped and level_mapped not in LEVELS:
            continue
        rows = json.loads(tr_path.read_text(encoding="utf-8"))
        for r in rows:
            correct = r.get("correct") or ""
            pred = r.get("predicted") or ""
            success = bool(r.get("success", False))
            pred_name, args = _parse_asp_cmd(correct)
            if not pred_name or not args:
                continue
            furniture = args[0]
            obj = args[1] if len(args) >= 2 else args[0]
            key = (pred_name, furniture)
            maj = majority.get(key)
            if maj is None:
                continue
            personalized = (obj != maj)
            bucket = "personalized" if personalized else "non_personalized"
            k = (MODE_MAP[mode], ask_mode, bucket)
            counts[k]["total"] += 1
            if success:
                counts[k]["success"] += 1

            ukey = (MODE_MAP[mode], ask_mode, user_id)
            user_counts[ukey]["total"] += 1
            if success:
                user_counts[ukey]["success"] += 1

    for (baseline, ask_mode, bucket), v in counts.items():
        total = v["total"]
        succ = v["success"]
        acc = succ / total if total else np.nan
        pers_rows.append(
            {
                "baseline": baseline,
                "ask_mode": ask_mode,
                "subset": bucket,
                "total": total,
                "success": succ,
                "accuracy": acc,
            }
        )

    for (baseline, ask_mode, user), v in user_counts.items():
        total = v["total"]
        succ = v["success"]
        acc = succ / total if total else np.nan
        user_acc_rows.append(
            {
                "baseline": baseline,
                "ask_mode": ask_mode,
                "user": user,
                "total": total,
                "success": succ,
                "overall_acc": acc,
            }
        )

    pers_df = pd.DataFrame(pers_rows)
    user_acc_df = pd.DataFrame(user_acc_rows)

    pers_df.to_csv(summary_dir / "personalized_scenarios.csv", index=False)
    user_acc_df.to_csv(summary_dir / "per_user_accuracy.csv", index=False)

    lines = []
    lines.append(f"# Consolidated Tables (L1/L2/L3/L4) - {summary_dir.name}\n")
    lines.append("Note: L5 has been re-labeled as L4 per your instruction.\n")

    lines.append("## 1) Main Results (all levels combined)\n")
    lines.append("**Metrics:** overall_acc, answer_acc, abstain_rate, hard_error_rate (weighted by total).\n")
    for ask_mode in ASK_MODES:
        lines.append(f"### ask_mode = {ask_mode}\n")
        sub = main_df[main_df["ask_mode"] == ask_mode][
            ["baseline", "overall_acc", "answer_acc", "abstain_rate", "hard_error_rate", "total"]
        ]
        lines.append(_md_table(sub))
        lines.append("")

    lines.append("## 2) By Ambiguity Level (L1-L4, all users)\n")
    for ask_mode in ASK_MODES:
        lines.append(f"### ask_mode = {ask_mode} (overall_acc)\n")
        sub = level_df[level_df["ask_mode"] == ask_mode][["level", "baseline", "overall_acc"]]
        lines.append(_md_table(sub))
        lines.append("")
        lines.append(f"### ask_mode = {ask_mode} (answer_acc)\n")
        sub = level_df[level_df["ask_mode"] == ask_mode][["level", "baseline", "answer_acc"]]
        lines.append(_md_table(sub))
        lines.append("")
        lines.append(f"### ask_mode = {ask_mode} (abstain_rate)\n")
        sub = level_df[level_df["ask_mode"] == ask_mode][["level", "baseline", "abstain_rate"]]
        lines.append(_md_table(sub))
        lines.append("")

    lines.append("## 3) Trend Across Levels (overall_acc per baseline)\n")
    for ask_mode in ASK_MODES:
        lines.append(f"### ask_mode = {ask_mode}\n")
        sub = trend_df[trend_df["ask_mode"] == ask_mode][
            ["baseline", "L1", "L2", "L3", "L4", "L1->L4_drop"]
        ]
        lines.append(_md_table(sub))
        lines.append("")

    lines.append("## 4) Baseline Gaps by Level (overall_acc)\n")
    for ask_mode in ASK_MODES:
        lines.append(f"### ask_mode = {ask_mode}\n")
        sub = gap_df[gap_df["ask_mode"] == ask_mode]
        lines.append(_md_table(sub))
        lines.append("")

    lines.append("## 5) Cross-User sigma (overall_acc)\n")
    for ask_mode in ASK_MODES:
        lines.append(f"### ask_mode = {ask_mode}\n")
        if sigma_df.empty:
            lines.append("N/A\n")
        else:
            sub = sigma_df[sigma_df["ask_mode"] == ask_mode][["baseline", "mean", "std"]]
            lines.append(_md_table(sub))
        lines.append("")

    lines.append("## 6) Stage1 Failure: History vs LLM (three_factor_compare)\n")
    if compare_df is None or compare_df.empty:
        lines.append("N/A - no three_factor_compare data found in this summary.\n")
    else:
        lines.append(_md_table(compare_df))

    lines.append("## 7) Personalized Scenarios\n")
    if pers_df.empty:
        lines.append("N/A - no personalized scenario data.\n")
    else:
        lines.append("### Personalized vs Non-personalized (accuracy)\n")
        lines.append(_md_table(pers_df))
        lines.append("")
        lines.append("### Per-user accuracy (overall)\n")
        lines.append(_md_table(user_acc_df))

    out_path = summary_dir / "summary_tables_l1234.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")

    all_path = summary_dir / "ALL_RESULTS.md"
    all_parts = [out_path.read_text(encoding="utf-8")]
    if not pers_df.empty:
        all_parts.append("\n## 8) Personalized Scenarios (CSV dump)\n")
        all_parts.append(pers_df.to_markdown(index=False))
    if not user_acc_df.empty:
        all_parts.append("\n## 9) Per-user Accuracy (CSV dump)\n")
        all_parts.append(user_acc_df.to_markdown(index=False))
    all_path.write_text("\n\n".join(all_parts).strip() + "\n", encoding="utf-8")
    print(f"Wrote {all_path}")


if __name__ == "__main__":
    main()
