#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
FUZZY_DIR = ROOT / "experiments" / "fuzzy_sets"
MU_DIR = ROOT / "experiments" / "multi_user_dataset"
RESULTS_ROOT = ROOT / "experiments" / "results"


@dataclass
class BaselineCfg:
    bid: str
    name: str
    disambig_mode: str
    env: Dict[str, str]


def _hist_file_for_target(target: str, history_dir: Path) -> Path:
    stem = target.replace(".txt", "")
    parts = stem.split("_")
    user_key = "_".join(parts[:-1])
    return history_dir / f"{user_key}_history.txt"


def _weights_env(w_sem: float, w_them: float, w_sal: float) -> Dict[str, str]:
    out = {
        "UNWEIGHTED_THREE_FACTOR": "0",
    }
    for lv in ("L1", "L2", "L3", "L4"):
        out[f"W_SEM_{lv}"] = f"{w_sem:.6f}"
        out[f"W_THEM_{lv}"] = f"{w_them:.6f}"
        out[f"W_SAL_{lv}"] = f"{w_sal:.6f}"
    return out


def _validate_cache_freshness(targets: List[str], history_dir: Path) -> List[Tuple[str, str, str]]:
    stale: List[Tuple[str, str, str]] = []
    seen = set()
    for t in targets:
        stem = t.replace(".txt", "")
        user_key = "_".join(stem.split("_")[:-1])
        if user_key in seen:
            continue
        seen.add(user_key)
        h = history_dir / f"{user_key}_history.txt"
        c = ROOT / "experiments" / "gold_state_cache" / f"{user_key}_history"
        if not h.exists() or not c.exists():
            continue
        h_ts = h.stat().st_mtime
        c_ts = c.stat().st_mtime
        if h_ts > c_ts:
            h_iso = datetime.fromtimestamp(h_ts).strftime("%Y-%m-%d %H:%M:%S")
            c_iso = datetime.fromtimestamp(c_ts).strftime("%Y-%m-%d %H:%M:%S")
            stale.append((user_key, h_iso, c_iso))
    return stale


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = os.environ.get("BASELINE8_RUN_TAG", "").strip() or f"baseline8_u1u5_a1a4_{ts}"
    run_root = RESULTS_ROOT / run_tag
    run_root.mkdir(parents=True, exist_ok=True)

    strict_no_fallback = os.environ.get("BASELINE8_STRICT_NO_FALLBACK", "1").strip() == "1"
    print(f"[CFG] BASELINE8_STRICT_NO_FALLBACK={'1' if strict_no_fallback else '0'}")

    users_filter_raw = os.environ.get("BASELINE8_USERS", "").strip()
    levels_filter_raw = os.environ.get("BASELINE8_LEVELS", "").strip()
    users_filter = {x.strip().lower() for x in users_filter_raw.split(",") if x.strip()}
    levels_filter = {x.strip().lower() for x in levels_filter_raw.split(",") if x.strip()}
    if users_filter:
        print(f"[CFG] BASELINE8_USERS={sorted(users_filter)}")
    if levels_filter:
        print(f"[CFG] BASELINE8_LEVELS={sorted(levels_filter)}")

    history_dir_raw = os.environ.get("BASELINE8_HISTORY_DIR", "").strip()
    history_dir = Path(history_dir_raw).expanduser().resolve() if history_dir_raw else MU_DIR
    print(f"[CFG] BASELINE8_HISTORY_DIR={history_dir}")

    targets: List[str] = []
    for p in sorted(FUZZY_DIR.glob("user*_*.txt")):
        n = p.name
        if not any(n.endswith(f"_{lv}.txt") for lv in ("a1", "a2", "a3", "a4")):
            continue
        stem = n.replace(".txt", "")
        user_id = "_".join(stem.split("_")[:-1]).lower()
        level = stem.split("_")[-1].lower()
        if users_filter and user_id not in users_filter:
            continue
        if levels_filter and level not in levels_filter:
            continue
        targets.append(n)

    allow_stale = os.environ.get("ALLOW_STALE_CACHE", "").strip() == "1"
    stale = _validate_cache_freshness(targets, history_dir)
    if stale and not allow_stale:
        print("[ERROR] history/cache mismatch detected (history newer than cache):")
        for u, h, c in stale:
            print(f"  - {u}: history={h} > cache={c}")
        print("Set ALLOW_STALE_CACHE=1 to bypass, or rerun gold cache first.")
        return 3

    force_no_ask_raw = os.environ.get("BASELINE8_FORCE_NO_ASK")
    if force_no_ask_raw is None:
        force_no_ask = "1"
    else:
        force_no_ask = force_no_ask_raw.strip()
    if force_no_ask not in {"0", "1"}:
        print(f"[WARN] invalid BASELINE8_FORCE_NO_ASK={force_no_ask!r}; fallback to '1'")
        force_no_ask = "1"
    fallback_policy = os.environ.get("BASELINE8_FALLBACK_POLICY", "fused_top1").strip() or "fused_top1"
    print(f"[CFG] BASELINE8_FORCE_NO_ASK={force_no_ask}")
    print(f"[CFG] BASELINE8_FALLBACK_POLICY={fallback_policy}")

    common_env = {
        "FORCE_NO_ASK": force_no_ask,
        "FORCE_NO_ASK_FALLBACK_POLICY": fallback_policy,
        "GOLD_STATE_CACHE": "use",
        "GOLD_STATE_CACHE_STRICT": "1",
        "EVAL_USE_GOLD_GROUP_HISTORY": "1",
    }
    if strict_no_fallback and force_no_ask == "1":
        print("[WARN] strict_no_fallback is ON but FORCE_NO_ASK=1 still implies ASK->top1 forced fallback.")

    full_profile = os.environ.get("BASELINE8_FULL_PROFILE", "legacy_c9").strip().lower() or "legacy_c9"
    if full_profile not in {"dynamic", "simple_equal", "legacy_c9", "legacy_c9_l1push"}:
        print(f"[WARN] invalid BASELINE8_FULL_PROFILE={full_profile!r}; fallback to 'dynamic'")
        full_profile = "dynamic"
    print(f"[CFG] BASELINE8_FULL_PROFILE={full_profile}")

    full_env_dynamic = {
        "DISABLE_HISTORY_COUNTER": "1",
        "THEMATIC_MODE": "concept",
        "UNWEIGHTED_THREE_FACTOR": "0",
        "W_SEM_L1": "0.20",
        "W_THEM_L1": "0.65",
        "W_SAL_L1": "0.15",
        "W_SEM_L2": "0.18",
        "W_THEM_L2": "0.62",
        "W_SAL_L2": "0.20",
        "W_SEM_L3": "0.24",
        "W_THEM_L3": "0.56",
        "W_SAL_L3": "0.20",
        "W_SEM_L4": "0.28",
        "W_THEM_L4": "0.52",
        "W_SAL_L4": "0.20",
        "CONCEPT_STHEM_MODE": "both",
        "CONCEPT_NMIN": "6",
        "CLEAR_LEAD_RATIO_L1": "0.18",
        "CLEAR_LEAD_RATIO_L2": "0.16",
        "CLEAR_LEAD_RATIO_L3": "0.15",
        "CLEAR_LEAD_RATIO_L4": "0.15",
    }
    full_env_simple_equal = {
        "DISABLE_HISTORY_COUNTER": "1",
        "THEMATIC_MODE": "concept",
        "UNWEIGHTED_THREE_FACTOR": "1",
    }
    full_env_legacy_c9 = {
        "DISABLE_HISTORY_COUNTER": "1",
        "THEMATIC_MODE": "concept",
        "UNWEIGHTED_THREE_FACTOR": "1",
        "CONCEPT_STHEM_MODE": "both",
        "CLEAR_LEAD_RATIO_L1": "0.25",
        "CLEAR_LEAD_RATIO_L2": "0.25",
        "CLEAR_LEAD_RATIO_L3": "0.25",
        "CLEAR_LEAD_RATIO_L4": "0.25",
    }
    full_env_legacy_c9_l1push = {
        **full_env_legacy_c9,
        "CLEAR_LEAD_RATIO_L1": "0.22",
    }
    if full_profile == "dynamic":
        full_env = full_env_dynamic
    elif full_profile == "simple_equal":
        full_env = full_env_simple_equal
    elif full_profile == "legacy_c9_l1push":
        full_env = full_env_legacy_c9_l1push
    else:
        full_env = full_env_legacy_c9

    full_override_prefix = "BASELINE8_FULL_SET_"
    full_overrides = {
        k[len(full_override_prefix) :]: v
        for k, v in os.environ.items()
        if k.startswith(full_override_prefix) and k[len(full_override_prefix) :]
    }
    if full_overrides:
        full_env = {**full_env, **full_overrides}
        print(f"[CFG] FULL overrides={full_overrides}")

    three_factor_base = {
        **full_env,
        "DISABLE_HISTORY_COUNTER": "1",
        "THEMATIC_MODE": "concept",
        "UNWEIGHTED_THREE_FACTOR": "0",
        "W_SEM_L1": "0.40", "W_THEM_L1": "0.45", "W_SAL_L1": "0.15",
        "W_SEM_L2": "0.25", "W_THEM_L2": "0.62", "W_SAL_L2": "0.13",
        "W_SEM_L3": "0.25", "W_THEM_L3": "0.62", "W_SAL_L3": "0.13",
        "W_SEM_L4": "0.35", "W_THEM_L4": "0.52", "W_SAL_L4": "0.13",
    }
    baselines = [
        BaselineCfg(
            "B0",
            "Random_ASP_filtered",
            "random_filtered",
            {},
        ),
        BaselineCfg(
            "X_LLM_ZS",
            "LLM_zero_shot",
            "llm_only",
            {
                "LLM_USE_RECENT_HISTORY": "0",
                "LLM_PERSONA_MODE": "0",
            },
        ),
        BaselineCfg(
            "X_LLM_ZS_ASP",
            "LLM_full_information_ASP_filtered",
            "llm_filtered",
            {
                "LLM_USE_RECENT_HISTORY": "0",
                "LLM_PERSONA_MODE": "0",
                "LLM_WORLD_STATE": "1",
                "LLM_STATIC_FEWSHOT": "1",
            },
        ),
        BaselineCfg(
            "X_LLM_FS",
            "LLM_few_shot",
            "llm_only",
            {
                "LLM_USE_RECENT_HISTORY": "1",
                "LLM_WORLD_STATE": "1",
            },
        ),
        BaselineCfg(
            "X_LLM_FS_ASP",
            "LLM_few_shot_ASP_filtered",
            "llm_filtered",
            {
                "LLM_USE_RECENT_HISTORY": "1",
                "LLM_WORLD_STATE": "1",
            },
        ),
        BaselineCfg(
            "X_LLM_FS_COT",
            "LLM_few_shot_CoT",
            "llm_only",
            {
                "LLM_USE_RECENT_HISTORY": "1",
                "LLM_WORLD_STATE": "1",
                "LLM_COT": "1",
            },
        ),
        BaselineCfg(
            "X_LLM_FH",
            "LLM_full_history",
            "llm_only",
            {
                "LLM_USE_RECENT_HISTORY": "1",
                "LLM_WORLD_STATE": "1",
                "LLM_CROSS_HISTORY": "1",
            },
        ),
        BaselineCfg(
            "B1",
            "LLM_full_history_CoT",
            "llm_only",
            {
                "LLM_USE_RECENT_HISTORY": "1",
                "LLM_WORLD_STATE": "1",
                "LLM_CROSS_HISTORY": "1",
                "LLM_COT": "1",
            },
        ),
        BaselineCfg(
            "B2",
            "LLM_full_history_CoT_ASP",
            "llm_filtered",
            {
                "LLM_USE_RECENT_HISTORY": "1",
                "LLM_WORLD_STATE": "1",
                "LLM_CROSS_HISTORY": "1",
                "LLM_COT": "1",
            },
        ),
        BaselineCfg(
            "B3",
            "Semantic_plus_Salience",
            "three_factor",
            {
                **three_factor_base,
                **_weights_env(0.5, 0.0, 0.5),
            },
        ),
        BaselineCfg(
            "B4",
            "Object_level_history",
            "three_factor",
            {
                **three_factor_base,
                "THEMATIC_MODE": "object",
                "NO_ASP_FILTER": "1",
            },
        ),
        BaselineCfg(
            "B6",
            "L1_only",
            "three_factor",
            {
                **three_factor_base,
                "CONCEPT_SOURCE_MODE": "l1",
                "CONCEPT_STHEM_MODE": "instruction",
            },
        ),
        BaselineCfg(
            "B7",
            "Full_no_ASP",
            "three_factor",
            {
                **three_factor_base,
                "CONCEPT_ENV_MODE": "legacy",
                "NO_ASP_FILTER": "1",
            },
        ),
        BaselineCfg(
            "X_TGT",
            "Target_only",
            "three_factor",
            {
                **three_factor_base,
                "CONCEPT_ENV_MODE": "habit",
                "CONCEPT_HABIT_BASKET_MODE": "target_only",
            },
        ),
        BaselineCfg(
            "X_TRG",
            "Trigger_only",
            "three_factor",
            {
                **three_factor_base,
                "CONCEPT_ENV_MODE": "habit",
                "CONCEPT_HABIT_BASKET_MODE": "trigger_only",
            },
        ),
        BaselineCfg(
            "Proposed",
            "Proposed_fusion",
            "three_factor",
            {
                **three_factor_base,
                "CONCEPT_ENV_MODE": "legacy",
            },
        ),
        BaselineCfg(
            "X_PTN",
            "Pattern_only",
            "three_factor",
            {
                **three_factor_base,
                "CONCEPT_ENV_MODE": "legacy",
                "CONCEPT_STHEM_MODE": "environment",
            },
        ),
        BaselineCfg(
            "B5",
            "L0_attributes_full",
            "three_factor",
            {
                **three_factor_base,
                "CONCEPT_ENV_MODE": "legacy",
                "CONCEPT_SOURCE_MODE": "l0",
            },
        ),
    ]

    baseline_filter_raw = os.environ.get("BASELINE8_ONLY", "").strip()
    if baseline_filter_raw:
        allow = {x.strip().upper() for x in baseline_filter_raw.split(",") if x.strip()}
        baselines = [b for b in baselines if b.bid.upper() in allow]
        print(f"[CFG] BASELINE8_ONLY={sorted(allow)} -> {len(baselines)} baseline(s)")

    for i, b in enumerate(baselines):
        prefix = f"BASELINE8_{b.bid}_SET_"
        ov = {
            k[len(prefix) :]: v
            for k, v in os.environ.items()
            if k.startswith(prefix) and k[len(prefix) :]
        }
        if ov:
            baselines[i] = BaselineCfg(
                bid=b.bid,
                name=b.name,
                disambig_mode=b.disambig_mode,
                env={**b.env, **ov},
            )
            print(f"[CFG] {b.bid} overrides={ov}")

    py = sys.executable
    rows: List[Dict[str, object]] = []
    fail_count = 0
    timeout_by_baseline = {
        "B1": 900,
        "B2": 900,
    }
    disable_timeout = os.environ.get("BASELINE8_DISABLE_TIMEOUT", "").strip() == "1"
    if disable_timeout:
        print("[CFG] BASELINE8_DISABLE_TIMEOUT=1 (no per-target timeout)")

    def _apply_timeout_override(bid: str) -> None:
        key = f"BASELINE8_TIMEOUT_{bid}"
        raw = os.environ.get(key, "").strip()
        if not raw:
            return
        try:
            timeout_by_baseline[bid] = max(1, int(raw))
            print(f"[CFG] {key}={timeout_by_baseline[bid]}")
        except ValueError:
            print(f"[WARN] invalid {key}={raw!r}; keep default {timeout_by_baseline[bid]}")

    for _bid in timeout_by_baseline:
        _apply_timeout_override(_bid)

    for b in baselines:
        print(f"\n=== {b.bid} {b.name} ({b.disambig_mode}) ===", flush=True)
        for idx, t in enumerate(targets, 1):
            stem = t.replace(".txt", "")
            out_dir = run_root / f"{stem}__{b.bid}"
            test_json = out_dir / "test_results.json"
            if test_json.exists():
                try:
                    arr = json.loads(test_json.read_text())
                    total = len(arr)
                    succ = sum(1 for x in arr if bool(x.get("success", False)))
                    acc = (succ / total) if total else 0.0
                    rows.append(
                        {
                            "baseline_id": b.bid,
                            "baseline_name": b.name,
                            "target": t,
                            "user_id": "_".join(stem.split("_")[:-1]),
                            "level": stem.split("_")[-1],
                            "total": total,
                            "success": succ,
                            "overall_acc": acc,
                            "status": "skipped_exists",
                        }
                    )
                    print(f"[{idx}/{len(targets)}] skip {t} acc={acc:.3f}", flush=True)
                    continue
                except Exception:
                    pass

            env = os.environ.copy()
            env.update(common_env)
            env.update(b.env)
            env["DISAMBIG_MODE"] = b.disambig_mode
            env["FUZZY_RUN_NAME"] = str(Path(run_tag) / out_dir.name)

            hist = _hist_file_for_target(t, history_dir)
            if hist.exists():
                env["ACTIVE_HISTORY_FILE"] = str(hist)
                env["THEMATIC_HISTORY_FILE"] = str(hist)

            print(f"[{idx}/{len(targets)}] run {t} -> {out_dir.name}", flush=True)
            timeout_s = None if disable_timeout else timeout_by_baseline.get(b.bid, 900)
            try:
                proc = subprocess.run(
                    [py, str(ROOT / "test_fuzzy_sets.py"), "--target", t],
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_s,
                )
                (out_dir / "runner_stdout.log").write_text(proc.stdout, encoding="utf-8")
                rc = proc.returncode
            except subprocess.TimeoutExpired as e:
                rc = 124
                out = e.stdout or b""
                if isinstance(out, bytes):
                    try:
                        out = out.decode("utf-8", errors="ignore")
                    except Exception:
                        out = ""
                partial = str(out) + "\n[TIMEOUT]\n"
                (out_dir / "runner_stdout.log").write_text(partial, encoding="utf-8")

            if rc != 0 or not test_json.exists():
                fail_count += 1
                rows.append(
                    {
                        "baseline_id": b.bid,
                        "baseline_name": b.name,
                        "target": t,
                        "user_id": "_".join(stem.split("_")[:-1]),
                        "level": stem.split("_")[-1],
                        "total": 0,
                        "success": 0,
                        "overall_acc": 0.0,
                        "status": f"failed_rc_{rc}",
                    }
                )
                print(f"  !! failed rc={rc}", flush=True)
                continue

            arr = json.loads(test_json.read_text())
            total = len(arr)
            succ = sum(1 for x in arr if bool(x.get("success", False)))
            acc = (succ / total) if total else 0.0
            rows.append(
                {
                    "baseline_id": b.bid,
                    "baseline_name": b.name,
                    "target": t,
                    "user_id": "_".join(stem.split("_")[:-1]),
                    "level": stem.split("_")[-1],
                    "total": total,
                    "success": succ,
                    "overall_acc": acc,
                    "status": "ok",
                }
            )
            print(f"  -> acc={acc:.3f}", flush=True)

    csv_path = run_root / "baseline8_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "baseline_id",
                "baseline_name",
                "target",
                "user_id",
                "level",
                "total",
                "success",
                "overall_acc",
                "status",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    import pandas as pd

    df = pd.DataFrame(rows)
    agg = (
        df[df["status"].isin(["ok", "skipped_exists"])]
        .groupby(["baseline_id", "baseline_name", "level"], as_index=False)["overall_acc"]
        .mean()
    )
    agg_path = run_root / "baseline8_results_by_level.csv"
    agg.to_csv(agg_path, index=False)

    agg_user = (
        df[df["status"].isin(["ok", "skipped_exists"])]
        .groupby(["baseline_id", "baseline_name", "user_id"], as_index=False)["overall_acc"]
        .mean()
    )
    agg_user_path = run_root / "baseline8_results_by_user.csv"
    agg_user.to_csv(agg_user_path, index=False)

    print("\n=== DONE ===")
    print(f"run_root: {run_root}")
    print(f"rows: {len(rows)}, fails: {fail_count}")
    print(f"csv: {csv_path}")
    print(f"agg(level): {agg_path}")
    print(f"agg(user): {agg_user_path}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
