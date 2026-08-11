#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
import shutil
from pathlib import Path

from commands.context_io import clear_context_file
from commands.command_planner import plan
from commands.confidence_scorer import set_repeat_pool
from main import clear_group_history
from kg.kg_updater import revert_kg_to_backup, update_kg_from_asp_outputs
from kg.loader import get_graph, load_kg, save_kg, EX
from asp.file_manager import (
    extract_initial_conditions,
    remove_initial_conditions_from_asp,
    insert_initial_conditions_to_asp,
)
import os
import lean_asp
from config.config import (
    ASP_FILE,
    INITIAL_CONDITIONS_FILE,
    SHOW_START_OUTPUT,
    SHOW_LAST_OUTPUT,
    SHOW_CHANGED_HOLDS_OUTPUT,
)


ROOT = Path(__file__).resolve().parent
MU_DIR = ROOT / "experiments" / "multi_user_dataset"
GOLD_CACHE_ROOT = ROOT / "experiments" / "gold_state_cache"
LEAN = os.environ.get("LEAN", "1") != "0"

_INIT_FACT_RE = re.compile(r"([a-z_]+)\(([^)]*)\)")


def _parse_init_line(s: str) -> list[tuple[str, list[str]]]:
    facts: list[tuple[str, list[str]]] = []
    low = s.strip().lower()
    if low.startswith("@init:") or low.startswith("% init:"):
        payload = s.split(":", 1)[1]
        for m in _INIT_FACT_RE.finditer(payload):
            facts.append((m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()]))
    return facts


def parse_init_blocks(text: str) -> list[list[tuple[str, list[str]]]]:
    head = text.split("[", 1)[0]
    shared: list[tuple[str, list[str]]] = []
    for raw in head.splitlines():
        shared += _parse_init_line(raw)
    blocks = re.findall(r"\[\s*(.*?)\s*\]", text, flags=re.DOTALL)
    out: list[list[tuple[str, list[str]]]] = []
    for blk in blocks:
        facts = list(shared)
        for raw in blk.splitlines():
            facts += _parse_init_line(raw)
        out.append(facts)
    return out


def apply_init_state(facts: list[tuple[str, list[str]]]) -> None:
    from rdflib import Literal
    from rdflib.namespace import XSD
    if not facts:
        return
    g = get_graph()
    for pred, args in facts:
        if pred in ("on", "inside") and len(args) == 2:
            f, x = args
            g.remove((EX[x], EX.on, None))
            g.remove((EX[x], EX.inside, None))
            g.add((EX[x], EX.on if pred == "on" else EX.inside, EX[f]))
        elif pred == "has" and len(args) == 2:
            who, x = args
            g.remove((EX[x], EX.on, None))
            g.remove((EX[x], EX.inside, None))
            g.add((EX[who], EX.has, EX[x]))
        elif pred == "heated" and len(args) == 1:
            g.add((EX[args[0]], EX.heated, Literal(True, datatype=XSD.boolean)))
        elif pred == "filled" and len(args) == 2:
            g.add((EX[args[0]], EX.filled, EX[args[1]]))
    save_kg(str(ROOT / "kg" / "living_room.ttl"))
    print(f"[INIT] applied {len(facts)} initial-state fact(s): {facts}")


def _apply_lean_sorts_for_clip(clip_lines):
    objs = lean_asp.clip_object_set(clip_lines)
    g = get_graph()
    g.remove((None, None, None))
    load_kg(str(ROOT / "kg" / "living_room.ttl"))
    objs = lean_asp.expand_with_kg(objs, get_graph())
    lean_sorts, allow = lean_asp.build_lean_sorts(objs, n=15)
    text = Path(ASP_FILE).read_text(encoding="utf-8")
    Path(ASP_FILE).write_text(lean_asp.replace_sorts_block(text, lean_sorts), encoding="utf-8")
    return allow


def _filter_ic_to_allow(allow):
    p = Path(INITIAL_CONDITIONS_FILE)
    p.write_text(lean_asp.filter_ic_lines(p.read_text(encoding="utf-8"), allow),
                 encoding="utf-8")


def _gold_cache_dir(cache_key: str, group_idx: int, line_idx: int) -> Path:
    return GOLD_CACHE_ROOT / cache_key / f"group_{group_idx}" / f"line_{line_idx}"


def save_gold_state(cache_keys: list[str], group_idx: int, line_idx: int) -> None:
    for key in cache_keys:
        cache_dir = _gold_cache_dir(key, group_idx, line_idx)
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "kg" / "living_room.ttl", cache_dir / "living_room.ttl")
        shutil.copyfile(ROOT / "context.txt", cache_dir / "context.txt")


def _parse_blocks(text: str) -> list[list[str]]:
    blocks = re.findall(r"\[\s*(.*?)\s*\]", text, flags=re.DOTALL)
    all_groups: list[list[str]] = []
    for blk in blocks:
        cmds = [l.strip() for l in blk.splitlines() if "|" in l]
        all_groups.append(cmds)
    return all_groups


def _write_summary(out_path: Path, payload: dict) -> None:
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        required=True,
        help="experiments/multi_user_dataset/ history filename, e.g. user1_history.txt",
    )
    parser.add_argument(
        "--cache-key",
        default="",
        help=(
            "cache key(s) to save under (comma-separated). "
            "Defaults to the history filename without .txt"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="maximum ASP steps; abort with an error above this",
    )
    parser.add_argument(
        "--only-clips",
        default="",
        help=(
            "only (re)record these clips (comma-separated, e.g. '6' or '3,5,12'); empty = all. "
            "Clips are independent (each starts from the clean backup plus fixed dirty state), "
            "so only the edited clip needs re-recording."
        ),
    )
    args = parser.parse_args()

    only_clips = None
    if args.only_clips.strip():
        only_clips = {int(x) for x in args.only_clips.split(",") if x.strip().isdigit()}

    history_path = MU_DIR / args.history
    if not history_path.exists():
        sys.stderr.write(f"[ERROR] history file not found: {history_path}\n")
        return 1

    file_stem = history_path.stem
    if args.cache_key.strip():
        cache_keys = [k.strip() for k in args.cache_key.split(",") if k.strip()]
    else:
        cache_keys = [file_stem]

    res_dir = ROOT / "experiments" / "results" / f"{file_stem}__gold_record"
    res_dir.mkdir(parents=True, exist_ok=True)
    log_path = res_dir / "execute_log.txt"
    summary_path = res_dir / "gold_record_summary.json"

    log_fp = log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_fp)
    sys.stderr = _Tee(sys.__stderr__, log_fp)

    clear_context_file()

    text = history_path.read_text(encoding="utf-8")
    groups = _parse_blocks(text)
    clip_inits = parse_init_blocks(text)
    if not groups:
        print("[ERROR] no [ ... ] blocks found")
        return 1

    failures = []
    total_lines = 0

    all_group_cmds = []
    for group in groups:
        cmds = [l.split("|", 1)[0].strip() for l in group if "|" in l]
        all_group_cmds.append(cmds)

    for grp_idx, lines in enumerate(groups, start=1):
        if only_clips is not None and grp_idx not in only_clips:
            continue
        print(f"Group{grp_idx}")

        clear_group_history(f"[{grp_idx}]")

        pool = [
            cmd
            for i, group in enumerate(all_group_cmds, start=1)
            if i != grp_idx
            for cmd in group
        ]
        set_repeat_pool(pool)

        revert_kg_to_backup()
        apply_init_state(clip_inits[grp_idx - 1] if grp_idx - 1 < len(clip_inits) else [])

        lean_allow = None
        if LEAN:
            try:
                lean_allow = _apply_lean_sorts_for_clip(lines)
                print(f"[LEAN] clip {grp_idx}: {len(lean_allow)} objects in lean sorts")
            except Exception as e:
                print(f"[ERROR] lean sorts failed: {e}")
                _write_summary(summary_path, {"ok": False, "failures": failures})
                return 1

        for line_idx, raw in enumerate(lines, start=1):
            asp_gold = raw.split("|", 1)[0].strip()
            nl = raw.split("|", 1)[1].strip()
            total_lines += 1
            print(f" [{grp_idx}-{line_idx}] {asp_gold} | {nl}")

            try:
                extract_initial_conditions()
                if LEAN and lean_allow is not None:
                    _filter_ic_to_allow(lean_allow)
                remove_initial_conditions_from_asp()
                insert_initial_conditions_to_asp()
            except Exception as e:
                failures.append(
                    {"group": grp_idx, "line": line_idx, "asp": asp_gold, "error": f"init:{e!r}"}
                )
                print(f"[ERROR] initial condition handling failed: {e}")
                _write_summary(summary_path, {"ok": False, "failures": failures})
                return 1

            try:
                plan_res = plan(asp_gold, line_id=f"[{grp_idx}-{line_idx}]", nl=nl)
            except Exception as e:
                failures.append(
                    {"group": grp_idx, "line": line_idx, "asp": asp_gold, "error": f"plan:{e!r}"}
                )
                print(f"[ERROR] ASP run failed: {e}")
                _write_summary(summary_path, {"ok": False, "failures": failures})
                return 1

            steps = int(plan_res.get("steps", 0) or 0)
            if steps > args.max_steps:
                failures.append(
                    {
                        "group": grp_idx,
                        "line": line_idx,
                        "asp": asp_gold,
                        "error": f"steps>{args.max_steps} ({steps})",
                    }
                )
                print(f"[ERROR] ASP step limit exceeded: {steps} > {args.max_steps}")
                _write_summary(summary_path, {"ok": False, "failures": failures})
                return 1

            try:
                update_kg_from_asp_outputs(
                    start_holds_file=SHOW_START_OUTPUT,
                    last_holds_file=SHOW_LAST_OUTPUT,
                    changed_holds_file=SHOW_CHANGED_HOLDS_OUTPUT,
                    changed_names_file=None,
                    touched_items=plan_res.get("touched", []),
                    auto_find_changed_names=True,
                )
            except Exception as e:
                failures.append(
                    {"group": grp_idx, "line": line_idx, "asp": asp_gold, "error": f"kg:{e!r}"}
                )
                print(f"[ERROR] KG update failed: {e}")
                _write_summary(summary_path, {"ok": False, "failures": failures})
                return 1

            try:
                save_gold_state(cache_keys, grp_idx, line_idx)
            except Exception as e:
                failures.append(
                    {"group": grp_idx, "line": line_idx, "asp": asp_gold, "error": f"save:{e!r}"}
                )
                print(f"[ERROR] cache save failed: {e}")
                _write_summary(summary_path, {"ok": False, "failures": failures})
                return 1

    set_repeat_pool([])
    _write_summary(summary_path, {"ok": True, "total_lines": total_lines, "failures": failures})
    print(f" gold state cache complete: {summary_path}")
    return 0


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        self.flush()

    def flush(self):
        for s in self.streams:
            if hasattr(s, "flush"):
                s.flush()


if __name__ == "__main__":
    raise SystemExit(main())
