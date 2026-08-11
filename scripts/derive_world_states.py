#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MU_DIR = ROOT / "experiments" / "multi_user_dataset"
OUT_ROOT = ROOT / "experiments" / "gold_state_cache"
ASP_FILE = ROOT / "two_goals.sp"
ASP_MASTER = ROOT / "two_goals.sp.master"
BACKUP = ROOT / "kg" / "living_room_backup.ttl"
SCENE = ROOT / "kg" / "living_room_scene.ttl"

USERS = [
    "user1",
    "user2",
    "user3",
    "user4",
    "user5",
]


def clips_for(user: str) -> list[int]:
    hist = MU_DIR / f"{user}_history.txt"
    if not hist.is_file():
        return []
    n = sum(1 for line in hist.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("["))
    return list(range(1, n + 1))


def build_clip_scene(clip_kb: Path) -> int:
    from rdflib import Graph, Namespace

    ex = Namespace("http://example.org/")
    scene = Graph()
    scene.parse(SCENE, format="turtle")
    overlay = Graph()
    overlay.parse(clip_kb, format="turtle")
    placements = [(s, p, o) for p in (ex.on, ex.inside)
                  for s, o in overlay.subject_objects(p)]
    for subj, _, _ in placements:
        scene.remove((subj, ex.on, None))
        scene.remove((subj, ex.inside, None))
    for subj, prop, obj in placements:
        scene.add((subj, prop, obj))
    scene.serialize(BACKUP, format="turtle")
    return len(placements)


def derive(user: str, clips: list[int], max_steps: int) -> int:
    kb_dir = MU_DIR / f"{user}_kb"
    hist = f"{user}_history.txt"
    failures = 0
    for clip in clips:
        clip_kb = kb_dir / f"clip{clip}.ttl"
        if not clip_kb.is_file():
            print(f"[SKIP] {user} clip {clip}: no {clip_kb.name}")
            continue
        shutil.copyfile(ASP_MASTER, ASP_FILE)
        shutil.copyfile(clip_kb, BACKUP)
        print(f"[DERIVE] {user} clip {clip}: planning with SPARC")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "record_gold_states.py"),
             "--history", hist, "--only-clips", str(clip),
             "--max-steps", str(max_steps)],
            cwd=str(ROOT),
        )
        if proc.returncode != 0:
            failures += 1
            print(f"[FAIL] {user} clip {clip}: planner returned {proc.returncode}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Derive the per-line world states with ASP, from scratch."
    )
    ap.add_argument("--user", action="append",
                    help="user to process; repeatable, defaults to all five")
    ap.add_argument("--clips", help="comma-separated clip numbers, defaults to all")
    ap.add_argument("--max-steps", type=int, default=15)
    args = ap.parse_args()

    if not ASP_MASTER.is_file():
        shutil.copyfile(ASP_FILE, ASP_MASTER)
    if not SCENE.is_file():
        shutil.copyfile(BACKUP, SCENE)

    users = args.user or USERS
    total_failures = 0
    try:
        for user in users:
            clips = ([int(c) for c in args.clips.split(",") if c.strip()]
                     if args.clips else clips_for(user))
            if not clips:
                print(f"[SKIP] {user}: no history")
                continue
            total_failures += derive(user, clips, args.max_steps)
            recorded = len(list((OUT_ROOT / f"{user}_history").rglob("living_room.ttl"))) \
                if (OUT_ROOT / f"{user}_history").is_dir() else 0
            print(f"[DONE] {user}: {recorded} world states written")
    finally:
        shutil.copyfile(SCENE, BACKUP)
        shutil.copyfile(ASP_MASTER, ASP_FILE)
        print("[RESET] pristine scene and ASP domain restored")

    if total_failures:
        print(f"[ERROR] {total_failures} clip(s) failed to plan")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
