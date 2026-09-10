#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def stage(label: str, cmd: list[str], env: dict) -> int:
    print(f"\n===== {label} =====", flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
    if rc != 0:
        print(f"[ERROR] {label} returned {rc}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Derive the world states, evaluate every baseline, write result/."
    )
    ap.add_argument("--user", action="append",
                    help="restrict to a user; repeatable, defaults to all five")
    ap.add_argument("--level", action="append",
                    help="restrict to a1, a2, a3 or a4; repeatable")
    ap.add_argument("--run-tag", default="full_run")
    args = ap.parse_args()

    env = dict(os.environ)
    env["BASELINE8_RUN_TAG"] = args.run_tag
    if args.user:
        env["BASELINE8_USERS"] = ",".join(args.user)
    if args.level:
        env["BASELINE8_LEVELS"] = ",".join(args.level)

    py = [sys.executable, "-u"]
    derive = py + ["scripts/derive_world_states.py"]
    for user in args.user or []:
        derive += ["--user", user]

    for label, cmd in (
        ("stage 1  derive the world states with ASP", derive),
        ("stage 2  evaluate every baseline", py + ["scripts/run_baseline8_matrix.py"]),
        ("collect  write result/", py + ["scripts/collect_results.py", args.run_tag]),
    ):
        rc = stage(label, cmd, env)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
