
from __future__ import annotations
import re
import pathlib
from typing import Set, Dict, Optional, Callable

from asp.goals import execute_user_goal
from config import config as cfg

_OCCURS_RE   = re.compile(r"occurs\(")
_TOUCHED_RE  = re.compile(r"\(([\w_]+)\)")

HistoryRecorder = Optional[Callable[[Optional[str], str, str], None]]

_HISTORY_RECORDER: HistoryRecorder = None

def register_history_recorder(fn: HistoryRecorder) -> None:
    global _HISTORY_RECORDER
    _HISTORY_RECORDER = fn


def _count_occurs(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return len(_OCCURS_RE.findall(txt))


def _parse_touched(path: pathlib.Path) -> Set[str]:
    if not path.exists():
        return set()
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return {m.group(1) for m in _TOUCHED_RE.finditer(txt)}


def plan(
    asp_goal: str,
    *,
    line_id: Optional[str] = None,
    nl: str = "",
    history_recorder: HistoryRecorder = None,
) -> Dict[str, object]:
    execute_user_goal(asp_goal)

    rec = history_recorder if history_recorder is not None else _HISTORY_RECORDER
    if rec is not None:
        try:
            rec(line_id, asp_goal, nl or asp_goal)
        except Exception:
            pass

    occurs_file   = pathlib.Path(cfg.OCCURS_OUTPUT)
    operated_file = pathlib.Path(cfg.OPERATED_OUTPUT)

    steps = _count_occurs(occurs_file)
    if steps == 0:
        steps = 1

    touched = _parse_touched(operated_file)

    return {"steps": steps, "touched": touched}
