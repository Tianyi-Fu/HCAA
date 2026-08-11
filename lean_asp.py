#!/usr/bin/env python3
from __future__ import annotations
import re
from typing import Dict, List, Set, Tuple

SORT_MEMBERS: Dict[str, List[str]] = {
    "#fruit":          ["apple", "banana", "orange", "peach", "grape", "pear"],
    "#vegetable":      ["tomato", "carrot", "potato", "lettuce", "mushroom"],
    "#baked":          ["bread", "cake", "cookie", "pizza", "muffin", "doughnut", "brownie", "croissant"],
    "#protein":        ["chicken", "steak", "salmon", "egg", "cheese"],
    "#drinkware":      ["cup", "mug", "glass", "bottle", "jar", "kettle", "teapot", "thermos", "beaker", "coffeepot", "cauldron", "flask", "carafe", "pitcher"],
    "#tableware":      ["plate", "bowl", "spoon", "fork", "knife", "spatula", "tongs", "colander", "ladle"],
    "#fillable":       ["cup", "mug", "glass", "bottle", "jar", "kettle", "teapot",
                         "thermos", "bowl", "bucket", "vase", "beaker", "coffeepot", "cauldron",
                         "flask", "carafe", "pitcher"],
    "#servingware":    ["plate", "bowl"],
    "#portable_media": ["book", "newspaper", "phone", "computer"],
    "#stationery":     ["pencil", "crayon"],
    "#linen":          ["pillow", "blanket", "rug", "curtain", "towel"],
    "#decor":          ["vase", "plant", "candle", "clock"],
    "#cleaning":       ["broom", "mop", "bucket", "basket"],
    "#sit_furn":       ["chair", "couch", "bed", "bench"],
    "#table_furn":     ["table", "drawers"],
    "#appliance":      ["refrigerator", "oven", "stove", "dishwasher", "microwave_oven", "hamper"],
    "#mounted_light":  ["lamp", "chandelier"],
    "#mounted_media":  ["television"],
    "#agent":          ["agent1"],
    "#user":           ["user"],
    "#room":           ["kitchen", "living_room", "bedroom", "bathroom"],
}

import os as _os, json as _json
_TWIN_MAP_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                               "data", "concepts", "twin_map.json")
try:
    with open(_TWIN_MAP_PATH, encoding="utf-8") as _tf:
        _TWIN_MAP = _json.load(_tf)
    for _sort, _members in list(SORT_MEMBERS.items()):
        for _old in list(_members):
            _tw = _TWIN_MAP.get(_old)
            if _tw and _tw not in _members:
                _members.append(_tw)
except FileNotFoundError:
    _TWIN_MAP = {}

REQUIRED_LEAVES = [
    "#fruit", "#vegetable", "#baked", "#protein", "#drinkware", "#tableware",
    "#servingware", "#fillable",
    "#portable_media", "#stationery", "#linen", "#decor", "#cleaning",
    "#sit_furn", "#table_furn",
    "#appliance", "#mounted_light", "#mounted_media",
]

REQUIRED_LITERALS: Dict[str, List[str]] = {
    "#appliance": ["microwave_oven", "stove", "dishwasher"],
    "#table_furn": ["drawers"],
}

COMPOSITE_BLOCK = """\
#food            = #fruit + #vegetable + #baked + #protein.
#kitchenware     = #drinkware + #tableware.
#large_household = #sit_furn + #table_furn + #appliance.
#household       = #linen + #decor + #cleaning.
#small_household = #portable_media + #stationery + #mounted_light + #mounted_media.
#heatable = #protein.
#item = #food + #kitchenware + #portable_media + #stationery + #household.
#on_furniture        = #sit_furn + #table_furn.
#surface             = #on_furniture + #servingware.
#inside_furniture    = #appliance.
#microwave_furniture = {microwave_oven}.
#light               = #mounted_light.
#switch_furniture    = #microwave_furniture + #light + #mounted_media + {stove} + {dishwasher}.
#container_furniture = #inside_furniture + #on_furniture.
#furniture           = #container_furniture + #switch_furniture.
#value = 0..10.
#sum_val = 0..100.
#step  = 0..n.
#thing = #item + #furniture.
#user_furniture = #furniture + #user.\
"""

_RESERVED = {
    "holds", "occurs", "on", "inside", "has", "location", "furniture_location",
    "user_location", "at_furniture", "at_user", "open", "closed", "locked",
    "changed", "switched_on", "switched_off", "heated", "dangerous", "in",
    "walk", "walktowards", "grab", "putin", "put", "give", "switchon",
    "switchoff", "close", "fill", "filled", "agent", "agent1", "user",
}

_ALL_MEMBERS = {m for ms in SORT_MEMBERS.values() for m in ms}


def clip_object_set(lines: List[str]) -> Set[str]:
    found: Set[str] = set()
    for raw in lines:
        lit = raw.split("|", 1)[0]
        for tok in re.findall(r"[a-z][a-z0-9_]*", lit):
            if tok in _ALL_MEMBERS:
                found.add(tok)
    found |= {"agent1", "user"}
    return found


def _placeholder(sort: str) -> str:
    return f"unused_{sort.lstrip('#')}"


def build_lean_sorts(obj_set: Set[str], n: int = 15) -> Tuple[str, Set[str]]:
    lines = [f"#const n = {n}.", "sorts",
             "#agent = {agent1}.",
             "#user  = {user}.",
             "#room  = {kitchen, living_room, bedroom, bathroom}."]

    allow: Set[str] = {"agent1", "user", "kitchen", "living_room", "bedroom", "bathroom"}

    for sort in REQUIRED_LEAVES:
        members = [m for m in SORT_MEMBERS[sort] if m in obj_set]
        for lit in REQUIRED_LITERALS.get(sort, []):
            if lit not in members:
                members.append(lit)
        if members:
            allow.update(members)
            lines.append(f"{sort} = {{{', '.join(members)}}}.")
        else:
            lines.append(f"{sort} = {{{_placeholder(sort)}}}.")
    lines.append(COMPOSITE_BLOCK)
    return "\n".join(lines), allow


def expand_with_kg(obj_set: Set[str], graph) -> Set[str]:
    from rdflib import Namespace
    EX = Namespace("http://example.org/")
    out = set(obj_set)
    for it in list(obj_set):
        for furn in (graph.value(EX[it], EX.on), graph.value(EX[it], EX.inside)):
            if furn is not None:
                out.add(str(furn).split("/")[-1])
    return out


def replace_sorts_block(asp_text: str, lean_sorts: str) -> str:
    start = "% ===== SORTS START ====="
    end = "% ===== SORTS END ====="
    block = f"{start}\n{lean_sorts}\n{end}"
    if start in asp_text and end in asp_text:
        return re.sub(rf"{re.escape(start)}.*?{re.escape(end)}", block, asp_text, flags=re.S)
    raise RuntimeError("SORTS markers not found in ASP file")


def filter_ic_lines(ic_text: str, allow: Set[str]) -> str:
    kept = []
    for line in ic_text.splitlines():
        toks = set(re.findall(r"[a-z][a-z0-9_]*", line.split("|", 1)[0]))
        objs = {t for t in toks if t in _ALL_MEMBERS}
        if objs <= allow:
            kept.append(line)
    return "\n".join(kept) + ("\n" if kept else "")


if __name__ == "__main__":
    sample = [
        "switched_on(lamp)            | switch on the lamp",
        "on(couch, book)              | put the book on the couch",
        "on(couch, computer)          | put the computer on the couch",
        "has(user, book)              | give the book to the user",
        "has(user, mug)               | give the mug of tea to the user",
        "has(user, computer)          | give the computer to the user",
        "has(user, cookie)            | give the cookie to the user",
        "switched_off(lamp)           | switch off the lamp",
    ]
    objs = clip_object_set(sample)
    print("clip objects:", sorted(objs))
    sorts, allow = build_lean_sorts(objs)
    print("\n--- lean sorts ---")
    print(sorts)
    print("\nallow-list (real objs in scene):", sorted(allow))
