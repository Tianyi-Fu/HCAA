from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rdflib import BNode, Graph, URIRef

from config.config import PROJECT_ROOT
from config.config import SHOW_CHANGED_HOLDS_NAME_OUTPUT, SHOW_CHANGED_HOLDS_NAME_OUTPUT_USER
from commands.context_io import parse_changed_names_file


ROOT = Path(PROJECT_ROOT)
GOLD_CACHE_ROOT = ROOT / "experiments" / "gold_state_cache"
CONCEPT_DIR = ROOT / "data" / "concepts"
OBJ2CONCEPTS_PATH = CONCEPT_DIR / "object_to_concepts.json"
DOMAIN_OBJECTS_PATH = CONCEPT_DIR / "domain_objects.json"
BACKUP_TTL_PATH = ROOT / "kg" / "living_room_backup.ttl"

_CHANGED_NAMES_CANDIDATES = [
    Path(SHOW_CHANGED_HOLDS_NAME_OUTPUT_USER),
    Path(SHOW_CHANGED_HOLDS_NAME_OUTPUT),
]
_FURNITURE_TYPES = {"OnFurniture", "InsideFurniture", "Light"}

_L2_TRIGGER_INCLUDE_FURNITURE = os.environ.get(
    "L2_TRIGGER_INCLUDE_FURNITURE", "1"
).strip().lower() in {"1", "true", "yes"}


def _tail(uri: URIRef | str) -> str:
    s = str(uri)
    return s.split("/")[-1] if "/" in s else s


@lru_cache(maxsize=4)
def _load_obj2concepts(path: str) -> Dict[str, List[str]]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _load_domain_entity_ids(path: str) -> Set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(x.get("object_id", "")).strip() for x in data.get("objects", []) if x.get("object_id")}


@lru_cache(maxsize=4)
def _load_object_type_map(path: str) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for x in data.get("objects", []):
        oid = str(x.get("object_id", "")).strip()
        typ = str(x.get("object_type", "")).strip()
        if oid:
            out[oid] = typ
    return out


def _diff_entities(prev_ttl: Path, cur_ttl: Path, entity_ids: Set[str]) -> Set[str]:
    if not prev_ttl.exists() or not cur_ttl.exists():
        return set()
    only_old, only_new = _diff_pair(str(prev_ttl), str(cur_ttl))

    out: Set[str] = set()
    for triples in (only_old, only_new):
        for s, _, o in triples:
            if isinstance(s, URIRef):
                name = _tail(s)
                if name in entity_ids:
                    out.add(name)
            if isinstance(o, URIRef):
                name = _tail(o)
                if name in entity_ids:
                    out.add(name)
    return out


@lru_cache(maxsize=2048)
def _load_graph_triples(path: str):
    g = Graph()
    g.parse(path, format="turtle")
    triples = set()
    for s, p, o in g.triples((None, None, None)):
        if isinstance(s, BNode) or isinstance(o, BNode):
            continue
        triples.add((s, p, o))
    return triples


@lru_cache(maxsize=4096)
def _diff_pair(prev_path: str, cur_path: str):
    g0 = _load_graph_triples(prev_path)
    g1 = _load_graph_triples(cur_path)
    only_old = g0 - g1
    only_new = g1 - g0
    return only_old, only_new


def _line_ttl(cache_key: str, group: str | int, line_idx: int) -> Path:
    return GOLD_CACHE_ROOT / cache_key / f"group_{group}" / f"line_{line_idx}" / "living_room.ttl"


def current_changed_entities_from_cache(
    cache_key: str,
    group: str | int,
    line_idx: int,
) -> List[str]:
    line_idx = int(line_idx)
    if line_idx <= 0:
        return []
    entity_ids = _load_domain_entity_ids(str(DOMAIN_OBJECTS_PATH))
    if line_idx == 1:
        prev_ttl = BACKUP_TTL_PATH
    else:
        prev_ttl = _line_ttl(cache_key, group, line_idx - 1)
    cur_ttl = _line_ttl(cache_key, group, line_idx)
    names = sorted(_diff_entities(prev_ttl, cur_ttl, entity_ids))
    return names


def prev_changed_entities_from_cache(
    cache_key: str,
    group: str | int,
    line_idx: int,
) -> List[str]:
    line_idx = int(line_idx)
    if line_idx <= 1:
        return []

    entity_ids = _load_domain_entity_ids(str(DOMAIN_OBJECTS_PATH))
    if line_idx == 2:
        prev_ttl = BACKUP_TTL_PATH
        cur_ttl = _line_ttl(cache_key, group, 1)
    else:
        prev_ttl = _line_ttl(cache_key, group, line_idx - 2)
        cur_ttl = _line_ttl(cache_key, group, line_idx - 1)

    names = sorted(_diff_entities(prev_ttl, cur_ttl, entity_ids))
    return names


def names_to_concepts(names: List[str], *, include_furniture: bool = False) -> List[str]:
    obj2concepts = _load_obj2concepts(str(OBJ2CONCEPTS_PATH))
    type_map = _load_object_type_map(str(DOMAIN_OBJECTS_PATH))
    out: Set[str] = set()
    for n in names:
        if not include_furniture and type_map.get(n, "") in _FURNITURE_TYPES:
            continue
        for c in obj2concepts.get(n, []):
            out.add(c)
    return sorted(out)


def prev_changed_concepts_from_cache(
    cache_key: str,
    group: str | int,
    line_idx: int,
) -> List[str]:
    return names_to_concepts(
        prev_changed_entities_from_cache(cache_key, group, line_idx),
        include_furniture=_L2_TRIGGER_INCLUDE_FURNITURE,
    )


def current_changed_concepts_from_cache(
    cache_key: str,
    group: str | int,
    line_idx: int,
) -> List[str]:
    return names_to_concepts(
        current_changed_entities_from_cache(cache_key, group, line_idx),
        include_furniture=_L2_TRIGGER_INCLUDE_FURNITURE,
    )


def prev_changed_concepts_current(changed_names_file: Optional[str] = None) -> List[str]:
    names: List[str] = []
    if changed_names_file:
        p = Path(changed_names_file)
        if p.exists() and p.stat().st_size > 0:
            names = parse_changed_names_file(str(p))
    else:
        for p in _CHANGED_NAMES_CANDIDATES:
            if p.exists() and p.stat().st_size > 0:
                names = parse_changed_names_file(str(p))
                if names:
                    break
    if not names:
        return []
    return names_to_concepts(names, include_furniture=False)


def reset_changed_concepts_cache() -> None:
    _load_obj2concepts.cache_clear()
    _load_domain_entity_ids.cache_clear()
    _load_object_type_map.cache_clear()
    _load_graph_triples.cache_clear()
    _diff_pair.cache_clear()


__all__ = [
    "current_changed_entities_from_cache",
    "current_changed_concepts_from_cache",
    "prev_changed_entities_from_cache",
    "prev_changed_concepts_from_cache",
    "prev_changed_concepts_current",
    "names_to_concepts",
    "reset_changed_concepts_cache",
]
