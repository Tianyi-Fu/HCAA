from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple, Set

from nltk.corpus import wordnet as wn


def wn_similarity(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0

    syn_a = wn.synsets(a)
    syn_b = wn.synsets(b)
    if not syn_a or not syn_b:
        return 0.0

    best = 0.0
    for s1 in syn_a:
        for s2 in syn_b:
            sim = s1.wup_similarity(s2)
            if sim and sim > best:
                best = sim
    return float(best)


_last_cn_source = "none"


@lru_cache(maxsize=1)


@lru_cache(maxsize=4096)
def cn_similarity(a: str, b: str, timeout: float = 0.5) -> float:
    a = (a or "").strip().lower().replace(" ", "_")
    b = (b or "").strip().lower().replace(" ", "_")
    if not a or not b:
        global _last_cn_source
        _last_cn_source = "none"
        return 0.0

    val = wn_similarity(a, b)
    _last_cn_source = "wordnet"
    return float(val or 0.0)


def get_last_cn_source() -> str:
    return _last_cn_source


def word_forms(verb: str, particle: str | None = None) -> Set[str]:
    verb = (verb or "").strip().lower()
    out: Set[str] = set()
    if not verb:
        return out

    base = verb
    out.add(base)

    out.add(base + "s")
    out.add(base + "ed")
    out.add(base + "ing")

    if particle:
        p = particle.strip().lower()
        if p:
            out.add(f"{base}_{p}")
            out.add(f"{base}s_{p}")
            out.add(f"{base}ed_{p}")
            out.add(f"{base}ing_{p}")

    return out


def top_k_scores(word: str, cands: List[str], k: int = 3) -> List[Tuple[str, float]]:
    word = (word or "").strip().lower()
    scored: List[Tuple[str, float]] = []
    for c in cands:
        s = wn_similarity(word, c)
        scored.append((c, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
