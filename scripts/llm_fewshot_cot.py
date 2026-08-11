#!/usr/bin/env python3
import json, sys, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.utils import _ask_llm_json

ROOT = Path(__file__).resolve().parent.parent
USER = sys.argv[1] if len(sys.argv) > 1 else "user1_reader"
RUN  = sys.argv[2] if len(sys.argv) > 2 else "u1v2"
MODE = sys.argv[3] if len(sys.argv) > 3 else "fewshot"
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
LEVELS = ("l1", "l2", "l3", "l4")

HIST_PATH = ROOT / "experiments" / "multi_user_dataset" / f"{USER}_history.txt"
import os
WITH_CTX = os.environ.get("WITH_CURRENT_CLIP", "0").strip() == "1"
_tag = "HISTCTX" if WITH_CTX else "HISTONLY"
OUT_DIR = ROOT / "experiments" / "results" / RUN / f"LLM{_tag}_{MODE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_history(path):
    clips, titles = {}, {}
    cur, lines = None, []
    hdr = re.compile(r"%\s*(\d+)\s+(.*)")
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("["):
            m = hdr.search(s)
            if m:
                cur = int(m.group(1)); titles[cur] = m.group(2).strip(); lines = []
            continue
        if s.startswith("]"):
            if cur is not None:
                clips[cur] = lines
            cur, lines = None, []
            continue
        if "|" in s and cur is not None:
            nl = s.split("|", 1)[1].strip()
            lines.append(nl)
    return clips, titles


CLIPS, TITLES = parse_history(HIST_PATH)


def history_block(skip_group):
    out = []
    idx = 0
    for g in sorted(CLIPS):
        if g == skip_group:
            continue
        idx += 1
        out.append(f"[Past session {idx}]")
        for nl in CLIPS[g]:
            out.append(f"  {nl}")
    return "\n".join(out)


def context_block(group, line):
    prev = CLIPS.get(group, [])[: max(0, line - 1)]
    if not prev:
        return "  (session just started - no actions yet)"
    return "\n".join(f"  {nl}   <- already done" for nl in prev)


def build_prompt(group, line, cur_fuzzy, cands, cot):
    hist = history_block(group)
    cand_line = ", ".join(cands)
    if cot:
        task = (
            "Think step by step BEFORE answering:\n"
            "  (1) From the PAST SESSIONS, what does THIS person habitually do in a "
            "situation like the current one?\n"
            "  (2) Which single candidate matches that habit"
            + (" AND the current session" if WITH_CTX else "") + "?\n"
            'Reply with JSON only: {"reasoning":"<your steps>","choice":"<one candidate>"}'
        )
    else:
        task = ('Pick the SINGLE most likely object. '
                'Reply with JSON only: {"choice":"<one candidate>"}')
    ctx_section = ""
    if WITH_CTX:
        ctx_section = (
            "===== CURRENT SESSION (in progress - what has happened so far) =====\n"
            f"{context_block(group, line)}\n\n"
        )
    new_cmd_note = (
        "The person now gives this command in the current session (shown above). "
        if WITH_CTX else
        "The person now gives a NEW command in a fresh session. You do NOT get to see "
        "anything else happening in this new session - ONLY the command itself. "
    )
    return (
        "You are a home-assistant robot serving ONE specific person. Below is the COMPLETE "
        "log of this person's PAST sessions. Each line is an action performed for them, "
        "showing the EXACT object involved. Study their habits.\n\n"
        "===== PAST SESSIONS (full history of this user) =====\n"
        f"{hist}\n\n"
        f"{ctx_section}"
        "===== NEW COMMAND =====\n"
        f"{new_cmd_note}Resolve the ambiguous command.\n\n"
        f'Command: "{cur_fuzzy}"\n'
        f"Candidates (in no particular order): {cand_line}\n\n" + task
    )


def load_level(level):
    p = ROOT / f"experiments/results/{RUN}/{USER}_{level}__FULL/test_results.json"
    return json.loads(p.read_text())


def _match_candidate(raw, cands):
    s = str(raw).strip().strip("'\"").lower()
    if not s:
        return None
    norm = {c.lower(): c for c in cands}
    if s in norm:
        return norm[s]
    s2 = re.sub(r"^(the|a|an)\s+", "", s)
    if s2 in norm:
        return norm[s2]
    hits = [c for c in cands if re.search(rf"\b{re.escape(c.lower())}\b", s)]
    if len(hits) == 1:
        return hits[0]
    return None


def resolve_row(r, cot):
    gold = r.get("gold_obj")
    cands = r.get("obj_candidates") or r.get("furn_candidates") or []
    if not gold or gold in ("__OBJ__", "__FURN__", None) or not cands:
        return None
    group = r.get("group"); line = r.get("line")
    cur_fuzzy = r.get("fuzzy", "").split("|")[-1].strip()
    cands = [c for c in cands if not (isinstance(c, str) and c.startswith("__") and c.endswith("__"))]
    if not cands:
        return None
    cands_neutral = sorted(cands)
    prompt = build_prompt(group, line, cur_fuzzy, cands_neutral, cot)
    data = _ask_llm_json(prompt) or {}
    matched = _match_candidate(data.get("choice", ""), cands_neutral)
    invalid = matched is None
    pred = matched if matched is not None else "<invalid>"
    return {
        "group": group, "line": line, "fuzzy": cur_fuzzy,
        "gold": gold, "pred": pred, "success": (pred == gold),
        "invalid": invalid, "ncand": len(cands), "reasoning": data.get("reasoning"),
    }


def main():
    cot = (MODE == "cot")
    summary = {}
    for level in LEVELS:
        rows = load_level(level)
        jobs = [r for r in rows
                if r.get("gold_obj") and r.get("gold_obj") not in ("__OBJ__", "__FURN__", None)
                and (r.get("obj_candidates") or r.get("furn_candidates"))]
        recs = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(resolve_row, r, cot): i for i, r in enumerate(jobs)}
            for f in as_completed(futs):
                recs[futs[f]] = f.result()
        recs = [x for x in recs if x]
        correct = sum(1 for x in recs if x["success"])
        invalid = sum(1 for x in recs if x.get("invalid"))
        total = len(recs)
        acc = correct / total if total else 0.0
        summary[level] = acc
        (OUT_DIR / f"{USER}_{level}.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2))
        print(f"{USER} {MODE} {level}: acc={acc:.3f} ({correct}/{total}) invalid={invalid}", flush=True)
    print(f"\n{USER} {MODE} summary: " +
          " ".join(f"{l}={summary[l]:.3f}" for l in LEVELS), flush=True)


if __name__ == "__main__":
    main()
