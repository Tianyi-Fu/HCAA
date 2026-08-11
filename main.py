from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional, Union
import re, math, os, json, time, random
from datetime import datetime
from pathlib import Path

from config.config import (
    LIVINGROOM_BACKUP_TTL,
    PROJECT_ROOT,
    HISTORY_FILE,
    DISAMBIG_MODE,
    COMPARE_LLM_HISTORY,
    FORCE_NO_ASK,
    CONCEPT_MATCH_MODE,
    LLM_USE_RECENT_HISTORY,
    LLM_PERSONA_MODE,
    LLM_PERSONA_TOPK,
)

from kg.history_manager import load_user_history, get_active_history_file
from kg.history_analyzer import (
    extract_asp_command as hx_extract,
    extract_keywords as hx_keywords,
    filter_valid_history_entries,
)

from rdflib import Namespace, RDF, RDFS, Literal
from kg.loader import load_kg, get_graph

from commands.ambiguity_checker import analyse_command_en
from commands.command_planner import plan
from commands.context_io import last_context
from commands.context_tokens import extract_context_tokens_current
from commands.confidence_scorer import ConfidenceScorer, dominant_context
from commands.kg_utils import list_inst
from commands.wordnet_utils import cn_similarity, get_last_cn_source
from commands.feasible_filter import filter_candidates
from commands.history_prior import HistoryPrior
from commands.concept_history import get_concept_history_index
from commands.changed_concepts import (
    prev_changed_concepts_current,
    prev_changed_concepts_from_cache,
)
from experiments.generate_fuzzy_sets import (
    HOT_DRINK_ITEMS,
    HOT_FOOD_ITEMS,
    INSIDE_FURN,
    OBJ_LEAVES,
    OBJ_TOKENS,
    ON_FURN,
    SWITCH_FURN,
    OBJ_HYPERNYM_MAP,
    PARENT_MAP_EXTRA,
)
try:
    from llm.utils import choose_instance_no_ctx, choose_instance_no_ctx_forced
except Exception:
    choose_instance_no_ctx = None
    choose_instance_no_ctx_forced = None

try:
    from config.config import USE_HISTORY_COUNTER
except Exception:
    USE_HISTORY_COUNTER = True

try:
    from config.config import COUNTER_USE_FULL_HISTORY
except Exception:
    COUNTER_USE_FULL_HISTORY = True


EX = Namespace("http://example.org/")

G = load_kg(LIVINGROOM_BACKUP_TTL)
scorer = ConfidenceScorer()
prior = HistoryPrior(HISTORY_FILE)

_MODE = str(DISAMBIG_MODE).lower()
LLM_ONLY_BASELINE = (_MODE == "llm_only")
LLM_FILTERED_BASELINE = (_MODE in {"llm_filtered", "filter_llm", "b_filter_llm"})
LLM_REPLACE_HISTORY = (_MODE == "llm_replace_history")
RANDOM_FILTERED_BASELINE = (_MODE in {"random_filtered", "b0_random_filtered"})
MOST_RECENT_FILTERED_BASELINE = (_MODE in {"most_recent_filtered", "b3_most_recent_filtered"})
MOST_FREQUENT_FILTERED_BASELINE = (_MODE in {"most_frequent_filtered", "b4_most_frequent_filtered"})
RANDOM_RAW_BASELINE = (_MODE in {"random_raw", "b0_random_raw"})
MOST_RECENT_RAW_BASELINE = (_MODE in {"most_recent_raw", "b3_most_recent_raw"})
MOST_FREQUENT_RAW_BASELINE = (_MODE in {"most_frequent_raw", "b4_most_frequent_raw"})
NO_HISTORY_BASELINE = (_MODE in {"no_history", "a1_no_history"})
THREE_FACTOR_ASK_ONLY = (_MODE in {"three_factor_ask_only", "three_factor_nofallback"})
NO_ASP_FILTER = os.environ.get("NO_ASP_FILTER", "0").strip().lower() in {"1", "true", "yes"}
DISABLE_HISTORY_COUNTER = os.environ.get("DISABLE_HISTORY_COUNTER", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
NO_STAGE2_FALLBACK = THREE_FACTOR_ASK_ONLY or DISABLE_HISTORY_COUNTER

BASELINE_RANDOM_SEED = int(os.environ.get("BASELINE_RANDOM_SEED", "42"))
_BASELINE_RNG = random.Random(BASELINE_RANDOM_SEED)

FORCE_NO_ASK_FALLBACK_POLICY = os.environ.get(
    "FORCE_NO_ASK_FALLBACK_POLICY", "fused_top1"
).strip().lower()
FORCE_NO_ASK_THEMATIC_MIN_TOP = float(os.environ.get("FORCE_NO_ASK_THEMATIC_MIN_TOP", "0.50"))
FORCE_NO_ASK_THEMATIC_MIN_GAP = float(os.environ.get("FORCE_NO_ASK_THEMATIC_MIN_GAP", "0.08"))

COUNTER_MATCH_SCOPE = os.environ.get("COUNTER_MATCH_SCOPE", "pred_furn").strip().lower()
if COUNTER_MATCH_SCOPE not in {"pred_furn", "pred_only", "global"}:
    COUNTER_MATCH_SCOPE = "pred_furn"

DISAMBIG_RUN_NAME = os.environ.get("DISAMBIG_RUN_NAME", "").strip() or "default_run"
DISAMBIG_DIR = Path(PROJECT_ROOT) / "experiments" / "results" / DISAMBIG_RUN_NAME
DISAMBIG_DIR.mkdir(parents=True, exist_ok=True)

HIST_JSONL = DISAMBIG_DIR / "group_history.jsonl"
HIST_TXT = DISAMBIG_DIR / "group_history.txt"

_RESULTS_JSONL = DISAMBIG_DIR / "disambig_results.jsonl"
_RESULTS_TXT = DISAMBIG_DIR / "disambig_results.txt"

CLEAR_LEAD_RATIO_BASE = 0.20
CLEAR_LEAD_RATIO_HYP = 0.15
CLEAR_LEAD_RATIO = CLEAR_LEAD_RATIO_BASE
W_SEM, W_THEM, W_SAL = 0.60, 0.30, 0.10
W_SEM_L1 = float(os.environ.get("W_SEM_L1", "0.50"))
W_THEM_L1 = float(os.environ.get("W_THEM_L1", "0.35"))
W_SAL_L1 = float(os.environ.get("W_SAL_L1", "0.15"))
W_SEM_L2 = float(os.environ.get("W_SEM_L2", "0.35"))
W_THEM_L2 = float(os.environ.get("W_THEM_L2", "0.50"))
W_SAL_L2 = float(os.environ.get("W_SAL_L2", "0.15"))
W_SEM_L3 = float(os.environ.get("W_SEM_L3", "0.40"))
W_THEM_L3 = float(os.environ.get("W_THEM_L3", "0.45"))
W_SAL_L3 = float(os.environ.get("W_SAL_L3", "0.15"))
W_SEM_L4 = float(os.environ.get("W_SEM_L4", str(W_SEM)))
W_THEM_L4 = float(os.environ.get("W_THEM_L4", str(W_THEM)))
W_SAL_L4 = float(os.environ.get("W_SAL_L4", str(W_SAL)))

UNWEIGHTED_THREE_FACTOR = os.environ.get("UNWEIGHTED_THREE_FACTOR", "1").lower() in {
    "1",
    "true",
    "yes",
}
FACTOR_DISTRIBUTION_NORMALIZE = os.environ.get("FACTOR_DISTRIBUTION_NORMALIZE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
FACTOR_DISTRIBUTION_ZERO_POLICY = os.environ.get("FACTOR_DISTRIBUTION_ZERO_POLICY", "uniform").strip().lower()

THEMATIC_MODE = os.environ.get("THEMATIC_MODE", "concept").strip().lower()
CONCEPT_NMIN = int(os.environ.get("CONCEPT_NMIN", "20"))
CONCEPT_ROOM_BACKOFF = float(os.environ.get("CONCEPT_ROOM_BACKOFF", "0.85"))
CONCEPT_PRED_BACKOFF = float(os.environ.get("CONCEPT_PRED_BACKOFF", "0.70"))


_LAST_SCORE_WEIGHTS_OBJ: Optional[tuple[float, float, float]] = None
_LAST_SCORE_WEIGHTS_FURN: Optional[tuple[float, float, float]] = None

def _run_level(run_name: str | None) -> Optional[int]:
    if not run_name:
        return None
    m = re.search(r"_l([1-4])(?=$|_)", run_name.lower())
    if not m:
        return None
    return int(m.group(1))


def _is_pronoun_sentence(sentence: str | None) -> bool:
    if not sentence:
        return False
    s = str(sentence).strip().lower().replace("_", " ")
    return re.search(r"\b(it|this|that|them|these|those|one|ones)\b", s) is not None


def _is_pronoun_run(run_name: str | None, sentence: str | None = None) -> bool:
    lname = (run_name or "").lower()
    if "pronoun" in lname:
        return True
    lv = _run_level(run_name)
    return lv == 4 and _is_pronoun_sentence(sentence)


def _weights_for_run(run_name: str | None, sentence: str | None = None) -> tuple[float, float, float]:
    level = _run_level(run_name)
    if level == 1:
        return W_SEM_L1, W_THEM_L1, W_SAL_L1
    if level == 2:
        return W_SEM_L2, W_THEM_L2, W_SAL_L2
    if level == 3:
        return W_SEM_L3, W_THEM_L3, W_SAL_L3
    if level == 4:
        return W_SEM_L4, W_THEM_L4, W_SAL_L4
    return W_SEM, W_THEM, W_SAL

def _lead_threshold(run_name: str | None) -> float:
    level = _run_level(run_name)
    if level == 1:
        return float(os.environ.get("CLEAR_LEAD_RATIO_L1", "0.30"))
    if level == 2:
        return float(os.environ.get("CLEAR_LEAD_RATIO_L2", "0.30"))
    if level == 3:
        return float(os.environ.get("CLEAR_LEAD_RATIO_L3", "0.32"))
    if level == 4:
        return float(os.environ.get("CLEAR_LEAD_RATIO_L4", str(CLEAR_LEAD_RATIO_BASE)))
    if run_name and "hypernym" in run_name.lower():
        return CLEAR_LEAD_RATIO_HYP
    return CLEAR_LEAD_RATIO_BASE


def _dynamic_lead_threshold(base: float, n_candidates: int) -> float:
    if n_candidates <= 2:
        return max(float(os.environ.get("LEAD_FLOOR_LE2", "0.03")), base - 0.08)
    if n_candidates <= 4:
        return max(float(os.environ.get("LEAD_FLOOR_LE4", "0.04")), base - 0.05)
    if n_candidates >= 25:
        return min(0.35, base + 0.08)
    if n_candidates >= 15:
        return min(0.30, base + 0.05)
    if n_candidates >= 10:
        return min(0.28, base + 0.03)
    return base


def _norm_symbol(x: str | None) -> str:
    return str(x or "").strip().lower().replace(" ", "_")

def _parse_level_set(raw: str) -> set[int]:
    out: set[int] = set()
    for x in str(raw or "").split(","):
        x = x.strip()
        if not x:
            continue
        try:
            v = int(x)
        except Exception:
            continue
        if 1 <= v <= 4:
            out.add(v)
    return out


def _normalize_candidate_distribution(
    keys: List[str],
    values: Dict[str, float],
    *,
    zero_policy: str = "uniform",
    eps: float = 1e-12,
) -> Dict[str, float]:
    if not keys:
        return {}
    nonneg: Dict[str, float] = {}
    for k in keys:
        try:
            v = float(values.get(k, 0.0))
        except Exception:
            v = 0.0
        nonneg[k] = max(0.0, v)
    z = sum(nonneg.values())
    if z <= eps:
        if str(zero_policy).strip().lower() == "zero":
            return {k: 0.0 for k in keys}
        u = 1.0 / max(1, len(keys))
        return {k: u for k in keys}
    return {k: (nonneg[k] / z) for k in keys}


def _prob_top_gap(keys: List[str], p_map: Dict[str, float]) -> tuple[float, float]:
    vals = sorted((float(p_map.get(k, 0.0)) for k in keys), reverse=True)
    if not vals:
        return 0.0, 0.0
    top = vals[0]
    second = vals[1] if len(vals) > 1 else 0.0
    return top, (top - second)
MAX_HISTORY_TO_PASS = 12

THEM_ALPHA = 0.5
THEM_BETA = 0.5
THEM_GAMMA = 0.2
THEM_EPS = 1.0

RISK_ENABLED = False

STAGE_NAMES = {
    1: "SCORE_FUSION",
    2: "HISTORY_COUNTER",
    3: "FALLBACK_TOP1",
}
_NOISY_TARGETS = {"hot_drink", "cold_drink", "hot_food", "snack", "item", "thing"}
_FURN_CLASS_TOKENS = {"inside_furniture", "on_furniture", "switch_furniture", "container_furniture", "furniture"}


def _is_noisy_target(name: Optional[str]) -> bool:
    if not name:
        return True
    s = str(name)
    return (s in _NOISY_TARGETS) or s.endswith("Context")


def _filter_noisy_candidates(cands: List[str]) -> List[str]:
    out: List[str] = []
    for c in cands:
        if not _is_noisy_target(c):
            out.append(c)
    return out


def _extract_target_for_persona(pred: str, subj: Optional[str], obj: Optional[str]) -> Optional[str]:
    if pred in {"inside", "on", "has"}:
        return obj
    if pred == "heated":
        return subj
    return obj or subj


def _recent_group_commands_for_llm(line_id: Optional[str]) -> List[str]:
    lines = _get_group_prev_lines_for_prompt(line_id)
    out: List[str] = []
    for s in lines:
        if "|" in s:
            nl = s.split("|", 1)[1].strip()
            if nl:
                out.append(nl)
        elif s.strip():
            out.append(s.strip())
    return out[-MAX_HISTORY_TO_PASS:]


def _build_user_persona_text(topk: int = 4) -> str:
    try:
        rows = list(load_user_history())
    except Exception:
        rows = []
    rows = filter_valid_history_entries(rows)
    if not rows:
        return ""

    pred_cnt: Dict[str, int] = {}
    tgt_cnt: Dict[str, int] = {}
    for row in rows:
        try:
            asp = row.split("|", 1)[0].strip() if "|" in row else row.strip()
            pred, subj, obj, _ = hx_keywords(asp)
        except Exception:
            continue
        if not pred:
            continue
        pred_cnt[pred] = pred_cnt.get(pred, 0) + 1
        tgt = _extract_target_for_persona(pred, subj, obj)
        if tgt:
            tgt_cnt[tgt] = tgt_cnt.get(tgt, 0) + 1

    if not pred_cnt and not tgt_cnt:
        return ""

    topk = max(1, int(topk))
    top_preds = [k for k, _ in sorted(pred_cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:topk]]
    top_tgts = [k for k, _ in sorted(tgt_cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:topk]]

    parts: List[str] = []
    if top_preds:
        parts.append("frequent actions: " + ", ".join(top_preds))
    if top_tgts:
        parts.append("frequently selected entities: " + ", ".join(top_tgts))
    if not parts:
        return ""
    return "; ".join(parts)


LLM_WORLD_STATE = os.environ.get("LLM_WORLD_STATE", "0").strip() == "1"
LLM_CROSS_HISTORY = os.environ.get("LLM_CROSS_HISTORY", "0").strip() == "1"
LLM_WORLD_STATE_CLIP_START = os.environ.get("LLM_WORLD_STATE_CLIP_START", "0").strip() == "1"
LLM_CLIP_START_KB_DIR = os.environ.get("LLM_CLIP_START_KB_DIR", "")
_CUR_PREV_CMD: Optional[str] = None


def _build_world_state_text(prev_cmd_asp: Optional[str], g_override=None) -> str:
    g = g_override if g_override is not None else get_graph()
    operated: set = set()
    if prev_cmd_asp:
        mm = re.findall(r"\(([^)]*)\)", prev_cmd_asp)
        if mm:
            operated = {a.strip() for a in mm[0].split(",") if a.strip()}

    def _b(s, p):
        v = next(g.objects(s, p), None)
        try:
            return isinstance(v, Literal) and bool(v.toPython())
        except Exception:
            return False

    lines: List[str] = []
    for item in g.subjects(RDF.type, EX.Item):
        name = _label(item)
        st: List[str] = []
        on = next(g.objects(item, EX.on), None)
        ins = next(g.objects(item, EX.inside), None)
        fil = next(g.objects(item, EX.filled), None)
        if on is not None:
            st.append(f"on {_label(on)}")
        if ins is not None:
            st.append(f"in {_label(ins)}")
        if (EX.user, EX.has, item) in g:
            st.append("held by user")
        if _b(item, EX.heated):
            st.append("heated")
        if fil is not None:
            st.append(f"filled from {_label(fil)}")
        mark = "  [just operated]" if name in operated else ""
        lines.append(f"- {name}: {', '.join(st) or 'in storage'}{mark}")
    for furn in g.subjects(RDF.type, EX.Furniture):
        name = _label(furn)
        st2: List[str] = []
        if _b(furn, EX.switched_on):
            st2.append("switched on")
        ov = next(g.objects(furn, EX.open), None)
        if isinstance(ov, Literal):
            try:
                st2.append("open" if bool(ov.toPython()) else "closed")
            except Exception:
                pass
        mark = "  [just operated]" if name in operated else ""
        if st2:
            lines.append(f"- {name} (furniture): {', '.join(st2)}{mark}")
    return "\n".join(sorted(lines))


def _llm_pick_no_ctx(
    sentence: str,
    cands: List[str],
    role: str,
    *,
    line_id: Optional[str] = None,
) -> Tuple[str, str, str]:
    if not cands:
        return "unknown", f"{role}: no-candidates", "unknown"
    recent_commands = _recent_group_commands_for_llm(line_id) if LLM_USE_RECENT_HISTORY else None
    if LLM_WORLD_STATE and LLM_WORLD_STATE_CLIP_START and LLM_CLIP_START_KB_DIR:
        grp, _ = _parse_line_id(line_id)
        try:
            from rdflib import Graph as _G
            _gcs = _G(); _gcs.parse(os.path.join(LLM_CLIP_START_KB_DIR, f"clip{grp}.ttl"), format="turtle")
            world_state_text = _build_world_state_text(None, g_override=_gcs)
        except Exception as _e:
            world_state_text = _build_world_state_text(_CUR_PREV_CMD)
    elif LLM_WORLD_STATE:
        world_state_text = _build_world_state_text(_CUR_PREV_CMD)
    else:
        world_state_text = ""
    if LLM_CROSS_HISTORY:
        try:
            g, _k = _parse_line_id(line_id)
        except Exception:
            g = None
        clips: List[List[str]] = []
        cur: Optional[List[str]] = None
        for _s in load_user_history():
            st = _s.strip()
            if st.startswith("["):
                cur = []
            elif st.startswith("]"):
                if cur is not None:
                    clips.append(cur); cur = None
            elif cur is not None and "|" in st:
                cur.append(st.split("|", 1)[1].strip())
        blocks: List[str] = []
        _sess = 0
        _llm_heldout = {x.strip() for x in os.environ.get("HELDOUT_GROUPS", "").split(",") if x.strip()}
        for _i, _clip in enumerate(clips, 1):
            if g is not None and str(_i) == str(g):
                continue
            if str(_i) in _llm_heldout:
                continue
            if not _clip:
                continue
            _sess += 1
            blocks.append(f"Session {_sess}:\n" + "\n".join(f"- {x}" for x in _clip))
        persona_text = "\n\n".join(blocks) if blocks else ""
    elif LLM_PERSONA_MODE and not LLM_WORLD_STATE:
        persona_text = _build_user_persona_text(LLM_PERSONA_TOPK)
    else:
        persona_text = ""
    if choose_instance_no_ctx_forced is None:
        if choose_instance_no_ctx is None:
            return cands[0], f"{role}: llm-unavailable", cands[0]
        kb_class = "object" if role == "obj" else "furniture"
        pick, reason = choose_instance_no_ctx(
            sentence,
            kb_class,
            cands,
            recent_commands=recent_commands,
            persona_text=persona_text,
            world_state_text=world_state_text,
        )
        forced = cands[0]
    else:
        kb_class = "object" if role == "obj" else "furniture"
        pick, forced, reason = choose_instance_no_ctx_forced(
            sentence,
            kb_class,
            cands,
            recent_commands=recent_commands,
            persona_text=persona_text,
            world_state_text=world_state_text,
        )
    if pick == "ASK_HUMAN":
        if FORCE_NO_ASK:
            return forced, f"{role}: force_no_ask ({reason})", forced
        return "ASK_HUMAN", f"{role}: {reason}", forced
    if pick not in cands:
        return forced, f"{role}: {reason} (fallback-first)", forced
    return pick, reason, forced


_SHOW_FILES = [
    "show_start_holds_output.txt",
    "show_last_holds_output.txt",
    "show_changed_holds_output.txt",
    "show_changed_holds_name_output.txt",
    "show_operated_holds_name_output.txt",
    "occurs_output.txt",
]

_GROUP_HDR_RE = re.compile(r'^\s*\[\s*(?:%\s*)?(\d+)\b')


def _load_history_excluding_group(exclude_group: Optional[str]) -> List[str]:
    try:
        lines = list(load_user_history())
    except Exception:
        return []

    kept: List[str] = []
    cur_group: Optional[str] = None

    if isinstance(exclude_group, (set, frozenset, list, tuple)):
        _excl_set = {str(x) for x in exclude_group}
    elif exclude_group is not None:
        _excl_set = {str(exclude_group)}
    else:
        _excl_set = set()

    for raw in lines:
        s = raw.strip()
        if not s:
            continue

        m = _GROUP_HDR_RE.match(s)
        if m:
            cur_group = m.group(1)
            continue
        if s == "]":
            cur_group = None
            continue

        if "|" in s:
            if cur_group is not None and cur_group in _excl_set:
                continue
            kept.append(s)

    if not kept and lines:
        return filter_valid_history_entries(lines)

    return kept


def _heldout_exclude_set(g_excl):
    excl = set()
    if g_excl is not None:
        excl.add(str(g_excl))
    h = os.environ.get("HELDOUT_GROUPS", "").strip()
    if h:
        excl |= {x.strip() for x in h.split(",") if x.strip()}
    return excl


def _pick_random_candidate(cands: List[str]) -> Optional[str]:
    if not cands:
        return None
    return _BASELINE_RNG.choice(list(cands))


def _extract_target_from_asp(asp_cmd: str, slot_type: str) -> Optional[str]:
    if not asp_cmd:
        return None
    try:
        p, s, o, _ = hx_keywords(asp_cmd)
    except Exception:
        return None
    if not p:
        return None
    if slot_type == "obj":
        if p in ("inside", "on", "has"):
            return o
        if p == "heated":
            return s
        return o or s
    return s or o


def _pick_most_recent_filtered(
    cands: List[str],
    slot_type: str,
    line_id: Optional[str],
) -> Optional[str]:
    if not cands:
        return None
    cand_set = set(cands)
    prev_lines = _get_group_prev_lines_for_prompt(line_id)
    for ln in reversed(prev_lines):
        asp = hx_extract(ln) or (ln.split("|", 1)[0].strip())
        target = _extract_target_from_asp(asp, slot_type)
        if target in cand_set:
            return target
    return None


def _best_by_count(counts: Dict[str, int]) -> Optional[str]:
    if not counts:
        return None
    max_v = max(counts.values()) if counts else 0
    if max_v <= 0:
        return None
    best = sorted([k for k, v in counts.items() if v == max_v])
    return best[0] if best else None


def _pick_most_frequent_filtered_obj(
    cands: List[str],
    pred: Optional[str],
    furn: Optional[str],
    *,
    exclude_group: Optional[str] = None,
) -> Optional[str]:
    if not cands:
        return None
    raw = _load_history_excluding_group(exclude_group)
    hist = filter_valid_history_entries(raw)
    cnt_pf: Dict[str, int] = {c: 0 for c in cands}
    cnt_p: Dict[str, int] = {c: 0 for c in cands}
    pred = (pred or "").strip()
    furn = (furn or "").strip() if furn else None

    scope = COUNTER_MATCH_SCOPE

    for line in hist:
        asp = hx_extract(line)
        if not asp:
            continue
        p2, s2, o2, _ = hx_keywords(asp)
        if not p2 or not o2:
            continue
        if _is_noisy_target(o2):
            continue
        if scope in {"pred_furn", "pred_only"}:
            if pred and p2 != pred:
                continue

        if o2 in cnt_p:
            cnt_p[o2] += 1

        if scope == "pred_furn" and pred in ("inside", "on") and furn:
            if s2 == furn and o2 in cnt_pf:
                cnt_pf[o2] += 1

    pick = None
    if scope == "pred_furn" and pred in ("inside", "on") and furn:
        pick = _best_by_count(cnt_pf)
    if not pick:
        pick = _best_by_count(cnt_p)
    return pick


def _pick_most_frequent_filtered_furn(
    cands: List[str],
    pred: Optional[str],
    *,
    exclude_group: Optional[str] = None,
) -> Optional[str]:
    if not cands:
        return None
    raw = _load_history_excluding_group(exclude_group)
    hist = filter_valid_history_entries(raw)
    cnt_p: Dict[str, int] = {c: 0 for c in cands}
    pred = (pred or "").strip()
    scope = "global" if COUNTER_MATCH_SCOPE == "global" else "pred_only"

    for line in hist:
        asp = hx_extract(line)
        if not asp:
            continue
        p2, s2, o2, _ = hx_keywords(asp)
        if not p2:
            continue
        if scope == "pred_only" and pred and p2 != pred:
            continue
        target = s2 or o2
        if target in cnt_p:
            cnt_p[target] += 1

    return _best_by_count(cnt_p)

def _hier_thematic_scores_furn(
    cands: List[str],
    pred: Optional[str],
    *,
    exclude_group: Optional[str] = None,
) -> Dict[str, float]:
    pred = (pred or "").strip()
    raw = _load_history_excluding_group(exclude_group)
    hist = filter_valid_history_entries(raw)

    cnt_p: Dict[str, Dict[str, int]] = {}
    cnt_all: Dict[str, int] = {}

    if not hist or not cands:
        return {c: 1.0 / max(1, len(cands)) for c in cands}

    for line in hist:
        asp = hx_extract(line)
        if not asp:
            continue
        try:
            p2, s2, o2, _ = hx_keywords(asp)
        except Exception:
            continue
        if not p2 or not s2:
            continue

        if p2 not in ("open", "closed", "switched_on", "switched_off", "inside", "on", "has"):
            continue

        tgt = s2
        if _is_noisy_target(tgt):
            continue

        cnt_p.setdefault(p2, {}).setdefault(tgt, 0)
        cnt_p[p2][tgt] += 1

        cnt_all.setdefault(tgt, 0)
        cnt_all[tgt] += 1

    raw_scores: Dict[str, float] = {}
    for c in cands:
        n_p = cnt_p.get(pred, {}).get(c, 0)
        n_all = cnt_all.get(c, 0)
        score = THEM_ALPHA * n_p + THEM_GAMMA * n_all + THEM_EPS
        raw_scores[c] = max(float(score), 1e-12)

    Z = sum(raw_scores.values()) or 1.0
    prob = {c: raw_scores[c] / Z for c in cands}

    print("[THEM-FURN] raw:", {k: round(v, 3) for k, v in raw_scores.items()})
    print("[THEM-FURN] prob:", {k: round(v, 3) for k, v in prob.items()})
    return prob


def _hier_thematic_scores(
    cands: List[str],
    pred: Optional[str],
    furn: Optional[str],
    *,
    exclude_group: Optional[str] = None,
) -> Dict[str, float]:
    pred = (pred or "").strip()
    furn = (furn or "").strip() if furn else None

    raw = _load_history_excluding_group(exclude_group)
    hist = filter_valid_history_entries(raw)

    cnt_pf: Dict[Tuple[str, Optional[str]], Dict[str, int]] = {}
    cnt_p: Dict[str, Dict[str, int]] = {}
    cnt_f: Dict[Optional[str], Dict[str, int]] = {}
    cnt_all: Dict[str, int] = {}

    for line in hist:
        asp = hx_extract(line)
        if not asp:
            continue
        p2, s2, o2, _ = hx_keywords(asp)
        if not p2 or not o2:
            continue
        if _is_noisy_target(o2):
            continue

        key_pf = (p2, s2 if s2 else None)
        cnt_pf.setdefault(key_pf, {}).setdefault(o2, 0)
        cnt_pf[key_pf][o2] += 1

        cnt_p.setdefault(p2, {}).setdefault(o2, 0)
        cnt_p[p2][o2] += 1

        kf = s2 if s2 else None
        cnt_f.setdefault(kf, {}).setdefault(o2, 0)
        cnt_f[kf][o2] += 1

        cnt_all.setdefault(o2, 0)
        cnt_all[o2] += 1

    if pred in ("inside", "on"):
        pf_map = cnt_pf.get((pred, furn), {})
    else:
        pf_map = cnt_p.get(pred, {})

    raw_scores: Dict[str, float] = {}
    for c in cands:
        n_pf = pf_map.get(c, 0)
        n_p = cnt_p.get(pred, {}).get(c, 0)
        n_f = cnt_f.get(furn if furn else None, {}).get(c, 0) if pred in ("inside", "on") else 0
        n_all = cnt_all.get(c, 0)
        score = n_pf + THEM_ALPHA * n_p + THEM_BETA * n_f + THEM_GAMMA * n_all + THEM_EPS
        raw_scores[c] = max(score, 1e-12)

    Z = sum(raw_scores.values()) or 1.0
    prob = {c: raw_scores[c] / Z for c in cands}

    print("[THEM] raw:", {k: round(v, 3) for k, v in raw_scores.items()})
    print("[THEM] prob:", {k: round(v, 3) for k, v in prob.items()})
    return prob


def _score_furniture_ctx_only(
    ctx_word: str,
    pred: Optional[str],
    cands: List[str],
    line_id: Optional[str],
    sentence: Optional[str] = None,
) -> List[tuple[str, float, float, float, float]]:
    if not cands:
        return []

    cands = sorted(_filter_noisy_candidates(list(dict.fromkeys(cands))))

    g_excl, _ = _parse_line_id(line_id)
    them_map = _hier_thematic_scores_furn(cands, pred, exclude_group=g_excl)

    local_lines = _get_group_prev_lines_for_prompt(line_id)
    sal_map = _compute_salience(cands, local_lines)

    sem_map, sem_src_map = _collect_semantic_scores(cands, ctx_word)
    if FACTOR_DISTRIBUTION_NORMALIZE:
        zp = FACTOR_DISTRIBUTION_ZERO_POLICY
        sem_map = _normalize_candidate_distribution(cands, sem_map, zero_policy=zp)
        them_map = _normalize_candidate_distribution(cands, them_map, zero_policy=zp)
        sal_map = _normalize_candidate_distribution(cands, sal_map, zero_policy=zp)
    if UNWEIGHTED_THREE_FACTOR and not NO_HISTORY_BASELINE:
        w_sem, w_them, w_sal = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    else:
        w_sem, w_them, w_sal = _weights_for_run(DISAMBIG_RUN_NAME, sentence)
        if NO_HISTORY_BASELINE:
            w_them = 0.0
            total = w_sem + w_sal
            if total > 0:
                scale = 1.0 / total
                w_sem *= scale
                w_sal *= scale
    global _LAST_SCORE_WEIGHTS_FURN
    _LAST_SCORE_WEIGHTS_FURN = (w_sem, w_them, w_sal)

    out: List[tuple[str, float, float, float, float]] = []
    for c in cands:
        s_sem = sem_map.get(c, 0.0)
        s_them = them_map.get(c, 0.0)
        s_sal = sal_map.get(c, 0.0)
        weighted = w_sem * s_sem + w_them * s_them + w_sal * s_sal
        sem_src = sem_src_map.get(c, "none")
        print(f"[SEM-FURN] cand={c:20s} head={_head(c):12s} ctx={ctx_word:10s} sim={s_sem:.3f} src={sem_src}")
        out.append((c, s_sem, s_them, s_sal, weighted))

    out.sort(key=lambda x: (-x[4], x[0]))
    print(
        f"[SCORE-FURN] furniture score (semantic/thematic/salience -> "
        f"weighted[{w_sem:.2f}/{w_them:.2f}/{w_sal:.2f}]): "
    )
    for (c, s1, s2, s3, m) in out:
        print(f" - {c:20s} sem(ctx)={s1:.3f} them={s2:.3f} sal={s3:.3f} weighted={m:.3f}")
    return out


def _uses_subject_as_furniture(pred: str) -> bool:
    return pred in {"inside", "on"}


def _target_of_next(pred_now: str, n_s: str | None, n_o: str | None) -> str | None:
    if pred_now in ("inside", "on", "has"):
        return n_o
    elif pred_now == "heated":
        return n_s
    else:
        return n_o or n_s


def decide_by_prev_command_counter(
    *,
    prev_cmd: str | None,
    pred_now: str,
    furn_now: str | None,
    candidates: list[str],
    exclude_group: Optional[str] = None,
) -> tuple[str | None, Dict[str, int], Dict[str, Any]]:
    g = get_graph()
    furn_uri = EX[furn_now] if furn_now else None
    user_has = set()
    for obj in g.objects(EX.user, EX.has):
        try:
            user_has.add(str(obj).split("/")[-1])
        except Exception:
            continue

    try:
        raw = list(load_user_history()) if COUNTER_USE_FULL_HISTORY else _load_history_excluding_group(exclude_group)
    except Exception:
        raw = []

    hist = filter_valid_history_entries(raw)

    if not candidates or not hist:
        return None, {c: 0 for c in candidates}, {
            "mode": "none",
            "total": 0,
            "t1": 0,
            "t2": 0,
            "ratio": 0.0,
        }

    prev_pred = None
    prev_subj = None
    if prev_cmd:
        try:
            p0, s0, _o0, _ = hx_keywords(prev_cmd)
            prev_pred, prev_subj = p0, s0
        except Exception:
            prev_pred, prev_subj = None, None

    def _prev_similar(curr_asp: str) -> bool:
        if not prev_cmd:
            return False
        try:
            p1, s1, _o1, _ = hx_keywords(curr_asp)
        except Exception:
            return False
        if not p1 or not prev_pred:
            return False
        if p1 != prev_pred:
            return False
        if _uses_subject_as_furniture(p1) and prev_subj and s1 and prev_subj != s1:
            return False
        return True

    transitions: List[Tuple[str, str, Optional[str]]] = []
    for i in range(len(hist) - 1):
        curr_asp = hx_extract(hist[i])
        next_asp = hx_extract(hist[i + 1])
        if not curr_asp or not next_asp:
            continue

        n_pred, n_s, n_o, _ = hx_keywords(next_asp)
        if not n_pred or n_pred != pred_now:
            continue

        tgt = _target_of_next(n_pred, n_s, n_o)
        if not tgt or tgt not in candidates or _is_noisy_target(tgt):
            continue

        transitions.append((curr_asp.strip(), tgt, n_s))

    if not transitions:
        return None, {c: 0 for c in candidates}, {
            "mode": "none",
            "total": 0,
            "t1": 0,
            "t2": 0,
            "ratio": 0.0,
        }

    def _filter_L1(rows: List[Tuple[str, str, Optional[str]]]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for curr_asp, tgt, n_s in rows:
            if _uses_subject_as_furniture(pred_now) and furn_now and n_s != furn_now:
                continue
            if furn_uri and (EX[tgt], EX.on if pred_now == "on" else EX.inside, furn_uri) in g:
                continue
            if pred_now in {"give", "has"} and tgt in user_has:
                continue
            if _prev_similar(curr_asp):
                out.append((curr_asp, tgt))
        return out

    def _filter_L2(rows: List[Tuple[str, str, Optional[str]]]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for curr_asp, tgt, n_s in rows:
            if _uses_subject_as_furniture(pred_now) and furn_now and n_s != furn_now:
                continue
            if furn_uri and (EX[tgt], EX.on if pred_now == "on" else EX.inside, furn_uri) in g:
                continue
            if pred_now in {"give", "has"} and tgt in user_has:
                continue
            out.append((curr_asp, tgt))
        return out

    def _filter_L3(rows: List[Tuple[str, str, Optional[str]]]) -> List[Tuple[str, str]]:
        return [(c, t) for (c, t, _ns) in rows]

    def _count_and_pick(pairs: List[Tuple[str, str]]) -> Tuple[Optional[str], Dict[str, int], Dict[str, Any]]:
        cnt = {c: 0 for c in candidates}
        for _c, tgt in pairs:
            if tgt in cnt:
                cnt[tgt] += 1

        ranked = sorted(candidates, key=lambda c: (-cnt[c], c))
        pick = ranked[0] if pairs else None

        counts_sorted = sorted(cnt.values(), reverse=True)
        t1 = counts_sorted[0] if counts_sorted else 0
        t2 = counts_sorted[1] if len(counts_sorted) > 1 else 0
        total = sum(cnt.values())
        ratio = (t1 / max(1, total)) if total > 0 else 0.0

        return pick, cnt, {
            "total": total,
            "t1": t1,
            "t2": t2,
            "ratio": ratio,
        }

    L1_pairs = _filter_L1(transitions)
    if L1_pairs:
        pick, cnt, stat = _count_and_pick(L1_pairs)
        stat.update({"mode": "action+furniture+prev"})
        return pick, cnt, stat

    L2_pairs = _filter_L2(transitions)
    if L2_pairs:
        pick, cnt, stat = _count_and_pick(L2_pairs)
        stat.update({"mode": "action+furniture"})
        return pick, cnt, stat

    L3_pairs = _filter_L3(transitions)
    if L3_pairs:
        _pick, cnt, stat = _count_and_pick(L3_pairs)
        stat.update({"mode": "action-only"})
        return None, cnt, stat

    return None, {c: 0 for c in candidates}, {
        "mode": "none",
        "total": 0,
        "t1": 0,
        "t2": 0,
        "ratio": 0.0,
    }


_GROUP_HISTORY: Dict[str, List[Tuple[int, str]]] = {}
_ID_RE = re.compile(r"\[\s*(\d+)\s*-\s*(\d+)\s*\]")

_SESSION_ID = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
_LAST_GROUP_ID: Optional[str] = None
_GROUPS_RESET_THIS_SESSION: set[str] = set()

RESET_LOG_FILES_ON_START = True
RESET_ON_GROUP_SWITCH = True
APPLY_RESETS_FROM_PREV_SESSIONS = False

_INITIALIZED = False


def _safe_truncate_file(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("")


def _truncate_logs_on_start():
    if RESET_LOG_FILES_ON_START:
        _safe_truncate_file(HIST_JSONL)
        _safe_truncate_file(HIST_TXT)
    _GROUP_HISTORY.clear()
    _GROUPS_RESET_THIS_SESSION.clear()
    global _LAST_GROUP_ID
    _LAST_GROUP_ID = None


def _persist_event(event: dict) -> None:
    event = dict(event)
    event.setdefault("ts", int(time.time()))
    event.setdefault("session", _SESSION_ID)

    HIST_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(HIST_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    if event.get("type") == "session_start":
        with open(HIST_TXT, "a", encoding="utf-8") as f:
            f.write(
                f"\n==== SESSION START {event['session']} @ "
                f"{datetime.fromtimestamp(event['ts']).isoformat()} ====\n"
            )
    elif event.get("type") == "group_reset":
        with open(HIST_TXT, "a", encoding="utf-8") as f:
            f.write(
                f"---- RESET group {event.get('group')} "
                f"(session {event.get('session')}) ----\n"
            )
    elif event.get("type") == "entry":
        with open(HIST_TXT, "a", encoding="utf-8") as f:
            f.write(
                f"[{event.get('group')}-{event.get('idx')}] "
                f"{event.get('line')}\n"
            )


def _load_group_history_from_disk() -> None:
    _GROUP_HISTORY.clear()
    if not HIST_JSONL.exists():
        return

    try:
        with open(HIST_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue

                ev_sess = str(ev.get("session", ""))
                ev_type = ev.get("type")

                if ev_type == "group_reset":
                    if APPLY_RESETS_FROM_PREV_SESSIONS or ev_sess == _SESSION_ID:
                        g = str(ev.get("group", "0"))
                        _GROUP_HISTORY[g] = []
                elif ev_type == "entry":
                    g = str(ev.get("group", "0"))
                    idx = int(ev.get("idx", 1))
                    s = str(ev.get("line", ""))
                    lst = _GROUP_HISTORY.setdefault(g, [])
                    lst.append((idx, s))
    except Exception:
        pass


def _ensure_session_start_marked() -> None:
    _persist_event({"type": "session_start"})


def _ensure_init():
    global _INITIALIZED
    if _INITIALIZED:
        return
    _truncate_logs_on_start()
    _ensure_session_start_marked()
    _load_group_history_from_disk()
    _INITIALIZED = True
    print(f"[INIT] logs reset={RESET_LOG_FILES_ON_START} @ {HIST_JSONL} / {HIST_TXT}")


def _parse_line_id(line_id: Optional[str]) -> Tuple[str, int]:
    if not line_id:
        return "0", 10**9
    m = _ID_RE.search(line_id)
    if not m:
        return "0", 10**9
    return m.group(1), int(m.group(2))


def _group_id_from_any(group_or_line_id: Optional[Union[str, int]]) -> str:
    if group_or_line_id is None:
        return "0"
    if isinstance(group_or_line_id, int):
        return str(group_or_line_id)
    s = str(group_or_line_id).strip()
    m = _ID_RE.search(s)
    if m:
        return m.group(1)
    m2 = re.match(r"^\[\s*(\d+)\s*\]$", s)
    if m2:
        return m2.group(1)
    return s


def _fmt_hist_line(asp_cmd: str, user_sentence: str) -> str:
    return f"{asp_cmd:<30s} | {user_sentence}"


def _get_group_prev_lines_for_prompt(line_id: Optional[str]) -> List[str]:
    g, k = _parse_line_id(line_id)
    rows = sorted(_GROUP_HISTORY.get(g, []), key=lambda x: x[0])
    prev = [line for (idx, line) in rows if idx < k]
    return prev[-MAX_HISTORY_TO_PASS:] if prev else []


def _mark_group_reset(g: str) -> None:
    if g in _GROUPS_RESET_THIS_SESSION:
        return
    _GROUPS_RESET_THIS_SESSION.add(g)
    _GROUP_HISTORY[g] = []
    _persist_event({"type": "group_reset", "group": g})


def _append_group_history(line_id: Optional[str], asp_cmd: str, user_sentence: str) -> None:
    global _LAST_GROUP_ID

    g, k = _parse_line_id(line_id)

    if RESET_ON_GROUP_SWITCH and (_LAST_GROUP_ID is not None) and (g != _LAST_GROUP_ID):
        _mark_group_reset(g)

    lst = _GROUP_HISTORY.setdefault(g, [])
    idx = k if k != 10**9 else (max((i for i, _ in lst), default=0) + 1) or 1

    lst = [row for row in lst if row[0] != idx]
    line = _fmt_hist_line(asp_cmd, user_sentence)
    lst.append((idx, line))
    lst.sort(key=lambda x: x[0])
    _GROUP_HISTORY[g] = lst

    _persist_event({"type": "entry", "group": g, "idx": idx, "line": line})
    _LAST_GROUP_ID = g


def log_command_external(line_id: Optional[str], asp_cmd: str, user_sentence: str) -> None:
    _ensure_init()
    _append_group_history(line_id, asp_cmd, user_sentence)


def add_group_history(line_id: Optional[str], asp_cmd: str, user_sentence: str) -> None:
    _ensure_init()
    _append_group_history(line_id, asp_cmd, user_sentence)


def clear_group_history(group_or_line_id: Optional[Union[str, int]]) -> None:
    _ensure_init()
    g = _group_id_from_any(group_or_line_id)
    _mark_group_reset(g)


def replace_group_history(group_or_line_id: Optional[Union[str, int]], *args) -> int:
    _ensure_init()
    g = _group_id_from_any(group_or_line_id)

    if len(args) == 2 and all(isinstance(x, str) for x in args):
        asp_cmd, nl = args
        rows = _GROUP_HISTORY.setdefault(g, [])
        next_idx = (max((idx for idx, _ in rows), default=0) + 1) or 1
        line = _fmt_hist_line(asp_cmd, nl)
        rows.append((next_idx, line))
        _GROUP_HISTORY[g] = rows
        _persist_event({"type": "entry", "group": g, "idx": next_idx, "line": line})
        return len(rows)

    if len(args) == 1:
        lines = args[0]
        _mark_group_reset(g)
        new_rows: List[Tuple[int, str]] = []
        if lines:
            for i, entry in enumerate(lines, start=1):
                if isinstance(entry, tuple) and len(entry) == 2:
                    idx, line = entry
                    try:
                        idx_int = int(idx)
                    except Exception:
                        idx_int = i
                    new_rows.append((idx_int, str(line)))
                else:
                    new_rows.append((i, str(entry)))
        _GROUP_HISTORY[g] = new_rows
        for idx, line in new_rows:
            _persist_event({"type": "entry", "group": g, "idx": idx, "line": line})
        return len(new_rows)

    raise TypeError("replace_group_history() expects (g, lines) or (g, asp_cmd, nl)")


def get_group_history(group_or_line_id: Optional[Union[str, int]]) -> List[str]:
    _ensure_init()
    g = _group_id_from_any(group_or_line_id)
    rows = sorted(_GROUP_HISTORY.get(g, []), key=lambda x: x[0])
    return [line for _, line in rows]


_SLOT_RE = re.compile(r"__([A-Z]+)__")


def _template_slots(tmpl: str) -> List[str]:
    return _SLOT_RE.findall(tmpl or "")


def _label(node) -> str:
    s = str(node)
    return s.split("/")[-1] if "/" in s else s


def _instances_of(ex_cls) -> list[str]:
    g = get_graph()
    return sorted({_label(s) for s in g.subjects(RDF.type, ex_cls)})


def _extract_pred_furn_from_template(asp_template: str) -> tuple[str | None, str | None]:
    m = re.match(r"\s*([a-zA-Z_][\w]*)\s*\(([^)]*)\)", asp_template or "")
    if not m:
        return None, None
    pred = m.group(1)
    args = [a.strip() for a in m.group(2).split(",")]
    furn = None
    if args and "__" not in args[0]:
        furn = args[0]
    return pred, furn


def _head(name: str) -> str:
    return (name or "").split("_")[-1].lower()


def residual_ambiguity_from_scores(means: List[float]) -> float:
    if not means:
        return 0.0
    n = len(means)
    if n == 1:
        return 0.0
    mu = sum(means) / n
    var = sum((x - mu) ** 2 for x in means) / n
    sigma = math.sqrt(var) if var > 0 else 1e-8
    z = [(x - mu) / sigma for x in means]
    ez = [math.exp(v) for v in z]
    s = sum(ez) or 1e-12
    p = [v / s for v in ez]
    import math as _m
    H = -sum(pi * _m.log(pi + 1e-12) for pi in p)
    return H / _m.log(n)


def _lead_ratio(scored: List[tuple[str, float, float, float, float]]) -> tuple[float, float, float, str, str]:
    if not scored:
        return 0.0, 0.0, 0.0, "", ""
    if len(scored) == 1:
        return float("inf"), scored[0][4], 0.0, scored[0][0], ""
    m1 = scored[0][4]
    m2 = scored[1][4]
    name1 = scored[0][0]
    name2 = scored[1][0]
    ratio = (m1 - m2) / max(m2, 1e-9)
    return ratio, m1, m2, name1, name2


def _context_base_word(ctx_label: str) -> str:
    if not ctx_label:
        return ""
    s = re.sub(r"[_\s]*Context$", "", ctx_label.strip())
    m = re.match(r"([A-Za-z]+)", s)
    base = (m.group(1) if m else s).lower()
    return base


def _semantic_ctx_score_with_source(cand: str, ctx_word: str) -> tuple[float, str]:
    if not ctx_word:
        return 0.0, "none"
    return (float(cn_similarity(_head(cand), ctx_word) or 0.0), get_last_cn_source())


def _semantic_ctx_score(cand: str, ctx_word: str) -> float:
    return _semantic_ctx_score_with_source(cand, ctx_word)[0]


def _collect_semantic_scores(
    cands: List[str], ctx_word: str | None
) -> tuple[Dict[str, float], Dict[str, str]]:
    sem_map: Dict[str, float] = {}
    src_map: Dict[str, str] = {}
    if not ctx_word:
        for c in cands:
            sem_map[c] = 0.0
            src_map[c] = "none"
        return sem_map, src_map
    for c in cands:
        sem_map[c], src_map[c] = _semantic_ctx_score_with_source(c, ctx_word)
    return sem_map, src_map


def _salience_from_group_asp(
    cands: List[str],
    local_lines: List[str],
    decay: float = 0.9,
    eps: float = 1e-9,
) -> Dict[str, float]:
    if not cands or not local_lines:
        return {c: 0.0 for c in cands}

    cand_set = set(cands)
    scores: Dict[str, float] = {c: 0.0 for c in cands}

    lines = list(local_lines)[-MAX_HISTORY_TO_PASS:]
    lines_rev = list(reversed(lines))

    t = 0
    for ln in lines_rev:
        t += 1
        w = decay ** (t - 1)
        asp_part = (ln.split("|", 1)[0] or "").strip()
        if not asp_part:
            continue
        try:
            p, s, o, _ = hx_keywords(asp_part)
        except Exception:
            continue
        if not p:
            continue

        target = None
        if p in ("inside", "on", "has"):
            target = o
        elif p in ("heated", "filled"):
            target = s
        else:
            target = o or s

        if target and (target in cand_set):
            scores[target] += w

    total = sum(scores.values())
    if total <= eps:
        return {c: 0.0 for c in cands}
    return {c: (scores[c] / total) for c in cands}


def _compute_salience(cands: List[str], local_lines: List[str]) -> Dict[str, float]:
    return _salience_from_group_asp(cands, local_lines)


def _ctx_uri_from_word(ctx_word: str | None):
    if not ctx_word:
        return None
    parts = [p.capitalize() for p in str(ctx_word).strip().split("_") if p]
    if not parts:
        return None
    return EX["".join(parts) + "Context"]


def _ctx_weight_factor(name: str, ctx_word: str | None) -> float:
    uri = _ctx_uri_from_word(ctx_word)
    if not uri:
        return 1.0
    g = get_graph()
    subj = EX[name]
    try:
        for bn in g.objects(subj, EX.hasContextWeight):
            ctx = next(g.objects(bn, EX.context), None)
            if ctx != uri:
                continue
            val = next(g.objects(bn, EX.importanceWeight), None)
            if isinstance(val, Literal):
                try:
                    num = float(val.toPython())
                    norm = max(0.0, min(num, 1.0))
                    return 0.7 + 0.6 * norm
                except Exception:
                    continue
    except Exception:
        return 1.0
    return 1.0


def _score_candidates_ctx_only(
    ctx_word: str,
    pred: str | None,
    furn: str | None,
    cands: List[str],
    line_id: Optional[str],
    sentence: Optional[str] = None,
) -> List[tuple[str, float, float, float, float]]:
    cands = sorted(_filter_noisy_candidates(list(cands)))

    sem_map, sem_src_map = _collect_semantic_scores(cands, ctx_word)
    if UNWEIGHTED_THREE_FACTOR and not NO_HISTORY_BASELINE:
        w_sem, w_them, w_sal = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    else:
        w_sem, w_them, w_sal = _weights_for_run(DISAMBIG_RUN_NAME, sentence)
        if NO_HISTORY_BASELINE:
            w_them = 0.0
            total = w_sem + w_sal
            if total > 0:
                scale = 1.0 / total
                w_sem *= scale
                w_sal *= scale
    global _LAST_SCORE_WEIGHTS_OBJ
    _LAST_SCORE_WEIGHTS_OBJ = (w_sem, w_them, w_sal)

    g_excl, line_no = _parse_line_id(line_id)
    if THEMATIC_MODE == "concept":
        ctx_tokens = extract_context_tokens_current(exclude_items_on_furn=True)
        prev_changed_cs: List[str] = []
        prev_changed_src = "none"
        if line_no > 1:
            try:
                cache_key = Path(get_active_history_file()).stem
                cache_prev = prev_changed_concepts_from_cache(cache_key, g_excl, line_no)
            except Exception:
                cache_prev = []
            if cache_prev:
                prev_changed_cs = cache_prev
                prev_changed_src = "cache"
            else:
                prev_changed_cs = prev_changed_concepts_current()
                prev_changed_src = "runtime"
        concept_idx = get_concept_history_index(
            nmin=CONCEPT_NMIN,
            room_backoff=CONCEPT_ROOM_BACKOFF,
            pred_backoff=CONCEPT_PRED_BACKOFF,
        )
        _excl_groups = {str(g_excl)}
        _heldout_env = os.environ.get("HELDOUT_GROUPS", "").strip()
        if _heldout_env:
            _excl_groups |= {x.strip() for x in _heldout_env.split(",") if x.strip()}
        concept_them_map = concept_idx.score_candidates(
            pred,
            furn,
            ctx_tokens,
            cands,
            exclude_group=_excl_groups,
            prev_changed_concepts=prev_changed_cs,
            level=_run_level(DISAMBIG_RUN_NAME),
        )
        them_map = concept_them_map
        dbg = concept_idx.last_debug()
        print("[THEM] mode=concept")
        print(f"[THEM] prev_changed_source={prev_changed_src}")
        if prev_changed_cs:
            print(f"[THEM] prev_changed_concepts={prev_changed_cs[:12]}")
        if isinstance(dbg, dict):
            sm = dbg.get("sthem_mode")
            if sm:
                print(f"[THEM] sthem_mode={sm}")
            mm = dbg.get("match_mode")
            if mm:
                print(f"[THEM] match_mode={mm}")
            em = dbg.get("env_mode")
            if em:
                print(f"[THEM] env_mode={em}")
            tk = dbg.get("concept_topk")
            if tk is not None:
                print(f"[THEM] concept_topk={tk}")
            ws = dbg.get("weights", {})
            if ws:
                print(f"[THEM] channel_weights: action={ws.get('action')} env={ws.get('env')}")
        print("[THEM] prob(concept):", {k: round(v, 3) for k, v in concept_them_map.items()})
    else:
        them_map = _hier_thematic_scores(cands, pred, furn, exclude_group=_heldout_exclude_set(g_excl))

    local_lines = _get_group_prev_lines_for_prompt(line_id)
    sal_map = _compute_salience(cands, local_lines)
    if FACTOR_DISTRIBUTION_NORMALIZE:
        zp = FACTOR_DISTRIBUTION_ZERO_POLICY
        sem_map = _normalize_candidate_distribution(cands, sem_map, zero_policy=zp)
        them_map = _normalize_candidate_distribution(cands, them_map, zero_policy=zp)
        sal_map = _normalize_candidate_distribution(cands, sal_map, zero_policy=zp)

    if os.environ.get("SEM_CONF_GATE", "") == "1" and them_map and w_them > 0:
        _top_them = max(them_map, key=them_map.get)
        if _top_them not in _reinforced_objects():
            w_sem = 0.0
            _tot = w_sem + w_them + w_sal
            if _tot > 0:
                w_sem, w_them, w_sal = w_sem / _tot, w_them / _tot, w_sal / _tot
            print(f"[SEM_GATE] them-top '{_top_them}' unseen -> drop sem; w_them={w_them:.2f} w_sal={w_sal:.2f}")
    _LAST_SCORE_WEIGHTS_OBJ = (w_sem, w_them, w_sal)

    out: List[tuple[str, float, float, float, float]] = []
    for c in cands:
        s_sem = sem_map.get(c, 0.0)
        sem_src = sem_src_map.get(c, "none")
        s_them = them_map.get(c, 0.0)
        s_sal = sal_map.get(c, 0.0)
        weighted = w_sem * s_sem + w_them * s_them + w_sal * s_sal
        print(f"[SEM] cand={c:20s} head={_head(c):12s} ctx={ctx_word:10s} sim={s_sem:.3f} src={sem_src}")
        out.append((c, s_sem, s_them, s_sal, weighted))
    out.sort(key=lambda x: (-x[4], x[0]))
    return out


def _now_iso() -> str:
    try:
        return datetime.now().isoformat(timespec="seconds")
    except Exception:
        return datetime.now().isoformat()


def _scores_to_list(scored: List[tuple[str, float, float, float, float]]) -> List[dict]:
    out = []
    for name, s_sem, s_them, s_sal, mean in scored:
        out.append(
            {
                "name": name,
                "semantic": round(float(s_sem), 6),
                "thematic": round(float(s_them), 6),
                "salience": round(float(s_sal), 6),
                "mean": round(float(mean), 6),
            }
        )
    return out


def _write_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_txt_block(path: Path, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(block + ("\n" if not block.endswith("\n") else ""))


def _fmt_top(scores: List[dict], k: int = 10) -> str:
    rows = []
    for i, d in enumerate(scores[:k], start=1):
        rows.append(
            f" {i:>2}. {d['name']:<24} "
            f"sem={d['semantic']:.3f} them={d['thematic']:.3f} "
            f"sal={d['salience']:.3f} mean={d['mean']:.3f}"
        )
    return "\n".join(rows) if rows else " (no candidates)"


def log_disambig_result(
    *,
    line_id: Optional[str],
    sentence: str,
    prev_cmd: Optional[str],
    prev_nl: Optional[str],
    asp_template: Optional[str],
    asp_cmd: str,
    context_label: str,
    ctx_word: str,
    pred: Optional[str],
    furniture: Optional[str],
    decision: str,
    decision_reason: str,
    confidence: float,
    residual_ambiguity: float,
    lead_ratio: float,
    top1: str,
    top2: str,
    m1: float,
    m2: float,
    scored_tuples: List[tuple[str, float, float, float, float]],
    history_prior: Optional[Dict[str, float]] = None,
    wilson_lbs: Optional[Dict[str, float]] = None,
    steps: int = 0,
    touched: Optional[List[str]] = None,
    disambig_stage: int = 0,
    disambig_steps: int = 0,
    stage_trace: Optional[List[Dict[str, Any]]] = None,
    would_ask: bool = False,
    compare: Optional[Dict[str, Any]] = None,
) -> None:
    scores_list = _scores_to_list(scored_tuples)
    hist_map = {k: float(v) for k, v in (history_prior or {}).items()}
    wilson_map = {k: float(v) for k, v in (wilson_lbs or {}).items()} if wilson_lbs else {}

    stage_trace = stage_trace or []

    entry = {
        "ts": _now_iso(),
        "line_id": line_id,
        "sentence": sentence,
        "prev_cmd": prev_cmd,
        "prev_nl": prev_nl,
        "asp_template": asp_template,
        "asp_cmd": asp_cmd,
        "context_label": context_label,
        "ctx_word": ctx_word,
        "predicate": pred,
        "furniture": furniture,
        "decision": decision,
        "decision_reason": decision_reason,
        "confidence": round(float(confidence), 6),
        "residual_ambiguity": round(float(residual_ambiguity), 6),
        "lead_ratio": (None if lead_ratio == float("inf") else round(float(lead_ratio), 6)),
        "top1": {"name": top1, "mean": round(float(m1), 6)},
        "top2": {"name": top2, "mean": round(float(m2), 6)},
        "scores": scores_list,
        "history_prior": hist_map,
        "wilson_lbs": wilson_map,
        "steps": int(steps or 0),
        "touched": list(touched or []),
        "disambig_stage": disambig_stage,
        "disambig_steps": disambig_steps,
        "stage_trace": stage_trace,
        "would_ask": bool(would_ask),
    }
    if compare:
        entry["compare"] = compare

    _write_jsonl(_RESULTS_JSONL, entry)

    path_str = (
        " -> ".join(
            f"{x.get('name', '?')}{'Y' if x.get('passed') else 'N'}" for x in stage_trace
        )
        or "(no stages)"
    )

    header = (
        f"\n===== DISAMBIG RESULT @ {entry['ts']} =====\n"
        f"[line] {line_id or '(None)'}\n"
        f"[nl] {sentence}\n"
        f"[prev] {prev_nl or '(None)'}\n"
        f"[tmpl] {asp_template}\n"
        f"[final] {asp_cmd}\n"
        f"[ctx] {context_label} (ctx_word={ctx_word})\n"
        f"[pred] {pred} [furn] {furniture}\n"
        f"[decide] {decision} | {decision_reason}\n"
        f"[conf] {entry['confidence']:.3f} "
        f"[residual_H] {entry['residual_ambiguity']:.3f}\n"
        f"[lead] ratio={entry['lead_ratio'] if entry['lead_ratio'] is not None else 'inf'} "
        f"top1={top1}({m1:.3f}) vs top2={top2 or '-'}({m2:.3f})\n"
        f"[steps] {steps} [touched] {', '.join(touched or []) or '(none)'}\n"
        f"[stage] disambig_stage={disambig_stage} "
        f"disambig_steps={disambig_steps} path={path_str}\n"
        f"[ask] would_ask={would_ask}\n"
        f"[scores]\n{_fmt_top(scores_list, k=10)}\n"
    )
    if compare:
        header += (
            f"[compare] stage1_failed={compare.get('stage1_failed')}\n"
            f"[compare] history_pick={compare.get('history_pick')} "
            f"mode={compare.get('history_mode')} "
            f"asp={compare.get('asp_history')}\n"
            f"[compare] llm_pick={compare.get('llm_pick')} "
            f"asp={compare.get('asp_llm')} "
            f"reason={compare.get('llm_reason')}\n"
        )

    if hist_map:
        header += f"[history_prior] { {k: round(v, 3) for k, v in hist_map.items()} }\n"
    if wilson_map:
        header += f"[wilson_lbs] { {k: round(v, 3) for k, v in wilson_map.items()} }\n"

    _write_txt_block(_RESULTS_TXT, header)


def _pick_furniture_by_history(cands: List[str], pred: Optional[str], *, exclude_group: Optional[str] = None) -> Optional[str]:
    if not cands:
        return None

    raw = _load_history_excluding_group(exclude_group)
    hist = filter_valid_history_entries(raw)

    freq: Dict[str, int] = {c: 0 for c in cands}
    for line in hist:
        asp = hx_extract(line)
        if not asp:
            continue
        p2, s2, _o2, _ = hx_keywords(asp)
        if not p2 or not s2:
            continue
        if pred and p2 != pred:
            continue
        if s2 in freq:
            freq[s2] += 1

    scored = sorted(((c, freq.get(c, 0)) for c in cands), key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else None

_REINFORCED_OBJ_CACHE: Dict[tuple, set] = {}


def _reinforced_objects() -> set:
    try:
        hist_path = str(get_active_history_file())
    except Exception:
        return set()
    heldout = os.environ.get("HELDOUT_GROUPS", "").strip()
    key = (hist_path, heldout)
    if key in _REINFORCED_OBJ_CACHE:
        return _REINFORCED_OBJ_CACHE[key]
    excl = {x.strip() for x in heldout.split(",") if x.strip()}
    seen: set = set()
    try:
        txt = Path(hist_path).read_text()
    except Exception:
        _REINFORCED_OBJ_CACHE[key] = seen
        return seen
    for b in re.split(r"(?=\[\s*%\s*\d+)", txt):
        if not b.strip():
            continue
        m = re.match(r"\[\s*%\s*(\d+)", b)
        if m and m.group(1) in excl:
            continue
        for line in b.splitlines():
            if "|" not in line:
                continue
            am = re.search(r"\(([^)]*)\)", line.split("|", 1)[0])
            if not am:
                continue
            for tok in am.group(1).split(","):
                tok = tok.strip()
                if tok and tok != "user":
                    seen.add(tok)
    _REINFORCED_OBJ_CACHE[key] = seen
    return seen


HISTORY_PRIOR_WEIGHT = 0.25


def pipeline(
    sentence: str,
    prev_cmd: str | None,
    prev_command_nl: str | None = None,
    asp_template: str | None = None,
    *,
    line_id: Optional[str] = None,
    run_planner: bool = False,
):
    _ensure_init()
    global _CUR_PREV_CMD
    _CUR_PREV_CMD = prev_cmd

    sys_ctx = (last_context() or "").strip()
    ctx_word = _context_base_word(sys_ctx)

    print(f"[CTX] prev_nl = {prev_command_nl if (prev_command_nl and prev_command_nl.strip()) else '(None)'}")
    print(f"[CTX] context = {sys_ctx if sys_ctx else '(None)'}")
    print(f"[CTX] ctx_word = {ctx_word if ctx_word else '(None)'}")
    print(f"[CFG] disambig_mode = {DISAMBIG_MODE}")
    if MOST_FREQUENT_FILTERED_BASELINE or MOST_FREQUENT_RAW_BASELINE:
        print(f"[CFG] counter_match_scope = {COUNTER_MATCH_SCOPE}")
    if THEMATIC_MODE == "concept":
        print(f"[CFG] concept_match_mode = {CONCEPT_MATCH_MODE}")

    is_clear = True

    if not asp_template:
        if run_planner:
            plan_res = plan(sentence)
            detail = {"steps": plan_res.get("steps", []), "confidence": 0.0}
            touched = plan_res.get("touched", [])
        else:
            _append_group_history(line_id, sentence, sentence)
            detail = {"steps": [], "confidence": 0.0}
            touched = []
        return sentence, 0.0, detail, touched, True

    info = analyse_command_en(sentence)
    if "error" in info:
        print("", info["error"])
        touched, steps = [], []
        if run_planner:
            plan_res = plan(asp_template)
            touched = plan_res.get("touched", [])
            steps = plan_res.get("steps", [])
        else:
            _append_group_history(line_id, asp_template, sentence)
        return asp_template, 0.0, {"steps": steps}, touched, True

    obj_class = info.get("obj_class")

    slots = _template_slots(asp_template)
    need_obj = "OBJ" in slots
    need_furn = "FURN" in slots

    pred, furn_in_tmpl = _extract_pred_furn_from_template(asp_template or "")
    furn = info.get("furniture") or furn_in_tmpl

    disambig_slot = None
    if need_obj and not need_furn:
        disambig_slot = "obj"
    elif need_furn and not need_obj:
        disambig_slot = "furn"
    elif need_obj and need_furn:
        disambig_slot = info.get("ambiguous_slot") or "obj"

    print(f"[CTX] disambig_slot = {disambig_slot or '(unknown)'}")

    if LLM_ONLY_BASELINE:
        cand_objs: List[str] = []
        cand_furn: List[str] = []

        if need_obj:
            cand_objs = list(info.get("members", []) or [])
            if not cand_objs and info.get("obj_status") == "instance" and info.get("object") not in {None, "__OBJ__"}:
                cand_objs = [info["object"]]
            if not cand_objs and info.get("object") and info.get("object") != "__OBJ__":
                try:
                    cand_objs = list_inst(info["object"])
                except Exception:
                    cand_objs = _instances_of(EX.Item)
            if not cand_objs:
                cand_objs = _instances_of(EX.Item)
            cand_objs = _dedup_preserve_order(cand_objs)
            if _is_pronoun_run(DISAMBIG_RUN_NAME, sentence):
                cand_objs = _dedup_preserve_order(cand_objs + _instances_of(EX.Furniture))

        if need_furn:
            cand_furn = list(info.get("furn_members", []) or [])
            if not cand_furn and info.get("furniture") and info.get("furniture") != "__FURN__":
                try:
                    cand_furn = list_inst(info["furniture"])
                except Exception:
                    cand_furn = _instances_of(EX.Furniture)
            if not cand_furn:
                cand_furn = _instances_of(EX.Furniture)
            cand_furn = _dedup_preserve_order(cand_furn)
            if _is_pronoun_run(DISAMBIG_RUN_NAME, sentence) and not need_obj:
                cand_furn = _dedup_preserve_order(cand_furn + _instances_of(EX.Item))

        obj_pick: Optional[str] = None
        furn_pick_llm: Optional[str] = furn or furn_in_tmpl
        obj_forced: Optional[str] = None
        furn_forced: Optional[str] = furn or furn_in_tmpl
        obj_reason = ""
        furn_reason = "explicit"

        if need_obj:
            obj_pick, obj_reason, obj_forced = _llm_pick_no_ctx(sentence, cand_objs, "obj", line_id=line_id)
            print(f"[LLM] obj_pick={obj_pick} reason={obj_reason}")

        if need_furn:
            if furn_pick_llm and furn_pick_llm not in {"__FURN__"} and furn_pick_llm not in _FURN_CLASS_TOKENS:
                furn_reason = "explicit"
            else:
                furn_pick_llm, furn_reason, furn_forced = _llm_pick_no_ctx(sentence, cand_furn, "furn", line_id=line_id)
            print(f"[LLM] furn_pick={furn_pick_llm} reason={furn_reason}")

        llm_ask = (obj_pick == "ASK_HUMAN") or (furn_pick_llm == "ASK_HUMAN")

        filled = asp_template
        if not llm_ask:
            if need_obj and obj_pick:
                filled = filled.replace("__OBJ__", obj_pick)
            if need_furn and furn_pick_llm:
                filled = filled.replace("__FURN__", furn_pick_llm)

        if need_obj and not obj_forced:
            if cand_objs:
                obj_forced = cand_objs[0]
            elif obj_pick and obj_pick != "ASK_HUMAN":
                obj_forced = obj_pick
            else:
                obj_forced = ""
        if need_furn and not furn_forced:
            if furn_pick_llm and furn_pick_llm != "ASK_HUMAN":
                furn_forced = furn_pick_llm
            elif cand_furn:
                furn_forced = cand_furn[0]
            else:
                furn_forced = ""

        fallback_filled = asp_template
        if need_obj:
            fallback_filled = fallback_filled.replace("__OBJ__", obj_forced or "")
        if need_furn:
            fallback_filled = fallback_filled.replace("__FURN__", furn_forced or "")

        parts = []
        if need_obj:
            parts.append(f"obj={obj_pick} ({obj_reason})")
        if need_furn:
            parts.append(f"furn={furn_pick_llm} ({furn_reason})")
        decision_reason = "; ".join(parts) if parts else "llm_only"
        decision = "ASK_HUMAN" if llm_ask else "LLM_ONLY"

        try:
            slot_cands = cand_objs if disambig_slot == "obj" else cand_furn
            scored_stub = [(c, 0.0, 0.0, 0.0, 0.0) for c in slot_cands]
            log_disambig_result(
                line_id=line_id,
                sentence=sentence,
                prev_cmd=prev_cmd,
                prev_nl=prev_command_nl,
                asp_template=asp_template,
                asp_cmd=filled,
                context_label=sys_ctx,
                ctx_word=ctx_word,
                pred=pred,
                furniture=furn_pick_llm,
                decision=decision,
                decision_reason=decision_reason,
                confidence=0.0,
                residual_ambiguity=0.0,
                lead_ratio=0.0,
                top1=(obj_pick or furn_pick_llm or ""),
                top2="",
                m1=0.0,
                m2=0.0,
                scored_tuples=scored_stub,
                history_prior={},
                wilson_lbs={},
                steps=0,
                touched=[],
                disambig_stage=0,
                disambig_steps=0,
                stage_trace=[],
                would_ask=llm_ask,
            )
        except Exception as _e:
            print("[WARN] llm_only logger failed:", _e)

        detail = {
            "decision": decision,
            "obj_reason": obj_reason,
            "furn_reason": furn_reason,
            "decision_reason": decision_reason,
            "fallback_prediction": (fallback_filled if llm_ask else filled),
        }
        _append_group_history(line_id, filled, sentence)
        return filled, 0.0, detail, [], (not llm_ask)

    cand_objs: List[str] = []
    cand_objs_raw: List[str] = []
    cand_objs_filtered: List[str] = []
    forced_ask_obj = False
    if need_obj:
        cand_objs = list(info.get("members", []) or [])
        if not cand_objs and info.get("obj_status") == "instance" and info.get("object") not in {None, "__OBJ__"}:
            cand_objs = [info["object"]]
        elif not cand_objs and info.get("object"):
            try:
                cand_objs = list_inst(info["object"])
            except Exception:
                cand_objs = _instances_of(EX.Item)
        if not cand_objs:
            cand_objs = _instances_of(EX.Item)
        cand_objs = _dedup_preserve_order(cand_objs)
        if _is_pronoun_run(DISAMBIG_RUN_NAME, sentence):
            cand_objs = _dedup_preserve_order(cand_objs + _instances_of(EX.Furniture))
        cand_objs_raw = list(cand_objs)

        if not NO_ASP_FILTER:
            before_type = len(cand_objs)
            cand_objs = _filter_obj_by_pred_types(pred or "", cand_objs)
            if len(cand_objs) != before_type:
                print(f"[FILTER] predicate type constraint: {before_type} -> {len(cand_objs)} (pred={pred})")

            if pred == "filled" and os.environ.get("FILL_FIX", "") == "1":
                _fm = re.match(r"\s*filled\s*\(([^)]*)\)", asp_template or "")
                if _fm:
                    _fargs = [a.strip() for a in _fm.group(1).split(",")]
                    if len(_fargs) >= 2 and _fargs[1]:
                        _before_src = len(cand_objs)
                        cand_objs = [c for c in cand_objs if c != _fargs[1]]
                        if len(cand_objs) != _before_src:
                            print(f"[FILTER] FILL_FIX excluded source {_fargs[1]}: {_before_src} -> {len(cand_objs)}")

            before = list(cand_objs)
            cand_objs = filter_candidates(pred or "", furn, cand_objs)
            print(f"[FILTER] feasibility filter: {len(before)} -> {len(cand_objs)} (pred={pred}, furn={furn})")
        else:
            print("[FILTER] NO_ASP_FILTER=1 -> skip predicate-type & ASP feasibility filter")

        cand_objs = _filter_noisy_candidates(cand_objs)
        print(f"[FILTER] candidates after denoise: {len(cand_objs)}")

        before_dedup = len(cand_objs)
        cand_objs = _dedup_preserve_order(cand_objs)
        if len(cand_objs) != before_dedup:
            print(f"[DEDUP] candidate dedup: {before_dedup} -> {len(cand_objs)}")

        cand_objs_filtered = list(cand_objs)

        if not cand_objs:
            forced_ask_obj = True
            print("[FILTER] empty candidate set -> ASK_HUMAN (no fallback to the unfiltered set)")
    furn_pick = info.get("furniture") or furn_in_tmpl
    cand_furn: List[str] = []
    cand_furn_raw: List[str] = []
    cand_furn_filtered: List[str] = []
    if need_furn:
        g_excl, _ = _parse_line_id(line_id)
        cand_furn = list(info.get("furn_members", []) or [])
        if not cand_furn and info.get("furniture"):
            try:
                cand_furn = list_inst(info["furniture"])
            except Exception:
                cand_furn = _instances_of(EX.Furniture)
        if not cand_furn:
            cand_furn = _instances_of(EX.Furniture)
        cand_furn = _dedup_preserve_order(cand_furn)
        if _is_pronoun_run(DISAMBIG_RUN_NAME, sentence) and not need_obj:
            cand_furn = _dedup_preserve_order(cand_furn + _instances_of(EX.Item))
        cand_furn_raw = list(cand_furn)

        def _is_allowed_furn(pred: str, furn: str) -> bool:
            if pred == "inside":
                if furn in {"inside_furniture"}:
                    return True
                return furn in INSIDE_FURN
            if pred == "on":
                if furn in {"on_furniture"}:
                    return True
                return furn in ON_FURN
            if pred in {"open", "close"}:
                if furn in {"inside_furniture"}:
                    return True
                return furn in INSIDE_FURN
            if pred in {"switched_on", "switched_off"}:
                if furn in {"switch_furniture"}:
                    return True
                return furn in SWITCH_FURN
            return True

        if not NO_ASP_FILTER and pred in {"inside", "on", "open", "close", "switched_on", "switched_off"}:
            cand_before = len(cand_furn)
            cand_furn = [f for f in cand_furn if _is_allowed_furn(pred, f)]
            print(f"[FILTER] furniture predicate filter: {cand_before} -> {len(cand_furn)} (pred={pred})")
            if pred == "switched_on":
                _sb = len(cand_furn)
                cand_furn = filter_candidates(pred, None, cand_furn)
                if len(cand_furn) != _sb:
                    print(f"[FILTER] furniture state filter (drop already-on): {_sb} -> {len(cand_furn)}")
        elif NO_ASP_FILTER:
            print("[FILTER] NO_ASP_FILTER=1 -> skip furniture predicate filter")
        cand_furn = _dedup_preserve_order(cand_furn)
        cand_furn_filtered = list(cand_furn)
        before_furn_class = len(cand_furn)
        cand_furn = _furn_class_shrink(cand_furn, pred, DISAMBIG_RUN_NAME)
        if len(cand_furn) != before_furn_class:
            print(f"[FILTER] furniture class narrowing: {before_furn_class} -> {len(cand_furn)}")

        if need_obj:
            if (not furn_pick) or (furn_pick not in cand_furn):
                furn_from_hist = _pick_furniture_by_history(cand_furn, pred, exclude_group=g_excl)
                if furn_from_hist:
                    furn_pick = furn_from_hist
                else:
                    furn_pick = cand_furn[0] if cand_furn else None
        else:
            if furn_pick and furn_pick not in cand_furn:
                cand_furn.insert(0, furn_pick)
                cand_furn = _dedup_preserve_order(cand_furn)

    baseline_raw_mode = (
        RANDOM_RAW_BASELINE
        or MOST_RECENT_RAW_BASELINE
        or MOST_FREQUENT_RAW_BASELINE
    )
    if baseline_raw_mode:
        slot_baseline = disambig_slot or ("obj" if need_obj else "furn")
        g_excl, _ = _parse_line_id(line_id)

        def _fallback_obj_raw() -> tuple[List[str], str]:
            if cand_objs_raw:
                return cand_objs_raw, ""
            if cand_objs:
                return cand_objs, "raw_empty->post_shrink"
            return _instances_of(EX.Item), "raw_empty->all_items"

        def _fallback_furn_raw() -> tuple[List[str], str]:
            if cand_furn_raw:
                return cand_furn_raw, ""
            if cand_furn:
                return cand_furn, "raw_empty->post_shrink"
            return _instances_of(EX.Furniture), "raw_empty->all_furniture"

        obj_pick = None
        furn_pick_bl = furn_pick or furn_in_tmpl
        decision = "HEURISTIC"
        decision_reason = ""
        note_parts: List[str] = []
        used_obj_candidates: List[str] = []
        used_furn_candidates: List[str] = []

        if slot_baseline == "obj" and need_obj:
            base_cands, base_note = _fallback_obj_raw()
            used_obj_candidates = list(base_cands)
            pick = None
            if base_note:
                note_parts.append(base_note)
            if RANDOM_RAW_BASELINE:
                pick = _pick_random_candidate(base_cands)
                decision = "RANDOM"
                decision_reason = f"random_raw seed={BASELINE_RANDOM_SEED}"
            elif MOST_RECENT_RAW_BASELINE:
                pick = _pick_most_recent_filtered(base_cands, "obj", line_id)
                if pick:
                    decision = "HEURISTIC"
                    decision_reason = "most_recent_raw"
                else:
                    pick = _pick_random_candidate(base_cands)
                    decision = "RANDOM"
                    decision_reason = f"most_recent_raw miss -> random_raw seed={BASELINE_RANDOM_SEED}"
            elif MOST_FREQUENT_RAW_BASELINE:
                pick = _pick_most_frequent_filtered_obj(base_cands, pred, furn_pick or furn, exclude_group=_heldout_exclude_set(g_excl))
                if pick:
                    decision = "HEURISTIC"
                    decision_reason = f"most_frequent_raw(scope={COUNTER_MATCH_SCOPE})"
                else:
                    pick = _pick_random_candidate(base_cands)
                    decision = "RANDOM"
                    decision_reason = (
                        f"most_frequent_raw(scope={COUNTER_MATCH_SCOPE}) miss "
                        f"-> random_raw seed={BASELINE_RANDOM_SEED}"
                    )
            obj_pick = pick
            if need_furn and (not furn_pick_bl or furn_pick_bl == "__FURN__"):
                furn_cands, furn_note = _fallback_furn_raw()
                used_furn_candidates = list(furn_cands)
                furn_pick_bl = furn_cands[0] if furn_cands else None
                if furn_note:
                    note_parts.append(furn_note)
            if note_parts:
                decision_reason = f"{decision_reason} ({'; '.join(note_parts)})".strip()

        elif slot_baseline == "furn" and need_furn:
            base_cands, base_note = _fallback_furn_raw()
            used_furn_candidates = list(base_cands)
            pick = None
            if base_note:
                note_parts.append(base_note)
            if RANDOM_RAW_BASELINE:
                pick = _pick_random_candidate(base_cands)
                decision = "RANDOM"
                decision_reason = f"random_raw seed={BASELINE_RANDOM_SEED}"
            elif MOST_RECENT_RAW_BASELINE:
                pick = _pick_most_recent_filtered(base_cands, "furn", line_id)
                if pick:
                    decision = "HEURISTIC"
                    decision_reason = "most_recent_raw"
                else:
                    pick = _pick_random_candidate(base_cands)
                    decision = "RANDOM"
                    decision_reason = f"most_recent_raw miss -> random_raw seed={BASELINE_RANDOM_SEED}"
            elif MOST_FREQUENT_RAW_BASELINE:
                pick = _pick_most_frequent_filtered_furn(base_cands, pred, exclude_group=g_excl)
                if pick:
                    decision = "HEURISTIC"
                    decision_reason = f"most_frequent_raw(scope={COUNTER_MATCH_SCOPE})"
                else:
                    pick = _pick_random_candidate(base_cands)
                    decision = "RANDOM"
                    decision_reason = (
                        f"most_frequent_raw(scope={COUNTER_MATCH_SCOPE}) miss "
                        f"-> random_raw seed={BASELINE_RANDOM_SEED}"
                    )
            furn_pick_bl = pick
            if need_obj and not obj_pick:
                obj_cands, obj_note = _fallback_obj_raw()
                used_obj_candidates = list(obj_cands)
                obj_pick = obj_cands[0] if obj_cands else None
                if obj_note:
                    note_parts.append(obj_note)
            if note_parts:
                decision_reason = f"{decision_reason} ({'; '.join(note_parts)})".strip()

        asp_cmd = asp_template
        if need_obj:
            asp_cmd = asp_cmd.replace("__OBJ__", obj_pick or "")
        if need_furn:
            asp_cmd = asp_cmd.replace("__FURN__", furn_pick_bl or "")

        slot_cands = used_obj_candidates if slot_baseline == "obj" else used_furn_candidates
        scored_stub = [(c, 0.0, 0.0, 0.0, 0.0) for c in (slot_cands or [])]

        try:
            log_disambig_result(
                line_id=line_id,
                sentence=sentence,
                prev_cmd=prev_cmd,
                prev_nl=prev_command_nl,
                asp_template=asp_template,
                asp_cmd=asp_cmd,
                context_label=sys_ctx,
                ctx_word=ctx_word,
                pred=pred,
                furniture=furn_pick_bl,
                decision=decision,
                decision_reason=decision_reason,
                confidence=0.0,
                residual_ambiguity=0.0,
                lead_ratio=0.0,
                top1=(obj_pick or furn_pick_bl or ""),
                top2="",
                m1=0.0,
                m2=0.0,
                scored_tuples=scored_stub,
                history_prior={},
                wilson_lbs={},
                steps=0,
                touched=[],
                disambig_stage=0,
                disambig_steps=0,
                stage_trace=[],
                would_ask=False,
            )
        except Exception as _e:
            print("[WARN] baseline_raw logger failed:", _e)

        detail = {
            "decision": decision,
            "decision_reason": decision_reason,
        }
        _append_group_history(line_id, asp_cmd, sentence)
        return asp_cmd, 0.0, detail, [], True

    baseline_filtered_mode = (
        RANDOM_FILTERED_BASELINE
        or MOST_RECENT_FILTERED_BASELINE
        or MOST_FREQUENT_FILTERED_BASELINE
    )
    if baseline_filtered_mode:
        slot_baseline = disambig_slot or ("obj" if need_obj else "furn")
        g_excl, _ = _parse_line_id(line_id)

        def _fallback_obj() -> tuple[List[str], str]:
            if cand_objs_filtered:
                return cand_objs_filtered, ""
            if cand_objs_raw:
                return cand_objs_raw, "filtered_empty->raw"
            if cand_objs:
                return cand_objs, "filtered_empty->post_shrink"
            return _instances_of(EX.Item), "filtered_empty->all_items"

        def _fallback_furn() -> tuple[List[str], str]:
            if cand_furn_filtered:
                return cand_furn_filtered, ""
            if cand_furn:
                return cand_furn, "filtered_empty->post_shrink"
            return _instances_of(EX.Furniture), "filtered_empty->all_furniture"

        obj_pick = None
        furn_pick_bl = furn_pick or furn_in_tmpl
        decision = "HEURISTIC"
        decision_reason = ""
        note_parts: List[str] = []
        used_obj_candidates: List[str] = []
        used_furn_candidates: List[str] = []

        if slot_baseline == "obj" and need_obj:
            base_cands, base_note = _fallback_obj()
            used_obj_candidates = list(base_cands)
            pick = None
            if base_note:
                note_parts.append(base_note)
            if RANDOM_FILTERED_BASELINE:
                pick = _pick_random_candidate(base_cands)
                decision = "RANDOM"
                decision_reason = f"random_filtered seed={BASELINE_RANDOM_SEED}"
            elif MOST_RECENT_FILTERED_BASELINE:
                pick = _pick_most_recent_filtered(base_cands, "obj", line_id)
                if pick:
                    decision = "HEURISTIC"
                    decision_reason = "most_recent_filtered"
                else:
                    pick = _pick_random_candidate(base_cands)
                    decision = "RANDOM"
                    decision_reason = f"most_recent_filtered miss -> random_filtered seed={BASELINE_RANDOM_SEED}"
            elif MOST_FREQUENT_FILTERED_BASELINE:
                pick = _pick_most_frequent_filtered_obj(base_cands, pred, furn_pick or furn, exclude_group=_heldout_exclude_set(g_excl))
                if pick:
                    decision = "HEURISTIC"
                    decision_reason = f"most_frequent_filtered(scope={COUNTER_MATCH_SCOPE})"
                else:
                    pick = _pick_random_candidate(base_cands)
                    decision = "RANDOM"
                    decision_reason = (
                        f"most_frequent_filtered(scope={COUNTER_MATCH_SCOPE}) miss "
                        f"-> random_filtered seed={BASELINE_RANDOM_SEED}"
                    )
            obj_pick = pick
            if need_furn and (not furn_pick_bl or furn_pick_bl == "__FURN__"):
                furn_cands, furn_note = _fallback_furn()
                used_furn_candidates = list(furn_cands)
                furn_pick_bl = furn_cands[0] if furn_cands else None
                if furn_note:
                    note_parts.append(furn_note)
            if note_parts:
                decision_reason = f"{decision_reason} ({'; '.join(note_parts)})".strip()

        elif slot_baseline == "furn" and need_furn:
            base_cands, base_note = _fallback_furn()
            used_furn_candidates = list(base_cands)
            pick = None
            if base_note:
                note_parts.append(base_note)
            if RANDOM_FILTERED_BASELINE:
                pick = _pick_random_candidate(base_cands)
                decision = "RANDOM"
                decision_reason = f"random_filtered seed={BASELINE_RANDOM_SEED}"
            elif MOST_RECENT_FILTERED_BASELINE:
                pick = _pick_most_recent_filtered(base_cands, "furn", line_id)
                if pick:
                    decision = "HEURISTIC"
                    decision_reason = "most_recent_filtered"
                else:
                    pick = _pick_random_candidate(base_cands)
                    decision = "RANDOM"
                    decision_reason = f"most_recent_filtered miss -> random_filtered seed={BASELINE_RANDOM_SEED}"
            elif MOST_FREQUENT_FILTERED_BASELINE:
                pick = _pick_most_frequent_filtered_furn(base_cands, pred, exclude_group=g_excl)
                if pick:
                    decision = "HEURISTIC"
                    decision_reason = f"most_frequent_filtered(scope={COUNTER_MATCH_SCOPE})"
                else:
                    pick = _pick_random_candidate(base_cands)
                    decision = "RANDOM"
                    decision_reason = (
                        f"most_frequent_filtered(scope={COUNTER_MATCH_SCOPE}) miss "
                        f"-> random_filtered seed={BASELINE_RANDOM_SEED}"
                    )
            furn_pick_bl = pick
            if need_obj and not obj_pick:
                obj_cands, obj_note = _fallback_obj()
                used_obj_candidates = list(obj_cands)
                obj_pick = obj_cands[0] if obj_cands else None
                if obj_note:
                    note_parts.append(obj_note)
            if note_parts:
                decision_reason = f"{decision_reason} ({'; '.join(note_parts)})".strip()

        asp_cmd = asp_template
        if need_obj:
            asp_cmd = asp_cmd.replace("__OBJ__", obj_pick or "")
        if need_furn:
            asp_cmd = asp_cmd.replace("__FURN__", furn_pick_bl or "")

        slot_cands = cand_objs_filtered if slot_baseline == "obj" else cand_furn_filtered
        scored_stub = [(c, 0.0, 0.0, 0.0, 0.0) for c in (slot_cands or [])]

        try:
            log_disambig_result(
                line_id=line_id,
                sentence=sentence,
                prev_cmd=prev_cmd,
                prev_nl=prev_command_nl,
                asp_template=asp_template,
                asp_cmd=asp_cmd,
                context_label=sys_ctx,
                ctx_word=ctx_word,
                pred=pred,
                furniture=furn_pick_bl,
                decision=decision,
                decision_reason=decision_reason,
                confidence=0.0,
                residual_ambiguity=0.0,
                lead_ratio=0.0,
                top1=(obj_pick or furn_pick_bl or ""),
                top2="",
                m1=0.0,
                m2=0.0,
                scored_tuples=scored_stub,
                history_prior={},
                wilson_lbs={},
                steps=0,
                touched=[],
                disambig_stage=0,
                disambig_steps=0,
                stage_trace=[],
                would_ask=False,
            )
        except Exception as _e:
            print("[WARN] baseline logger failed:", _e)

        _append_group_history(line_id, asp_cmd, sentence)
        detail = {
            "decision": decision,
            "decision_reason": decision_reason,
            "obj_candidates": used_obj_candidates or cand_objs_filtered,
            "furn_candidates": used_furn_candidates or cand_furn_filtered,
            "fallback_prediction": asp_cmd,
        }
        return asp_cmd, 0.0, detail, [], True

    if LLM_FILTERED_BASELINE:
        slot_baseline = disambig_slot or ("obj" if need_obj else "furn")

        def _fallback_obj_llm() -> tuple[List[str], str]:
            if cand_objs_filtered:
                return cand_objs_filtered, ""
            if cand_objs_raw:
                return cand_objs_raw, "filtered_empty->raw"
            if cand_objs:
                return cand_objs, "filtered_empty->post_shrink"
            return _instances_of(EX.Item), "filtered_empty->all_items"

        def _fallback_furn_llm() -> tuple[List[str], str]:
            if cand_furn_filtered:
                return cand_furn_filtered, ""
            if cand_furn:
                return cand_furn, "filtered_empty->post_shrink"
            return _instances_of(EX.Furniture), "filtered_empty->all_furniture"

        obj_pick = None
        furn_pick_llm = furn_pick or furn_in_tmpl
        obj_forced = None
        furn_forced = furn_pick or furn_in_tmpl
        obj_reason = ""
        furn_reason = "explicit"

        if slot_baseline == "obj" and need_obj:
            base_cands, base_note = _fallback_obj_llm()
            obj_pick, obj_reason, obj_forced = _llm_pick_no_ctx(sentence, base_cands, "obj", line_id=line_id)
            if base_note:
                obj_reason = f"{obj_reason} [{base_note}]"
        elif slot_baseline == "furn" and need_furn:
            base_cands, base_note = _fallback_furn_llm()
            furn_pick_llm, furn_reason, furn_forced = _llm_pick_no_ctx(sentence, base_cands, "furn", line_id=line_id)
            if base_note:
                furn_reason = f"{furn_reason} [{base_note}]"

        llm_ask = (obj_pick == "ASK_HUMAN") or (furn_pick_llm == "ASK_HUMAN")

        filled = asp_template
        if not llm_ask:
            if need_obj and obj_pick:
                filled = filled.replace("__OBJ__", obj_pick)
            if need_furn and furn_pick_llm:
                filled = filled.replace("__FURN__", furn_pick_llm)

        if need_obj and not obj_forced:
            obj_forced = obj_pick if obj_pick and obj_pick != "ASK_HUMAN" else ""
        if need_furn and not furn_forced:
            if furn_pick_llm and furn_pick_llm != "ASK_HUMAN":
                furn_forced = furn_pick_llm
            else:
                furn_forced = ""

        fallback_filled = asp_template
        if need_obj:
            fallback_filled = fallback_filled.replace("__OBJ__", obj_forced or "")
        if need_furn:
            fallback_filled = fallback_filled.replace("__FURN__", furn_forced or "")

        parts = []
        if need_obj:
            parts.append(f"obj={obj_pick} ({obj_reason})")
        if need_furn:
            parts.append(f"furn={furn_pick_llm} ({furn_reason})")
        decision_reason = "; ".join(parts) if parts else "llm_filtered"
        decision = "ASK_HUMAN" if llm_ask else "LLM_FILTERED"

        detail = {
            "decision": decision,
            "obj_reason": obj_reason,
            "furn_reason": furn_reason,
            "decision_reason": decision_reason,
            "fallback_prediction": (fallback_filled if llm_ask else filled),
        }
        _append_group_history(line_id, filled, sentence)
        return filled, 0.0, detail, [], (not llm_ask)

    llm_forced_obj: Optional[str] = None
    llm_forced_furn: Optional[str] = None
    scored: List[tuple[str, float, float, float, float]] = []
    slot_type: Optional[str] = None

    if forced_ask_obj:
        slot_type = "obj"

    if not forced_ask_obj:
        if need_obj and need_furn and disambig_slot == "furn":
            slot_type = "furn"
            scored = _score_furniture_ctx_only(ctx_word, pred, cand_furn, line_id, sentence)
            if cand_objs:
                obj_pick = cand_objs[0]
        elif need_obj and need_furn and disambig_slot == "obj":
            slot_type = "obj"
            scored = _score_candidates_ctx_only(ctx_word, pred, furn_pick or furn, cand_objs, line_id, sentence)
            if cand_furn:
                furn_pick = furn_pick or cand_furn[0]
        elif need_obj:
            scored = _score_candidates_ctx_only(ctx_word, pred, furn_pick or furn, cand_objs, line_id, sentence)
            slot_type = "obj"
        elif need_furn:
            scored = _score_furniture_ctx_only(ctx_word, pred, cand_furn, line_id, sentence)
            slot_type = "furn"

    if scored and slot_type == "obj":
        w_obj = _LAST_SCORE_WEIGHTS_OBJ or (W_SEM, W_THEM, W_SAL)
        print(
            f"[SCORE-1] score (semantic/thematic/salience -> "
            f"weighted[{w_obj[0]:.2f}/{w_obj[1]:.2f}/{w_obj[2]:.2f}]): "
        )
        for (c, s1, s2, s3, m) in scored:
            print(f" - {c:20s} sem(ctx)={s1:.3f} them={s2:.3f} sal={s3:.3f} weighted={m:.3f}")
    elif scored and slot_type == "furn":
        w_furn = _LAST_SCORE_WEIGHTS_FURN or (W_SEM, W_THEM, W_SAL)
        print(
            f"[SCORE-1] furniture score (semantic/thematic/salience -> "
            f"weighted[{w_furn[0]:.2f}/{w_furn[1]:.2f}/{w_furn[2]:.2f}]): "
        )
        for (c, s1, s2, s3, m) in scored:
            print(f" - {c:20s} sem(ctx)={s1:.3f} them={s2:.3f} sal={s3:.3f} weighted={m:.3f}")

    stage_trace: List[Dict[str, Any]] = []
    disambig_stage = 0
    disambig_steps = 0

    obj_pick: Optional[str] = None
    decision = "DIRECT"
    decision_reason = ""
    lead_thres = _lead_threshold(DISAMBIG_RUN_NAME)
    is_hyper_run = "hypernym" in DISAMBIG_RUN_NAME.lower()
    if slot_type == "obj" and scored:
        lead_thres = _dynamic_lead_threshold(lead_thres, len(scored))

    if forced_ask_obj and need_obj:
        decision = "ASK_HUMAN"
        decision_reason = "no object candidates after filtering"
        is_clear = False

    counter_ratio_for_conf: Optional[float] = None

    if scored and slot_type == "obj":
        ratio, m1, m2, n1, n2 = _lead_ratio(scored)

        compare_stage1_failed = False
        compare_history_pick: Optional[str] = None
        compare_history_mode: Optional[str] = None
        compare_history_decision: Optional[str] = None
        compare_llm_pick: Optional[str] = None
        compare_llm_reason: Optional[str] = None

        passed_stage1 = (ratio == float("inf")) or (ratio >= lead_thres)
        stage1_guard = ""
        lv = _run_level(DISAMBIG_RUN_NAME)
        stage_trace.append({
            "stage": 1,
            "name": STAGE_NAMES[1],
            "lead_ratio": (None if ratio == float("inf") else ratio),
            "threshold": lead_thres,
            "passed": bool(passed_stage1),
            "guard": stage1_guard,
        })

        if passed_stage1:
            obj_pick = scored[0][0]
            decision = "DIRECT"

            if ratio == float("inf"):
                decision_reason = f"only one candidate; weighted={m1:.3f}"
                print(f"[DECIDE-1] single candidate -> pick {obj_pick}")
            else:
                decision_reason = f"lead_ratio={ratio:.3f} (delta={m1 - m2:.3f}) crosses threshold"
                print(
                    f"[DECIDE-1] top1={n1} w={m1:.3f} | top2={n2} w={m2:.3f} | "
                    f"lead_ratio={ratio:.3f} >= {lead_thres:.3f} -> pick {obj_pick}"
                )

            disambig_stage = 1
            disambig_steps = 1

        elif NO_HISTORY_BASELINE:
            decision = "ASK_HUMAN"
            decision_reason = (
                f"no_history: lead_ratio={ratio:.3f} < {lead_thres:.3f} -> ASK_HUMAN"
            )
            stage_trace.append({
                "stage": 2,
                "name": STAGE_NAMES[2],
                "mode": "no_history",
                "passed": False,
                "picked": None,
            })
            stage_trace.append({"stage": 3, "name": STAGE_NAMES[3], "passed": True})
            disambig_stage = 3
            disambig_steps = 3
            is_clear = False

        elif NO_STAGE2_FALLBACK:
            mode_name = "three_factor_ask_only" if THREE_FACTOR_ASK_ONLY else "history_counter_disabled"
            decision = "ASK_HUMAN"
            decision_reason = (
                f"{mode_name}: lead_ratio={ratio:.3f} < {lead_thres:.3f} -> ASK_HUMAN"
            )
            stage_trace.append({
                "stage": 2,
                "name": STAGE_NAMES[2],
                "mode": ("ask_only" if THREE_FACTOR_ASK_ONLY else "no_counter_fallback"),
                "passed": False,
                "picked": None,
            })
            stage_trace.append({"stage": 3, "name": STAGE_NAMES[3], "passed": True})
            disambig_stage = 3
            disambig_steps = 3
            is_clear = False

        elif not USE_HISTORY_COUNTER:
            obj_pick = scored[0][0]
            decision = "DIRECT"
            decision_reason = "history disabled -> fallback top1"
            disambig_stage = 1
            disambig_steps = 1

        else:
            if is_hyper_run:
                obj_pick = scored[0][0]
                decision = "DIRECT"
                decision_reason = (
                    f"hyper fallback top1; lead_ratio={ratio:.3f} < {lead_thres:.3f}"
                )
                disambig_stage = 1
                disambig_steps = 1
                print(f"[DECIDE-1] hyper mode below threshold, skipping history -> pick {obj_pick}")
                stage_trace.append({
                    "stage": 2,
                    "name": STAGE_NAMES[2],
                    "mode": "skipped_hyper",
                    "total": 0,
                    "t1": 0,
                    "t2": 0,
                    "ratio": 0.0,
                    "passed": False,
                    "picked": None,
                })
            else:
                print(f"[DECIDE-1] lead too small ({ratio:.3f} < {lead_thres:.3f}) -> trying HISTORY_COUNTER")

                cand_names = [c for (c, *_rest) in scored]
                class_focus = _class_focus_for_history(obj_class, scored)
                hist_candidates = cand_names
                if class_focus:
                    hist_candidates = [c for c in cand_names if _categories_of(c) & class_focus]
                    if not hist_candidates:
                        hist_candidates = cand_names

                focus_list = sorted(class_focus) if class_focus else []
                focus_kind = "class"
                if focus_list:
                    print(
                        f"[HISTORY] {focus_kind}_focus={focus_list} "
                        f"candidates={len(hist_candidates)}/{len(cand_names)}"
                    )
                else:
                    print(
                        f"[HISTORY] {focus_kind}_focus=none candidates={len(hist_candidates)}/{len(cand_names)}"
                    )
                g_excl, _k = _parse_line_id(line_id)

                if COMPARE_LLM_HISTORY:
                    compare_stage1_failed = True
                    compare_llm_pick, compare_llm_reason, _compare_llm_forced = _llm_pick_no_ctx(
                        sentence, hist_candidates, "obj", line_id=line_id
                    )
                    print(f"[COMPARE-LLM] pick={compare_llm_pick} reason={compare_llm_reason}")

                if LLM_REPLACE_HISTORY:
                    obj_pick, obj_reason, obj_forced = _llm_pick_no_ctx(sentence, hist_candidates, "obj", line_id=line_id)
                    llm_forced_obj = obj_forced
                    if obj_pick == "ASK_HUMAN":
                        decision = "ASK_HUMAN"
                        decision_reason = obj_reason
                        disambig_stage = 2
                        disambig_steps = 2
                        stage_trace.append({
                            "stage": 2,
                            "name": STAGE_NAMES[2],
                            "mode": "llm_replace_history",
                            "total": len(hist_candidates),
                            "t1": 0,
                            "t2": 0,
                            "ratio": 0.0,
                            "passed": False,
                            "picked": None,
                            "class_focus": focus_list,
                            "focus_kind": focus_kind,
                            "cand_total": len(cand_names),
                            "cand_filtered": len(hist_candidates),
                        })
                        print(f"[HISTORY] replaced by LLM -> ASK_HUMAN")
                    else:
                        decision = "LLM_HISTORY"
                        decision_reason = f"llm_replace_history: {obj_reason}"
                        disambig_stage = 2
                        disambig_steps = 2
                        stage_trace.append({
                            "stage": 2,
                            "name": STAGE_NAMES[2],
                            "mode": "llm_replace_history",
                            "total": len(hist_candidates),
                            "t1": 0,
                            "t2": 0,
                            "ratio": 0.0,
                            "passed": True,
                            "picked": obj_pick,
                            "class_focus": focus_list,
                            "focus_kind": focus_kind,
                            "cand_total": len(cand_names),
                            "cand_filtered": len(hist_candidates),
                        })
                        print(f"[HISTORY] replaced by LLM -> pick {obj_pick}")
                else:
                    picked_by_counter: Optional[str] = None
                    cnt_map: Dict[str, int] = {c: 0 for c in hist_candidates}
                    hist_stat: Dict[str, Any] = {
                        "mode": "none",
                        "total": 0,
                        "t1": 0,
                        "t2": 0,
                        "ratio": 0.0,
                    }

                    if USE_HISTORY_COUNTER:
                        picked_by_counter, cnt_map, hist_stat = decide_by_prev_command_counter(
                            prev_cmd=prev_cmd,
                            pred_now=(pred or ""),
                            furn_now=(furn_pick or None),
                            candidates=hist_candidates,
                            exclude_group=g_excl,
                        )
                        t1 = hist_stat.get("t1", 0)
                        t2 = hist_stat.get("t2", 0)
                        if not (t1 >= 2 and (t1 - t2) >= 1):
                            picked_by_counter = None
                            stage_trace.append({
                                "stage": 2,
                                "name": STAGE_NAMES[2],
                                "mode": hist_stat.get("mode"),
                                "total": hist_stat.get("total"),
                                "t1": hist_stat.get("t1"),
                                "t2": hist_stat.get("t2"),
                                "ratio": hist_stat.get("ratio"),
                                "passed": False,
                                "picked": None,
                                "class_focus": focus_list,
                                "focus_kind": focus_kind,
                                "cand_total": len(cand_names),
                                "cand_filtered": len(hist_candidates),
                            })
                            obj_pick = None
                            decision = "ASK_HUMAN"
                            decision_reason = (
                                "insufficient evidence: lead_ratio below threshold "
                                "and history votes not decisive"
                            )
                            print("[HISTORY] votes not decisive -> ASK_HUMAN")
                            disambig_stage = 2
                            disambig_steps = 2

                    if COMPARE_LLM_HISTORY:
                        compare_history_pick = picked_by_counter
                        compare_history_mode = str(hist_stat.get("mode") or "none")
                        compare_history_decision = "HISTORY" if picked_by_counter else "ASK_HUMAN"

                    stage_trace.append({
                        "stage": 2,
                        "name": STAGE_NAMES[2],
                        "mode": hist_stat.get("mode"),
                        "total": hist_stat.get("total"),
                        "t1": hist_stat.get("t1"),
                        "t2": hist_stat.get("t2"),
                        "ratio": hist_stat.get("ratio"),
                        "passed": bool(picked_by_counter),
                        "picked": picked_by_counter,
                        "class_focus": focus_list,
                        "focus_kind": focus_kind,
                        "cand_total": len(cand_names),
                        "cand_filtered": len(hist_candidates),
                    })

                    mode_hist = str(hist_stat.get("mode") or "none")

                    if picked_by_counter:
                        is_hyper = is_hyper_run
                        top_names = [name for (name, *_rest) in scored[:2]]
                        if is_hyper and picked_by_counter not in top_names:
                            obj_pick = scored[0][0]
                            decision = "DIRECT"
                            decision_reason = "history pick outside top2 in hyper -> fallback top1"
                            disambig_stage = 1
                            disambig_steps = 1
                            print(f"[HISTORY] hyper mode, {picked_by_counter} not in the top 2 -> falling back to top1 {obj_pick}")
                        else:
                            obj_pick = picked_by_counter
                            decision = "HISTORY"
                            decision_reason = f"COUNTER decided ({mode_hist})"
                            print(f"[HISTORY] COUNTER({mode_hist}) -> pick {obj_pick}")

                            disambig_stage = 2
                            disambig_steps = 2

                        try:
                            counter_ratio_for_conf = float(hist_stat.get("ratio", 0.0))
                        except Exception:
                            counter_ratio_for_conf = 0.0

                    else:
                        obj_pick = None
                        decision = "ASK_HUMAN"
                        decision_reason = (
                            "history not decisive and lead below threshold -> ASK_HUMAN"
                        )
                        print("[HISTORY] not decisive -> ASK_HUMAN")

                        disambig_stage = 2
                        disambig_steps = 2
                        is_clear = False

    elif scored and slot_type == "furn":
        ratio, m1, m2, n1, n2 = _lead_ratio(scored)

        passed_stage1 = (ratio == float("inf")) or (ratio >= lead_thres)
        stage_trace.append({
            "stage": 1,
            "name": STAGE_NAMES[1],
            "lead_ratio": (None if ratio == float("inf") else ratio),
            "threshold": lead_thres,
            "passed": bool(passed_stage1),
        })

        if passed_stage1:
            furn_pick = scored[0][0]
            decision = "DIRECT"

            if ratio == float("inf"):
                decision_reason = f"only one furniture candidate; weighted={m1:.3f}"
                print(f"[DECIDE-FURN-1] single furniture candidate -> pick {furn_pick}")
            else:
                decision_reason = f"lead_ratio={ratio:.3f} (delta={m1 - m2:.3f}) crosses threshold"
                print(
                    f"[DECIDE-FURN-1] top1={n1} w={m1:.3f} | top2={n2} w={m2:.3f} | "
                    f"lead_ratio={ratio:.3f} >= {lead_thres:.3f} -> pick {furn_pick}"
                )

            disambig_stage = 1
            disambig_steps = 1

        elif NO_HISTORY_BASELINE:
            decision = "ASK_HUMAN"
            decision_reason = (
                f"no_history: lead_ratio={ratio:.3f} < {lead_thres:.3f} -> ASK_HUMAN"
            )
            stage_trace.append({
                "stage": 2,
                "name": STAGE_NAMES[2],
                "mode": "no_history",
                "passed": False,
                "picked": None,
            })
            stage_trace.append({"stage": 3, "name": STAGE_NAMES[3], "passed": True})
            disambig_stage = 3
            disambig_steps = 3
            is_clear = False

        elif NO_STAGE2_FALLBACK:
            mode_name = "three_factor_ask_only" if THREE_FACTOR_ASK_ONLY else "history_counter_disabled"
            decision = "ASK_HUMAN"
            decision_reason = (
                f"{mode_name}: lead_ratio={ratio:.3f} < {lead_thres:.3f} -> ASK_HUMAN"
            )
            stage_trace.append({
                "stage": 2,
                "name": STAGE_NAMES[2],
                "mode": ("ask_only" if THREE_FACTOR_ASK_ONLY else "no_counter_fallback"),
                "passed": False,
                "picked": None,
            })
            stage_trace.append({"stage": 3, "name": STAGE_NAMES[3], "passed": True})
            disambig_stage = 3
            disambig_steps = 3
            is_clear = False

        else:
            print(f"[DECIDE-FURN-1] lead too small ({ratio:.3f} < {lead_thres:.3f}) -> trying FURN_HISTORY")

            g_excl, _k = _parse_line_id(line_id)
            cand_names = [c for (c, *_rest) in scored]
            furn_from_hist = None
            if LLM_REPLACE_HISTORY:
                furn_pick_llm, furn_reason, furn_forced = _llm_pick_no_ctx(sentence, cand_names, "furn", line_id=line_id)
                llm_forced_furn = furn_forced
                if furn_pick_llm == "ASK_HUMAN":
                    furn_pick = None
                    decision = "ASK_HUMAN"
                    decision_reason = furn_reason
                    stage_trace.append({
                        "stage": 2,
                        "name": STAGE_NAMES[2],
                        "mode": "llm_replace_history",
                        "passed": False,
                        "picked": None,
                    })
                    print(f"[FURN-HISTORY] replaced by LLM -> ASK_HUMAN")
                else:
                    furn_pick = furn_pick_llm
                    decision = "LLM_HISTORY"
                    decision_reason = f"llm_replace_history: {furn_reason}"
                    stage_trace.append({
                        "stage": 2,
                        "name": STAGE_NAMES[2],
                        "mode": "llm_replace_history",
                        "passed": True,
                        "picked": furn_pick,
                    })
                    print(f"[FURN-HISTORY] replaced by LLM -> pick {furn_pick}")

                disambig_stage = 2
                disambig_steps = 2
            else:
                furn_from_hist = _pick_furniture_by_history(cand_names, pred, exclude_group=g_excl)

            if not LLM_REPLACE_HISTORY:
                stage_trace.append({
                    "stage": 2,
                    "name": STAGE_NAMES[2],
                    "mode": "furn_history",
                    "passed": bool(furn_from_hist),
                    "picked": furn_from_hist,
                })

            if not LLM_REPLACE_HISTORY:
                if furn_from_hist:
                    furn_pick = furn_from_hist
                    decision = "HISTORY"
                    decision_reason = "FURN history decided"
                    print(f"[FURN-HISTORY] -> pick {furn_pick}")

                    disambig_stage = 2
                    disambig_steps = 2
                else:
                    furn_pick = scored[0][0]
                    decision = "DIRECT"
                    decision_reason = "no clear lead; no furniture history; pick top1"
                    print(f"[FURN-HISTORY] inactive -> taking top1 furniture: {furn_pick}")

                    stage_trace.append({"stage": 3, "name": STAGE_NAMES[3], "passed": True})
                disambig_stage = 3
                disambig_steps = 3

    asp_cmd = asp_template

    if decision == "ASK_HUMAN" and FORCE_NO_ASK:
        fallback_reason = "force_no_ask: fallback_top1"

        def _fallback_pick_from_scored(rows: List[tuple[str, float, float, float, float]]) -> tuple[Optional[str], str]:
            if not rows:
                return None, "empty"
            if FORCE_NO_ASK_FALLBACK_POLICY == "thematic_if_clear_else_fused":
                by_them = sorted(rows, key=lambda t: float(t[2]), reverse=True)
                t1 = float(by_them[0][2]) if by_them else 0.0
                t2 = float(by_them[1][2]) if len(by_them) > 1 else 0.0
                if (t1 >= FORCE_NO_ASK_THEMATIC_MIN_TOP) and ((t1 - t2) >= FORCE_NO_ASK_THEMATIC_MIN_GAP):
                    return by_them[0][0], "thematic_if_clear"
                best = max(rows, key=lambda t: float(t[4]))
                return best[0], "fused_if_them_unclear"
            if FORCE_NO_ASK_FALLBACK_POLICY == "thematic_top1":
                best = max(rows, key=lambda t: (float(t[2]), float(t[4])))
                return best[0], "thematic_top1"
            best = max(rows, key=lambda t: float(t[4]))
            return best[0], "fused_top1"

        if need_obj and (obj_pick is None or obj_pick == "ASK_HUMAN"):
            if scored and slot_type == "obj":
                _pick, _policy = _fallback_pick_from_scored(scored)
                obj_pick = _pick or scored[0][0]
                fallback_reason = f"force_no_ask: fallback_{_policy}"
            elif cand_objs:
                obj_pick = cand_objs[0]
            else:
                obj_pick = (_instances_of(EX.Item) or [""])[0]
        if need_furn and (furn_pick is None or furn_pick == "ASK_HUMAN"):
            if scored and slot_type == "furn":
                _pick, _policy = _fallback_pick_from_scored(scored)
                furn_pick = _pick or scored[0][0]
                fallback_reason = f"force_no_ask: fallback_{_policy}"
            elif cand_furn:
                furn_pick = cand_furn[0]
            else:
                furn_pick = (_instances_of(EX.Furniture) or [""])[0]
        decision = "FALLBACK"
        decision_reason = f"{decision_reason}; {fallback_reason}".strip("; ")

    if decision != "ASK_HUMAN":
        if need_obj:
            asp_cmd = asp_cmd.replace("__OBJ__", obj_pick or "")
        if need_furn:
            asp_cmd = asp_cmd.replace("__FURN__", furn_pick or "")

        def _first_non_placeholder(items: List[str]) -> Optional[str]:
            for v in items or []:
                vv = (v or "").strip()
                if vv and "__" not in vv:
                    return vv
            return None

        obj_cand_fallback = _first_non_placeholder(cand_objs) or _first_non_placeholder(cand_furn)
        furn_cand_fallback = _first_non_placeholder(cand_furn) or _first_non_placeholder(cand_objs)

        if "__OBJ__" in asp_cmd:
            obj_fallback = (obj_pick if obj_pick and "__" not in obj_pick else None) or (furn_pick if furn_pick and "__" not in furn_pick else None) or obj_cand_fallback or ""
            asp_cmd = asp_cmd.replace("__OBJ__", obj_fallback)
        if "__FURN__" in asp_cmd:
            furn_fallback = (furn_pick if furn_pick and "__" not in furn_pick else None) or (obj_pick if obj_pick and "__" not in obj_pick else None) or furn_cand_fallback or ""
            asp_cmd = asp_cmd.replace("__FURN__", furn_fallback)

        if "__OBJ__" in asp_cmd or "__FURN__" in asp_cmd:
            generic_fallback = (obj_pick if obj_pick and "__" not in obj_pick else None) \
                or (furn_pick if furn_pick and "__" not in furn_pick else None) \
                or obj_cand_fallback \
                or furn_cand_fallback \
                or ""
            asp_cmd = re.sub(r"__([A-Z]+)__", generic_fallback, asp_cmd)
            if "__OBJ__" in asp_cmd or "__FURN__" in asp_cmd:
                asp_cmd = asp_cmd.replace("__OBJ__", "").replace("__FURN__", "")
            print(f"[WARN] unresolved placeholders patched in asp_cmd -> {asp_cmd}")
    else:
        pass

    fallback_asp_cmd = asp_cmd
    if decision == "ASK_HUMAN":
        fallback_obj = None
        fallback_furn = None
        if need_obj:
            if llm_forced_obj:
                fallback_obj = llm_forced_obj
            elif scored and slot_type == "obj":
                fallback_obj = scored[0][0]
            elif cand_objs:
                fallback_obj = cand_objs[0]
            else:
                fallback_obj = (_instances_of(EX.Item) or [""])[0]
        if need_furn:
            if llm_forced_furn:
                fallback_furn = llm_forced_furn
            elif scored and slot_type == "furn":
                fallback_furn = scored[0][0]
            elif cand_furn:
                fallback_furn = cand_furn[0]
            else:
                fallback_furn = (_instances_of(EX.Furniture) or [""])[0]
        fallback_asp_cmd = asp_template
        if need_obj:
            fallback_asp_cmd = fallback_asp_cmd.replace("__OBJ__", fallback_obj or "")
        if need_furn:
            fallback_asp_cmd = fallback_asp_cmd.replace("__FURN__", fallback_furn or "")

    touched: List[str] = []
    steps: int = 0
    if run_planner and decision != "ASK_HUMAN":
        try:
            plan_res = plan(asp_cmd)
            steps = plan_res.get("steps", 0) or 0
            touched = list(plan_res.get("touched", []))
        except Exception as e:
            print("[WARN] planner failed:", e)

    _append_group_history(line_id, asp_cmd, sentence)

    compare_info: Optional[Dict[str, Any]] = None
    if (
        COMPARE_LLM_HISTORY
        and slot_type == "obj"
        and "compare_stage1_failed" in locals()
        and compare_stage1_failed
    ):
        def _fill_alt_cmd(obj_pick_alt: Optional[str]) -> Optional[str]:
            cmd = asp_template
            if "__OBJ__" in cmd:
                if not obj_pick_alt or obj_pick_alt == "ASK_HUMAN":
                    return None
                cmd = cmd.replace("__OBJ__", obj_pick_alt)
            if "__FURN__" in cmd:
                if not furn_pick:
                    return None
                cmd = cmd.replace("__FURN__", furn_pick)
            if "__OBJ__" in cmd or "__FURN__" in cmd:
                return None
            return cmd

        compare_info = {
            "stage1_failed": True,
            "history_pick": compare_history_pick,
            "history_mode": compare_history_mode,
            "history_decision": compare_history_decision,
            "history_reason": "history_counter" if compare_history_pick else "history_not_decisive",
            "llm_pick": compare_llm_pick,
            "llm_decision": ("ASK_HUMAN" if compare_llm_pick == "ASK_HUMAN" else "LLM"),
            "llm_reason": compare_llm_reason,
            "asp_history": _fill_alt_cmd(compare_history_pick),
            "asp_llm": _fill_alt_cmd(compare_llm_pick),
        }

    conf = 0.0
    primary_name: Optional[str] = None
    if slot_type == "obj" and obj_pick:
        primary_name = obj_pick
    elif slot_type == "furn" and furn_pick:
        primary_name = furn_pick

    if scored and primary_name:
        for (c, _s1, _s2, _s3, m) in scored:
            if c == primary_name:
                conf = m
                break

    if counter_ratio_for_conf is not None and slot_type == "obj" and obj_pick:
        conf = 0.5 * conf + 0.5 * float(counter_ratio_for_conf)

    residual_H = residual_ambiguity_from_scores(
        [m for (_c, _s1, _s2, _s3, m) in scored]
    ) if scored else 0.0

    would_ask = (
        decision == "ASK_HUMAN"
        or (scored and disambig_stage >= 2 and (conf < 0.30 or residual_H > 0.80))
    )

    detail: Dict[str, Any] = {
        "steps": steps,
        "picked_obj": obj_pick if need_obj else None,
        "picked_furn": furn_pick if need_furn else None,
        "pred": pred,
        "scores": [
            {"name": c, "semantic": s1, "thematic": s2, "salience": s3, "mean": m}
            for (c, s1, s2, s3, m) in scored
        ] if scored else [],
        "picked_cost_norm": 0.0,
        "picked_risk": 0.0,
        "avg_risk": 0.0,
        "residual_ambiguity": residual_H,
        "need_confirm": (decision == "ASK_HUMAN"),
        "prev_ctx": ctx_word or sys_ctx,
        "curr_ctx": dominant_context((obj_pick or "") or (furn_pick or "")),
        "obj_candidates": [c for c, *_ in scored] if (scored and slot_type == "obj") else [],
        "furn_candidates": [c for c, *_ in scored] if (scored and slot_type == "furn") else [],
        "confidence": conf,
        "decision": decision,
        "decision_reason": decision_reason,
        "insufficient_evidence": (decision == "ASK_HUMAN"),
        "fallback_prediction": fallback_asp_cmd,
    }
    if compare_info:
        detail["compare"] = compare_info

    try:
        r_ratio, r_m1, r_m2, r_n1, r_n2 = _lead_ratio(scored)
        log_disambig_result(
            line_id=line_id,
            sentence=sentence,
            prev_cmd=prev_cmd,
            prev_nl=prev_command_nl,
            asp_template=asp_template,
            asp_cmd=asp_cmd,
            context_label=sys_ctx,
            ctx_word=ctx_word,
            pred=pred,
            furniture=furn_pick,
            decision=decision,
            decision_reason=decision_reason,
            confidence=conf,
            residual_ambiguity=residual_H,
            lead_ratio=r_ratio,
            top1=r_n1,
            top2=r_n2,
            m1=r_m1,
            m2=r_m2,
            scored_tuples=scored,
            history_prior={},
            wilson_lbs={},
            steps=steps,
            touched=touched,
            disambig_stage=disambig_stage,
            disambig_steps=disambig_steps,
            stage_trace=stage_trace,
            would_ask=would_ask,
            compare=compare_info,
        )
    except Exception as _e:
        print("[WARN] disambig logger failed:", _e)

    print(f"[DECISION] pick={(obj_pick or furn_pick)} via={decision} reason={decision_reason}")
    print(f" confidence(weighted/fused) = {detail['confidence']:.3f}")

    return asp_cmd, detail["confidence"], detail, touched, is_clear


def _dedup_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _types_of_instance(name: str) -> List:
    g = get_graph()
    return list(g.objects(EX[name], RDF.type))


def _is_subclass_of(g, cls, target) -> bool:
    if cls == target:
        return True
    seen = set()
    stack = [cls]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for sup in g.objects(cur, RDFS.subClassOf):
            if sup == target:
                return True
            stack.append(sup)
    return False


def _is_instance_of_any(name: str, supers: List) -> bool:
    g = get_graph()
    types = _types_of_instance(name)
    for t in types:
        for sup in supers:
            if t == sup or _is_subclass_of(g, t, sup):
                return True
    return False


def _categories_of(name: str) -> set[str]:
    cats: set[str] = set()
    cls = OBJ_HYPERNYM_MAP.get(name)
    if cls:
        cats.add(str(cls).lower())
    try:
        for t in _types_of_instance(name):
            cats.add(str(t).split("/")[-1].lower())
    except Exception:
        pass
    return cats


_GENERIC_CLASSES = {"item", "thing", "object", "furniture"}

_FURN_BUCKETS = {"inside_furniture", "on_furniture", "switch_furniture", "furniture"}


def _furn_bucket_set(name: str) -> set[str]:
    buckets: set[str] = set()
    if name in INSIDE_FURN:
        buckets.add("inside_furniture")
    if name in ON_FURN:
        buckets.add("on_furniture")
    if name in SWITCH_FURN:
        buckets.add("switch_furniture")
    if buckets:
        buckets.add("furniture")
    return buckets


def _furn_class_shrink(
    cands: List[str],
    pred: str | None,
    run_name: str | None,
) -> List[str]:
    if not cands:
        return cands
    if _run_level(run_name) not in (2, 3):
        return cands

    if pred == "inside":
        buckets = {"inside_furniture"}
    elif pred == "on":
        buckets = {"on_furniture"}
    elif pred in {"open", "close"}:
        buckets = {"inside_furniture"}
    elif pred in {"switched_on", "switched_off", "switchon", "switchoff"}:
        buckets = {"switch_furniture"}
    else:
        buckets = _FURN_BUCKETS

    kept = [c for c in cands if _furn_bucket_set(c) & buckets]
    if kept:
        print(f"[CLASS-FURN] buckets={sorted(buckets)} keep {len(kept)}/{len(cands)}")
        return kept
    return cands


def _class_focus_for_history(obj_class: str | None, scored: List[tuple[str, float, float, float, float]]) -> set[str]:
    focus: set[str] = set()
    if obj_class:
        oc = str(obj_class).lower().strip()
        if oc and oc not in _GENERIC_CLASSES:
            focus.add(oc)

    if not focus and scored:
        top_names = [scored[0][0]]
        if len(scored) > 1:
            top_names.append(scored[1][0])
        for name in top_names:
            for c in _categories_of(name):
                if c and c not in _GENERIC_CLASSES:
                    focus.add(c)
    return focus


def _filter_obj_by_pred_types(pred: str | None, objs: List[str]) -> List[str]:
    if not pred or not objs:
        return objs
    req_classes: list = []
    allow_tokens: set[str] = set()

    if pred in {"grab", "put", "putin", "on", "inside", "give", "has"}:
        req_classes = [EX.Item]
        allow_tokens = OBJ_TOKENS
    elif pred == "heated":
        req_classes = [EX.HotDrink, EX.HotFood]
        allow_tokens = HOT_DRINK_ITEMS | HOT_FOOD_ITEMS | {"hot_drink", "hot_food"}

    if not req_classes and not allow_tokens:
        return objs

    filtered: List[str] = []
    for o in objs:
        if o in allow_tokens:
            filtered.append(o)
            continue
        if o in OBJ_LEAVES:
            filtered.append(o)
            continue
        if req_classes and _is_instance_of_any(o, req_classes):
            filtered.append(o)

    return filtered


def _history_pred_furn_obj_set(pred: str | None, furn: str | None) -> set[str]:
    return set()


def _abs_root(name: str) -> str:
    return str(Path(PROJECT_ROOT) / name)


def main():
    _ensure_init()
    prev_cmd = None
    prev_nl: Optional[str] = None
    print(" Ready. Type English commands; 'exit' quits.")
    while True:
        sent = input("\nCommand: ").strip()
        if sent.lower() in {"exit", "quit"}:
            break
        asp_tmpl = input(" Template (e.g., inside(fridge, __OBJ__)): ").strip()
        line_id = input(" Line ID (e.g., [1-2], optional): ").strip() or None
        asp_cmd, conf, detail, touched_items, is_clear = pipeline(
            sent,
            prev_cmd,
            prev_command_nl=prev_nl,
            asp_template=asp_tmpl,
            line_id=line_id,
            run_planner=True,
        )
        print(f"\n ASP: {asp_cmd}")
        print(
            f" confidence(weighted/fused)={conf:.3f}, "
            f"residual_ambiguity={detail.get('residual_ambiguity')}"
        )
        if is_clear:
            prev_cmd = asp_cmd
            prev_nl = sent


if __name__ == "__main__":
    main()
