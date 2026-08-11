import os
from rdflib import URIRef, Literal, RDF, RDFS
from kg.loader import get_graph, EX
from experiments.generate_fuzzy_sets import OBJ_HYPERNYM_MAP

_FILLABLE_ITEMS = {
    "cup", "mug", "glass", "bottle", "jar", "kettle", "teapot", "thermos",
    "bowl", "bucket", "vase", "beaker", "coffeepot", "cauldron",
    "flask", "carafe", "pitcher",
}

def _ln(x):
    s = str(x)
    return s.split("/")[-1] if "/" in s else s

def filter_candidates(pred: str, furn: str | None, candidates: list[str]) -> list[str]:
    g = get_graph()
    kept: list[str] = []

    def _camel(tok: str) -> str:
        parts = tok.replace("-", "_").split("_")
        return "".join(p.capitalize() for p in parts if p)

    def _hypernyms(name: str) -> set[str]:
        hyps = set()
        try:
            for t in g.objects(EX[name], RDF.type):
                hyps.add(_ln(t))
                for sup in g.objects(t, RDFS.subClassOf):
                    hyps.add(_ln(sup))
        except Exception:
            pass
        if not hyps:
            cls = OBJ_HYPERNYM_MAP.get(name)
            if cls:
                hyps.add(_camel(cls))
        return hyps

    furn_alias = {"bowl": "dish_bowl"}
    if furn in furn_alias:
        furn = furn_alias[furn]

    furn_uri = EX[furn] if furn else None
    furn_open = None
    if furn_uri:
        v = next(g.objects(furn_uri, EX.open), None)
        if isinstance(v, Literal):
            try:
                furn_open = bool(v.toPython())
            except Exception:
                furn_open = None

    for it in candidates:
        iu = EX[it]
        ok = True

        if next(g.objects(iu, RDF.type), None) is None:
            ok = False
            kept_skip = True
            continue

        if (iu, EX.inside, EX.dishwasher) in g:
            ok = False
        if (iu, EX.inside, EX.hamper) in g:
            ok = False

        if pred == "inside" and furn_uri:
            if (iu, EX.inside, furn_uri) in g:
                ok = False

        if pred == "on" and furn_uri:
            if (iu, EX.on, furn_uri) in g:
                ok = False
        if pred and isinstance(it, str) and it.startswith("__"):
            ok = False
        if pred == "has":
            if (EX.user, EX.has, iu) in g:
                ok = False

        if pred == "heated":
            v = next(g.objects(iu, EX.heated), None)
            if isinstance(v, Literal):
                try:
                    if bool(v.toPython()) is True:
                        ok = False
                except Exception:
                    pass

        if pred == "filled":
            if next(g.objects(iu, EX.filled), None) is not None:
                ok = False

        if pred == "switched_on":
            v = next(g.objects(iu, EX.switched_on), None)
            if isinstance(v, Literal):
                try:
                    if bool(v.toPython()) is True:
                        ok = False
                except Exception:
                    pass

        if pred in {"has", "give"}:
            if (EX.user, EX.has, iu) in g:
                ok = False

        if ok and pred == "inside" and furn == "fridge":
            hyps = _hypernyms(it)
            allowed = {"ColdDrink", "Fruit", "HotDrink"}
            if not hyps & allowed:
                ok = False

        if ok and pred == "inside" and furn in ("microwave", "microwave_oven"):
            hyps = _hypernyms(it)
            allowed = {"HotDrink", "HotFood", "Protein", "Food"}
            if not hyps & allowed:
                ok = False

        if ok and pred == "on" and furn == "dish_bowl":
            hyps = _hypernyms(it)
            allowed = {"Fruit", "Snack", "Food", "Protein", "HotFood"}
            if not hyps & allowed:
                ok = False

        if ok and pred == "filled" and os.environ.get("FILL_FIX", "") == "1":
            if it not in _FILLABLE_ITEMS:
                ok = False
            elif furn and it == furn:
                ok = False

        if ok:
            kept.append(it)

    return kept
