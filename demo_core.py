"""TRIVENI demo core: every function here is mirrored line for line in
web/triveni.js, so the browser reproduces the trained models exactly.

Stages: prep (normalize) -> block (IDF-weighted keys + trigram re-rank)
-> stage-1 pair features -> stage-2 context features -> (models elsewhere).
All similarity functions are written by hand (no rapidfuzz) for parity.
"""
from __future__ import annotations

import math
import re

from normalize import Normalizer, skeleton

NORM = Normalizer()
K_RETRIEVE, K_MAX = 40, 8
STREET = re.compile(r"(?:^|\s)(\d+)\s+([a-z]{3,})")


# ---------------------------------------------------------------- prep
def prep(r: dict) -> dict:
    n = NORM.name(r["business_name"])
    a = NORM.address(r["business_address"])
    core = n["name_core"]
    return {
        "id": r["entity_id"], "country": r["country"], "src": r["entity_id"][:2],
        "name": r["business_name"], "addr": r["business_address"],
        "core": core, "alt": n["name_alt"], "suffix": n["name_suffix"],
        "skel": n["name_skel"], "acr": n["name_acronym"],
        "toks": core.split(), "sk": n["name_skel"].split(),
        "anorm": a["addr_norm"], "awords": a["addr_words"].split(),
        "pc": a["addr_postcode"].split(), "nums": a["addr_nums"].split(),
        "landmark": a["addr_landmark"],
    }


# ---------------------------------------------------------------- similarity
def jaro_winkler(s: str, t: str) -> float:
    if s == t:
        return 1.0 if s else 0.0
    ls, lt = len(s), len(t)
    if ls == 0 or lt == 0:
        return 0.0
    rng = max(max(ls, lt) // 2 - 1, 0)
    sm, tm = [False] * ls, [False] * lt
    m = 0
    for i in range(ls):
        lo, hi = max(0, i - rng), min(i + rng + 1, lt)
        for j in range(lo, hi):
            if not tm[j] and s[i] == t[j]:
                sm[i] = tm[j] = True
                m += 1
                break
    if m == 0:
        return 0.0
    k = tr = 0
    for i in range(ls):
        if sm[i]:
            while not tm[k]:
                k += 1
            if s[i] != t[k]:
                tr += 1
            k += 1
    jaro = (m / ls + m / lt + (m - tr / 2) / m) / 3
    p = 0
    while p < min(4, ls, lt) and s[p] == t[p]:
        p += 1
    return jaro + p * 0.1 * (1 - jaro)


def lev_ratio(s: str, t: str) -> float:
    if not s and not t:
        return 1.0
    prev = list(range(len(t) + 1))
    for i in range(1, len(s) + 1):
        cur = [i] + [0] * len(t)
        for j in range(1, len(t) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (s[i - 1] != t[j - 1]))
        prev = cur
    return 1 - prev[len(t)] / max(len(s), len(t))


def trigram_cos(s: str, t: str) -> float:
    if not s or not t:
        return 0.0

    def grams(x):
        x = " " + x + " "
        d = {}
        for i in range(len(x) - 2):
            g = x[i:i + 3]
            d[g] = d.get(g, 0) + 1
        return d
    a, b = grams(s), grams(t)
    dot = 0.0
    for g in sorted(a):
        if g in b:
            dot += a[g] * b[g]
    na = math.sqrt(sum(a[g] * a[g] for g in sorted(a)))
    nb = math.sqrt(sum(b[g] * b[g] for g in sorted(b)))
    return dot / (na * nb)


def jaccard(A, B) -> float:
    A, B = set(A), set(B)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def overlap(A, B) -> float:
    A, B = set(A), set(B)
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


def soft_match(A, B, idf, dflt):
    """Symmetric Monge-Elkan with Jaro-Winkler + IDF of rarest unmatched token."""
    if not A or not B:
        return 0.0, 0.0
    tot, worst = 0.0, 0.0
    for X, Y in ((A, B), (B, A)):
        acc = 0.0
        for x in X:
            best = 0.0
            for y in Y:
                v = jaro_winkler(x, y)
                if v > best:
                    best = v
            acc += best
            if best < 0.88:
                worst = max(worst, idf.get(x, dflt))
        tot += acc / len(X)
    return tot / 2, worst


def idf_jaccard(A, B, idf, dflt) -> float:
    A, B = sorted(set(A)), sorted(set(B))
    if not A or not B:
        return 0.0
    sb = set(B)
    inter = sum(idf.get(t, dflt) for t in A if t in sb)
    union = sum(idf.get(t, dflt) for t in sorted(set(A) | sb))
    return inter / union if union else 0.0


FEATURES = ["n_jw", "n_lev", "n_jac", "n_ovl", "n_cos3", "sk_jac", "sk_eq", "first_eq",
            "comp_lev", "soft_me", "idf_jac", "unmatched_idf", "suf_both", "suf_eq",
            "acr_hit", "alt_best", "a_jac", "a_cos3", "pc_both", "pc_eq", "pc_conf",
            "num_both", "num_eq", "num_conf", "a_empty", "len_diff", "ntok_diff"]


def pair_features(a: dict, b: dict, idf: dict, dflt: float) -> list:
    me, worst = soft_match(a["toks"], b["toks"], idf, dflt)
    ca, cb = a["core"].replace(" ", ""), b["core"].replace(" ", "")
    alts_a = [a["toks"]] + ([a["alt"].split()] if a["alt"] else [])
    alts_b = [b["toks"]] + ([b["alt"].split()] if b["alt"] else [])
    alt_best = max(overlap(x, y) for x in alts_a for y in alts_b)
    pcb = 1.0 if a["pc"] and b["pc"] else 0.0
    pce = 1.0 if pcb and set(a["pc"]) & set(b["pc"]) else 0.0
    nb = 1.0 if a["nums"] and b["nums"] else 0.0
    ne = 1.0 if nb and set(a["nums"]) & set(b["nums"]) else 0.0
    acr = 1.0 if ((len(a["acr"]) >= 2 and a["acr"] == cb) or (len(b["acr"]) >= 2 and b["acr"] == ca)) else 0.0
    return [
        jaro_winkler(a["core"], b["core"]), lev_ratio(a["core"], b["core"]),
        jaccard(a["toks"], b["toks"]), overlap(a["toks"], b["toks"]),
        trigram_cos(a["core"], b["core"]), jaccard(a["sk"], b["sk"]),
        1.0 if a["skel"] and a["skel"] == b["skel"] else 0.0,
        1.0 if a["toks"] and b["toks"] and a["toks"][0] == b["toks"][0] else 0.0,
        lev_ratio(ca, cb), me, idf_jaccard(a["toks"], b["toks"], idf, dflt), worst,
        1.0 if a["suffix"] and b["suffix"] else 0.0,
        1.0 if a["suffix"] and a["suffix"] == b["suffix"] else 0.0,
        acr, alt_best, jaccard(a["awords"], b["awords"]), trigram_cos(a["anorm"], b["anorm"]),
        pcb, pce, pcb - pce, nb, ne, nb - ne,
        1.0 if not a["anorm"] or not b["anorm"] else 0.0,
        float(abs(len(a["core"]) - len(b["core"]))), float(abs(len(a["toks"]) - len(b["toks"]))),
    ]


# ---------------------------------------------------------------- blocking
def keys(r: dict) -> list:
    out = []
    for t in r["sk"]:
        if len(t) >= 2:
            out.append("t|" + t)
    for t in r["toks"]:
        if len(t) >= 4:
            out.append("q|" + t[:4])
    if len(r["sk"]) >= 2:
        out.append("n|" + "_".join(sorted(r["sk"][:2])))
    comp = r["core"].replace(" ", "")
    if len(comp) >= 4:
        out.append("c|" + comp[:6])
    if r["pc"] and r["sk"]:
        out.append("p|" + r["pc"][0] + "|" + r["sk"][0])
    m = STREET.search(r["anorm"])
    if m:
        out.append("a|" + m.group(1) + "|" + skeleton(m.group(2)))
    seen, uniq = set(), []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def build_index(pool: list) -> dict:
    """Inverted index over one country's pool: key -> [pool positions]."""
    inv = {}
    for i, r in enumerate(pool):
        for k in keys(r):
            inv.setdefault(k, []).append(i)
    n = len(pool)
    cap = max(50, int(0.05 * n))
    weights = {k: math.log(1 + n / len(v)) for k, v in inv.items() if len(v) <= cap}
    return {"inv": inv, "w": weights, "n": n, "cap": cap}


def block(s: dict, pool: list, index: dict) -> list:
    """Return [(pool position, kb_score, block_score)] best first, at most K_MAX."""
    score = {}
    for k in keys(s):
        w = index["w"].get(k)
        if w is None:
            continue
        for j in index["inv"][k]:
            score[j] = score.get(j, 0.0) + w
    if not score:
        return []
    ranked = sorted(score.items(), key=lambda x: (-x[1], x[0]))[:K_RETRIEVE]
    top = ranked[0][1]
    res = []
    for j, kb in ranked:
        p = pool[j]
        bs = trigram_cos(s["core"], p["core"]) + 0.5 * trigram_cos(s["anorm"], p["anorm"]) + 0.2 * kb / top
        res.append((j, kb, bs))
    res.sort(key=lambda x: (-x[2], x[0]))
    return res[:K_MAX]


# ---------------------------------------------------------------- context (stage 2)
CTX = ["p1", "s1_rank", "s1_max", "s1_gap", "s1_top2_gap", "s1_sum", "t_rank",
       "t_best_other", "t_margin", "t_n_claims", "tri"]


def context(pairs: list, p1: list, recs: dict) -> list:
    """pairs: [(s1_id, t_id)] grouped by s1 in order; returns rows of CTX values."""
    by_s, by_t = {}, {}
    for i, (s, t) in enumerate(pairs):
        by_s.setdefault(s, []).append(i)
        by_t.setdefault(t, []).append(i)
    rank_s, rank_t = [0] * len(pairs), [0] * len(pairs)
    smax, s2, ssum = [0.0] * len(pairs), [0.0] * len(pairs), [0.0] * len(pairs)
    tbest, tn = [0.0] * len(pairs), [0] * len(pairs)
    for idx in by_s.values():
        order = sorted(idx, key=lambda i: (-p1[i], i))
        vals = [p1[i] for i in order]
        for r, i in enumerate(order):
            rank_s[i] = r + 1
            smax[i] = vals[0]
            s2[i] = vals[1] if len(vals) > 1 else 0.0
        tot = 0.0
        for i in idx:
            tot += p1[i]
        for i in idx:
            ssum[i] = tot
    for idx in by_t.values():
        order = sorted(idx, key=lambda i: (-p1[i], i))
        for r, i in enumerate(order):
            rank_t[i] = r + 1
            tn[i] = len(idx)
            others = [p1[k] for k in order if k != i]
            tbest[i] = others[0] if others else 0.0
    rows = []
    for i, (s, t) in enumerate(pairs):
        x = recs[t]
        tri = 0.0
        if p1[i] >= 0.02:
            for k in by_s[s]:
                if k == i or p1[k] < 0.2:
                    continue
                y = recs[pairs[k][1]]
                n = overlap(x["toks"], y["toks"])
                a = overlap(x["awords"], y["awords"]) if x["awords"] and y["awords"] else n
                v = (0.6 * n + 0.4 * a) * p1[k]
                if v > tri:
                    tri = v
        rows.append([p1[i], float(rank_s[i]), smax[i], smax[i] - p1[i], smax[i] - s2[i], ssum[i],
                     float(rank_t[i]), tbest[i], p1[i] - tbest[i], float(tn[i]), tri])
    return rows
