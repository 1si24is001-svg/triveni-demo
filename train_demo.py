"""Train the TRIVENI demo models and export a bundle for the web app.

Train split (US + India): block -> stage-1 LightGBM (5-fold out-of-fold) ->
stage-2 LightGBM on context features -> isotonic calibration.
Test split (US + India + France, France unseen): full pipeline, metrics.
Writes web/bundle.json with test data, trees, calibrator, IDF and the Python
predictions the browser must reproduce.
"""
import json
import math
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

import demo_core as D
from decode import expected_fbeta_set

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=15, min_data_in_leaf=20,
              feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=3, deterministic=True, num_threads=1)


def load(name):
    d = json.load(open(f"data/{name}.json"))
    s1 = [D.prep(r) for r in d["s1"]]
    pool = [D.prep(r) for r in d["pool"]]
    return d, s1, pool


def candidates(s1, pool):
    by_c = {}
    for p in pool:
        by_c.setdefault(p["country"], []).append(p)
    idx = {c: D.build_index(v) for c, v in by_c.items()}
    pairs, kb = [], []
    total_pairs = 0
    for s in s1:
        P = by_c.get(s["country"], pool)
        I = idx.get(s["country"]) or D.build_index(pool)
        total_pairs += len(P)
        for j, k, b in D.block(s, P, I):
            pairs.append((s["id"], P[j]["id"]))
            kb.append((k, b))
    return pairs, kb, total_pairs


def fbeta(pred, truth, b2=0.25):
    pred, truth = set(pred), set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + b2) * p * r / (b2 * p + r)


def decode(pairs, probs, s1_ids, method="expected", th=0.5):
    groups = {}
    for (s, t), p in zip(pairs, probs):
        groups.setdefault(s, []).append((p, t))
    out = {}
    for s in s1_ids:
        g = sorted(groups.get(s, []), key=lambda x: -x[0])
        if method == "threshold":
            out[s] = [t for p, t in g if p >= th]
        else:
            k, _ = expected_fbeta_set([p for p, _ in g]) if g and g[0][0] >= 1e-3 else (0, 0)
            out[s] = [t for _, t in g[:k]]
    return out


def oof(X, y, groups, rounds=400):
    o = np.zeros(len(y))
    its = []
    for tr, va in GroupKFold(5).split(X, y, groups):
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), rounds, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(25, verbose=False)])
        o[va] = m.predict(X[va], num_iteration=m.best_iteration)
        its.append(m.best_iteration)
    return o, max(20, int(np.mean(its)))


def export_trees(model):
    """LightGBM trees as flat arrays: internal [f, thr, left, right], leaf [-1, value]."""
    out = []
    for t in model.dump_model()["tree_info"]:
        nodes = []

        def walk(n):
            i = len(nodes)
            nodes.append(None)
            if "leaf_value" in n:
                nodes[i] = [-1, n["leaf_value"]]
            else:
                assert n["decision_type"] == "<=" and n["missing_type"] in ("None", "NaN"), n
                l, r = walk(n["left_child"]), walk(n["right_child"])
                nodes[i] = [n["split_feature"], n["threshold"], l, r]
            return i
        walk(t["tree_structure"])
        out.append(nodes)
    return out


def metrics(pred, truth, ids):
    s = [fbeta(pred[i], truth.get(i, [])) for i in ids]
    tp = sum(len(set(pred[i]) & set(truth.get(i, []))) for i in ids)
    npred = sum(len(pred[i]) for i in ids)
    ntrue = sum(len(truth.get(i, [])) for i in ids)
    sing = [i for i in ids if not truth.get(i)]
    return {"f05": float(np.mean(s)), "precision": tp / max(npred, 1), "recall": tp / max(ntrue, 1),
            "singleton_acc": float(np.mean([not pred[i] for i in sing])) if sing else None,
            "n": len(ids)}


def main():
    dtr, s1, pool = load("train")
    names = [r["core"] for r in s1 + pool]
    df = Counter(t for c in names for t in set(c.split()))
    n = len(names)
    idf = {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}
    dflt = max(idf.values())
    recs = {r["id"]: r for r in s1 + pool}
    truth = dtr["truth"]
    true_set = {(s, t) for s, ts in truth.items() for t in ts}

    pairs, kb, total = candidates(s1, pool)
    y = np.array([(s, t) in true_set for s, t in pairs], dtype=int)
    groups = np.array([s for s, _ in pairs])
    n_true = len(true_set)
    print(f"train: {len(s1)} S1, {len(pool)} pool, all pairs {total:,} -> candidates {len(pairs):,} "
          f"({len(pairs)/len(s1):.1f}/entity), recall {y.sum()/n_true:.4f}")
    X1 = np.array([D.pair_features(recs[s], recs[t], idf, dflt) for s, t in pairs])
    o1, it1 = oof(X1, y, groups)
    X2 = np.array(D.context(pairs, list(o1), recs))
    o2, it2 = oof(X2, y, groups)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(o2, y)
    pc = iso.predict(o2)
    ids = [r["id"] for r in s1]
    oofm = {"stage1_threshold": metrics(decode(pairs, o1, ids, "threshold"), truth, ids),
            "stage2_threshold": metrics(decode(pairs, pc, ids, "threshold"), truth, ids),
            "stage2_expected": metrics(decode(pairs, pc, ids, "expected"), truth, ids)}
    for k, v in oofm.items():
        print(f"  OOF {k:18s} F0.5={v['f05']:.4f} P={v['precision']:.4f} R={v['recall']:.4f}")
    m1 = lgb.train(PARAMS, lgb.Dataset(X1, y), it1)
    m2 = lgb.train(PARAMS, lgb.Dataset(X2, y), it2)
    imp1 = sorted(zip(D.FEATURES, m1.feature_importance("gain")), key=lambda x: -x[1])
    imp2 = sorted(zip(D.CTX, m2.feature_importance("gain")), key=lambda x: -x[1])

    # ---------------- test
    dte, t1, tpool = load("test")
    trecs = {r["id"]: r for r in t1 + tpool}
    tpairs, tkb, ttotal = candidates(t1, tpool)
    tX1 = np.array([D.pair_features(trecs[s], trecs[t], idf, dflt) for s, t in tpairs])
    p1 = m1.predict(tX1)
    tX2 = np.array(D.context(tpairs, list(p1), trecs))
    p2 = m2.predict(tX2)
    p3 = iso.predict(p2)
    tids = [r["id"] for r in t1]
    pred = decode(tpairs, p3, tids, "expected")
    tt = dte["truth"]
    tm = metrics(pred, tt, tids)
    tm_thr = metrics(decode(tpairs, p3, tids, "threshold"), tt, tids)
    tm_s1 = metrics(decode(tpairs, p1, tids, "threshold"), tt, tids)
    tset = {(s, t) for s, ts in tt.items() for t in ts}
    tm["blocking_recall"] = sum((s, t) in tset for s, t in tpairs) / max(len(tset), 1)
    by_c = {}
    for c in ("US", "India", "France"):
        cid = [r["id"] for r in t1 if r["country"] == c]
        by_c[c] = metrics(pred, tt, cid)
    print(f"test: all pairs {ttotal:,} -> {len(tpairs):,} candidates; blocking recall {tm['blocking_recall']:.4f}")
    print(f"  stage-1 @0.5: F0.5={tm_s1['f05']:.4f} | stage-2 @0.5: {tm_thr['f05']:.4f} | expected: {tm['f05']:.4f}")
    print("  by country:", {c: round(v["f05"], 4) for c, v in by_c.items()})

    bundle = {
        "test": {"s1": dte["s1"], "pool": dte["pool"], "truth": tt, "script": dte["script"]},
        "idf": idf, "idf_default": dflt, "features": D.FEATURES, "ctx": D.CTX,
        "k_retrieve": D.K_RETRIEVE, "k_max": D.K_MAX,
        "m1": export_trees(m1), "m2": export_trees(m2),
        "iso": {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()},
        "python": {"pairs": tpairs, "p1": p1.tolist(), "p3": p3.tolist(), "pred": pred,
                   "metrics": tm, "metrics_threshold": tm_thr, "metrics_stage1": tm_s1, "by_country": by_c,
                   "all_pairs": ttotal},
        "train": {"n_s1": len(s1), "n_pool": len(pool), "n_pairs": len(pairs), "oof": oofm,
                  "imp1": [[a, float(b)] for a, b in imp1[:10]], "imp2": [[a, float(b)] for a, b in imp2[:8]],
                  "trees1": m1.num_trees(), "trees2": m2.num_trees()},
    }
    Path("web").mkdir(exist_ok=True)
    json.dump(bundle, open("web/bundle.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print("bundle KB:", round(Path("web/bundle.json").stat().st_size / 1024), "| trees", m1.num_trees(), m2.num_trees())


if __name__ == "__main__":
    main()
