"""Steps 3–7 of the MetroCare pipeline: label, engineer features, train, evaluate and explain.

Run after prepare_data.py:   python train_models.py
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve

from metrocare.config import (FAILURES, METRICS_FILE, MINUTE_FILE, MODEL_DIR, MODEL_NAMES, PRE_FAILURE_HOURS,
                              SCORES_FILE, SPLIT_DATE)
from metrocare.features import build_features, feature_signal, make_labels
from metrocare.models import HybridDetector, make_model
from metrocare.scoring import evaluate, model_scores, smooth

THRESHOLD = 0.5
NEG_STRIDE = 3  # keep every 3rd normal minute for training (all failure minutes are kept)


def subsample(X, y):
    keep = (y == 1) | (np.arange(len(y)) % NEG_STRIDE == 0)
    return X[keep], y[keep]


def fit(name, X, y):
    Xs, ys = subsample(X, y)
    return make_model(name).fit(Xs, ys)


def pr_points(y, s, n=150):
    p, r, t = precision_recall_curve(y, s)
    idx = np.unique(np.linspace(0, len(p) - 1, n).astype(int))
    return {"precision": p[idx].round(4).tolist(), "recall": r[idx].round(4).tolist()}


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    m = pd.read_parquet(MINUTE_FILE)
    X = build_features(m)
    print(f"Features: {X.shape} in {time.time() - t0:.0f}s")

    y_pre = make_labels(X.index, PRE_FAILURE_HOURS)   # failure window + 2 h warning
    y_win = make_labels(X.index, 0)                    # failure window only
    tr = X.index < SPLIT_DATE
    te = ~tr

    metrics = {"threshold": THRESHOLD, "split_date": str(SPLIT_DATE.date()), "n_features": X.shape[1],
               "train_minutes": int(tr.sum()), "test_minutes": int(te.sum()),
               "train_failure_minutes": int(y_pre[tr].sum()), "test_failure_minutes": int(y_pre[te].sum()),
               "models": {}, "label_comparison": {}, "pr_curves": {}}
    scores = pd.DataFrame(index=X.index)
    fitted, fitted_w = {}, {}

    for key, label in MODEL_NAMES.items():
        ts = time.time()
        if key == "hybrid":  # combines models already trained above, no extra fitting
            fwd = HybridDetector(fitted["logreg"][0], fitted["iforest"][0])
            bwd = HybridDetector(fitted["logreg"][1], fitted["iforest"][1])
        else:
            fwd = fit(key, X[tr], y_pre[tr])      # trained on Feb–May → tested on Jun–Sep
            bwd = fit(key, X[te], y_pre[te])      # trained on Jun–Sep → scores Feb–May (for replaying failures 1–2)
        fitted[key] = (fwd, bwd)
        s_te = pd.Series(model_scores(fwd, X[te]), index=X.index[te])
        s_tr = pd.Series(model_scores(bwd, X[tr]), index=X.index[tr])
        scores[f"risk_{key}"] = pd.concat([s_tr, s_te]).astype("float32")

        res = evaluate(smooth(s_te), y_pre[te], THRESHOLD)
        res["backward"] = evaluate(smooth(s_tr), y_pre[tr], THRESHOLD)
        res["fit_seconds"] = round(time.time() - ts, 1)
        metrics["models"][key] = res
        metrics["pr_curves"][key] = pr_points(y_pre[te], smooth(s_te))
        joblib.dump(fwd, MODEL_DIR / f"{key}_fwd.joblib", compress=3)
        joblib.dump(bwd, MODEL_DIR / f"{key}_bwd.joblib", compress=3)

        # Labelling comparison: train on the failure window only
        if key == "iforest":
            w = fwd
        elif key == "hybrid":
            w = HybridDetector(fitted_w["logreg"], fwd.anomaly)
        else:
            w = fit(key, X[tr], y_win[tr])
        fitted_w[key] = w
        if key == "iforest":
            res_w = evaluate(smooth(s_te), y_win[te], THRESHOLD)
        else:
            res_w = evaluate(smooth(pd.Series(model_scores(w, X[te]), index=X.index[te])), y_win[te], THRESHOLD)
        metrics["label_comparison"][key] = {k: res_w[k] for k in ("precision", "recall", "f1", "pr_auc", "false_alarms_per_day", "failures_detected", "mean_lead_h")}
        print(f"{label:22s} F1={res['f1']:.3f} PR-AUC={res['pr_auc']:.3f} FA/day={res['false_alarms_per_day']:.2f} "
              f"detected={res['failures_detected']}/{res['failures_total']} ({res['fit_seconds']}s)")

    # Best model: most failures caught, then highest PR-AUC
    best = max(metrics["models"], key=lambda k: (metrics["models"][k]["failures_detected"], metrics["models"][k]["pr_auc"]))
    if best == "iforest":  # explanations need a supervised tree model; fall back to the best of rf/gb
        best_tree = max(("rf", "gb"), key=lambda k: metrics["models"][k]["pr_auc"])
    else:
        best_tree = best if best in ("rf", "gb") else max(("rf", "gb"), key=lambda k: metrics["models"][k]["pr_auc"])
    metrics["best_model"] = best
    metrics["explain_model"] = best_tree

    # Global importance per model
    imp = {}
    lr = joblib.load(MODEL_DIR / "logreg_fwd.joblib")
    imp["logreg"] = dict(zip(X.columns, np.abs(lr[-1].coef_[0]).tolist()))
    rf = joblib.load(MODEL_DIR / "rf_fwd.joblib")
    imp["rf"] = dict(zip(X.columns, rf.feature_importances_.tolist()))
    gb = joblib.load(MODEL_DIR / "gb_fwd.joblib")
    rng = np.random.default_rng(0)
    sample_idx = np.sort(np.r_[np.flatnonzero(y_pre[te].to_numpy()), rng.choice(int(te.sum()), 6000, replace=False)])
    Xs, ys = X[te].iloc[sample_idx], y_pre[te].iloc[sample_idx]
    pi = permutation_importance(gb, Xs, ys, scoring="average_precision", n_repeats=2, random_state=0, n_jobs=-1)
    imp["gb"] = dict(zip(X.columns, np.clip(pi.importances_mean, 0, None).tolist()))
    try:  # Isolation Forest: mean |SHAP| of isolation depth on test minutes
        import shap
        ad = joblib.load(MODEL_DIR / "iforest_fwd.joblib")
        sx = Xs.sample(400, random_state=0)
        sv = shap.TreeExplainer(ad.pipe[-1]).shap_values(ad.pipe[0].transform(sx))
        imp["iforest"] = dict(zip(X.columns, np.abs(sv).mean(0).tolist()))
    except Exception as e:
        print("Isolation Forest importance skipped:", e)
    for k in imp:
        tot = sum(imp[k].values()) or 1
        imp[k] = {f: v / tot for f, v in imp[k].items()}
    metrics["importance"] = imp

    # Global SHAP summary for the explanation model (sample of test minutes)
    try:
        import shap
        model = joblib.load(MODEL_DIR / f"{best_tree}_fwd.joblib")
        sx = Xs.sample(min(1500, len(Xs)), random_state=0)
        sv = shap.TreeExplainer(model).shap_values(sx)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv)
        pd.DataFrame(sv, columns=X.columns, index=sx.index).astype("float32").to_parquet(MODEL_DIR / "shap_sample.parquet")
        sx.to_parquet(MODEL_DIR / "shap_sample_X.parquet")
        metrics["shap_by_signal"] = (pd.Series(np.abs(sv).mean(0), index=X.columns)
                                     .groupby(feature_signal).sum().sort_values(ascending=False).round(5).to_dict())
    except Exception as e:  # SHAP is optional
        print("SHAP skipped:", e)

    scores.to_parquet(SCORES_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"Best model: {MODEL_NAMES[best]} · explanations use {MODEL_NAMES[best_tree]} · total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
