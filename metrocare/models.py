"""Model definitions. Kept in the package so joblib can reload them inside the app."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """Isolation Forest trained on normal minutes only, with scores mapped to a 0–1 risk.

    Risk = 0.5 at the 99.5th percentile of anomaly scores seen during normal training data.
    """

    def __init__(self, random_state: int = 42):
        self.pipe = make_pipeline(StandardScaler(), IsolationForest(n_estimators=200, max_samples=4096, random_state=random_state, n_jobs=-1))

    def fit(self, X, y=None):
        Xn = X if y is None else X[np.asarray(y) == 0]
        self.pipe.fit(Xn)
        s = self._raw(Xn)
        self.q_mid = float(np.quantile(s, 0.995))
        self.scale = float(max(np.quantile(s, 0.9999) - self.q_mid, 1e-6) / 3)
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def _raw(self, X):
        return -self.pipe.score_samples(X)

    def risk(self, X):
        return 1 / (1 + np.exp(-(self._raw(X) - self.q_mid) / self.scale))


class HybridDetector:
    """Risk = the higher of a supervised model (knows past leak signatures) and an anomaly detector (flags new ones)."""

    def __init__(self, supervised, anomaly):
        self.supervised, self.anomaly = supervised, anomaly
        self.feature_names_in_ = anomaly.feature_names_in_

    def risk(self, X):
        return np.maximum(self.supervised.predict_proba(X)[:, 1], self.anomaly.risk(X))


def make_model(name: str):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=0.05, max_iter=3000))
    if name == "rf":
        return RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=20, max_features="sqrt",
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    if name == "gb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                              class_weight="balanced", random_state=42)
    if name == "iforest":
        return AnomalyDetector()
    raise KeyError(name)
