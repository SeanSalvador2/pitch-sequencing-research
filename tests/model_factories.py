"""Small, fast, seeded LightGBM model factories for the falsification tests.

The falsification / harness code (decision D17) takes a caller-supplied
``model_factory(view) -> model`` with ``fit(X, y)`` / ``predict_proba(X)`` (and
``classes_``). These wrappers adapt a compact, deterministic LightGBM classifier to that
contract and prepare the mixed-dtype state-view frames (categoricals stay categorical;
booleans and strings are coerced to LightGBM-friendly dtypes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

_FAST_PARAMS = dict(
    n_estimators=60,
    num_leaves=15,
    min_child_samples=30,
    learning_rate=0.10,
    max_depth=-1,
    reg_lambda=1.0,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
    random_state=0,
    verbose=-1,
)


def _prep(X: pd.DataFrame) -> pd.DataFrame:
    """Coerce a state-view frame to dtypes LightGBM accepts (categoricals preserved)."""
    X = X.copy()
    for c in X.columns:
        dt = X[c].dtype
        if isinstance(dt, pd.CategoricalDtype):
            continue
        if dt == bool:
            X[c] = X[c].astype("int8")
        elif dt == object:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


class LGBMProbaModel:
    """Adapter exposing ``fit`` / ``predict_proba`` / ``classes_`` over ``LGBMClassifier``."""

    def __init__(self, **params) -> None:
        self.params = {**_FAST_PARAMS, **params}
        self.model: LGBMClassifier | None = None
        self.classes_ = None

    def fit(self, X, y):
        self.model = LGBMClassifier(**self.params)
        self.model.fit(_prep(X), np.asarray(y))
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(_prep(X))


def make_lgbm_factory(**overrides):
    """Return a ``model_factory(view) -> LGBMProbaModel`` ignoring the view name."""

    def factory(view: str) -> LGBMProbaModel:
        return LGBMProbaModel(**overrides)

    return factory
