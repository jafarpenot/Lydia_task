"""Transaction classifier: deterministic refund rule + XGBoost on the remaining transactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

from txlabel.preprocessing import (
    CATEGORICAL_COLS,
    MINOR_TO_MAJOR,
    NO_MAJOR,
    NUMERIC_COLS,
    REFUND,
    TEXT_COL,
    build_features,
    is_refund,
)

DEFAULT_XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.5,
    tree_method="hist",
    eval_metric="mlogloss",
)


def build_pipeline(seed: int = 42, **xgb_params) -> Pipeline:
    """Feature transformer + XGBoost. Input: the DataFrame produced by ``build_features``."""
    features = ColumnTransformer(
        [
            # presence of character 2-4-grams of the normalized merchant name (capped vocabulary)
            (
                "name",
                CountVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=2000, binary=True),
                TEXT_COL,
            ),
            # no min_frequency: rare digitalization values are kept on purpose (strong signal, see EDA notes)
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS),
        ]
    )
    params = {**DEFAULT_XGB_PARAMS, **xgb_params}
    clf = XGBClassifier(objective="multi:softprob", random_state=seed, n_jobs=-1, **params)
    return Pipeline([("features", features), ("clf", clf)])


class TransactionClassifier:
    """Predicts ``minor_category`` (and the derived ``major_category``) from raw transaction columns.

    * amount > 0  -> ``refund`` (rule, confidence 1.0, major = ``none``)
    * otherwise   -> XGBoost trained on non-refund rows only
    The major category is derived from the minor one with the strict hierarchy found in the data.
    """

    def __init__(self, seed: int = 42, **xgb_params):
        self.seed = seed
        self.xgb_params = xgb_params
        self.pipeline_: Pipeline | None = None
        self.label_encoder_: LabelEncoder | None = None

    # ------------------------------------------------------------------ training
    def fit(self, raw: pd.DataFrame, y_minor: pd.Series) -> "TransactionClassifier":
        refund = is_refund(raw)
        bad = (refund != (y_minor == REFUND)).sum()
        if bad:
            raise ValueError(f"refund rule disagrees with labels on {bad} rows (rule: amount > 0)")
        X = build_features(raw[~refund])
        self.label_encoder_ = LabelEncoder().fit(y_minor[~refund])
        self.pipeline_ = build_pipeline(self.seed, **self.xgb_params)
        self.pipeline_.fit(X, self.label_encoder_.transform(y_minor[~refund]))
        return self

    # ------------------------------------------------------------------ inference
    @property
    def classes_(self) -> list[str]:
        return [*self.label_encoder_.classes_, REFUND]

    def predict_proba(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Probability of every minor class (columns = ``classes_``), one row per input row."""
        refund = is_refund(raw).to_numpy()
        proba = pd.DataFrame(0.0, index=raw.index, columns=self.classes_)
        proba.loc[refund, REFUND] = 1.0
        if (~refund).any():
            X = build_features(raw[~refund])
            p = self.pipeline_.predict_proba(X)
            proba.loc[~refund, list(self.label_encoder_.classes_)] = p
        return proba

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(raw)
        minor = proba.idxmax(axis=1)
        return pd.DataFrame(
            {
                "minor_category": minor,
                "major_category": minor.map(MINOR_TO_MAJOR).fillna(NO_MAJOR),
                "confidence": proba.max(axis=1),
            },
            index=raw.index,
        )

    # ------------------------------------------------------------------ introspection
    def feature_importances(self) -> pd.Series:
        names = self.pipeline_.named_steps["features"].get_feature_names_out()
        imp = self.pipeline_.named_steps["clf"].feature_importances_
        return pd.Series(imp, index=names).sort_values(ascending=False)
