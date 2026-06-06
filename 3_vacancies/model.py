from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor


TEXT_COLS: List[str] = [
    "name",
    "employer_name",
    "experience_name",
    "schedule_name",
    "key_skills_name",
    "unified_address_city",
    "unified_address_state",
    "unified_address_region",
    "unified_address_country",
    "specializations_profarea_name",
    "professional_roles_name",
    "languages_name",
    "raw_description",
    "raw_branded_description",
    "lemmaized_wo_stopwords_raw_description",
    "lemmaized_wo_stopwords_raw_branded_description",
    "name_clean",
    "employment_name",
    "employer_industries",
]

CAT_COLS: List[str] = [
    "accept_handicapped",
    "accept_kids",
    "if_foreign_language",
    "is_branded_description",
]

DROP_COLS: List[str] = [
    "id",
    "salary_mean_net",
]


def join_text_columns(df: pd.DataFrame) -> pd.Series:
    """Concatenate many text columns into one string per row."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    data = df.copy()
    for col in data.columns:
        data[col] = data[col].fillna("").astype(str)

    joined = data.agg(" ".join, axis=1)
    joined = joined.str.replace(r"\s+", " ", regex=True).str.strip()
    return joined


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup before model pipeline."""
    out = df.copy()

    for col in TEXT_COLS + CAT_COLS:
        if col not in out.columns:
            out[col] = np.nan

    for col in TEXT_COLS:
        out[col] = out[col].fillna("").astype(str)

    for col in CAT_COLS:
        out[col] = out[col].fillna("unknown").astype(str)


    if "employer_id" in out.columns:
        out["employer_id"] = out["employer_id"].fillna(-1).astype(str)
        out["employer_id_token"] = "employer_" + out["employer_id"]
        if "employer_id_token" not in TEXT_COLS:
            out["employer_id_token"] = out["employer_id_token"].astype(str)

    return out


def make_model() -> Pipeline:
    text_union = FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    max_features=60000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    max_features=40000,
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )

    text_pipe = Pipeline(
        steps=[
            ("join", FunctionTransformer(join_text_columns, validate=False)),
            ("union", text_union),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("text", text_pipe, TEXT_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    reg = Ridge(alpha=12.0, random_state=42)

    model = Pipeline(
        steps=[
            ("prep", preprocess),
            (
                "reg",
                TransformedTargetRegressor(
                    regressor=reg,
                    func=np.log1p,
                    inverse_func=np.expm1,
                    check_inverse=False,
                ),
            ),
        ]
    )
    return model


def evaluate_cv(df: pd.DataFrame, target_col: str = "salary_mean_net") -> None:
    if target_col not in df.columns:
        return

    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col].astype(float).values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = make_model()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        preds = np.clip(preds, 0, None)
        score = mean_absolute_percentage_error(y_val, preds)
        scores.append(score)
        print(f"Fold {fold}: MAPE = {score:.5f}")

    print(f"CV mean MAPE: {np.mean(scores):.5f} ± {np.std(scores):.5f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to train CSV")
    parser.add_argument("--test", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--out", type=str, default="submission.csv", help="Output CSV")
    parser.add_argument("--no-cv", action="store_true", help="Skip 5-fold CV printing")
    args = parser.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    out_path = Path(args.out)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "salary_mean_net" not in train_df.columns:
        raise ValueError("Train file must contain salary_mean_net")

    train_df = clean_frame(train_df)
    test_df = clean_frame(test_df)

    train_df = train_df.loc[train_df["salary_mean_net"].notna()].copy()
    train_df["salary_mean_net"] = train_df["salary_mean_net"].astype(float).clip(lower=0)

    if not args.no_cv:
        evaluate_cv(train_df, target_col="salary_mean_net")

    X_train = train_df.drop(columns=DROP_COLS, errors="ignore")
    y_train = train_df["salary_mean_net"].values

    X_test = test_df.drop(columns=DROP_COLS, errors="ignore")

    model = make_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)

    if "id" in test_df.columns:
        out_df = pd.DataFrame(
            {
                "id": test_df["id"].values,
                "salary_mean_net": preds,
            }
        )
    else:
        out_df = pd.DataFrame(
            {
                "id": np.arange(len(test_df)),
                "salary_mean_net": preds,
            }
        )

    out_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
