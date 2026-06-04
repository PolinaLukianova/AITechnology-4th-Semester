from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
TRAIN_PATH = DATA_DIR / "prices_train.csv"
TEST_PATH = DATA_DIR / "prices_test.csv"
NAMED_OUT_PATH = SUBMISSIONS_DIR / "submission_03_with_log_target.csv"

TARGET = "Y house price of unit area"


SUBMISSIONS_DIR.mkdir(exist_ok=True)

train = pd.read_csv(TRAIN_PATH, index_col=0)
test  = pd.read_csv(TEST_PATH, index_col=0)
y_train = train[TARGET].copy()


lat0 = train["X5 latitude"].median()
lon0 = train["X6 longitude"].median()
date0 = train["X1 transaction date"].median()

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.replace([np.inf, -np.inf], np.nan)

    date = d["X1 transaction date"]
    age  = d["X2 house age"]
    dist = d["X3 distance to the nearest MRT station"]
    stores = d["X4 number of convenience stores"]
    lat = d["X5 latitude"]
    lon = d["X6 longitude"]

    d["date_centered"] = date - date0
    frac = date - np.floor(date)
    d["date_sin"] = np.sin(2 * np.pi * frac)
    d["date_cos"] = np.cos(2 * np.pi * frac)

    d["log_dist"] = np.log1p(dist)
    d["inv_dist"] = 1 / (1 + dist)

    d["age_sq"] = age ** 2
    d["age_log"] = np.log1p(age)

    d["lat_c"] = lat - lat0
    d["lon_c"] = lon - lon0
    d["geo_dist"] = np.sqrt(d["lat_c"] ** 2 + d["lon_c"] ** 2)

    d["lat_sq"] = d["lat_c"] ** 2
    d["lon_sq"] = d["lon_c"] ** 2
    d["lat_lon_interaction"] = d["lat_c"] * d["lon_c"]

    d["stores_x_logdist"] = stores * d["log_dist"]
    d["stores_x_age"] = stores * age
    d["stores_x_lat"] = stores * d["lat_c"]
    d["stores_x_lon"] = stores * d["lon_c"]
    d["dist_x_age"] = d["log_dist"] * age

    return d


train_fe = make_features(train)
test_fe  = make_features(test)


FEATURE_SETS = {
    "balanced": [
        "X2 house age",
        "X4 number of convenience stores",
        "date_centered",
        "date_sin",
        "date_cos",
        "log_dist",
        "inv_dist",
        "age_sq",
        "lat_c",
        "lon_c",
        "geo_dist",
        "lat_lon_interaction",
        "stores_x_logdist",
        "stores_x_age",
    ],
    "geo_rich": [
        "X2 house age",
        "X4 number of convenience stores",
        "date_centered",
        "date_sin",
        "date_cos",
        "log_dist",
        "inv_dist",
        "age_sq",
        "age_log",
        "lat_c",
        "lon_c",
        "geo_dist",
        "lat_sq",
        "lon_sq",
        "lat_lon_interaction",
        "stores_x_logdist",
        "stores_x_age",
        "stores_x_lat",
        "stores_x_lon",
        "dist_x_age",
    ],
    "compact": [
        "X2 house age",
        "X4 number of convenience stores",
        "date_centered",
        "log_dist",
        "inv_dist",
        "age_sq",
        "lat_c",
        "lon_c",
        "geo_dist",
        "stores_x_logdist",
        "lat_lon_interaction",
    ],
}


def clip_by_quantiles(train_df, test_df, cols, low_q, high_q):
    tr = train_df.copy()
    te = test_df.copy()
    for col in cols:
        lo = tr[col].quantile(low_q)
        hi = tr[col].quantile(high_q)
        tr[col] = tr[col].clip(lo, hi)
        te[col] = te[col].clip(lo, hi)
    return tr, te


def make_model():
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    return TransformedTargetRegressor(
        regressor=base,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False
    )


clip_grid = [
    (0.005, 0.995),
    (0.01, 0.99),
    (0.02, 0.98),
    (0.03, 0.97),
]

best = {
    "mse": np.inf,
    "low_q": None,
    "high_q": None,
    "features": None,
}

for feats in FEATURE_SETS.values():
    for low_q, high_q in clip_grid:
        tr_clipped, _ = clip_by_quantiles(train_fe, test_fe, feats, low_q, high_q)

        X = tr_clipped[feats]
        model = make_model()

        oof = cross_val_predict(model, X, y_train, cv=5, n_jobs=-1)
        mse = mean_squared_error(y_train, oof)

        if mse < best["mse"]:
            best.update({
                "mse": mse,
                "low_q": low_q,
                "high_q": high_q,
                "features": feats,
            })


X_train_final, X_test_final = clip_by_quantiles(
    train_fe, test_fe, best["features"], best["low_q"], best["high_q"]
)

final_model = make_model()
final_model.fit(X_train_final[best["features"]], y_train)

y_pred = final_model.predict(X_test_final[best["features"]])

submission = pd.DataFrame({
    "index": test.index,
    TARGET: np.round(y_pred, 2)
})

submission.to_csv(NAMED_OUT_PATH, index=False)
