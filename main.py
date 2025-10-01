import argparse
import math
import time
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from tqdm import tqdm

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import optuna


def set_global_seed(seed: int) -> None:
    """Set seeds across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def read_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    # Prefer common ID names, case-insensitive
    for candidate in ["ID", "Id", "id"]:
        if candidate in df.columns:
            return candidate
    # Fallback: choose the first column if it's unique length
    first_col = df.columns[0] if len(df.columns) > 0 else None
    if first_col is not None and df[first_col].nunique(dropna=False) == len(df):
        return first_col
    return None


def basic_preprocess(
    df: pd.DataFrame,
    id_col: str = "ID",
    target_col: Optional[str] = "BeatsPerMinute",
    engineer_features: bool = False,
    fe_clean: bool = False,
    add_kmeans: Optional[List[int]] = None,
    add_pca: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    - Separates target if present
    - Adds missing-value indicators for numeric columns
    - Returns feature frame X and optional y
    """
    y = None
    if target_col is not None and target_col in df.columns:
        y = df[target_col].copy()
    feature_cols = [c for c in df.columns if c not in {id_col, target_col}]
    X = df[feature_cols].copy()

    # Add missing indicators for numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if X[col].isna().any():
            X[f"{col}__isna"] = X[col].isna().astype(np.int8)

    # Fill remaining NaNs with column medians for numeric
    for col in numeric_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Simple handling for non-numeric columns (rare in this dataset): fillna with "missing"
    non_numeric_cols = [c for c in X.columns if c not in numeric_cols]
    for col in non_numeric_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna("missing")

    # Feature engineering based on top importances
    if engineer_features:
        # Clean mode restricts to a small, high-signal set
        if fe_clean:
            if "TrackDurationMs" in X.columns:
                X["TrackDurationMs_log"] = np.log1p(X["TrackDurationMs"])
            if "RhythmScore" in X.columns:
                X["RhythmScore_log"] = np.log1p(X["RhythmScore"] + 1)
            if "Energy" in X.columns:
                X["Energy_log"] = np.log1p(X["Energy"])
            if "MoodScore" in X.columns:
                X["MoodScore_log"] = np.log1p(X["MoodScore"])
            if "TrackDurationMs" in X.columns and "Energy" in X.columns:
                X["TrackDurationMs_x_Energy"] = X["TrackDurationMs"] * X["Energy"]
            if "RhythmScore" in X.columns and "Energy" in X.columns:
                X["RhythmScore_x_Energy"] = X["RhythmScore"] * X["Energy"]
            if "MoodScore" in X.columns and "Energy" in X.columns:
                X["MoodScore_x_Energy"] = X["MoodScore"] * X["Energy"]
            return X, y
        # Log transforms for skewed features
        if "TrackDurationMs" in X.columns:
            X["TrackDurationMs_log"] = np.log1p(X["TrackDurationMs"])
        if "RhythmScore" in X.columns:
            X["RhythmScore_log"] = np.log1p(X["RhythmScore"] + 1)  # +1 to handle negatives
        if "Energy" in X.columns:
            X["Energy_log"] = np.log1p(X["Energy"])
        if "MoodScore" in X.columns:
            X["MoodScore_log"] = np.log1p(X["MoodScore"])

        # Interactions between top features
        if "TrackDurationMs" in X.columns and "Energy" in X.columns:
            X["TrackDurationMs_x_Energy"] = X["TrackDurationMs"] * X["Energy"]
        if "RhythmScore" in X.columns and "Energy" in X.columns:
            X["RhythmScore_x_Energy"] = X["RhythmScore"] * X["Energy"]
        if "MoodScore" in X.columns and "Energy" in X.columns:
            X["MoodScore_x_Energy"] = X["MoodScore"] * X["Energy"]

        # Ratios
        if "TrackDurationMs" in X.columns and "Energy" in X.columns:
            X["TrackDurationMs_div_Energy"] = X["TrackDurationMs"] / (X["Energy"] + 1e-8)
        if "RhythmScore" in X.columns and "MoodScore" in X.columns:
            X["RhythmScore_div_MoodScore"] = X["RhythmScore"] / (X["MoodScore"] + 1e-8)

        # Polynomial features for top 2
        if "TrackDurationMs" in X.columns:
            X["TrackDurationMs_sq"] = X["TrackDurationMs"] ** 2
        if "RhythmScore" in X.columns:
            X["RhythmScore_sq"] = X["RhythmScore"] ** 2

        # Binning for TrackDurationMs (most important)
        if "TrackDurationMs" in X.columns:
            X["TrackDurationMs_binned"] = pd.cut(X["TrackDurationMs"], bins=5, labels=False, duplicates="drop")

        # Statistical features for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col in ["TrackDurationMs", "RhythmScore", "Energy", "MoodScore"]:
                # Quantile features
                X[f"{col}_q25"] = X[col].quantile(0.25)
                X[f"{col}_q75"] = X[col].quantile(0.75)
                X[f"{col}_iqr"] = X[f"{col}_q75"] - X[f"{col}_q25"]
                
                # Z-score normalization
                X[f"{col}_zscore"] = (X[col] - X[col].mean()) / (X[col].std() + 1e-8)
                
                # Min-max scaling
                X[f"{col}_minmax"] = (X[col] - X[col].min()) / (X[col].max() - X[col].min() + 1e-8)

        # Advanced interactions (3-way)
        if all(col in X.columns for col in ["TrackDurationMs", "Energy", "MoodScore"]):
            X["TrackDurationMs_x_Energy_x_MoodScore"] = X["TrackDurationMs"] * X["Energy"] * X["MoodScore"]
        
        if all(col in X.columns for col in ["RhythmScore", "Energy", "MoodScore"]):
            X["RhythmScore_x_Energy_x_MoodScore"] = X["RhythmScore"] * X["Energy"] * X["MoodScore"]

        # Feature combinations
        if "TrackDurationMs" in X.columns and "RhythmScore" in X.columns:
            X["TrackDurationMs_plus_RhythmScore"] = X["TrackDurationMs"] + X["RhythmScore"]
            X["TrackDurationMs_minus_RhythmScore"] = X["TrackDurationMs"] - X["RhythmScore"]
        
        if "Energy" in X.columns and "MoodScore" in X.columns:
            X["Energy_plus_MoodScore"] = X["Energy"] + X["MoodScore"]
            X["Energy_minus_MoodScore"] = X["Energy"] - X["MoodScore"]

    # Optional KMeans clusters and PCA components (fit on this frame only; CV safeguards leakage)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if add_kmeans:
        scaler = StandardScaler()
        Z = scaler.fit_transform(X[numeric_cols]) if len(numeric_cols) > 0 else None
        if Z is not None:
            for k in add_kmeans:
                try:
                    km = KMeans(n_clusters=k, n_init=10, random_state=0)
                    X[f"kmeans_{k}"] = km.fit_predict(Z)
                except Exception:
                    pass
    if add_pca and add_pca > 0:
        scaler = StandardScaler()
        Z = scaler.fit_transform(X[numeric_cols]) if len(numeric_cols) > 0 else None
        if Z is not None:
            pca = PCA(n_components=min(add_pca, Z.shape[1]))
            comps = pca.fit_transform(Z)
            for i in range(comps.shape[1]):
                X[f"pca_{i+1}"] = comps[:, i]

    return X, y


def build_stratified_folds(y: pd.Series, n_splits: int, seed: int) -> StratifiedKFold:
    # Bin the continuous target into quantiles for stratification
    # Use duplicates='drop' to handle constant targets robustly
    bins = min(20, max(2, int(round(np.sqrt(n_splits * 10)))))
    y_binned = pd.qcut(y, q=bins, labels=False, duplicates="drop")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), y_binned


@dataclass
class ModelOutputs:
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    oof_rmse: float


def train_lightgbm_cpu(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    folds: StratifiedKFold,
    y_binned: pd.Series,
    seed: int,
    param_overrides: Optional[Dict[str, object]] = None,
) -> ModelOutputs:
    oof = np.zeros(len(X), dtype=np.float64)
    test_preds = np.zeros(len(X_test), dtype=np.float64)
    params = {
        "objective": "rmse",
        "metric": "rmse",
        "verbosity": -1,
        "seed": seed,
        "device_type": "cpu",
        "num_threads": os.cpu_count() or 4,
        "feature_pre_filter": False,
    }
    if param_overrides:
        params.update(param_overrides)

    start_ts = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(tqdm(list(folds.split(X, y_binned)), total=folds.get_n_splits(), desc="LGBM folds", leave=False), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va)
        model = lgb.train(
            params,
            dtr,
            valid_sets=[dva],
            num_boost_round=5000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=0),
            ],
        )
        oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
        fold_rmse = compute_rmse(y_va.values, oof[va_idx])
        print(f"[LGBM][Fold {fold_idx}/{folds.get_n_splits()}] best_iter={model.best_iteration} rmse={fold_rmse:.5f}")
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

    duration = time.time() - start_ts
    rmse = compute_rmse(y.values, oof)
    print(f"[LGBM] OOF RMSE={rmse:.5f} time={duration:.1f}s")
    return ModelOutputs(oof, test_preds, rmse)


def train_xgboost_cpu(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    folds: StratifiedKFold,
    y_binned: pd.Series,
    seed: int,
) -> ModelOutputs:
    oof = np.zeros(len(X), dtype=np.float64)
    test_preds = np.zeros(len(X_test), dtype=np.float64)

    start_ts = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(tqdm(list(folds.split(X, y_binned)), total=folds.get_n_splits(), desc="XGB folds", leave=False), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=800,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed + fold_idx,
            tree_method="hist",
            n_jobs=os.cpu_count() or 4,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        oof[va_idx] = model.predict(X_va)
        fold_rmse = compute_rmse(y_va.values, oof[va_idx])
        best_iter = getattr(model, "best_iteration", None)
        print(f"[XGB][Fold {fold_idx}/{folds.get_n_splits()}] best_iter={best_iter} rmse={fold_rmse:.5f}")
        test_preds += model.predict(X_test) / folds.n_splits

    duration = time.time() - start_ts
    rmse = compute_rmse(y.values, oof)
    print(f"[XGB] OOF RMSE={rmse:.5f} time={duration:.1f}s")
    return ModelOutputs(oof, test_preds, rmse)


def train_catboost_gpu(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    folds: StratifiedKFold,
    y_binned: pd.Series,
    seed: int,
    task_type: str = "GPU",
    param_overrides: Optional[Dict[str, object]] = None,
) -> ModelOutputs:
    oof = np.zeros(len(X), dtype=np.float64)
    test_preds = np.zeros(len(X_test), dtype=np.float64)

    # Identify categorical columns for CatBoost (non-numeric)
    cat_cols = np.where(~X.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)))[0].tolist()

    start_ts = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(tqdm(list(folds.split(X, y_binned)), total=folds.get_n_splits(), desc="CAT folds", leave=False), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_cols)
        valid_pool = Pool(X_va, label=y_va, cat_features=cat_cols)

        base_params: Dict[str, object] = {
            "loss_function": "RMSE",
            "depth": 8,
            "learning_rate": 0.03,
            "n_estimators": 5000,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "random_seed": seed + fold_idx,
            "task_type": task_type,
            "devices": "0" if task_type == "GPU" else None,
            "verbose": False,
        }
        if param_overrides:
            base_params.update(param_overrides)
        model = CatBoostRegressor(**base_params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=200)
        oof[va_idx] = model.predict(valid_pool)
        fold_rmse = compute_rmse(y_va.values, oof[va_idx])
        best_iter = model.get_best_iteration()
        print(f"[CAT][Fold {fold_idx}/{folds.get_n_splits()}] best_iter={best_iter} rmse={fold_rmse:.5f}")
        test_preds += model.predict(Pool(X_test, cat_features=cat_cols)) / folds.n_splits

    duration = time.time() - start_ts
    rmse = compute_rmse(y.values, oof)
    print(f"[CAT] OOF RMSE={rmse:.5f} time={duration:.1f}s")
    return ModelOutputs(oof, test_preds, rmse)


def blend_predictions(preds: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    if weights is None:
        # Equal-weight blend by default
        weights = {name: 1.0 for name in preds}
    total_weight = sum(weights.values())
    blended = np.zeros_like(next(iter(preds.values())))
    for name, arr in preds.items():
        blended += weights[name] * arr
    blended /= max(1e-12, total_weight)
    return blended


def clip_to_train_range(preds: np.ndarray, y_train: pd.Series) -> np.ndarray:
    y_min, y_max = float(y_train.min()), float(y_train.max())
    return np.clip(preds, y_min, y_max)


def search_blend_weights(oof_preds: Dict[str, np.ndarray], y: np.ndarray, step: float = 0.05) -> Dict[str, float]:
    names = list(oof_preds.keys())
    best_weights: Dict[str, float] = {n: 1.0 / len(names) for n in names}
    best_rmse = float("inf")

    # Grid over weights that sum to 1. Simple 3-model case is fast; for N>3 this grows.
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    if len(names) == 2:
        for w0 in grid:
            w1 = 1.0 - w0
            blended = w0 * oof_preds[names[0]] + w1 * oof_preds[names[1]]
            rmse = compute_rmse(y, blended)
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = {names[0]: float(w0), names[1]: float(w1)}
    elif len(names) == 3:
        for w0 in grid:
            for w1 in grid:
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9:
                    continue
                blended = w0 * oof_preds[names[0]] + w1 * oof_preds[names[1]] + w2 * oof_preds[names[2]]
                rmse = compute_rmse(y, blended)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = {names[0]: float(w0), names[1]: float(w1), names[2]: float(w2)}
    else:
        # Fallback to equal weights for >3 models
        best_weights = {n: 1.0 / len(names) for n in names}
    return best_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BPM Prediction — tabular models with CV and blending")
    parser.add_argument("--train", type=str, default="train.csv", help="Path to train.csv")
    parser.add_argument("--test", type=str, default="test.csv", help="Path to test.csv")
    parser.add_argument("--sample-sub", type=str, default="sample_submission.csv", help="Path to sample submission")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds")
    parser.add_argument("--id-col", type=str, default=None, help="ID column name (auto-detect if not set)")
    parser.add_argument("--use-cat-gpu", action="store_true", help="Use GPU for CatBoost (if available)")
    parser.add_argument("--no-xgb", action="store_true", help="Skip training XGBoost to save time")
    parser.add_argument("--no-clip", action="store_true", help="Disable clipping predictions to train range")
    parser.add_argument("--out", type=str, default="submission.csv", help="Output submission CSV path")
    parser.add_argument("--save-oof", action="store_true", help="Save OOF predictions to artifacts dir")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to save artifacts")
    parser.add_argument("--search-blend", action="store_true", help="Search blend weights on OOF predictions")
    parser.add_argument("--blend-step", type=float, default=0.05, help="Weight grid step for blend search")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose training progress logs")
    parser.add_argument("--save-importances", action="store_true", help="Save feature importances for LGBM/XGB")
    parser.add_argument("--tune-lgbm", action="store_true", help="Run Optuna tuning for LightGBM before training")
    parser.add_argument("--tune-cat", action="store_true", help="Run Optuna tuning for CatBoost before training")
    parser.add_argument("--tune-trials", type=int, default=30, help="Number of Optuna trials per tuner")
    parser.add_argument("--engineer-features", action="store_true", help="Add engineered features (log, interactions, ratios)")
    parser.add_argument("--fe-clean", action="store_true", help="Use a minimal, high-signal engineered feature set")
    parser.add_argument("--cv-bins", type=int, default=None, help="Override number of bins used for stratification")
    parser.add_argument("--seed-ensemble", type=int, default=1, help="Run N seeds and average predictions (>=1)")
    parser.add_argument("--kmeans", type=str, default="", help="Comma-separated k values for KMeans clusters (e.g., 8,16)")
    parser.add_argument("--pca", type=int, default=0, help="Number of PCA components to add (0=off)")
    parser.add_argument("--log-target", action="store_true", help="Train on log1p(target) and inverse-transform at the end")
    parser.add_argument("--meta-stack", action="store_true", help="Enable stacking meta-model (Ridge) on OOF preds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    train_df, test_df = read_data(args.train, args.test)
    # Determine ID column
    id_col = args.id_col or detect_id_column(train_df)
    if id_col is None:
        id_col = "ID"  # default if not found
    # Parse KMeans list
    kmeans_list: Optional[List[int]] = None
    if args.kmeans:
        try:
            kmeans_list = [int(x) for x in args.kmeans.split(',') if x.strip()]
        except Exception:
            kmeans_list = None

    X_train, y = basic_preprocess(
        train_df,
        id_col=id_col,
        engineer_features=args.engineer_features,
        fe_clean=args.fe_clean,
        add_kmeans=kmeans_list,
        add_pca=(args.pca if args.pca and args.pca > 0 else None),
    )
    X_test, _ = basic_preprocess(
        test_df,
        id_col=id_col,
        target_col=None,
        engineer_features=args.engineer_features,
        fe_clean=args.fe_clean,
        add_kmeans=kmeans_list,
        add_pca=(args.pca if args.pca and args.pca > 0 else None),
    )

    # Optional log-transform of target
    y_train_raw = y.copy()
    if args.log_target:
        y = pd.Series(np.log1p(y.values), index=y.index)

    # Allow overriding number of bins
    if args.cv_bins is not None:
        bins = args.cv_bins
        y_binned = pd.qcut(y, q=bins, labels=False, duplicates="drop")
        folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    else:
        folds, y_binned = build_stratified_folds(y, n_splits=args.folds, seed=args.seed)

    # Optional tuning
    lgbm_best_params: Optional[Dict[str, object]] = None
    cat_best_params: Optional[Dict[str, object]] = None

    if args.tune_lgbm:
        print("[Start] Tuning LightGBM with Optuna")
        def lgb_objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "rmse",
                "metric": "rmse",
                "verbosity": -1,
                "seed": args.seed,
                "device_type": "cpu",
                "feature_pre_filter": False,
                "num_leaves": trial.suggest_int("num_leaves", 16, 512),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
            }
            oof = np.zeros(len(X_train), dtype=np.float64)
            for tr_idx, va_idx in folds.split(X_train, y_binned):
                X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
                dtr = lgb.Dataset(X_tr, label=y_tr)
                dva = lgb.Dataset(X_va, label=y_va)
                model = lgb.train(
                    params,
                    dtr,
                    valid_sets=[dva],
                    num_boost_round=3000,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=200),
                        lgb.log_evaluation(period=0),
                    ],
                )
                oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
            return compute_rmse(y.values, oof)

        study = optuna.create_study(direction="minimize")
        study.optimize(lgb_objective, n_trials=args.tune_trials, show_progress_bar=True)
        lgbm_best_params = {k: v for k, v in study.best_params.items()}
        print(f"[Tuned] LightGBM best params: {lgbm_best_params}")

    if args.tune_cat:
        print("[Start] Tuning CatBoost with Optuna")
        def cat_objective(trial: optuna.Trial) -> float:
            params: Dict[str, object] = {
                "loss_function": "RMSE",
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "bootstrap_type": "Bernoulli",
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "random_seed": args.seed,
                "task_type": ("GPU" if args.use_cat_gpu else "CPU"),
                "devices": "0" if args.use_cat_gpu else None,
                "verbose": False,
                "n_estimators": 5000,
            }
            oof = np.zeros(len(X_train), dtype=np.float64)
            for tr_idx, va_idx in folds.split(X_train, y_binned):
                X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
                train_pool = Pool(X_tr, label=y_tr)
                valid_pool = Pool(X_va, label=y_va)
                model = CatBoostRegressor(**params)
                model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=200)
                oof[va_idx] = model.predict(valid_pool)
            return compute_rmse(y.values, oof)

        study_c = optuna.create_study(direction="minimize")
        study_c.optimize(cat_objective, n_trials=args.tune_trials, show_progress_bar=True)
        cat_best_params = {k: v for k, v in study_c.best_params.items()}
        print(f"[Tuned] CatBoost best params: {cat_best_params}")

    # Train models
    # Seed ensembling
    seed_count = max(1, int(args.seed_ensemble))
    lgbm_out_list = []
    xgb_out_list = []
    cat_out_list = []
    for s in range(seed_count):
        seed_i = args.seed + s
        print(f"[Start] Training LightGBM (CPU) seed={seed_i}")
        l_out = train_lightgbm_cpu(X_train, y, X_test, folds, y_binned, seed=seed_i, param_overrides=lgbm_best_params)
        print("[Done] LightGBM")
        lgbm_out_list.append(l_out)

        if not args.no_xgb:
            print(f"[Start] Training XGBoost (CPU) seed={seed_i}")
            x_out = train_xgboost_cpu(X_train, y, X_test, folds, y_binned, seed=seed_i)
            print("[Done] XGBoost")
            xgb_out_list.append(x_out)
        print(f"[Start] Training CatBoost ({'GPU' if args.use_cat_gpu else 'CPU'}) seed={seed_i}")
        c_out = train_catboost_gpu(
            X_train,
            y,
            X_test,
            folds,
            y_binned,
            seed=seed_i,
            task_type=("GPU" if args.use_cat_gpu else "CPU"),
            param_overrides=cat_best_params,
        )
        print("[Done] CatBoost")
        cat_out_list.append(c_out)

    # Average across seeds
    def avg_outputs(outputs: List[ModelOutputs]) -> ModelOutputs:
        oof_stack = np.vstack([o.oof_predictions for o in outputs])
        test_stack = np.vstack([o.test_predictions for o in outputs])
        oof_avg = oof_stack.mean(axis=0)
        test_avg = test_stack.mean(axis=0)
        rmse_avg = compute_rmse(y.values, oof_avg)
        return ModelOutputs(oof_avg, test_avg, rmse_avg)

    lgbm_out = avg_outputs(lgbm_out_list)
    xgb_out = avg_outputs(xgb_out_list) if (not args.no_xgb and len(xgb_out_list) > 0) else ModelOutputs(np.zeros(len(X_train)), np.zeros(len(X_test)), 999.0)
    cat_out = avg_outputs(cat_out_list)

    # Report OOF RMSE on original scale if log-target is enabled
    if args.log_target:
        lgbm_oof_rmse_disp = compute_rmse(y_train_raw.values, np.expm1(lgbm_out.oof_predictions))
        print(f"OOF RMSE — LightGBM (CPU): {lgbm_oof_rmse_disp:.5f}")
        if not args.no_xgb:
            xgb_oof_rmse_disp = compute_rmse(y_train_raw.values, np.expm1(xgb_out.oof_predictions))
            print(f"OOF RMSE — XGBoost (CPU):  {xgb_oof_rmse_disp:.5f}")
        cat_oof_rmse_disp = compute_rmse(y_train_raw.values, np.expm1(cat_out.oof_predictions))
        print(f"OOF RMSE — CatBoost ({'GPU' if args.use_cat_gpu else 'CPU'}): {cat_oof_rmse_disp:.5f}")
    else:
        print(f"OOF RMSE — LightGBM (CPU): {lgbm_out.oof_rmse:.5f}")
        if not args.no_xgb:
            print(f"OOF RMSE — XGBoost (CPU):  {xgb_out.oof_rmse:.5f}")
        print(f"OOF RMSE — CatBoost ({'GPU' if args.use_cat_gpu else 'CPU'}): {cat_out.oof_rmse:.5f}")

    # Save OOF predictions if requested
    if args.save_oof:
        os.makedirs(args.artifacts_dir, exist_ok=True)
        oof_df = pd.DataFrame({
            (id_col if id_col in train_df.columns else "row_index"): train_df[id_col] if id_col in train_df.columns else np.arange(len(train_df)),
            "BeatsPerMinute": y_train_raw,
            "oof_lgbm": lgbm_out.oof_predictions,
            "oof_xgb": xgb_out.oof_predictions,
            "oof_cat": cat_out.oof_predictions,
        })
        oof_path = os.path.join(args.artifacts_dir, "oof_predictions.csv")
        oof_df.to_csv(oof_path, index=False)
        print(f"Saved OOF predictions to: {oof_path}")

    # Save global feature importances (quick full-data fit) if requested
    if args.save_importances:
        os.makedirs(args.artifacts_dir, exist_ok=True)
        # LightGBM
        lgb_model_full = lgb.train(
            {
                "objective": "rmse",
                "metric": "rmse",
                "verbosity": -1,
                "seed": args.seed,
                "device_type": "cpu",
            },
            lgb.Dataset(X_train, label=y),
            num_boost_round=400,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        lgb_imp = pd.Series(lgb_model_full.feature_importance(importance_type="gain"), index=X_train.columns, name="gain")
        lgb_imp.sort_values(ascending=False).to_csv(os.path.join(args.artifacts_dir, "feature_importance_lgbm.csv"))
        print(f"Saved LightGBM importances → {os.path.join(args.artifacts_dir, 'feature_importance_lgbm.csv')}")

        # XGBoost
        xgb_model_full = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=600,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.seed,
            tree_method="hist",
            n_jobs=os.cpu_count() or 4,
        )
        xgb_model_full.fit(X_train, y, verbose=False)
        booster = xgb_model_full.get_booster()
        fmap = {f"f{idx}": name for idx, name in enumerate(X_train.columns)}
        score = booster.get_score(importance_type="gain")
        xgb_imp = pd.Series({fmap.get(k, k): v for k, v in score.items()}, name="gain").reindex(X_train.columns, fill_value=0.0)
        xgb_imp.sort_values(ascending=False).to_csv(os.path.join(args.artifacts_dir, "feature_importance_xgb.csv"))
        print(f"Saved XGBoost importances → {os.path.join(args.artifacts_dir, 'feature_importance_xgb.csv')}")

    # Blending (skip XGBoost if weight is 0)
    test_pred_dict = {
        "lgbm": lgbm_out.test_predictions,
        "cat": cat_out.test_predictions,
    }
    oof_pred_dict = {
        "lgbm": lgbm_out.oof_predictions,
        "cat": cat_out.oof_predictions,
    }
    # Only include XGBoost if it has reasonable performance
    if (not args.no_xgb) and xgb_out.oof_rmse < 50:  # Only include if reasonable
        test_pred_dict["xgb"] = xgb_out.test_predictions
        oof_pred_dict["xgb"] = xgb_out.oof_predictions
    # Optional stacking meta-model
    if args.meta_stack:
        print("[Start] Stacking meta-model (Ridge)")
        meta_feature_names = sorted(oof_pred_dict.keys())
        meta_X = np.vstack([oof_pred_dict[name] for name in meta_feature_names]).T
        meta_test_X = np.vstack([test_pred_dict[name] for name in meta_feature_names]).T
        meta_oof = np.zeros(meta_X.shape[0], dtype=float)
        meta_test_pred = np.zeros(meta_test_X.shape[0], dtype=float)
        for tr_idx, va_idx in folds.split(meta_X, y_binned):
            X_tr, X_va = meta_X[tr_idx], meta_X[va_idx]
            y_tr, y_va = y.values[tr_idx], y.values[va_idx]
            meta = Ridge(alpha=1.0, random_state=args.seed)
            meta.fit(X_tr, y_tr)
            meta_oof[va_idx] = meta.predict(X_va)
            meta_test_pred += meta.predict(meta_test_X) / args.folds
        if args.log_target:
            meta_rmse = compute_rmse(y_train_raw.values, np.expm1(meta_oof))
        else:
            meta_rmse = compute_rmse(y.values, meta_oof)
        print(f"[Stack] Meta OOF RMSE={meta_rmse:.5f}")
        test_blend = meta_test_pred
        print("[Done] Stacking")
    else:
        weights: Optional[Dict[str, float]] = None
        if args.search_blend:
            weights = search_blend_weights(oof_pred_dict, y.values, step=args.blend_step)
            print(f"Blend weights (searched): {weights}")
            if args.save_oof:
                import json
                weights_path = os.path.join(args.artifacts_dir, "blend_weights.json")
                with open(weights_path, "w", encoding="utf-8") as f:
                    json.dump(weights, f)
                print(f"Saved blend weights to: {weights_path}")
        test_blend = blend_predictions(test_pred_dict, weights=weights)
        print("[Done] Blending predictions")

    # Inverse-transform if needed and clip to train range
    if args.log_target:
        final_preds = np.expm1(test_blend)
        if not args.no_clip:
            final_preds = clip_to_train_range(final_preds, y_train_raw)
    else:
        final_preds = test_blend
        if not args.no_clip:
            final_preds = clip_to_train_range(final_preds, y)

    submission = pd.read_csv(args.sample_sub)
    submission["BeatsPerMinute"] = final_preds
    submission.to_csv(args.out, index=False)
    print(f"Wrote submission to: {args.out}")


if __name__ == "__main__":
    main()


