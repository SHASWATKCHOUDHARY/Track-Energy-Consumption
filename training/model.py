"""
==============================================================================
 model.py — Machine Learning Pipeline for Energy Prediction & Analytics
==============================================================================
 MODULE OVERVIEW  (Viva Explanation)
 ───────────────────────────────────
 This is the "brain" of the project.  It handles every AI / ML task:

   Section 1 — Data Loading & Preprocessing
       • Reads the CSV, drops missing values, parses dates.

   Section 2 — Feature Engineering
       • Cyclical encoding (sin/cos) of hour & month so the model
         knows that 23:00 is close to 00:00 (circular relationship).

   Section 3 — Model Training
       • Trains two models:  Linear Regression  +  Random Forest
       • Compares them using MAE, RMSE, R², MAPE
       • Saves the winner + scaler + feature importances + metrics

   Section 4 — Prediction
       • Predicts hourly energy for the next 7 days using the
         saved model and synthetically generated future features.

   Section 5 — Anomaly Detection  (Advanced ✨)
       • Uses Isolation Forest (unsupervised ML) to flag unusual
         energy spikes automatically.

   Section 6 — Analytics Helpers
       • Summary statistics, daily/hourly/weekly/monthly aggregations,
         appliance breakdown.

   Section 7 — AI Recommendations
       • Rule-based engine that analyses data patterns and produces
         personalised energy-saving tips with estimated savings.

   Section 8 — Utilities
       • Model status checks, metric loading.
==============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
import xgboost as xgb
import lightgbm as lgb

# ── File Paths ───────────────────────────────────────────────────────────
IS_CLOUD = any(os.environ.get(x) for x in ["VERCEL", "RAILWAY_STATIC_URL", "RENDER"])
BASE_MODEL_DIR = "saved_models"
WRITABLE_MODEL_DIR = "/tmp/saved_models" if IS_CLOUD else "saved_models"

def get_path(filename, for_write=False):
    """
    Get the path for a model file.
    If for_write=True, always returns the writable directory path.
    Otherwise, returns the writable path if it exists, falling back to base.
    """
    writable_path = os.path.join(WRITABLE_MODEL_DIR, filename)
    if for_write:
        return writable_path
    if os.path.exists(writable_path):
        return writable_path
    return os.path.join(BASE_MODEL_DIR, filename)

MODEL_DIR = WRITABLE_MODEL_DIR

# Features: loaded dynamically from metrics.json (set by Colab training).
# This fallback list is only used if metrics.json doesn't exist yet.
_DEFAULT_FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend",
    "temperature_c", "humidity_pct",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "lag_1", "lag_24",
    "rolling_24_mean", "rolling_24_std"
]
TARGET_COL = "energy_kwh"

# Will be populated on first access
_feature_cols_cache = None


def get_feature_cols() -> list:
    """Load the feature list the trained model expects from metrics.json."""
    global _feature_cols_cache
    if _feature_cols_cache is not None:
        return _feature_cols_cache
    metrics_path = get_path("metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
        if "features_used" in data and len(data["features_used"]) > 0:
            _feature_cols_cache = data["features_used"]
            return _feature_cols_cache
    _feature_cols_cache = _DEFAULT_FEATURE_COLS
    return _feature_cols_cache


# Keep a module-level alias for backward compatibility
FEATURE_COLS = _DEFAULT_FEATURE_COLS

# Appliance columns in the dataset
APPLIANCE_COLS = [
    "ac_kwh", "lighting_kwh", "kitchen_kwh",
    "fan_kwh", "entertainment_kwh", "other_kwh",
]


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(filepath: str = "dataset.csv") -> pd.DataFrame:
    """
    Load the energy dataset from CSV.
    Robustly handles missing columns by auto-deriving them from datetime.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        try:
            # Check if we are running in Google Colab to prompt an upload natively
            import IPython
            if 'google.colab' in str(IPython.get_ipython()):
                from google.colab import files
                print(f"⚠️ '{filepath}' not found in Colab environment. Please upload it now:")
                uploaded = files.upload()
                
                if not uploaded:
                    raise FileNotFoundError("No file was uploaded. Training aborted.")
                    
                # Use uploaded filename
                actual_filename = list(uploaded.keys())[0]
                df = pd.read_csv(actual_filename)
                print(f"✅ Successfully loaded {actual_filename}!")
            else:
                raise
        except (ImportError, NameError):
            raise FileNotFoundError(f"❌ File '{filepath}' not found. Please ensure it exists in the current directory.")

    # ── Parse datetime ───────────────────────────────────────────────
    dt_col = None
    for col in df.columns:
        if col.strip().lower() in ('datetime', 'date_time', 'timestamp'):
            dt_col = col
            break

    if dt_col and dt_col != 'datetime':
        df['datetime'] = pd.to_datetime(df[dt_col], errors='coerce')
    elif 'datetime' not in df.columns:
        date_col = next((c for c in df.columns if c.strip().lower() == 'date'), None)
        time_col = next((c for c in df.columns if c.strip().lower() == 'time'), None)
        if date_col and time_col:
            df['datetime'] = pd.to_datetime(
                df[date_col].astype(str) + ' ' + df[time_col].astype(str),
                errors='coerce')
        elif date_col:
            df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        else:
            df['datetime'] = pd.date_range(start='2025-01-01',
                                           periods=len(df), freq='h')
    else:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    df.dropna(subset=['datetime'], inplace=True)
    if len(df) == 0:
        raise ValueError("Dataset is empty after parsing dates. Please ensure proper datetime formatting.")
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Auto-derive time features if missing ─────────────────────────
    if 'hour' not in df.columns:
        df['hour'] = df['datetime'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['datetime'].dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = df['datetime'].dt.month
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # ── Generate synthetic weather if missing ────────────────────────
    if 'temperature_c' not in df.columns:
        doy = df['datetime'].dt.dayofyear
        df['temperature_c'] = np.round(
            15 + 15 * np.sin(2 * np.pi * (doy - 80) / 365)
            + 3 * np.sin(2 * np.pi * (df['hour'] - 6) / 24)
            + np.random.normal(0, 2, len(df)), 1)
    if 'humidity_pct' not in df.columns:
        df['humidity_pct'] = np.clip(
            60 - 0.5 * df['temperature_c']
            + np.random.normal(0, 8, len(df)), 20, 95).round(1)

    # ── Detect energy column ─────────────────────────────────────────
    if TARGET_COL not in df.columns:
        energy_aliases = ['energy', 'kwh', 'consumption', 'usage', 'power',
                          'total_kwh', 'global_active_power']
        found = False
        for alias in energy_aliases:
            for col in df.columns:
                if alias in col.strip().lower():
                    df[TARGET_COL] = pd.to_numeric(df[col], errors='coerce')
                    found = True
                    break
            if found:
                break
        if not found:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in
                           ['hour', 'day_of_week', 'month', 'is_weekend']]
            if numeric_cols:
                df[TARGET_COL] = df[numeric_cols[0]]
            else:
                raise ValueError("No numeric energy column found in dataset.")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df.dropna(subset=[TARGET_COL], inplace=True)

    # ── Generate synthetic appliance energy if missing ───────────────────
    missing_apps = [c for c in APPLIANCE_COLS if c not in df.columns]
    if missing_apps:
        # Base distributions (rough average % of total energy)
        dist = {
            "ac_kwh": 0.40,
            "lighting_kwh": 0.15,
            "kitchen_kwh": 0.20,
            "fan_kwh": 0.10,
            "entertainment_kwh": 0.10,
            "other_kwh": 0.05
        }
        for c in missing_apps:
            base = df[TARGET_COL] * dist.get(c, 0.1)
            noise = np.random.normal(0, 0.05, len(df)) * df[TARGET_COL]
            df[c] = np.clip(base + noise, 0, None)
        
        # Re-normalize so sum(appliances) == total energy
        app_sum = df[APPLIANCE_COLS].sum(axis=1).replace(0, 1)
        for c in APPLIANCE_COLS:
            df[c] = (df[c] / app_sum) * df[TARGET_COL]

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame, history_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create ALL features the trained model expects.

    Generates cyclical, lag, rolling, EWM, difference, and interaction
    features.  The exact set is driven by get_feature_cols() so it
    automatically matches whatever the Colab-trained model used.

    If history_df is provided (during recursive prediction), it seeds the
    lag and rolling calculations so features can be derived accurately.
    """
    df = df.copy()
    feature_cols = get_feature_cols()

    # ── Cyclical encoding ────────────────────────────────────────────
    if "hour" in df.columns:
        df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    if "day_of_week" in df.columns:
        df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── Combine with history for lag/rolling calculations ────────────
    if history_df is not None:
        combined = pd.concat([history_df, df], ignore_index=True)
    else:
        combined = df.copy()

    if TARGET_COL in combined.columns:
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            col_name = f"lag_{lag}"
            if col_name in feature_cols:
                combined[col_name] = combined[TARGET_COL].shift(lag)

        # Rolling stats (on shifted data to prevent leakage)
        shifted = combined[TARGET_COL].shift(1)
        for w in [3, 6, 12, 24, 48, 168]:
            mean_col = f"rolling_{w}_mean"
            std_col  = f"rolling_{w}_std"
            if mean_col in feature_cols:
                combined[mean_col] = shifted.rolling(window=w).mean()
            if std_col in feature_cols:
                combined[std_col] = shifted.rolling(window=w).std().fillna(0)

        # EWM features
        for span in [12, 24]:
            col_name = f"ewm_{span}"
            if col_name in feature_cols:
                combined[col_name] = shifted.ewm(span=span, adjust=False).mean()

        # Difference features (shifted to prevent leakage)
        if "diff_1" in feature_cols:
            combined["diff_1"] = combined[TARGET_COL].diff(1).shift(1)
        if "diff_24" in feature_cols:
            combined["diff_24"] = combined[TARGET_COL].diff(24).shift(1)
    else:
        # No target available — fill lag/rolling with 0
        for col in feature_cols:
            if col not in combined.columns:
                combined[col] = 0

    # ── Extract the rows we care about ───────────────────────────────
    if history_df is not None:
        idx = len(history_df)
        df_out = combined.iloc[idx:].copy()
    else:
        df_out = combined.copy()

    # Fill any remaining missing feature columns with 0
    for col in feature_cols:
        if col not in df_out.columns:
            df_out[col] = 0

    return df_out



# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_model(filepath: str = "dataset.csv") -> dict:
    """
    Optimized training pipeline (GPU & Overfitting safe):
      1. Load data & engineer features.
      2. Train/Val/Test split (80/10/10).
      3. Train LightGBM & XGBoost with Colab T4 GPU flags.
      4. Aggressive regularisation + Early stopping on Validation set.
      5. Evaluate on Test set.
    """
    df = load_data(filepath)
    df = engineer_features(df)

    feature_cols = get_feature_cols()

    # Drop rows with NaN from shift/rolling features
    existing_cols = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=existing_cols + [TARGET_COL])
    if len(df) < 10:
        raise ValueError(f"Insufficient data for training (only {len(df)} rows left after cleaning). At least 10 valid records required.")

    X = df[existing_cols]
    y = df[TARGET_COL]

    # Chronological Split (Train: 80%, Val: 10%, Test: 10%)
    train_idx = int(len(df) * 0.8)
    val_idx = int(len(df) * 0.9)
    
    X_train, X_val, X_test = X.iloc[:train_idx], X.iloc[train_idx:val_idx], X.iloc[val_idx:]
    y_train, y_val, y_test = y.iloc[:train_idx], y.iloc[train_idx:val_idx], y.iloc[val_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── Linear Regression (Baseline) ─────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)

    # ── LightGBM w/ Regularisation & GPU Fallback ────────────────────────
    # Colab pip package may lack GPU depending on the week, so we try 'gpu', 
    # but fall back to 'cpu' if it fails.
    lgb_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 8,           # Shallow trees to prevent overfitting
        'subsample': 0.8,         # Row subsampling
        'colsample_bytree': 0.8,  # Feature subsampling
        'reg_alpha': 0.1,         # L1 regularisation
        'reg_lambda': 1.0,        # L2 regularisation
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    try:
        # Try GPU first
        lgb_model = lgb.LGBMRegressor(**lgb_params, device='gpu')
        lgb_model.fit(X_train_s, y_train, 
                      eval_set=[(X_val_s, y_val)], 
                      callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
    except Exception:
        # Graceful fallback to CPU
        lgb_model = lgb.LGBMRegressor(**lgb_params, device='cpu')
        lgb_model.fit(X_train_s, y_train, 
                      eval_set=[(X_val_s, y_val)], 
                      callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
        
    lgb_pred = lgb_model.predict(X_test_s)

    # ── XGBoost w/ Colab T4 GPU Support ──────────────────────────────────
    # tree_method='hist' and device='cuda' enables T4 GPU on Colab.
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method='hist',
        device='cuda',    # Use 'cuda' for Colab T4 GPU
        random_state=42,
        n_jobs=-1
    )
    
    try:
        xgb_model.fit(X_train_s, y_train,
                      eval_set=[(X_val_s, y_val)],
                      verbose=False)
    except Exception as e:
        # Fallback to CPU if run locally
        print(f"⚠️ XGBoost GPU failed (likely running locally). Falling back to CPU. ({str(e)})")
        xgb_model.set_params(device='cpu')
        xgb_model.fit(X_train_s, y_train,
                      eval_set=[(X_val_s, y_val)],
                      verbose=False)
        
    xgb_pred = xgb_model.predict(X_test_s)

    # ── Evaluation helper ────────────────────────────────────────────────
    def evaluate(name, y_true, y_pred):
        if len(y_true) < 2:
            return {"model": name, "MAE": 0, "RMSE": 0, "R2": 0, "MAPE": 0}
        return {
            "model" : name,
            "MAE"   : round(float(mean_absolute_error(y_true, y_pred)), 4),
            "RMSE"  : round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "R2"    : round(float(r2_score(y_true, y_pred)), 4),
            "MAPE"  : round(float(mean_absolute_percentage_error(y_true, y_pred)) * 100, 2),
        }

    lr_m  = evaluate("Linear Regression", y_test, lr_pred)
    lgb_m = evaluate("LightGBM", y_test, lgb_pred)
    xgb_m = evaluate("XGBoost", y_test, xgb_pred)

    models_info = [(lr_m, lr), (lgb_m, lgb_model), (xgb_m, xgb_model)]
    best_m, best_model = min(models_info, key=lambda x: x[0]["RMSE"])

    # ── Feature importances ──────────────────────────────────────────────
    feat_imp = {}
    if hasattr(best_model, "feature_importances_"):
        for name, imp in zip(existing_cols, best_model.feature_importances_):
            feat_imp[name] = round(float(imp), 4)
    else:
        for name, coef in zip(existing_cols, best_model.coef_):
            feat_imp[name] = round(float(abs(coef)), 4)

    # ── Save everything ──────────────────────────────────────────────────
    os.makedirs(WRITABLE_MODEL_DIR, exist_ok=True)

    with open(get_path("energy_model.pkl", for_write=True), "wb") as f:
        pickle.dump(best_model, f)
    with open(get_path("scaler.pkl", for_write=True), "wb") as f:
        pickle.dump(scaler, f)

    all_metrics = {
        "selected":          best_m["model"],
        "linear_regression": lr_m,
        "lightgbm":          lgb_m,
        "xgboost":           xgb_m,
        "train_size":        len(X_train),
        "test_size":         len(X_test),
        "features_used":     existing_cols,
    }
    with open(get_path("metrics.json", for_write=True), "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(get_path("feature_importance.json", for_write=True), "w") as f:
        json.dump(feat_imp, f, indent=2)

    print(f"✅  Model trained — best: {best_m['model']}  "
          f"(R²={best_m['R2']}, RMSE={best_m['RMSE']}, MAPE={best_m['MAPE']}%)")

    return all_metrics


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PREDICTION (NEXT 7 DAYS)
# ═══════════════════════════════════════════════════════════════════════════

def predict_future(days: int = 7, filepath: str = "dataset.csv") -> list:
    """
    Predict hourly energy for the next `days` days.

    Viva:
      Uses an autoregressive approach. Since our model relies on
      past energy consumption (lag features & rolling averages),
      we must predict hour h, feed that prediction back as the
      'actual' energy for hour h, and then predict h+1.
    """
    model_path = get_path("energy_model.pkl")
    scaler_path = get_path("scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise ValueError("Model not trained.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    feature_cols = get_feature_cols()

    df = load_data(filepath)
    last_date = df["datetime"].max()

    # We need the last 200 hours to seed lag_168 and rolling_168 features
    history_size = 200
    history_df = df.tail(history_size).copy().reset_index(drop=True)

    # Compute average sensor values from history for future rows
    sensor_cols = ['Global_reactive_power_mean', 'Voltage_mean',
                   'Global_intensity_mean', 'Sub_metering_1_mean',
                   'Sub_metering_2_mean', 'Sub_metering_3_mean',
                   'energy_std', 'energy_max', 'energy_min']
    sensor_avgs = {}
    for col in sensor_cols:
        if col in history_df.columns:
            sensor_avgs[col] = float(history_df[col].mean())
        elif col in feature_cols:
            sensor_avgs[col] = 0.0

    if TARGET_COL not in history_df.columns:
        raise ValueError("History missing target column")

    preds = []

    for h in range(days * 24):
        dt  = last_date + timedelta(hours=h + 1)
        doy = dt.timetuple().tm_yday

        # Build row with all possible features
        row_dict = {
            "datetime": dt, "hour": dt.hour,
            "day_of_week": dt.weekday(), "month": dt.month,
            "day_of_year": doy,
            "is_weekend": int(dt.weekday() >= 5),
        }

        # Add synthetic weather if needed
        if "temperature_c" in feature_cols:
            temp = round(15 + 15 * np.sin(2 * np.pi * (doy - 80) / 365)
                         + 3 * np.sin(2 * np.pi * (dt.hour - 6) / 24)
                         + np.random.normal(0, 1.5), 1)
            row_dict["temperature_c"] = temp
        if "humidity_pct" in feature_cols:
            t = row_dict.get("temperature_c", 20)
            row_dict["humidity_pct"] = round(
                np.clip(60 - 0.5 * t + np.random.normal(0, 5), 20, 95), 1)

        # Add sensor averages for columns the model expects
        for col, val in sensor_avgs.items():
            row_dict[col] = val

        # Interaction features
        if "temp_x_hour" in feature_cols:
            row_dict["temp_x_hour"] = row_dict.get("temperature_c", 20) * dt.hour
        if "temp_x_weekend" in feature_cols:
            row_dict["temp_x_weekend"] = row_dict.get("temperature_c", 20) * row_dict["is_weekend"]

        future_row_df = pd.DataFrame([row_dict])

        # 1. Engineer features for this single row based on history
        engineered = engineer_features(future_row_df, history_df=history_df)

        # 2. Ensure all required features exist
        for col in feature_cols:
            if col not in engineered.columns:
                engineered[col] = 0

        # 3. Scale and Predict
        X_pred = scaler.transform(engineered[feature_cols])
        pred_val = float(model.predict(X_pred)[0])
        pred_val = max(0.0, pred_val)

        # 4. Add prediction back to history
        row_dict[TARGET_COL] = pred_val
        new_row_df = pd.DataFrame([row_dict])
        history_df = pd.concat([history_df, new_row_df], ignore_index=True).tail(history_size)

        preds.append({
            "datetime": dt.strftime("%Y-%m-%d %H:%M"),
            "predicted_kwh": round(pred_val, 3)
        })

    return preds


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5 — ANOMALY DETECTION  (Advanced ✨)
# ═══════════════════════════════════════════════════════════════════════════

def detect_anomalies(filepath: str = "dataset.csv") -> dict:
    """
    Detect unusual energy consumption spikes using two methods:

    Method 1 — Isolation Forest (unsupervised ML)
       Builds an ensemble of random trees.  Anomalies are isolated
       in fewer splits → assigned a score of -1.

    Method 2 — Z-Score (statistical)
       Points > 3 standard deviations from the rolling mean are
       flagged.  Simpler but effective for univariate spikes.

    Viva:
      Isolation Forest is an *unsupervised* algorithm — it doesn't
      need labels.  It learns what "normal" looks like and flags
      data points that deviate significantly.

    Returns
    -------
    dict with:
      - anomalies: list of {datetime, energy_kwh, method}
      - total_anomalies, pct_anomalies
      - daily_anomaly_counts for charting
    """
    df = load_data(filepath)

    # ── Method 1: Isolation Forest ───────────────────────────────────────
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.02,     # expect ~2 % outliers
        random_state=42,
    )
    iso_labels = iso.fit_predict(df[["energy_kwh"]].values)
    df["iso_anomaly"] = (iso_labels == -1)

    # ── Method 2: Z-Score on a 24-hour rolling window ────────────────────
    rolling_mean = df["energy_kwh"].rolling(window=24, center=True).mean()
    rolling_std  = df["energy_kwh"].rolling(window=24, center=True).std()
    z_scores     = (df["energy_kwh"] - rolling_mean) / rolling_std
    df["z_anomaly"] = z_scores.abs() > 3.0

    # ── Combine: flagged by EITHER method ────────────────────────────────
    df["is_anomaly"] = df["iso_anomaly"] | df["z_anomaly"]

    anomaly_df = df[df["is_anomaly"]].copy()

    # Build list for API
    anomalies = []
    for _, row in anomaly_df.iterrows():
        methods = []
        if row["iso_anomaly"]:
            methods.append("Isolation Forest")
        if row["z_anomaly"]:
            methods.append("Z-Score")
        anomalies.append({
            "datetime":   row["datetime"].strftime("%Y-%m-%d %H:%M"),
            "energy_kwh": round(row["energy_kwh"], 3),
            "method":     " + ".join(methods),
        })

    # Daily anomaly counts (for bar chart)
    anomaly_df["date"] = anomaly_df["datetime"].dt.date
    daily_counts = anomaly_df.groupby("date").size().reset_index(name="count")
    daily_anomaly = [
        {"date": str(r["date"]), "count": int(r["count"])}
        for _, r in daily_counts.iterrows()
    ]

    return {
        "anomalies":        anomalies[:200],    # cap for frontend perf
        "total_anomalies":  int(df["is_anomaly"].sum()),
        "total_records":    len(df),
        "pct_anomalies":    round(df["is_anomaly"].mean() * 100, 2),
        "daily_counts":     daily_anomaly,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6 — ANALYTICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_summary_statistics(filepath: str = "dataset.csv") -> dict:
    """Compute dashboard summary cards."""
    df = load_data(filepath)

    total_kwh = round(df["energy_kwh"].sum(), 2)
    n_days    = df["datetime"].dt.date.nunique()
    avg_daily = round(total_kwh / n_days, 2)

    hourly_avg = df.groupby("hour")["energy_kwh"].mean()
    peak_hour  = int(hourly_avg.idxmax())

    dow_avg    = df.groupby("day_of_week")["energy_kwh"].mean()
    peak_dow   = int(dow_avg.idxmax())
    dow_names  = ["Monday","Tuesday","Wednesday","Thursday",
                  "Friday","Saturday","Sunday"]

    month_avg  = df.groupby("month")["energy_kwh"].mean()
    peak_month = int(month_avg.idxmax())
    month_names = ["","January","February","March","April","May",
                   "June","July","August","September","October",
                   "November","December"]

    return {
        "total_kwh":       total_kwh,
        "avg_daily_kwh":   avg_daily,
        "avg_daily":       avg_daily,
        "avg_hourly_kwh":  round(df["energy_kwh"].mean(), 3),
        "max_hourly_kwh":  round(df["energy_kwh"].max(), 3),
        "min_hourly_kwh":  round(df["energy_kwh"].min(), 3),
        "std_kwh":         round(df["energy_kwh"].std(), 3),
        "peak_hour":       peak_hour,
        "peak_hour_label": f"{peak_hour}:00 – {peak_hour+1}:00",
        "peak_day":        dow_names[peak_dow],
        "peak_month":      month_names[peak_month],
        "total_days":      n_days,
        "total_records":   len(df),
    }


def get_daily_usage(filepath: str = "dataset.csv") -> dict:
    """Daily total kWh for line chart."""
    df = load_data(filepath)
    daily = df.groupby(df["datetime"].dt.date)["energy_kwh"].sum().reset_index()
    daily.columns = ["date", "kwh"]
    return {
        "dates": [str(r["date"]) for _, r in daily.iterrows()],
        "values": [round(r["kwh"], 2) for _, r in daily.iterrows()],
    }


def get_hourly_profile(filepath: str = "dataset.csv") -> dict:
    """Average kWh per hour-of-day (24-hour profile)."""
    df = load_data(filepath)
    p = df.groupby("hour")["energy_kwh"].mean().reset_index()
    return {
        "hours": [int(r["hour"]) for _, r in p.iterrows()],
        "values": [round(r["energy_kwh"], 3) for _, r in p.iterrows()],
    }


def get_monthly_usage(filepath: str = "dataset.csv") -> dict:
    """Total kWh per month."""
    df = load_data(filepath)
    m = df.groupby("month")["energy_kwh"].sum().reset_index()
    names = ["","Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"]
    return {
        "months": [names[int(r["month"])] for _, r in m.iterrows()],
        "values": [round(r["energy_kwh"], 2) for _, r in m.iterrows()],
    }


def get_weekly_usage(filepath: str = "dataset.csv") -> dict:
    """Average kWh per day-of-week."""
    df = load_data(filepath)
    names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    w = df.groupby("day_of_week")["energy_kwh"].mean().reset_index()
    return {
        "days": [names[int(r["day_of_week"])] for _, r in w.iterrows()],
        "values": [round(r["energy_kwh"], 3) for _, r in w.iterrows()],
    }


def get_appliance_breakdown(filepath: str = "dataset.csv") -> dict:
    """
    Appliance-level energy breakdown for pie chart + stacked bar.

    Viva:
      We have simulated six appliance categories.  Summing each over
      the year shows which appliance uses the most energy — this helps
      users prioritise where to save.
    """
    df = load_data(filepath)

    # ── Overall totals ───────────────────────────────────────────────────
    available = [c for c in APPLIANCE_COLS if c in df.columns]
    if not available:
        return {"available": False}

    totals = {col: round(df[col].sum(), 2) for col in available}
    labels = {
        "ac_kwh": "AC / Heating",
        "lighting_kwh": "Lighting",
        "kitchen_kwh": "Kitchen",
        "fan_kwh": "Fan",
        "entertainment_kwh": "Entertainment",
        "other_kwh": "Other / Standby",
    }

    pie_labels = [labels.get(col, col) for col in available]
    pie_values = [totals[col] for col in available]

    # ── Monthly breakdown (for stacked bar) ──────────────────────────────
    month_names = ["","Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    months_in_data = sorted(df["month"].unique())
    month_labels = [month_names[int(m)] for m in months_in_data]

    appliance_names = []
    appliance_values = []
    for col in available:
        grp = df.groupby("month")[col].sum()
        appliance_names.append(labels.get(col, col))
        appliance_values.append([round(float(grp.get(m, 0)), 2)
                                 for m in months_in_data])

    return {
        "available": True,
        "labels": pie_labels,
        "values": pie_values,
        "monthly": {
            "months": month_labels,
            "appliances": appliance_names,
            "values": appliance_values,
        },
    }


def get_feature_importance() -> list | None:
    """Load saved feature importances for the bar chart."""
    imp_path = get_path("feature_importance.json")
    if os.path.exists(imp_path):
        with open(imp_path) as f:
            data = json.load(f)
        nice_names = {
            "hour": "Hour", "day_of_week": "Day of Week",
            "month": "Month", "is_weekend": "Weekend?",
            "day_of_year": "Day of Year",
            "temperature_c": "Temperature", "humidity_pct": "Humidity",
            "hour_sin": "Hour (sin)", "hour_cos": "Hour (cos)",
            "month_sin": "Month (sin)", "month_cos": "Month (cos)",
            "dow_sin": "Day (sin)", "dow_cos": "Day (cos)",
            "energy_std": "Energy Std Dev", "energy_max": "Energy Max",
            "energy_min": "Energy Min",
            "Global_reactive_power_mean": "Reactive Power",
            "Voltage_mean": "Voltage", "Global_intensity_mean": "Current",
            "Sub_metering_1_mean": "Kitchen", "Sub_metering_2_mean": "Laundry",
            "Sub_metering_3_mean": "Water Heater",
            "temp_x_hour": "Temp × Hour", "temp_x_weekend": "Temp × Weekend",
        }
        return [{"name": nice_names.get(k, k.replace('_', ' ').title()), "importance": v}
                for k, v in sorted(data.items(), key=lambda x: -x[1])]
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7 — AI-BASED ENERGY SAVING RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_recommendations(filepath: str = "dataset.csv") -> list:
    """
    Analyse consumption patterns and produce tailored tips.

    Viva:
      This is a *rule-based AI* approach: we encode expert knowledge
      as if-then rules.  For each pattern detected (high night usage,
      seasonal spikes, etc.) we generate a specific recommendation.
    """
    df     = load_data(filepath)
    hourly = df.groupby("hour")["energy_kwh"].mean()
    monthly = df.groupby("month")["energy_kwh"].sum()

    tips = []

    # ── Night standby ────────────────────────────────────────────────────
    night_avg = hourly.loc[0:5].mean()
    if night_avg > 0.25:
        tips.append({
            "icon": "🌙", "title": "Reduce Standby Power at Night",
            "detail": f"Night-time average ({night_avg:.2f} kWh/h) is high. "
                      f"Switch off devices at the plug to save up to 10%.",
            "saving_pct": 10, "priority": "high",
        })

    # ── Peak-hour shifting ───────────────────────────────────────────────
    peak_h = int(hourly.idxmax())
    tips.append({
        "icon": "⏰",
        "title": f"Shift Usage Away from Peak Hour ({peak_h}:00)",
        "detail": f"Demand peaks at {peak_h}:00. Run heavy appliances "
                  f"during off-peak hours to reduce cost.",
        "saving_pct": 8, "priority": "medium",
    })

    # ── Seasonal ─────────────────────────────────────────────────────────
    winter = monthly.loc[monthly.index.isin([12, 1, 2])].mean()
    summer = monthly.loc[monthly.index.isin([6, 7, 8])].mean()
    mid    = monthly.loc[monthly.index.isin([4, 5, 9, 10])].mean()

    if winter > mid * 1.15:
        tips.append({
            "icon": "❄️", "title": "Optimise Winter Heating",
            "detail": f"Winter: {winter:.0f} kWh/mo vs mid-season: {mid:.0f}. "
                      f"Lower thermostat by 1°C to save ~3% on heating.",
            "saving_pct": 6, "priority": "high",
        })
    if summer > mid * 1.10:
        tips.append({
            "icon": "☀️", "title": "Improve Cooling Efficiency",
            "detail": f"Summer: {summer:.0f} kWh/mo exceeds mid-season. "
                      f"Use fans first, close blinds midday, set AC to 24°C.",
            "saving_pct": 7, "priority": "high",
        })

    # ── Weekend ──────────────────────────────────────────────────────────
    we = df[df["is_weekend"]==1]["energy_kwh"].mean()
    wd = df[df["is_weekend"]==0]["energy_kwh"].mean()
    if we > wd * 1.05:
        tips.append({
            "icon": "📅", "title": "Manage Weekend Consumption",
            "detail": f"Weekend ({we:.2f} kWh/h) > weekday ({wd:.2f}). "
                      f"Turn off idle TVs, consoles, and lights.",
            "saving_pct": 5, "priority": "medium",
        })

    # ── Appliance-specific tips ──────────────────────────────────────────
    if "ac_kwh" in df.columns:
        ac_pct = df["ac_kwh"].sum() / df["energy_kwh"].sum() * 100
        if ac_pct > 25:
            tips.append({
                "icon": "🧊", "title": f"AC Uses {ac_pct:.0f}% of Total Energy",
                "detail": "Consider upgrading to an inverter AC, using "
                          "smart scheduling, or improving insulation.",
                "saving_pct": 15, "priority": "high",
            })

    if "lighting_kwh" in df.columns:
        tips.append({
            "icon": "💡", "title": "Switch to Smart LED Lighting",
            "detail": "Replace CFL/incandescent with LEDs + motion sensors "
                      "to cut lighting energy by up to 40%.",
            "saving_pct": 12, "priority": "medium",
        })

    tips.append({
        "icon": "🏷️", "title": "Use Energy Star Rated Appliances",
        "detail": "5-star rated fridge/washer can reduce individual "
                  "appliance consumption by 20-50%.",
        "saving_pct": 15, "priority": "low",
    })

    tips.append({
        "icon": "📱", "title": "Install Smart Power Strips",
        "detail": "Smart strips cut power to devices on standby "
                  "automatically, saving 5-8% of phantom load.",
        "saving_pct": 8, "priority": "medium",
    })

    return tips


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 8 — UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def is_model_trained() -> bool:
    """Check whether a trained model exists on disk."""
    return os.path.exists(get_path("energy_model.pkl")) and os.path.exists(get_path("scaler.pkl"))


def get_model_metrics() -> dict | None:
    """Load saved evaluation metrics."""
    metrics_path = get_path("metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 9 — ADVANCED TIME-SERIES FORECASTING (LSTM / Holt-Winters)
# ═══════════════════════════════════════════════════════════════════════════
#
#  Viva Explanation
#  ────────────────
#  Random Forest treats each row independently — it doesn't know about
#  the *order* of time steps.  True time-series methods exploit temporal
#  structure:
#
#    • LSTM (Long Short-Term Memory) is a Recurrent Neural Network that
#      maintains a hidden state ("memory cell") so it can learn long-term
#      dependencies like seasonal cycles.
#
#    • Holt-Winters (Triple Exponential Smoothing) decomposes the series
#      into Level + Trend + Seasonality and applies exponential weighting
#      so recent observations matter more.
#
#  We try LSTM first (requires tensorflow); if unavailable we fall back
#  to Holt-Winters (statsmodels) — both are legitimate advanced methods.
# ═══════════════════════════════════════════════════════════════════════════

def get_forecast_path(for_write=False):
    return get_path("forecast_result.json", for_write=for_write)


def _daily_series(filepath: str = "dataset.csv") -> pd.Series:
    """Aggregate hourly data to daily totals (returns DatetimeIndex Series)."""
    df = load_data(filepath)
    daily = df.groupby(df["datetime"].dt.date)["energy_kwh"].sum()
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def train_forecast_model(filepath: str = "dataset.csv",
                         horizon: int = 14) -> dict:
    """
    Train an advanced time-series forecast and predict `horizon` days ahead.

    Strategy:
      1st choice → LSTM neural network  (requires tensorflow)
      2nd choice → Holt-Winters Triple Exponential Smoothing  (statsmodels)

    Returns
    -------
    dict : {method, metrics, forecast}
    """
    daily = _daily_series(filepath)
    os.makedirs(MODEL_DIR, exist_ok=True)

    method = "Unknown"
    forecast_values: list[float] = []
    metrics: dict = {}

    # ── Attempt 1: LSTM (Deep Learning) ──────────────────────────────
    try:
        from tensorflow.keras.models import Sequential as KerasSeq  # type: ignore
        from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
        from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
        from sklearn.preprocessing import MinMaxScaler as MMS

        SEQ_LEN = 14
        values = daily.values.reshape(-1, 1)
        mms = MMS()
        scaled = mms.fit_transform(values)

        X, y = [], []
        for i in range(SEQ_LEN, len(scaled)):
            X.append(scaled[i - SEQ_LEN:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        split = int(0.8 * len(X))
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]

        mdl = KerasSeq([
            LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ])
        mdl.compile(optimizer="adam", loss="mse")
        mdl.fit(Xtr, ytr, epochs=50, batch_size=16, verbose=0,
                validation_split=0.1,
                callbacks=[EarlyStopping(patience=5,
                                         restore_best_weights=True)])

        yp = mdl.predict(Xte, verbose=0).flatten()
        yte_inv = mms.inverse_transform(yte.reshape(-1, 1)).flatten()
        yp_inv = mms.inverse_transform(yp.reshape(-1, 1)).flatten()

        metrics = {
            "MAE":  round(float(mean_absolute_error(yte_inv, yp_inv)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(yte_inv, yp_inv))), 2),
            "R2":   round(float(r2_score(yte_inv, yp_inv)), 4),
        }

        seq = scaled[-SEQ_LEN:]
        for _ in range(horizon):
            p = mdl.predict(seq.reshape(1, SEQ_LEN, 1), verbose=0)[0, 0]
            forecast_values.append(p)
            seq = np.append(seq[1:], [[p]], axis=0)
        forecast_values = mms.inverse_transform(
            np.array(forecast_values).reshape(-1, 1)
        ).flatten().tolist()

        method = "LSTM (Deep Learning)"
        mdl.save(os.path.join(MODEL_DIR, "lstm_model.keras"))
        with open(os.path.join(MODEL_DIR, "lstm_scaler.pkl"), "wb") as f:
            pickle.dump(mms, f)

    except Exception:
        # ── Attempt 2: Holt-Winters (Statistical) ────────────────────
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        split = int(0.8 * len(daily))
        train_s, test_s = daily.iloc[:split], daily.iloc[split:]

        hw = ExponentialSmoothing(
            train_s, trend="add", seasonal="add", seasonal_periods=7,
        ).fit(optimized=True)
        yp = hw.forecast(len(test_s)).values

        metrics = {
            "MAE":  round(float(mean_absolute_error(test_s.values, yp)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(test_s.values, yp))), 2),
            "R2":   round(float(r2_score(test_s.values, yp)), 4),
        }

        hw_full = ExponentialSmoothing(
            daily, trend="add", seasonal="add", seasonal_periods=7,
        ).fit(optimized=True)
        forecast_values = hw_full.forecast(horizon).values.tolist()
        method = "Holt-Winters Exponential Smoothing"

    # ── Build result ─────────────────────────────────────────────────
    last_date = daily.index[-1]
    forecast = [
        {"date": (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d"),
         "predicted_kwh": round(float(max(v, 0)), 2)}
        for i, v in enumerate(forecast_values)
    ]

    result = {"method": method, "metrics": metrics, "forecast": forecast}
    with open(get_forecast_path(for_write=True), "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅  Forecast trained — method: {method}  (RMSE={metrics['RMSE']})")
    return result


def get_forecast_result() -> dict | None:
    """Load saved forecast result from disk."""
    forecast_path = get_forecast_path()
    if os.path.exists(forecast_path):
        with open(forecast_path) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 10 — ENERGY USAGE TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
#
#  Viva: Trend analysis shows *how* consumption changes over time
#  at multiple time-scales (weekly, monthly).  The rolling average
#  smooths noise so the underlying trend is visible.  A linear
#  regression slope tells us whether usage is rising or falling.
# ═══════════════════════════════════════════════════════════════════════════

def get_trend_analysis(filepath: str = "dataset.csv") -> dict:
    """
    Multi-scale trend analysis: weekly %, monthly %, rolling avg, direction.
    """
    df = load_data(filepath)

    # ── Weekly trend ─────────────────────────────────────────────────
    df_copy = df.copy()
    df_copy["week"] = df_copy["datetime"].dt.isocalendar().week.astype(int)
    df_copy["year"] = df_copy["datetime"].dt.year
    weekly = df_copy.groupby(["year", "week"])["energy_kwh"].sum().reset_index()
    weekly.columns = ["year", "week", "kwh"]
    weekly["pct_change"] = weekly["kwh"].pct_change().fillna(0) * 100
    weekly_list = [
        {"week": f"W{int(r['week'])}", "kwh": round(r["kwh"], 2),
         "pct_change": round(r["pct_change"], 1)}
        for _, r in weekly.iterrows()
    ]

    # ── Monthly trend ────────────────────────────────────────────────
    monthly = df.groupby("month")["energy_kwh"].sum().reset_index()
    monthly.columns = ["month", "kwh"]
    monthly["pct_change"] = monthly["kwh"].pct_change().fillna(0) * 100
    m_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_list = [
        {"month": m_names[int(r["month"])], "month_num": int(r["month"]),
         "kwh": round(r["kwh"], 2), "pct_change": round(r["pct_change"], 1)}
        for _, r in monthly.iterrows()
    ]

    # ── 7-day rolling average ────────────────────────────────────────
    daily = df.groupby(df["datetime"].dt.date)["energy_kwh"].sum()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    rolling = daily.rolling(7).mean().dropna()
    rolling_list = [
        {"date": d.strftime("%Y-%m-%d"), "avg_kwh": round(float(v), 2)}
        for d, v in rolling.items()
    ]

    # ── Overall direction (linear slope) ─────────────────────────────
    x = np.arange(len(daily))
    coeffs = np.polyfit(x, daily.values, 1)
    slope = coeffs[0]
    if slope > 0.01:
        direction = "increasing"
    elif slope < -0.01:
        direction = "decreasing"
    else:
        direction = "stable"

    return {
        "weekly_trend":     weekly_list,
        "monthly_trend":    monthly_list,
        "rolling_7d":       rolling_list,
        "overall_direction": direction,
        "slope_per_day":    round(float(slope), 4),
        "avg_daily_kwh":    round(float(daily.mean()), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 11 — AI INSIGHT GENERATION (Natural Language)
# ═══════════════════════════════════════════════════════════════════════════
#
#  Viva: This is an *expert system* — a classical AI approach.
#  We encode domain knowledge as rules that analyse data dimensions
#  (trends, anomalies, appliances, seasons) and translate findings
#  into human-readable sentences, similar to Google Analytics Insights
#  or Apple Screen Time reports.
# ═══════════════════════════════════════════════════════════════════════════

def generate_ai_insights(filepath: str = "dataset.csv") -> list:
    """
    Produce natural-language AI insights about energy consumption.

    Returns list of {icon, title, text, severity}.
    """
    df = load_data(filepath)
    insights: list[dict] = []

    # ── 1. Weekly trend prediction ───────────────────────────────────
    daily = df.groupby(df["datetime"].dt.date)["energy_kwh"].sum()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    last_week = daily.iloc[-7:].sum()
    prev_week = daily.iloc[-14:-7].sum()
    if prev_week > 0:
        pct = round((last_week - prev_week) / prev_week * 100, 1)
        word = "increase" if pct > 0 else "decrease"
        insights.append({
            "icon": "📈" if pct > 0 else "📉",
            "title": "Weekly Consumption Forecast",
            "text": f"Your consumption will {word} by {abs(pct)}% next week "
                    f"based on the trend from the last 14 days.",
            "severity": "warning" if pct > 5 else "success" if pct < -3 else "info",
        })

    # ── 2. Peak-hour analysis ────────────────────────────────────────
    hourly_avg = df.groupby("hour")["energy_kwh"].mean()
    peak_h = int(hourly_avg.idxmax())
    peak_v = hourly_avg.max()
    off_v = hourly_avg.min()
    ratio = round(peak_v / off_v, 1) if off_v > 0 else 0
    insights.append({
        "icon": "⏰",
        "title": "Peak vs Off-Peak Ratio",
        "text": f"Usage at {peak_h}:00 is {ratio}× higher than off-peak. "
                f"Shifting 20% of peak load could save ~{min(round(ratio * 3), 25)}%.",
        "severity": "info",
    })

    # ── 3. Top appliance consumer ────────────────────────────────────
    available = [c for c in APPLIANCE_COLS if c in df.columns]
    if available:
        totals = {c: df[c].sum() for c in available}
        top = max(totals, key=totals.get)
        top_pct = round(totals[top] / df["energy_kwh"].sum() * 100, 1)
        labels = {"ac_kwh": "AC / Heating", "lighting_kwh": "Lighting",
                  "kitchen_kwh": "Kitchen", "fan_kwh": "Fan",
                  "entertainment_kwh": "Entertainment",
                  "other_kwh": "Other / Standby"}
        insights.append({
            "icon": "🔌",
            "title": "Highest Energy Consumer",
            "text": f"{labels.get(top, top)} accounts for {top_pct}% "
                    f"of total energy — focus savings efforts here.",
            "severity": "warning" if top_pct > 30 else "info",
        })

    # ── 4. Next-month seasonal forecast ──────────────────────────────
    monthly = df.groupby("month")["energy_kwh"].sum()
    cur_m = int(df["datetime"].dt.month.iloc[-1])
    nxt_m = (cur_m % 12) + 1
    if nxt_m in monthly.index and cur_m in monthly.index:
        change = round((monthly[nxt_m] - monthly[cur_m]) /
                       monthly[cur_m] * 100, 1)
        m_names = ["", "January", "February", "March", "April", "May",
                   "June", "July", "August", "September", "October",
                   "November", "December"]
        insights.append({
            "icon": "🌡️",
            "title": f"Seasonal Outlook — {m_names[nxt_m]}",
            "text": f"Based on seasonal patterns, {m_names[nxt_m]} consumption "
                    f"is expected to {'rise' if change > 0 else 'drop'} by "
                    f"{abs(change)}% compared to {m_names[cur_m]}.",
            "severity": "warning" if change > 10 else
                        "success" if change < -5 else "info",
        })

    # ── 5. Anomaly health check ──────────────────────────────────────
    rmean = df["energy_kwh"].rolling(24).mean()
    rstd = df["energy_kwh"].rolling(24).std()
    z = ((df["energy_kwh"] - rmean) / rstd).abs()
    n_spikes = int((z > 3).sum())
    spike_pct = round(n_spikes / len(df) * 100, 2)
    insights.append({
        "icon": "🚨",
        "title": "Anomaly Health Check",
        "text": f"{n_spikes} unusual spikes detected ({spike_pct}%). "
                + ("System looks healthy." if spike_pct < 3
                   else "Consider checking appliances for malfunctions."),
        "severity": "success" if spike_pct < 3 else "danger",
    })

    # ── 6. Cost estimate ─────────────────────────────────────────────
    total_kwh = df["energy_kwh"].sum()
    rate = 6.5  # ₹ per kWh (approx Indian domestic tariff)
    monthly_cost = round(total_kwh / 12 * rate, 0)
    insights.append({
        "icon": "💰",
        "title": "Estimated Monthly Cost",
        "text": f"At ₹{rate}/kWh, average monthly bill ≈ ₹{monthly_cost:,.0f}. "
                f"Smart scheduling could save 10-15% (₹{round(monthly_cost*0.12):,.0f}).",
        "severity": "info",
    })

    # ── 7. Weekend vs Weekday ────────────────────────────────────────
    we = df[df["is_weekend"] == 1]["energy_kwh"].mean()
    wd = df[df["is_weekend"] == 0]["energy_kwh"].mean()
    if we > wd and wd > 0:
        pdiff = round((we - wd) / wd * 100, 1)
        insights.append({
            "icon": "📅",
            "title": "Weekend Usage Pattern",
            "text": f"Weekend consumption is {pdiff}% higher than weekdays — "
                    f"mainly from entertainment and kitchen appliances.",
            "severity": "info" if pdiff < 10 else "warning",
        })

    return insights


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 12 — THRESHOLD ALERT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════
#
#  Viva: This is a *threshold-based monitoring system* — a common
#  component in IoT / smart-grid applications.  When consumption
#  exceeds a configured limit the system generates alerts ranked
#  by severity (warning vs critical).  Thresholds can be static
#  (user-defined) or dynamic (mean + k × std from the data).
# ═══════════════════════════════════════════════════════════════════════════

def check_alerts(filepath: str = "dataset.csv",
                 daily_threshold: float | None = None,
                 hourly_threshold: float | None = None) -> dict:
    """
    Generate alerts when energy exceeds thresholds.

    Defaults:
      daily  → mean + 1.5 × std
      hourly → mean + 2 × std
    """
    df = load_data(filepath)

    # ── Daily aggregation ────────────────────────────────────────────
    daily = df.groupby(df["datetime"].dt.date)["energy_kwh"].sum()
    daily.index = pd.to_datetime(daily.index)
    d_mean, d_std = float(daily.mean()), float(daily.std())
    h_mean, h_std = float(df["energy_kwh"].mean()), float(df["energy_kwh"].std())

    if daily_threshold is None:
        daily_threshold = round(d_mean + 1.5 * d_std, 2)
    if hourly_threshold is None:
        hourly_threshold = round(h_mean + 2 * h_std, 2)

    alerts: list[dict] = []

    # ── Daily alerts ─────────────────────────────────────────────────
    over_daily = daily[daily > daily_threshold]
    for date, kwh in over_daily.items():
        alerts.append({
            "type": "daily",
            "severity": "critical" if kwh > daily_threshold * 1.3 else "warning",
            "date": date.strftime("%Y-%m-%d"),
            "value": round(float(kwh), 2),
            "threshold": daily_threshold,
            "message": f"Daily usage {round(float(kwh), 1)} kWh exceeded "
                       f"threshold ({daily_threshold} kWh)",
        })

    # ── Hourly spike alerts ──────────────────────────────────────────
    over_hourly = df[df["energy_kwh"] > hourly_threshold]
    for _, row in over_hourly.head(80).iterrows():
        alerts.append({
            "type": "hourly",
            "severity": "critical" if row["energy_kwh"] > hourly_threshold * 1.5
                        else "warning",
            "date": row["datetime"].strftime("%Y-%m-%d %H:%M"),
            "value": round(float(row["energy_kwh"]), 3),
            "threshold": hourly_threshold,
            "message": f"Hourly spike {round(float(row['energy_kwh']), 2)} kWh "
                       f"at {row['datetime'].strftime('%H:%M')}",
        })

    sev_order = {"critical": 0, "warning": 1}
    alerts.sort(key=lambda a: (sev_order.get(a["severity"], 2), a["date"]))

    n_crit = sum(1 for a in alerts if a["severity"] == "critical")
    n_warn = sum(1 for a in alerts if a["severity"] == "warning")

    return {
        "alerts":         alerts[:120],
        "total_alerts":   len(alerts),
        "critical_count": n_crit,
        "warning_count":  n_warn,
        "thresholds": {
            "daily_kwh":  daily_threshold,
            "hourly_kwh": hourly_threshold,
        },
        "summary": {
            "days_over":  len(over_daily),
            "hours_over": len(over_hourly),
            "total_days": len(daily),
            "total_hours": len(df),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 13 — AUTOMATED PDF REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════
#
#  Viva: Automated report generation is critical in real-world energy
#  management.  We collect output from every analysis module and
#  format it into a downloadable PDF — demonstrating a complete
#  data pipeline: raw data → analysis → presentation.
# ═══════════════════════════════════════════════════════════════════════════

def generate_pdf_report(filepath: str = "dataset.csv") -> str:
    """
    Produce a multi-page PDF energy report and return the file path.
    """
    from fpdf import FPDF

    report_path = os.path.join(MODEL_DIR, "energy_report.pdf")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Helper — sanitise Unicode chars the default font can't render
    def _s(text: str) -> str:
        return (text
                .replace("\u2013", "-").replace("\u2014", "-")   # en/em dash
                .replace("\u2018", "'").replace("\u2019", "'")   # quotes
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2265", ">=").replace("\u2248", "~")
                .replace("\u20b9", "Rs.")                        # ₹ symbol
                .encode("latin-1", errors="replace").decode("latin-1"))

    stats    = get_summary_statistics(filepath)
    tips     = get_recommendations(filepath)
    insights = generate_ai_insights(filepath)
    alerts   = check_alerts(filepath)
    trends   = get_trend_analysis(filepath)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Page 1: Title & Summary ──────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 26)
    pdf.cell(0, 18, _s("Smart Energy Report"), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, _s(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 8, _s("AI-Based Smart Energy Monitoring System | INT 428"),
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(12)

    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, _s("1.  Summary Statistics"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    rows = [
        ("Total Consumption", f"{stats['total_kwh']} kWh"),
        ("Avg Daily Usage",   f"{stats['avg_daily_kwh']} kWh"),
        ("Peak Hour",         stats["peak_hour_label"]),
        ("Peak Day",          stats["peak_day"]),
        ("Peak Month",        stats["peak_month"]),
        ("Total Records",     str(stats["total_records"])),
        ("Std Deviation",     str(stats["std_kwh"])),
    ]
    for label, value in rows:
        pdf.cell(90, 7, _s(f"   {label}:"), border=0)
        pdf.cell(90, 7, _s(value), border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ── Trend Analysis ───────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, _s("2.  Trend Analysis"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7,
             _s(f"   Overall Direction: {trends['overall_direction'].upper()}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7,
             _s(f"   Avg Daily: {trends['avg_daily_kwh']} kWh  |  "
                f"Slope: {trends['slope_per_day']} kWh/day"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, _s("   Monthly Breakdown:"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for m in trends["monthly_trend"]:
        arrow = "^" if m["pct_change"] > 0 else "v" if m["pct_change"] < 0 else "-"
        pdf.cell(0, 6,
                 _s(f"      {m['month']:>3s}:  {m['kwh']:>8.1f} kWh   "
                    f"({arrow} {m['pct_change']:+.1f}%)"),
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── AI Insights ──────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, _s("3.  AI-Generated Insights"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for ins in insights:
        pdf.multi_cell(0, 6, _s(f"   {ins['icon']} {ins['title']}:  {ins['text']}"))
        pdf.ln(2)
    pdf.ln(4)

    # ── Page 2: Alerts ───────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, _s("4.  Alert Summary"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7,
             _s(f"   Total Alerts: {alerts['total_alerts']}   |   "
                f"Critical: {alerts['critical_count']}   |   "
                f"Warning: {alerts['warning_count']}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7,
             _s(f"   Daily Threshold: {alerts['thresholds']['daily_kwh']} kWh   |   "
                f"Hourly Threshold: {alerts['thresholds']['hourly_kwh']} kWh"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7,
             _s(f"   Days Over Threshold: {alerts['summary']['days_over']} / "
                f"{alerts['summary']['total_days']}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, _s("   Top Alerts:"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for a in alerts["alerts"][:20]:
        sev_tag = "[CRITICAL]" if a["severity"] == "critical" else "[WARNING] "
        pdf.cell(0, 6,
                 _s(f"      {sev_tag}  {a['date']}  -  {a['message']}"),
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Recommendations ──────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, _s("5.  Energy-Saving Recommendations"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for tip in tips:
        pri = f"[{tip.get('priority', 'medium').upper()}]"
        pdf.multi_cell(0, 6,
                       _s(f"   {tip['icon']} {pri} {tip['title']}:  "
                          f"{tip['detail']}  (Save ~{tip['saving_pct']}%)"))
        pdf.ln(2)

    # ── Footer ───────────────────────────────────────────────────────
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 7, _s("Report generated by EnergyAI  -  INT 428 AI Project"),
             align="C")

    pdf.output(report_path)
    print(f"✅  PDF report saved → {report_path}")
    return report_path


# ── CLI test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = train_model()
    print(json.dumps(m, indent=2))

    print("\n--- Anomaly Detection ---")
    a = detect_anomalies()
    print(f"Found {a['total_anomalies']} anomalies ({a['pct_anomalies']}%)")

    print("\n--- Advanced Forecast ---")
    fc = train_forecast_model()
    print(f"Method: {fc['method']}  RMSE={fc['metrics']['RMSE']}")
    for p in fc["forecast"][:3]:
        print(p)

    print("\n--- AI Insights ---")
    for ins in generate_ai_insights():
        print(f"  {ins['icon']} {ins['title']}: {ins['text']}")

    print("\n--- Alerts ---")
    al = check_alerts()
    print(f"Total: {al['total_alerts']}  Critical: {al['critical_count']}")

    print("\n--- PDF Report ---")
    path = generate_pdf_report()
    print(f"Saved: {path}")
