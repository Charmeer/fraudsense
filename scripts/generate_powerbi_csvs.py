from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import shap

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
OUTPUT_DIR = BASE

RAW_DATA_PATH = DATA_DIR / "creditcard_with_hour.csv"
FEATURE_NAMES_PATH = DATA_DIR / "feature_names.pkl"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
SHAP_MODEL_PATH = MODELS_DIR / "lightgbm.pkl"

POWERBI_EXPORT_PATH = OUTPUT_DIR / "powerbi_export.csv"
SHAP_IMPORTANCE_PATH = OUTPUT_DIR / "shap_importance.csv"
THRESHOLD_CURVE_PATH = OUTPUT_DIR / "threshold_curve.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Time").reset_index(drop=True)
    df["time_gap"] = df["Time"].diff().fillna(0)
    df["rolling_count_10"] = df["Amount"].rolling(window=10, min_periods=1).count()
    df["rolling_avg_amt_10"] = df["Amount"].rolling(window=10, min_periods=1).mean()
    df["rolling_std_amt_10"] = df["Amount"].rolling(window=10, min_periods=1).std().fillna(0)
    df["amount_deviation"] = df["Amount"] - df["rolling_avg_amt_10"]
    df["is_round_amount"] = (df["Amount"] % 1 == 0).astype(int)
    bins = [-1, 1, 10, 100, 500, 1000, float("inf")]
    labels = [0, 1, 2, 3, 4, 5]
    df["amount_bucket"] = pd.cut(df["Amount"], bins=bins, labels=labels).astype(int)
    df["hour_x_amount"] = df["Hour"] * df["Amount"]
    df["Amount"] = df["Amount"] - df["Amount"].mean()
    df["Hour"] = df["Hour"] - df["Hour"].mean()
    return df


def load_features() -> list[str]:
    return joblib.load(FEATURE_NAMES_PATH)


def compute_threshold_curve(y_true: pd.Series, probs: np.ndarray) -> pd.DataFrame:
    rows = []
    thresholds = np.arange(0.05, 0.951, 0.05)
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_positive_count": int(preds.sum()),
            }
        )
    return pd.DataFrame(rows)


def normalize_shap_values(shap_values):
    shap_arr = np.asarray(shap_values)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_arr = np.asarray(shap_values[1])
    if shap_arr.ndim == 3 and shap_arr.shape[0] == 2:
        shap_arr = shap_arr[1]
    return shap_arr


def generate_shap_importance(model, features: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    sample = features.sample(2000, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    shap_arr = normalize_shap_values(shap_values)
    shap_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean_SHAP": np.abs(shap_arr).mean(axis=0),
            "Fraud_direction": shap_arr.mean(axis=0),
        }
    )
    return shap_importance.sort_values("Mean_SHAP", ascending=False)


def main() -> None:
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    df = engineer_features(df)

    print("Loading model and features...")
    feature_names = load_features()
    model = joblib.load(MODEL_PATH)
    shap_model = joblib.load(SHAP_MODEL_PATH)

    print("Computing fraud probabilities and predictions...")
    X = df[feature_names]
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df_export = df.copy()
    df_export["fraud_prob"] = probs
    df_export["prediction"] = preds
    df_export["prediction_label"] = np.where(df_export["prediction"] == 1, "Flagged", "Not flagged")
    df_export["fn"] = ((df_export["Class"] == 1) & (df_export["prediction"] == 0)).astype(int)
    df_export["fp"] = ((df_export["Class"] == 0) & (df_export["prediction"] == 1)).astype(int)

    print(f"Saving {POWERBI_EXPORT_PATH}...")
    df_export.to_csv(POWERBI_EXPORT_PATH, index=False)

    print(f"Creating threshold curve {THRESHOLD_CURVE_PATH}...")
    threshold_df = compute_threshold_curve(df_export["Class"], probs)
    threshold_df.to_csv(THRESHOLD_CURVE_PATH, index=False)

    print(f"Computing SHAP importance and saving {SHAP_IMPORTANCE_PATH}...")
    shap_imp = generate_shap_importance(shap_model, X, feature_names)
    shap_imp.to_csv(SHAP_IMPORTANCE_PATH, index=False)

    print("Done. Generated CSV files:")
    print(f" - {POWERBI_EXPORT_PATH}")
    print(f" - {THRESHOLD_CURVE_PATH}")
    print(f" - {SHAP_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
