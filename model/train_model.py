import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# =========================================================
# PATH SETTINGS
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

DATASET_PATH = os.path.join(PROJECT_DIR, "Dataset", "CrimesOnWomenData.csv")
MODEL_DIR = CURRENT_DIR
GRAPH_DIR = os.path.join(MODEL_DIR, "confusion_matrices")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
RESULTS_CSV_PATH = os.path.join(MODEL_DIR, "model_results.csv")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "thresholds.pkl")
BEST_MODEL_INFO_PATH = os.path.join(MODEL_DIR, "best_model_info.pkl")
BEST_CM_PATH = os.path.join(GRAPH_DIR, "best_model_cm.png")

# =========================================================
# LOAD DATA
# =========================================================
print("===================================")
print("Loading dataset...")
print("Dataset path:", DATASET_PATH)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
df.columns = [str(col).strip() for col in df.columns]

unnamed_cols = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]
if unnamed_cols:
    df.drop(columns=unnamed_cols, inplace=True)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))

# =========================================================
# IDENTIFY STATE & YEAR
# =========================================================
year_col = next((c for c in df.columns if str(c).strip().lower() == "year"), None)
if year_col is None:
    raise ValueError("Year column not found in dataset")

state_col = next((c for c in df.columns if str(c).strip().lower() in ["state", "area_name"]), None)
if state_col is None:
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(object_cols) == 0:
        raise ValueError("State column not found and no object column available")
    state_col = object_cols[0]

print("Detected State column:", state_col)
print("Detected Year column:", year_col)

# =========================================================
# CLEAN DATA
# =========================================================
df[state_col] = df[state_col].astype(str).str.strip()
df[year_col] = pd.to_numeric(df[year_col], errors='coerce')

crime_cols = [c for c in df.columns if c not in [state_col, year_col]]
valid_crime_cols = []
for col in crime_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if not df[col].isna().all():
        valid_crime_cols.append(col)
crime_cols = valid_crime_cols

for col in crime_cols:
    median_val = df[col].median()
    if pd.isna(median_val):
        median_val = 0
    df[col].fillna(median_val, inplace=True)

df.dropna(subset=[year_col], inplace=True)
df[year_col] = df[year_col].astype(int)
df = df[df[state_col].astype(str).str.strip() != ""]
df.reset_index(drop=True, inplace=True)

print("Cleaned dataset shape:", df.shape)
print("Using crime columns:", crime_cols)

# =========================================================
# DERIVED FEATURES
# =========================================================
df["Total_Crime"] = df[crime_cols].sum(axis=1)
df = df.sort_values([state_col, year_col])

# Crime last year per state
df["Crime_Last_Year"] = df.groupby(state_col)["Total_Crime"].shift(1).fillna(0)
# 3-year rolling average
df["Crime_3Year_Avg"] = df.groupby(state_col)["Total_Crime"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
# Growth rate
df["Crime_Growth_Rate"] = (df["Total_Crime"] - df["Crime_Last_Year"]) / df["Crime_Last_Year"].replace(0, 1)

# =========================================================
# CREATE TARGET (RISK LEVEL)
# =========================================================
q1 = df["Total_Crime"].quantile(0.33)
q2 = df["Total_Crime"].quantile(0.66)
if q1 == q2:
    q1 = df["Total_Crime"].quantile(0.25)
    q2 = df["Total_Crime"].quantile(0.75)
if q1 == q2:
    min_val = df["Total_Crime"].min()
    max_val = df["Total_Crime"].max()
    q1 = min_val + (max_val - min_val) / 3.0
    q2 = min_val + 2 * (max_val - min_val) / 3.0

def get_risk_label(x):
    if x <= q1:
        return "Low"
    elif x <= q2:
        return "Medium"
    else:
        return "High"

df["Risk_Level"] = df["Total_Crime"].apply(get_risk_label)
class_counts = df["Risk_Level"].value_counts()
print("Risk level distribution:")
print(class_counts)
if df["Risk_Level"].nunique() < 2:
    raise ValueError("Only one class generated for Risk_Level. Classification cannot proceed.")

# =========================================================
# ENCODING
# =========================================================
state_encoder = LabelEncoder()
df["State_Encoded"] = state_encoder.fit_transform(df[state_col])

risk_order = ["Low", "Medium", "High"]
present_risks = [r for r in risk_order if r in df["Risk_Level"].unique()]
risk_encoder = LabelEncoder()
risk_encoder.fit(present_risks)
df["Risk_Encoded"] = risk_encoder.transform(df["Risk_Level"])

print("State classes count:", len(state_encoder.classes_))
print("Risk classes:", list(risk_encoder.classes_))

# =========================================================
# FEATURES & TARGET
# =========================================================
feature_cols = ["State_Encoded", year_col, "Crime_Last_Year", "Crime_3Year_Avg", "Crime_Growth_Rate"]
X = df[feature_cols]
y = df["Risk_Encoded"]

use_stratify = y if class_counts.min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=use_stratify)

print("Train size:", len(X_train))
print("Test size :", len(X_test))

# =========================================================
# MODEL DICTIONARY
# =========================================================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoost": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []
best_model = None
best_model_name = None
best_score = -1
best_y_pred = None
all_predictions = {}

# =========================================================
# TRAINING LOOP
# =========================================================
print("\n===================================")
print("Training models...")

for name, model in models.items():
    print(f"\nTraining {name}...")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_predictions[name] = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4)
        })

        print(f"{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        if best_model is None or acc > best_score or (acc == best_score and f1 > best_f1):
            best_model = model
            best_model_name = name
            best_score = acc
            best_y_pred = y_pred
            best_f1 = f1

    except Exception as e:
        print(f"Error training {name}: {str(e)}")

if len(results) == 0 or best_model is None:
    raise ValueError("No model trained successfully. Check dataset and preprocessing.")

# =========================================================
# SAVE RESULTS CSV
# =========================================================
results_df = pd.DataFrame(results).sort_values(by=["Accuracy", "F1_Score"], ascending=[False, False]).reset_index(drop=True)
results_df.to_csv(RESULTS_CSV_PATH, index=False)
print("\nSaved model results to:", RESULTS_CSV_PATH)
print(results_df)

# =========================================================
# CONFUSION MATRICES
# =========================================================
available_class_labels = [label for label in ["Low", "Medium", "High"] if label in risk_encoder.classes_]
y_test_labels = risk_encoder.inverse_transform(y_test)

for name, y_pred_each in all_predictions.items():
    try:
        y_pred_labels = risk_encoder.inverse_transform(y_pred_each)
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=available_class_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=available_class_labels)

        plt.figure(figsize=(6, 5))
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f"{name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIR, f"{name}_cm.png"))
        plt.close()
    except Exception as e:
        print(f"Could not save confusion matrix for {name}: {str(e)}")

# Best model CM
try:
    best_y_pred_labels = risk_encoder.inverse_transform(best_y_pred)
    best_cm = confusion_matrix(y_test_labels, best_y_pred_labels, labels=available_class_labels)
    best_disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=available_class_labels)
    plt.figure(figsize=(6, 5))
    best_disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{best_model_name} Best Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(BEST_CM_PATH)
    plt.close()
except Exception as e:
    print("Could not save best model confusion matrix:", str(e))

# =========================================================
# SAVE MODELS + ENCODERS + META
# =========================================================
pickle.dump(best_model, open(BEST_MODEL_PATH, "wb"))
pickle.dump({"state_encoder": state_encoder, "risk_encoder": risk_encoder}, open(ENCODER_PATH, "wb"))
pickle.dump({"q1": q1, "q2": q2}, open(THRESHOLD_PATH, "wb"))
pickle.dump({"best_model_name": best_model_name, "best_accuracy": float(best_score)}, open(BEST_MODEL_INFO_PATH, "wb"))

# =========================================================
# SUMMARY
# =========================================================
print("\n===================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("Best Model   :", best_model_name)
print("Best Accuracy:", round(float(best_score), 4))
print("\nGenerated files:")
print("1.", RESULTS_CSV_PATH)
print("2.", BEST_MODEL_PATH)
print("3.", ENCODER_PATH)
print("4.", THRESHOLD_PATH)
print("5.", BEST_MODEL_INFO_PATH)
print("6.", BEST_CM_PATH)
print("7. All model confusion matrices inside:", GRAPH_DIR)