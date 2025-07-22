from sklearn.metrics import accuracy_score # type: ignore
from sklearn.metrics import precision_score, recall_score, roc_auc_score # type: ignore

import pandas as pd
import pickle
import json
import os # Import the os module for creating directories

# Define paths (good practice to define them at the top or pass as arguments)
MODEL_PATH = "models/random_forest_model.pkl"
TEST_DATA_PATH = "data/interim/test_bow.csv"
PREDICTIONS_OUTPUT_PATH = "data/eval/predictions.csv"
METRICS_OUTPUT_PATH = "metrics.json" # Change this to match dvc.yaml's expected location

# --- Model Loading and Prediction ---
print(f"Loading model from: {MODEL_PATH}")
model = pickle.load(open(MODEL_PATH, "rb"))

print(f"Loading test data from: {TEST_DATA_PATH}")
test_data = pd.read_csv(TEST_DATA_PATH)

# Ensure 'label' column exists for y_test
if 'label' not in test_data.columns:
    raise ValueError("The 'test_bow.csv' must contain a 'label' column for evaluation.")

X_test = test_data.drop(columns=['label']).values
y_test = test_data['label'].values

print("Making predictions...")
y_pred = model.predict(X_test)

# --- Save Predictions ---
print(f"Saving predictions to: {PREDICTIONS_OUTPUT_PATH}")
# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(PREDICTIONS_OUTPUT_PATH), exist_ok=True)
# Create a DataFrame for predictions and save it
predictions_df = pd.DataFrame({'predicted_label': y_pred}) # Use a descriptive column name
predictions_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)


# --- Calculate and Save Metrics ---
print("Calculating metrics...")
metrics_dict = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_pred)
}

print(f"Saving metrics to: {METRICS_OUTPUT_PATH}")
# Ensure the directory for metrics exists (though for root, it's usually not needed)
os.makedirs(os.path.dirname(METRICS_OUTPUT_PATH) or '.', exist_ok=True) # '. ' handles root path safely
with open(METRICS_OUTPUT_PATH, "w") as f:
    json.dump(metrics_dict, f, indent=4)

print("Model evaluation complete.")