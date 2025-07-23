import os
import pickle
import json
import logging
from typing import Any, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

MODEL_PATH = "models/random_forest_model.pkl"
TEST_DATA_PATH = "data/interim/test_tfidf.csv"
PREDICTIONS_OUTPUT_PATH = "data/eval/predictions.csv"
METRICS_OUTPUT_PATH = "metrics.json"

def load_model(model_path: str) -> Any:
    """Load a trained model from a file."""
    try:
        logging.info(f"Loading model from: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def load_test_data(test_data_path: str) -> Tuple[pd.DataFrame, Any, Any]:
    """Load test data and split into features and labels."""
    try:
        logging.info(f"Loading test data from: {test_data_path}")
        test_data = pd.read_csv(test_data_path)
        if 'label' not in test_data.columns:
            raise ValueError("The 'test_bow.csv' must contain a 'label' column for evaluation.")
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values
        return test_data, X_test, y_test
    except Exception as e:
        logging.error(f"Failed to load or process test data: {e}")
        raise

def make_predictions(model: Any, X_test: Any) -> Any:
    """Make predictions using the trained model."""
    try:
        logging.info("Making predictions...")
        y_pred = model.predict(X_test)
        return y_pred
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise

def save_predictions(y_pred: Any, output_path: str) -> None:
    """Save predictions to a CSV file."""
    try:
        logging.info(f"Saving predictions to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df = pd.DataFrame({'predicted_label': y_pred})
        predictions_df.to_csv(output_path, index=False)
    except Exception as e:
        logging.error(f"Failed to save predictions: {e}")
        raise

def calculate_metrics(y_test: Any, y_pred: Any) -> dict:
    """Calculate evaluation metrics."""
    try:
        logging.info("Calculating metrics...")
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        return metrics_dict
    except Exception as e:
        logging.error(f"Failed to calculate metrics: {e}")
        raise

def save_metrics(metrics: dict, output_path: str) -> None:
    """Save metrics to a JSON file."""
    try:
        logging.info(f"Saving metrics to: {output_path}")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to save metrics: {e}")
        raise

def main() -> None:
    """Main function to orchestrate model prediction and evaluation."""
    try:
        model = load_model(MODEL_PATH)
        _, X_test, y_test = load_test_data(TEST_DATA_PATH)
        y_pred = make_predictions(model, X_test)
        save_predictions(y_pred, PREDICTIONS_OUTPUT_PATH)
        metrics = calculate_metrics(y_test, y_pred)
        save_metrics(metrics, METRICS_OUTPUT_PATH)
        logging.info("Model evaluation complete.")
    except Exception as e:
        logging.error(f"Prediction pipeline failed: {e}")

if __name__ == "__main__":
    main()