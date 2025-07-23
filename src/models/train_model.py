import pandas as pd
import numpy as np
import pickle
import yaml
import os
import logging
from typing import Any
from sklearn.ensemble import RandomForestClassifier # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_params(param_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {param_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def load_train_data(train_path: str) -> pd.DataFrame:
    """Load training data from CSV."""
    try:
        train_data = pd.read_csv(train_path)
        logging.info(f"Training data loaded from {train_path} with shape {train_data.shape}")
        return train_data
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        raise

def train_random_forest(
    x_train: np.ndarray, y_train: np.ndarray, n_estimators: int, max_depth: int
) -> RandomForestClassifier:
    """Train a RandomForestClassifier."""
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(x_train, y_train)
        logging.info("RandomForestClassifier trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise

def save_model(model: Any, model_path: str) -> None:
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise

def main() -> None:
    """Main function to orchestrate model training."""
    try:
        params = load_params('params.yaml')
        n_estimators = params['model_training']['n_estimators']
        max_depth = params['model_training']['max_depth']

        train_data = load_train_data("data/interim/train_tfidf.csv")
        x_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values

        model = train_random_forest(x_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")
        logging.info("Model training pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Model training pipeline failed: {e}")

if __name__ == "__main__":
    main()