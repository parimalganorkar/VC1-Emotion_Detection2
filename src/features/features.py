import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

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

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train and test data."""
    try:
        train_data = pd.read_csv(train_path).dropna(subset=['content'])
        test_data = pd.read_csv(test_path).dropna(subset=['content'])
        logging.info(f"Loaded train data from {train_path} and test data from {test_path}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def extract_features_and_labels(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract features and labels from train and test data."""
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        logging.info("Extracted features and labels from train and test data")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Failed to extract features and labels: {e}")
        raise

def vectorize_text(
    X_train: np.ndarray, X_test: np.ndarray, max_features: int
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Apply Bag of Words (CountVectorizer) to train and test data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logging.info("Text vectorization completed using Bag of Words")
        return X_train_bow, X_test_bow, vectorizer
    except Exception as e:
        logging.error(f"Text vectorization failed: {e}")
        raise

def save_features(
    X_train_bow: np.ndarray, y_train: np.ndarray,
    X_test_bow: np.ndarray, y_test: np.ndarray,
    output_dir: str
) -> None:
    """Save the processed feature data to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        train_path = os.path.join(output_dir, "train_tfidf.csv")
        test_path = os.path.join(output_dir, "test_tfidf.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logging.info(f"Saved feature data to {train_path} and {test_path}")
    except Exception as e:
        logging.error(f"Failed to save feature data: {e}")
        raise

def main() -> None:
    """Main function to orchestrate feature engineering."""
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")
        X_train, y_train, X_test, y_test = extract_features_and_labels(train_data, test_data)
        X_train_bow, X_test_bow, _ = vectorize_text(X_train, X_test, max_features)
        save_features(X_train_bow, y_train, X_test_bow, y_test, "data/interim")
        logging.info("Feature engineering completed successfully.")
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")

if __name__ == "__main__":
    main()