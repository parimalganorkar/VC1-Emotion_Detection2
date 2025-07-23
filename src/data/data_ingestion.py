import numpy as np
import pandas as pd
import os
import yaml
import logging

from typing import Tuple
from pandas import DataFrame
from sklearn.model_selection import train_test_split # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
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

def load_dataset(url: str) -> DataFrame:
    """Load dataset from a remote CSV file."""
    try:
        df = pd.read_csv(url)
        logging.info(f"Dataset loaded from {url} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

def preprocess_data(df: DataFrame) -> DataFrame:
    """Preprocess the dataset: drop columns, filter, and encode labels."""
    try:
        df = df.drop(columns=['tweet_id'])
        logging.info("Dropped 'tweet_id' column.")
        df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        logging.info("Filtered dataset for 'happiness' and 'sadness' sentiments.")
        df['sentiment'] = df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.info("Encoded sentiment labels to binary.")
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def split_data(df: DataFrame, test_size: float, random_state: int = 42) -> Tuple[DataFrame, DataFrame]:
    """Split the data into training and testing sets."""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Split data into train ({train_data.shape}) and test ({test_data.shape}) sets.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error during data splitting: {e}")
        raise

def save_data(train_data: DataFrame, test_data: DataFrame, output_dir: str) -> None:
    """Save the training and testing data to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, 'train.csv')
        test_path = os.path.join(output_dir, 'test.csv')
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info(f"Saved train data to {train_path} and test data to {test_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main() -> None:
    """Main function to orchestrate data ingestion."""
    try:
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        df = load_dataset('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        processed_df = preprocess_data(df)
        train_data, test_data = split_data(processed_df, test_size)
        save_data(train_data, test_data, 'data/raw')
        logging.info("Data ingestion completed successfully.")
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")

if __name__ == "__main__":
    main()