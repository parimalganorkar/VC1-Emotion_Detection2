import os
import re
import numpy as np
import pandas as pd
import nltk # type: ignore
import string
import logging
from typing import Any
from nltk.corpus import stopwords # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text: str) -> str:
    """Lemmatize each word in the text."""
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized)
    except Exception as e:
        logging.error(f"Lemmatization failed: {e}")
        return text

def remove_stop_words(text: str) -> str:
    """Remove stop words from the text."""
    try:
        stop_words = set(stopwords.words("english"))
        filtered = [word for word in str(text).split() if word not in stop_words]
        return " ".join(filtered)
    except Exception as e:
        logging.error(f"Removing stop words failed: {e}")
        return text

def removing_numbers(text: str) -> str:
    """Remove all digits from the text."""
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logging.error(f"Removing numbers failed: {e}")
        return text

def lower_case(text: str) -> str:
    """Convert all words in the text to lowercase."""
    try:
        words = text.split()
        lowered = [word.lower() for word in words]
        return " ".join(lowered)
    except Exception as e:
        logging.error(f"Lowercasing failed: {e}")
        return text

def removing_punctuations(text: str) -> str:
    """Remove punctuations and extra whitespace from the text."""
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logging.error(f"Removing punctuations failed: {e}")
        return text

def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logging.error(f"Removing URLs failed: {e}")
        return text

def remove_small_sentences(df: pd.DataFrame) -> None:
    """Set text to NaN if sentence has fewer than 3 words."""
    try:
        for i in range(len(df)):
            if len(str(df.text.iloc[i]).split()) < 3:
                df.text.iloc[i] = np.nan
        logging.info("Removed small sentences with fewer than 3 words.")
    except Exception as e:
        logging.error(f"Removing small sentences failed: {e}")

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to the 'content' column of the DataFrame."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logging.info("Text normalization completed.")
        return df
    except Exception as e:
        logging.error(f"Text normalization failed: {e}")
        return df

def normalized_sentence(sentence: str) -> str:
    """Apply all preprocessing steps to a single sentence."""
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error(f"Sentence normalization failed: {e}")
        return sentence

def load_and_preprocess_data(train_path: str, test_path: str) -> None:
    """Load, preprocess, and save train and test data."""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info(f"Loaded train data from {train_path} and test data from {test_path}")

        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)

        os.makedirs("data/processed", exist_ok=True)
        train_data.to_csv("data/processed/train.csv", index=False)
        test_data.to_csv("data/processed/test.csv", index=False)
        logging.info("Processed data saved to data/processed/")
    except Exception as e:
        logging.error(f"Data loading or preprocessing failed: {e}")

def main() -> None:
    """Main entry point for data preprocessing."""
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"
    load_and_preprocess_data(train_path, test_path)

if __name__ == "__main__":
    main()