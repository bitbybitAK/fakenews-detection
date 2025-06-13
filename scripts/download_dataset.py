"""
Script to download and prepare the Kaggle fake news dataset.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import kaggle
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset configuration
DATASET_NAME = "clmentbisaillon/fake-and-real-news-dataset"
DATA_DIR = Path("data/raw")

def download_dataset():
    """Download the Kaggle dataset."""
    try:
        # Create data directory if it doesn't exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            DATASET_NAME,
            path=DATA_DIR,
            unzip=True
        )
        
        logger.info(f"Successfully downloaded dataset to {DATA_DIR}")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise

def prepare_dataset():
    """Prepare and clean the dataset."""
    try:
        # Read fake news data
        fake_df = pd.read_csv(DATA_DIR / "Fake.csv")
        fake_df['label'] = 1  # 1 for fake news
        
        # Read real news data
        real_df = pd.read_csv(DATA_DIR / "True.csv")
        real_df['label'] = 0  # 0 for real news
        
        # Combine datasets
        df = pd.concat([fake_df, real_df], ignore_index=True)
        
        # Clean column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Combine title and text
        df['text'] = df['title'] + ' ' + df['text']
        
        # Drop unnecessary columns
        df = df.drop(['title', 'subject', 'date'], axis=1)
        
        # Shuffle data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save processed dataset
        output_path = DATA_DIR / "fake_news_dataset.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"Successfully prepared dataset and saved to {output_path}")
        
        # Print dataset statistics
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Fake news samples: {len(df[df['label'] == 1])}")
        logger.info(f"Real news samples: {len(df[df['label'] == 0])}")
        
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise

def validate_dataset():
    """Validate the prepared dataset."""
    try:
        # Read processed dataset
        df = pd.read_csv(DATA_DIR / "fake_news_dataset.csv")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning("Found missing values in dataset:")
            logger.warning(missing_values[missing_values > 0])
        
        # Check for duplicate entries
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate entries")
        
        # Check label distribution
        label_dist = df['label'].value_counts()
        logger.info("Label distribution:")
        logger.info(f"Fake news (1): {label_dist[1]}")
        logger.info(f"Real news (0): {label_dist[0]}")
        
        # Check text length statistics
        df['text_length'] = df['text'].str.len()
        length_stats = df['text_length'].describe()
        logger.info("Text length statistics:")
        logger.info(length_stats)
        
        logger.info("Dataset validation completed")
        
    except Exception as e:
        logger.error(f"Error validating dataset: {e}")
        raise

def main():
    """Main function to download and prepare the dataset."""
    try:
        # Download dataset
        download_dataset()
        
        # Prepare dataset
        prepare_dataset()
        
        # Validate dataset
        validate_dataset()
        
        logger.info("Dataset preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in dataset preparation: {e}")
        raise

if __name__ == "__main__":
    main() 