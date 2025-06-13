"""
ETL processing module for TruthGuard.
Handles data transformation, cleaning, and feature engineering.
"""

import logging
from typing import List, Optional

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ETLProcessor:
    """Handles ETL processing of news data."""

    def __init__(self):
        """Initialize ETL processor with Spark session."""
        self.spark = SparkSession.builder \
            .appName("TruthGuard ETL") \
            .config("spark.sql.warehouse.dir", "spark-warehouse") \
            .getOrCreate()
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = ''.join(c for c in text if c.isalpha() or c.isspace())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return ' '.join(tokens)

    def process_spark_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process DataFrame using Spark for scalability.
        
        Args:
            df: Input pandas DataFrame
            
        Returns:
            Processed pandas DataFrame
        """
        try:
            # Convert to Spark DataFrame
            spark_df = self.spark.createDataFrame(df)
            
            # Register UDF for text cleaning
            clean_text_udf = udf(self.clean_text, StringType())
            
            # Apply text cleaning
            processed_df = spark_df.withColumn(
                'cleaned_text',
                clean_text_udf(col('text'))
            )
            
            # Convert back to pandas
            return processed_df.toPandas()
        except Exception as e:
            logger.error(f"Error processing Spark DataFrame: {e}")
            raise

    def extract_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract TF-IDF features from text data.
        
        Args:
            texts: List of text documents
            
        Returns:
            DataFrame with TF-IDF features
        """
        try:
            # Fit and transform TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Convert to DataFrame
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            df_features = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=feature_names
            )
            
            return df_features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise

    def process_batch(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of news articles.
        
        Args:
            input_df: Input DataFrame with news articles
            
        Returns:
            Processed DataFrame with features
        """
        try:
            # Clean text
            processed_df = self.process_spark_dataframe(input_df)
            
            # Extract features
            features_df = self.extract_features(processed_df['cleaned_text'].tolist())
            
            # Combine with original data
            result_df = pd.concat([
                processed_df.drop('cleaned_text', axis=1),
                features_df
            ], axis=1)
            
            logger.info(f"Successfully processed batch of {len(result_df)} articles")
            return result_df
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise

    def process_stream(self, text: str) -> pd.DataFrame:
        """
        Process a single news article for real-time inference.
        
        Args:
            text: Input text to process
            
        Returns:
            DataFrame with processed features
        """
        try:
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Extract features
            features = self.tfidf_vectorizer.transform([cleaned_text])
            
            # Convert to DataFrame
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            df_features = pd.DataFrame(
                features.toarray(),
                columns=feature_names
            )
            
            return df_features
        except Exception as e:
            logger.error(f"Error processing stream: {e}")
            raise

    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save processed data to storage.
        
        Args:
            df: Processed DataFrame
            output_path: Path to save the data
        """
        try:
            # Convert to Spark DataFrame
            spark_df = self.spark.createDataFrame(df)
            
            # Save as parquet
            spark_df.write.parquet(output_path, mode='overwrite')
            
            logger.info(f"Successfully saved processed data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise

    def __del__(self):
        """Clean up Spark session."""
        if hasattr(self, 'spark'):
            self.spark.stop() 