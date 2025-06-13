"""
Data ingestion module for TruthGuard.
Handles data collection from various sources including Kaggle dataset and RSS feeds.
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from kafka import KafkaProducer
from minio import Minio
from minio.error import S3Error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DataIngestion:
    """Handles data ingestion from various sources."""

    def __init__(self):
        """Initialize data ingestion components."""
        self.minio_client = self._setup_minio()
        self.kafka_producer = self._setup_kafka()
        self.bucket_name = "raw-news"

    def _setup_minio(self) -> Minio:
        """Set up MinIO client for S3-like storage."""
        try:
            client = Minio(
                os.getenv("MINIO_ENDPOINT", "localhost:9000"),
                access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
                secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
                secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
            )
            
            # Create bucket if it doesn't exist
            if not client.bucket_exists(self.bucket_name):
                client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            
            return client
        except S3Error as e:
            logger.error(f"Error setting up MinIO client: {e}")
            raise

    def _setup_kafka(self) -> KafkaProducer:
        """Set up Kafka producer for streaming data."""
        try:
            return KafkaProducer(
                bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
                value_serializer=lambda x: str(x).encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Error setting up Kafka producer: {e}")
            raise

    def load_kaggle_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load the Kaggle fake news dataset.
        
        Args:
            file_path: Path to the Kaggle dataset CSV file
            
        Returns:
            DataFrame containing the dataset
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded Kaggle dataset from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading Kaggle dataset: {e}")
            raise

    def fetch_rss_feed(self, feed_url: str) -> List[dict]:
        """
        Fetch news articles from an RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            
        Returns:
            List of dictionaries containing article data
        """
        try:
            response = requests.get(feed_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            articles = []
            
            for item in soup.find_all('item'):
                article = {
                    'title': item.title.text if item.title else '',
                    'description': item.description.text if item.description else '',
                    'link': item.link.text if item.link else '',
                    'pubDate': item.pubDate.text if item.pubDate else '',
                    'source': feed_url
                }
                articles.append(article)
            
            logger.info(f"Successfully fetched {len(articles)} articles from {feed_url}")
            return articles
        except Exception as e:
            logger.error(f"Error fetching RSS feed {feed_url}: {e}")
            raise

    def store_to_minio(self, data: pd.DataFrame, filename: str) -> None:
        """
        Store data to MinIO (S3-like storage).
        
        Args:
            data: DataFrame to store
            filename: Name of the file to store
        """
        try:
            # Convert DataFrame to CSV
            csv_data = data.to_csv(index=False).encode('utf-8')
            
            # Upload to MinIO
            self.minio_client.put_object(
                bucket_name=self.bucket_name,
                object_name=filename,
                data=csv_data,
                length=len(csv_data),
                content_type='text/csv'
            )
            
            logger.info(f"Successfully stored data to MinIO: {filename}")
        except Exception as e:
            logger.error(f"Error storing data to MinIO: {e}")
            raise

    def stream_to_kafka(self, data: pd.DataFrame, topic: str) -> None:
        """
        Stream data to Kafka topic.
        
        Args:
            data: DataFrame to stream
            topic: Kafka topic to stream to
        """
        try:
            for _, row in data.iterrows():
                message = row.to_json()
                self.kafka_producer.send(topic, value=message)
            
            self.kafka_producer.flush()
            logger.info(f"Successfully streamed data to Kafka topic: {topic}")
        except Exception as e:
            logger.error(f"Error streaming data to Kafka: {e}")
            raise

    def process_kaggle_data(self, file_path: str) -> None:
        """
        Process and store Kaggle dataset.
        
        Args:
            file_path: Path to the Kaggle dataset
        """
        try:
            # Load data
            df = self.load_kaggle_dataset(file_path)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kaggle_fake_news_{timestamp}.csv"
            
            # Store to MinIO
            self.store_to_minio(df, filename)
            
            # Stream to Kafka
            self.stream_to_kafka(df, os.getenv("KAFKA_TOPIC_RAW", "raw_news"))
            
            logger.info("Successfully processed Kaggle dataset")
        except Exception as e:
            logger.error(f"Error processing Kaggle dataset: {e}")
            raise

    def process_rss_feeds(self, feed_urls: List[str]) -> None:
        """
        Process and store RSS feeds.
        
        Args:
            feed_urls: List of RSS feed URLs
        """
        try:
            all_articles = []
            
            # Fetch articles from all feeds
            for url in feed_urls:
                articles = self.fetch_rss_feed(url)
                all_articles.extend(articles)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_articles)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rss_news_{timestamp}.csv"
            
            # Store to MinIO
            self.store_to_minio(df, filename)
            
            # Stream to Kafka
            self.stream_to_kafka(df, os.getenv("KAFKA_TOPIC_RAW", "raw_news"))
            
            logger.info("Successfully processed RSS feeds")
        except Exception as e:
            logger.error(f"Error processing RSS feeds: {e}")
            raise 