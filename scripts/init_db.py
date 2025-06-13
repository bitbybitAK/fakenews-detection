"""
Database initialization script for TruthGuard.
Creates necessary tables and indexes.
"""

import logging
import os
from typing import List

import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB', 'airflow'),
    'user': os.getenv('POSTGRES_USER', 'airflow'),
    'password': os.getenv('POSTGRES_PASSWORD', 'airflow'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}

def create_database():
    """Create the database if it doesn't exist."""
    try:
        # Connect to default database
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Create cursor
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", 
                   (DB_CONFIG['dbname'],))
        exists = cur.fetchone()
        
        if not exists:
            # Create database
            cur.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
            logger.info(f"Created database: {DB_CONFIG['dbname']}")
        
        # Close connection
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise

def create_tables():
    """Create necessary tables."""
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Create model_metrics table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            accuracy FLOAT NOT NULL,
            precision FLOAT NOT NULL,
            recall FLOAT NOT NULL,
            f1_score FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create predictions table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            prediction BOOLEAN NOT NULL,
            confidence FLOAT NOT NULL,
            processed_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create data_quality_checks table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS data_quality_checks (
            id SERIAL PRIMARY KEY,
            check_name VARCHAR(255) NOT NULL,
            status BOOLEAN NOT NULL,
            details JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create indexes
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp 
        ON model_metrics(timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_predictions_created_at 
        ON predictions(created_at);
        
        CREATE INDEX IF NOT EXISTS idx_data_quality_checks_created_at 
        ON data_quality_checks(created_at);
        """)
        
        # Commit changes
        conn.commit()
        logger.info("Successfully created tables and indexes")
        
        # Close connection
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise

def create_views():
    """Create useful database views."""
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Create daily metrics view
        cur.execute("""
        CREATE OR REPLACE VIEW daily_model_metrics AS
        SELECT 
            DATE(timestamp) as date,
            AVG(accuracy) as avg_accuracy,
            AVG(precision) as avg_precision,
            AVG(recall) as avg_recall,
            AVG(f1_score) as avg_f1_score
        FROM model_metrics
        GROUP BY DATE(timestamp)
        ORDER BY date DESC;
        """)
        
        # Create prediction statistics view
        cur.execute("""
        CREATE OR REPLACE VIEW prediction_statistics AS
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN prediction = true THEN 1 ELSE 0 END) as fake_count,
            SUM(CASE WHEN prediction = false THEN 1 ELSE 0 END) as real_count,
            AVG(confidence) as avg_confidence
        FROM predictions
        GROUP BY DATE(created_at)
        ORDER BY date DESC;
        """)
        
        # Commit changes
        conn.commit()
        logger.info("Successfully created views")
        
        # Close connection
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating views: {e}")
        raise

def main():
    """Main function to initialize database."""
    try:
        # Create database
        create_database()
        
        # Create tables
        create_tables()
        
        # Create views
        create_views()
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    main() 