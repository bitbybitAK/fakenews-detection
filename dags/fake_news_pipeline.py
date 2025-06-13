"""
Airflow DAG for TruthGuard fake news detection pipeline.
Orchestrates data ingestion, processing, and model training.
"""

from datetime import datetime, timedelta
from typing import List

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.dates import days_ago

from src.data.ingestion import DataIngestion
from src.processing.etl import ETLProcessor
from src.ml.model import FakeNewsClassifier

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'fake_news_pipeline',
    default_args=default_args,
    description='Fake news detection pipeline',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
    tags=['fake_news', 'ml'],
)

# Task 1: Data Ingestion
def ingest_data(**kwargs):
    """Ingest data from Kaggle dataset and RSS feeds."""
    ingestion = DataIngestion()
    
    # Process Kaggle dataset
    kaggle_path = 'data/raw/fake_news_dataset.csv'
    ingestion.process_kaggle_data(kaggle_path)
    
    # Process RSS feeds
    rss_feeds = [
        'https://news.google.com/rss',
        'https://www.reutersagency.com/feed/',
        'https://www.bbc.com/news/10628494'
    ]
    ingestion.process_rss_feeds(rss_feeds)

ingest_task = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data,
    dag=dag,
)

# Task 2: Wait for S3 data
wait_for_data = S3KeySensor(
    task_id='wait_for_data',
    bucket_key='raw-news/*.csv',
    bucket_name='raw-news',
    aws_conn_id='aws_default',
    dag=dag,
)

# Task 3: Data Processing
def process_data(**kwargs):
    """Process and transform the data."""
    etl = ETLProcessor()
    
    # Process data from S3
    input_path = 's3://raw-news/'
    output_path = 's3://processed-news/'
    
    # Read and process data
    df = etl.process_batch(input_path)
    
    # Save processed data
    etl.save_processed_data(df, output_path)

process_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag,
)

# Task 4: Model Training
def train_model(**kwargs):
    """Train the fake news classifier."""
    classifier = FakeNewsClassifier()
    
    # Load processed data
    data_path = 's3://processed-news/'
    df = pd.read_parquet(data_path)
    
    # Prepare data
    X, y = classifier.prepare_data(df)
    
    # Train model
    metrics = classifier.train(X, y)
    
    # Save model
    model_path = 'models/fake_news_model'
    classifier.save_model(model_path)
    
    return metrics

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

# Task 5: Database Update
update_db = PostgresOperator(
    task_id='update_database',
    postgres_conn_id='postgres_default',
    sql="""
    INSERT INTO model_metrics (
        timestamp,
        accuracy,
        precision,
        recall,
        f1_score
    ) VALUES (
        CURRENT_TIMESTAMP,
        {{ ti.xcom_pull(task_ids='train_model')['accuracy'] }},
        {{ ti.xcom_pull(task_ids='train_model')['precision'] }},
        {{ ti.xcom_pull(task_ids='train_model')['recall'] }},
        {{ ti.xcom_pull(task_ids='train_model')['f1'] }}
    );
    """,
    dag=dag,
)

# Task 6: Data Validation
def validate_data(**kwargs):
    """Validate data quality using Great Expectations."""
    import great_expectations as ge
    
    # Create context
    context = ge.get_context()
    
    # Create expectation suite
    suite = context.create_expectation_suite(
        "fake_news_suite",
        overwrite_existing=True
    )
    
    # Load data
    validator = context.get_validator(
        batch_request={
            "datasource_name": "fake_news_data",
            "data_connector_name": "default_inferred_data_connector_name",
            "data_asset_name": "processed_news",
            "limit": 1000,
        },
        expectation_suite=suite,
    )
    
    # Add expectations
    validator.expect_column_values_to_not_be_null("text")
    validator.expect_column_values_to_not_be_null("label")
    validator.expect_column_values_to_be_in_set("label", [0, 1])
    
    # Save results
    results = validator.validate()
    context.save_expectation_suite(suite)
    
    return results

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

# Define task dependencies
ingest_task >> wait_for_data >> process_task >> train_task >> update_db
process_task >> validate_task 