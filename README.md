# TruthGuard: Scalable Fake News Detection System

TruthGuard is an end-to-end data engineering project that implements a production-ready system for detecting fake news using machine learning and modern data engineering practices.

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Sources   │────▶│  Data Pipeline  │────▶│  ML Pipeline    │
│  - Kaggle       │     │  - Airflow      │     │  - XGBoost      │
│  - RSS Feeds    │     │  - PySpark      │     │  - MLflow       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Lake      │     │  Data Quality   │     │  Inference      │
│  - S3/MinIO     │     │  - Great        │     │  - Kafka        │
└─────────────────┘     │  Expectations   │     └─────────────────┘
        │               └─────────────────┘             │
        │                                               │
        ▼                                               ▼
┌─────────────────┐                           ┌─────────────────┐
│  Data Warehouse │                           │  Dashboard      │
│  - PostgreSQL   │                           │  - Streamlit    │
└─────────────────┘                           └─────────────────┘
```

## 🚀 Features

- **Data Ingestion**: Automated collection of news articles from multiple sources
- **Data Processing**: Scalable ETL pipeline using PySpark
- **ML Pipeline**: 
  - NLP preprocessing (stopwords, lemmatization, TF-IDF)
  - XGBoost classifier for fake news detection
  - Model versioning with MLflow
- **Data Quality**: Automated validation using Great Expectations
- **Real-time Inference**: Kafka-based prediction service
- **Monitoring**: Interactive Streamlit dashboard with:
  - Real-time predictions
  - Model performance metrics
  - Data quality reports
  - Word cloud visualizations

## 📁 Project Structure

```
truthguard/
├── dags/               # Airflow DAGs
├── scripts/            # Utility scripts
├── models/            # Trained models
├── data/              # Data storage
├── notebooks/         # Jupyter notebooks
├── tests/             # Unit tests
├── config/            # Configuration files
└── src/               # Source code
    ├── data/          # Data ingestion
    ├── processing/    # ETL pipelines
    ├── ml/            # ML models
    ├── api/           # API endpoints
    └── utils/         # Utility functions
```

## 🛠️ Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/truthguard.git
cd truthguard
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Start the services:
```bash
docker-compose up -d
```

## 🚀 Usage

1. Start the Airflow webserver:
```bash
airflow webserver -p 8080
```

2. Start the Airflow scheduler:
```bash
airflow scheduler
```

3. Launch the Streamlit dashboard:
```bash
streamlit run src/api/dashboard.py
```

## 📊 Dashboard

The Streamlit dashboard provides:
- Real-time fake news detection
- Model performance metrics
- Data quality reports
- Interactive visualizations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Kaggle for the fake news dataset
- Apache Airflow, PySpark, and Streamlit communities 