"""
Streamlit dashboard for TruthGuard.
Provides a user interface for fake news detection and visualization.
"""

import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from wordcloud import WordCloud

from src.ml.model import FakeNewsClassifier
from src.processing.etl import ETLProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Dashboard:
    """Streamlit dashboard for TruthGuard."""

    def __init__(self):
        """Initialize dashboard components."""
        self.etl_processor = ETLProcessor()
        self.classifier = FakeNewsClassifier()
        self._load_model()

    def _load_model(self):
        """Load the trained model."""
        try:
            model_path = os.getenv("MODEL_PATH", "models/fake_news_model")
            self.classifier.load_model(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error("Error loading model. Please check the model path.")

    def _create_wordcloud(self, text: str) -> plt.Figure:
        """
        Create a word cloud from text.
        
        Args:
            text: Input text
            
        Returns:
            Matplotlib figure with word cloud
        """
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig

    def _plot_confusion_matrix(self, cm: List[List[int]]) -> go.Figure:
        """
        Create a confusion matrix plot.
        
        Args:
            cm: Confusion matrix
            
        Returns:
            Plotly figure with confusion matrix
        """
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Fake', 'Predicted Real'],
            y=['Actual Fake', 'Actual Real'],
            colorscale='RdBu'
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label'
        )
        
        return fig

    def _plot_metrics(self, metrics: Dict[str, float]) -> go.Figure:
        """
        Create a metrics bar plot.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Plotly figure with metrics
        """
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                text=[f'{v:.2f}' for v in metrics.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Model Performance Metrics',
            yaxis_title='Score',
            yaxis_range=[0, 1]
        )
        
        return fig

    def run(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="TruthGuard - Fake News Detection",
            page_icon="🛡️",
            layout="wide"
        )
        
        st.title("🛡️ TruthGuard - Fake News Detection")
        st.markdown("""
        Welcome to TruthGuard! This tool helps you detect fake news articles using
        machine learning. Enter or upload a news article to get started.
        """)
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Go to",
            ["Article Analysis", "Model Performance", "About"]
        )
        
        if page == "Article Analysis":
            self._show_article_analysis()
        elif page == "Model Performance":
            self._show_model_performance()
        else:
            self._show_about()
    
    def _show_article_analysis(self):
        """Show the article analysis page."""
        st.header("Article Analysis")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method",
            ["Enter text", "Upload file"]
        )
        
        if input_method == "Enter text":
            text = st.text_area(
                "Enter news article text",
                height=200
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file",
                type=['txt']
            )
            if uploaded_file is not None:
                text = uploaded_file.getvalue().decode()
            else:
                text = ""
        
        if text:
            # Process text
            processed_text = self.etl_processor.clean_text(text)
            features = self.etl_processor.process_stream(text)
            
            # Make prediction
            prediction = self.classifier.predict(features)[0]
            probability = self.classifier.predict_proba(features)[0]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction")
                if prediction == 1:
                    st.error("⚠️ This article is likely FAKE")
                else:
                    st.success("✅ This article is likely REAL")
                
                st.metric(
                    "Confidence",
                    f"{probability[prediction]*100:.2f}%"
                )
            
            with col2:
                st.subheader("Word Cloud")
                fig = self._create_wordcloud(processed_text)
                st.pyplot(fig)
            
            # Show detailed analysis
            st.subheader("Detailed Analysis")
            st.write("Cleaned text:", processed_text)
            
            # Show feature importance
            if hasattr(self.classifier.model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'Feature': self.classifier.feature_columns,
                    'Importance': self.classifier.model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.subheader("Top Important Features")
                st.dataframe(importance.head(10))
    
    def _show_model_performance(self):
        """Show the model performance page."""
        st.header("Model Performance")
        
        # Load evaluation metrics
        try:
            metrics = {
                'Accuracy': 0.85,  # Replace with actual metrics
                'Precision': 0.83,
                'Recall': 0.86,
                'F1 Score': 0.84
            }
            
            # Plot metrics
            st.plotly_chart(self._plot_metrics(metrics))
            
            # Show confusion matrix
            cm = [[100, 20], [15, 95]]  # Replace with actual confusion matrix
            st.plotly_chart(self._plot_confusion_matrix(cm))
            
            # Show detailed metrics
            st.subheader("Detailed Metrics")
            st.dataframe(pd.DataFrame(metrics.items(), columns=['Metric', 'Value']))
            
        except Exception as e:
            logger.error(f"Error showing model performance: {e}")
            st.error("Error loading model performance metrics.")
    
    def _show_about(self):
        """Show the about page."""
        st.header("About TruthGuard")
        
        st.markdown("""
        TruthGuard is an advanced fake news detection system that uses machine learning
        to analyze news articles and determine their authenticity.
        
        ### Features
        - Real-time article analysis
        - Machine learning-based classification
        - Detailed performance metrics
        - Word cloud visualization
        - Feature importance analysis
        
        ### How it works
        1. The system processes the input text using NLP techniques
        2. Features are extracted using TF-IDF vectorization
        3. A trained XGBoost model makes the prediction
        4. Results are displayed with confidence scores
        
        ### Technology Stack
        - Python
        - Streamlit
        - XGBoost
        - PySpark
        - MLflow
        - Great Expectations
        
        ### Contact
        For more information, please visit our [GitHub repository](https://github.com/yourusername/truthguard).
        """)

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run() 