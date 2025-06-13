"""
Script to run the entire TruthGuard pipeline.
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Handles running the entire TruthGuard pipeline."""

    def __init__(self):
        """Initialize pipeline runner."""
        self.project_root = Path(__file__).parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.dags_dir = self.project_root / "dags"

    def run_command(self, command: List[str], cwd: Optional[Path] = None) -> None:
        """
        Run a shell command.
        
        Args:
            command: List of command arguments
            cwd: Working directory
        """
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd or self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Stream output
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(output.strip())
            
            # Check for errors
            if process.returncode != 0:
                error = process.stderr.read()
                raise subprocess.CalledProcessError(process.returncode, command, error)
            
        except Exception as e:
            logger.error(f"Error running command {' '.join(command)}: {e}")
            raise

    def start_services(self):
        """Start required services using Docker Compose."""
        try:
            logger.info("Starting services...")
            self.run_command(["docker-compose", "up", "-d"])
            time.sleep(30)  # Wait for services to start
            logger.info("Services started successfully")
        except Exception as e:
            logger.error(f"Error starting services: {e}")
            raise

    def initialize_database(self):
        """Initialize the database schema."""
        try:
            logger.info("Initializing database...")
            self.run_command(["python", str(self.scripts_dir / "init_db.py")])
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def download_dataset(self):
        """Download and prepare the dataset."""
        try:
            logger.info("Downloading dataset...")
            self.run_command(["python", str(self.scripts_dir / "download_dataset.py")])
            logger.info("Dataset downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise

    def start_airflow(self):
        """Start Airflow services."""
        try:
            logger.info("Starting Airflow...")
            
            # Initialize Airflow
            self.run_command(["airflow", "db", "init"])
            
            # Create admin user
            self.run_command([
                "airflow", "users", "create",
                "--username", "admin",
                "--firstname", "Admin",
                "--lastname", "User",
                "--role", "Admin",
                "--email", "admin@example.com",
                "--password", "admin"
            ])
            
            # Start webserver
            self.run_command(["airflow", "webserver", "--port", "8080", "-D"])
            
            # Start scheduler
            self.run_command(["airflow", "scheduler", "-D"])
            
            logger.info("Airflow started successfully")
        except Exception as e:
            logger.error(f"Error starting Airflow: {e}")
            raise

    def start_dashboard(self):
        """Start the Streamlit dashboard."""
        try:
            logger.info("Starting dashboard...")
            self.run_command(["streamlit", "run", str(self.project_root / "src/api/dashboard.py")])
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            raise

    def run_pipeline(self):
        """Run the entire pipeline."""
        try:
            # Start services
            self.start_services()
            
            # Initialize database
            self.initialize_database()
            
            # Download dataset
            self.download_dataset()
            
            # Start Airflow
            self.start_airflow()
            
            # Start dashboard
            self.start_dashboard()
            
            logger.info("Pipeline started successfully")
            
        except Exception as e:
            logger.error(f"Error running pipeline: {e}")
            raise

def main():
    """Main function to run the pipeline."""
    try:
        runner = PipelineRunner()
        runner.run_pipeline()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 