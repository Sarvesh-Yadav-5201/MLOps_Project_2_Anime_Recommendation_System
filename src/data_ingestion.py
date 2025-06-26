import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exceptions import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

# Initialize logger
logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        """
        Initialize the DataIngestion class with configuration details.

        Args:
            config (dict): Configuration dictionary containing data ingestion parameters.
        """
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_names = self.config["bucket_file_names"]

        # Ensure the RAW_DIR exists
        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info("### ................................Data Ingestion Initialized. .......................... ###")

    def download_csv_from_gcp(self):
        """
        Download CSV files from Google Cloud Storage bucket to the local RAW_DIR.
        Handles large files by limiting the number of rows downloaded.
        """
        try:
            # Initialize GCP storage client
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            # Iterate through the list of file names to download
            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)

                # Check if the file already exists locally to avoid redundant downloads
                if os.path.exists(file_path):
                    logger.info(f"File {file_name} already exists locally. Skipping download.")
                    continue

                # Download the file from the GCP bucket
                blob = bucket.blob(file_name)
                blob.download_to_filename(file_path)

                # Handle large files (e.g., animelist.csv) by limiting rows
                if file_name == "animelist.csv":
                    logger.info(f"Large file detected: {file_name}. Limiting to 5M rows.")
                    data = pd.read_csv(file_path, nrows=5000000)
                    data.to_csv(file_path, index=False)
                else:
                    logger.info(f"Downloading smaller file: {file_name}")

                logger.info(f"Successfully downloaded {file_name} to {file_path}")

        except Exception as e:
            logger.error("Error occurred while downloading data from GCP.")
            raise CustomException("Failed to download data", e)

    def validate_config(self):
        """
        Validate the configuration to ensure all required parameters are present.
        """
        try:
            if not self.bucket_name:
                raise ValueError("Bucket name is missing in the configuration.")
            if not self.file_names or not isinstance(self.file_names, list):
                raise ValueError("File names are missing or not in a list format.")
            logger.info("Configuration validation successful.")
        except ValueError as ve:
            logger.error(f"Configuration validation error: {str(ve)}")
            raise CustomException("Invalid configuration", ve)

    def run(self):
        """
        Execute the data ingestion process.
        """
        try:
            logger.info("Starting Data Ingestion Process...")
            self.validate_config()  # Validate configuration before proceeding
            self.download_csv_from_gcp()
            logger.info("Data Ingestion Completed Successfully.")
        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
        finally:
            logger.info("Data Ingestion Process Finished.")

if __name__ == "__main__":
    try:
        # Read configuration from YAML file
        config = read_yaml(CONFIG_PATH)
        data_ingestion = DataIngestion(config)
        data_ingestion.run()
    except Exception as e:
        logger.error(f"Failed to execute Data Ingestion: {str(e)}")