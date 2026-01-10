import sys
from us_visa.exception import USVisaException
from us_visa.logger import logging

from us_visa.components.data_ingestion import DataIngestion

from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact



class TrainingPipeline:
    """
    Class Name: TrainingPipeline
    Description: This class is used to run the training pipeline.
    Output: None
    On Failure: Raise Exception
    """

    def __init__(self):
        """
        Method Name: __init__
        Description: This method initializes the TrainingPipeline object.
        Output: None
        On Failure: Raise Exception
        """
        try:
            self.data_ingestion_config = DataIngestionConfig()
        except Exception as e:
            raise USVisaException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name: start_data_ingestion
        Description: This method starts the data ingestion component.
        Output: DataIngestionArtifact
        On Failure: Raise Exception
        """
        try:
            logging.info("Starting data ingestion component of training pipeline")
            logging.info("Getting the data from MongoDB")
            data_ingestion = DataIngestion(data_ingestion_config =self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from MongoDB data")
            logging.info("Exited  data ingestion component of training pipeline")

            return data_ingestion_artifact
        except Exception as e:
            raise USVisaException(e, sys)
        
    def run_pipeline(self) -> None:
        """
        Method Name: run_pipeline
        Description: This method runs the training pipeline.
        Output: None
        On Failure: Raise Exception
        """
        try:
            logging.info(f"{'='*20} Training Pipeline Started {'='*20}")
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info(f"{'='*20} Training Pipeline Completed {'='*20}")
        except Exception as e:
            raise USVisaException(e, sys)