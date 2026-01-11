import json
import sys

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from pandas import DataFrame

from us_visa.constants import DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
from us_visa.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.exception import USVisaException   
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa.constants import SCHEMA_FILE_PATH
from us_visa.logger import logging

class DataValidation:
    def __init__(self,data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_info = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise USVisaException(e, sys)
        
    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self.schema_info['columns'])
            logging.info(f"Number of columns validation status: {status}")
            
            return status   
        except Exception as e:
            raise USVisaException(e, sys)


    def is_column_exist(self, dataframe: DataFrame) -> bool:
        """
        Method Name: is_column_exist
        Description: This method checks whether all the columns mentioned in the schema file are present in the dataframe.
        Output: True if all columns are present, False otherwise"""
        try:
            schema_columns = self.schema_info['columns']
            dataframe_columns = dataframe.columns
            missing_columns = []
            for column in schema_columns:
                if column not in dataframe_columns:
                    missing_columns.append(column)
            if len(missing_columns) > 0:
                logging.info(f"Missing columns: {missing_columns}")
                return False
            logging.info("All columns are present")
            return True
        except Exception as e:
            raise USVisaException(e, sys)
    
    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USVisaException(e, sys)
    
    def detect_data_drift(self, base_df: DataFrame, current_df: DataFrame) -> bool:
        try:
            data_drift_report = Report(metrics=[DataDriftPreset()])

            data_drift_report.run(reference_data=base_df, current_data=current_df)
            report_dict = data_drift_report.as_dict()

            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path,
            content=report_dict,replace=True)

            drift_metrics = report_dict["metrics"][0]["result"]

            n_features = drift_metrics["number_of_columns"]
            n_drifted_features = drift_metrics["number_of_drifted_columns"]
            drift_status = drift_metrics["dataset_drift"]
            logging.info(f"Number of drifted features: {n_drifted_features} out of {n_features}")
            return drift_status
        except Exception as e:
            raise USVisaException(e, sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_msg = ""
            logging.info("Reading training and testing data for data validation")
            train_df = self.read_data(self.data_ingestion_artifact.training_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.testing_file_path)
            
            logging.info("Validating number of columns in training data")
            train_column_status = self.validate_number_of_columns(train_df)
            logging.info(f"All columns are present in training data:{train_column_status}")
            
            
            if not train_column_status:
                validation_error_msg += f"Training data does not have the required number of columns"
            

            logging.info("Validating number of columns in testing data")
            test_column_status = self.validate_number_of_columns(test_df)
            logging.info(f"All columns are present in testing data:{test_column_status}")

            if not test_column_status:
                validation_error_msg += f"Testing data does not have the required number of columns"
            
            logging.info("Validating column existence in training data")
            train_columns_exist = self.is_column_exist(train_df)

            logging.info("Validating column existence in testing data")
            test_columns_exist = self.is_column_exist(test_df)
            
            if not train_columns_exist:
                validation_error_msg += f"Training data is missing some columns"
            
            if not test_columns_exist:
                validation_error_msg += f"Testing data is missing some columns"
            
            validation_status = len(validation_error_msg)== 0

            if validation_status:
                 drift_status = self.detect_data_drift(base_df=train_df, current_df=test_df)
                 if drift_status:
                     logging.info("Data drift detected between training and testing data")
                     validation_error_msg = "Data drift detected"
                 else:
                     validation_error_msg = "No data drift detected"
            else:
                logging.info(f"Validation errors found: {validation_error_msg}")
                     

            logging.info("Detecting data drift between training and testing data")
           
            
            validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            logging.info(f"Data Validation Artifact: {validation_artifact}")
            return validation_artifact
        except Exception as e:
            raise USVisaException(e, sys)