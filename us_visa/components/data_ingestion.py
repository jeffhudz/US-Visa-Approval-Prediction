import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.entity.config_entity import DataIngestionConfig    
from us_visa.exception import USVisaException
from us_visa.logger import logging  
from us_visa.data_access.usvisa_data import USVisaData


class DataIngestion:
    """
    Class Name: DataIngestion
    Description: This class is used to fetch the data from MongoDB database and split it into training and testing sets.
    Output: DataIngestionArtifact
    On Failure: Raise Exception
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig=DataIngestionArtifact()):
        """
        :param data_ingestion_config: Configuration for data ingestion
        """
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise USVisaException(e, sys)
    
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name: export_data_into_feature_store
        Description: This method exports the data from MongoDB database to feature store.
        Output: DataFrame
        On Failure: Raise Exception
        """
        try:
            logging.info("Exporting data from MongoDB to feature store")
            usvisa_data = USVisaData()
            df: DataFrame = usvisa_data.get_collection_as_dataframe(
                collection_name=self.data_ingestion_config.data_ingestion_collection_name
            )
            logging.info(f"Exported data from collection: {self.data_ingestion_config.data_ingestion_collection_name} to feature store")
            logging.info(f"Rows and columns in df: {df.shape}")
            # create feature store directory
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)

            # save data to feature store
            df.to_csv(self.data_ingestion_config.feature_store_file_path, index=False, header=True)
            logging.info(f"Data saved to feature store at: {self.data_ingestion_config.feature_store_file_path}")   

            return df

        except Exception as e:
            raise USVisaException(e, sys)

    def split_data_as_train_test(self, df: DataFrame) -> None:
        """
        Method Name: split_data_as_train_test
        Description: This method splits the data into training and testing sets.
        Output: None
        On Failure: Raise Exception
        """
        try:
            logging.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            # create ingested directory
            ingested_dir = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(ingested_dir, exist_ok=True)

            # save train set
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            logging.info(f"Training data saved at: {self.data_ingestion_config.training_file_path}")

            # save test set
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info(f"Testing data saved at: {self.data_ingestion_config.testing_file_path}")

        except Exception as e:
            raise USVisaException(e, sys)
        
        def initiate_data_ingestion(self) -> DataIngestionArtifact:

            """
            Method Name: initiate_data_ingestion
            Description: This method initiates the data ingestion process.
            Output: DataIngestionArtifact
            On Failure: Raise Exception
            """
            try:
                logging.info("Initiating data ingestion process")
                df = self.export_data_into_feature_store()
                logging.info("Exported data from MongoDBinto feature store")
                self.split_data_as_train_test(df=df)
                logging.info("Split data into train and test sets")

                data_ingestion_artifact = DataIngestionArtifact(
                    training_file_path=self.data_ingestion_config.training_file_path,
                    testing_file_path=self.data_ingestion_config.testing_file_path
                )

                logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
                return data_ingestion_artifact

            except Exception as e:
                raise USVisaException(e, sys)
                