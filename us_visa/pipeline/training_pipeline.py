import sys

from us_visa.exception import USVisaException
from us_visa.logger import logging
from dotenv import load_dotenv
load_dotenv()

from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_validation import DataValidation
from us_visa.components.data_transformation import DataTransformation 
from us_visa.components.model_trainer import ModelTrainer
from us_visa.components.model_evaluation import ModelEvaluation
from us_visa.components.model_pusher import ModelPusher

from us_visa.entity.config_entity import (DataIngestionConfig,
                                          DataValidationConfig,
                                          DataTransformationConfig,
                                          ModelTrainerConfig, 
                                          ModelEvaluationConfig,
                                          ModelPusherConfig)

from us_visa.entity.artifact_entity import (DataIngestionArtifact,
                                            DataValidationArtifact,
                                            DataTransformationArtifact,
                                            ModelTrainerArtifact, 
                                            ModelEvaluationArtifact,
                                            ModelPusherArtifact)





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
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()  
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
        

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
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Method Name: start_data_validation
        Description: This method starts the data validation component.
        Output: DataValidationArtifact
        On Failure: Raise Exception
        """
        try:
            logging.info("Starting data validation component of training pipeline")
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation completed")
            logging.info("Exited data validation component of training pipeline")
            return data_validation_artifact
        except Exception as e:
            raise USVisaException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact,
                                  data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation component of training pipeline")
            data_transformation = DataTransformation(data_transformation_config=self.data_transformation_config,
                                                     data_ingestion_artifact=data_ingestion_artifact,
                                                     data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed")
            logging.info("Exited data transformation component of training pipeline")
            return data_transformation_artifact
        except Exception as e:
            raise USVisaException(e, sys)
    def start_model_trainer(self,data_transformation_artifact:DataTransformation)->ModelTrainerArtifact:
        """
        This method of the training class is responsible for starting the training
        """
        try:
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact
        except Exception as e:
            raise USVisaException(e,sys) from e
    
    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                            model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        This method of TrainPipeline class is responsible for starting modle evaluation
        """
        try:
            model_evaluation = ModelEvaluation(model_eval_config=self.model_evaluation_config,
                                                data_ingestion_artifact=data_ingestion_artifact,
                                                model_trainer_artifact=model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise USVisaException(e, sys)
        

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        Method Name: start_model_pusher
        Description: This method starts the model pusher component.
        Output: ModelPusherArtifact
        On Failure: Raise Exception
        """
        try:
            logging.info("Starting model pusher component of training pipeline")
           
            model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact,
                                       model_pusher_config=self.model_pusher_config)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Model pusher completed")
            logging.info("Exited model pusher component of training pipeline")
            return model_pusher_artifact
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
            
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
      
            data_transformation_artifact = self.start_data_transformation(
                                                            data_validation_artifact=data_validation_artifact,
                                                            data_ingestion_artifact=data_ingestion_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact,
                                                                   data_ingestion_artifact=data_ingestion_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                logging.info("Trained model is not better than the best model present in s3 hence not pushing the model to s3")

                return None
            
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
            logging.info(f"{'='*20} Training Pipeline Completed {'='*20}")
        except Exception as e:
            raise USVisaException(e, sys)