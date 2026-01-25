import sys
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from neuro_mf import ModelFactory

from us_visa.exception import USVisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_numpy_array_data,load_object,read_yaml_file,save_object
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact,ModelTrainerArtifact,ClassificationMetricArtifact
from us_visa.entity.estimator import USvisaModel


class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: reference to the data transformation artifact
        :param model_trainer_config: Configuration for model training
        """

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

        
    def get_model_and_report(self, train :np.array, test: np.array) -> Tuple[object,object]:
        """
        Method Name: get_model_and_report
        Description: uses neuro_mf to get the best model and report
        Output: Returns metric artifact object and best model object
        Onfailure: Write an excpetion log and raise an exception 
        """
        try:

            logging.info("Using neuro_mf to get the best model and report")
            model_factory = ModelFactory(model_config_path = self.model_trainer_config.model_config_file_path)
            x_train,y_train,x_test,y_test = train[:, :-1],train[:, -1],test[:, :-1],test[:, -1]
            
            best_model_detail = model_factory.get_best_model(X= x_train, y= y_train,
                                                             base_accuracy= self.model_trainer_config.expected_score)
            
            model_obj = best_model_detail.best_model

            y_pred = model_obj.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall =recall_score(y_test,y_pred)
            precision = precision_score(y_test, y_pred)

            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision,
                                                           recall_score= recall)
            return best_model_detail,metric_artifact
        except Exception as e:
            raise USVisaException(e,sys) from e
    
    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        """
        Output: Returns ModelTrainerArtifact
        On Failure: Raises exception
        """
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            best_model_detail, metric_Artifact = self.get_model_and_report(train=train_arr,test=test_arr)

            preprocessing_obj = load_object(self.data_transformation_artifact.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_score:
                logging.info("No best model found with score better than base model")
                raise USVisaException("No best model found with score better than base model")

            usvisaModel = USvisaModel(preprocessiong_object=preprocessing_obj,trained_model_object=best_model_detail.best_model)

            logging.info("Created best model with preprocessing object")
            logging.info("Created path of best model")

            save_object(self.model_trainer_config.trained_model_file_path,usvisaModel)

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                        metric_artifact=metric_Artifact)
            logging.info(f"Model trainer Artifact:{model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise USVisaException(e,sys) from e


