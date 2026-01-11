import numpy as np
import pandas as pd 
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import sys
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.exception import USVisaException
from us_visa.logger import logging      
from us_visa.utils.main_utils import save_object,save_numpy_array_data,read_yaml_file,drop_columns
from us_visa.constants import SCHEMA_FILE_PATH,TARGET_COLUMN,CURRENT_YEAR
from us_visa.entity.artifact_entity import DataIngestionArtifact,DataTransformationArtifact,DataValidationArtifact
from us_visa.entity.estimator import TargetValueMapping


class DataTransformation:
    def __init__(self,data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact  
            self.data_validation_artifact = data_validation_artifact
            self.schema_info = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise USVisaException(e, sys)
        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USVisaException(e, sys)
    
    def get_data_transformer_object(self) -> Pipeline:
        try:
            schema_info = self.schema_info
            
            numeric_transformer = StandardScaler()
            onehot_transformer = OneHotEncoder()
            ordinal_transformer = OrdinalEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder and OrdinalEncoder")

            oh_columns = schema_info['oh_columns']
            or_columns = schema_info['or_columns'] 
            num_columns = schema_info['num_columns']
            transform_columns = schema_info['transform_columns']

            logging.info("Initialized PowerTransformer for columns: {transform_columns}")

            transform_pipe = Pipeline(steps=[
                ('power_transformer',PowerTransformer(method ='yeo-johnson'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', numeric_transformer, num_columns),
                    ('oh_pipeline', onehot_transformer, oh_columns),
                    ('or_pipeline', ordinal_transformer, or_columns),
                    ('transform_pipeline', transform_pipe, transform_columns)
                ]  )
            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of Data_Transformation class")
            return preprocessor
        except Exception as e:
            raise USVisaException(e, sys)   
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:   
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor_obj = self.get_data_transformer_object()
                logging.info("Got the data transformer object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.training_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.testing_file_path)

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Got the input and target features from training data")
                input_feature_train_df['company_age'] = CURRENT_YEAR - input_feature_train_df['yr_of_estab']
                logging.info("Added company_age feature to training data")
                drop_cols = self.schema_info['drop_columns']
                logging.info(f"Dropping columns: {drop_cols} from training data")
                input_feature_train_df = drop_columns(dataframe=input_feature_train_df,columns=drop_cols )
                target_feature_train_df = target_feature_train_df.replace(TargetValueMapping().to_dict())
                logging.info("Replaced target feature values with numerical mapping for training data")

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]     

                logging.info("Got the input and target features from testing data")
                input_feature_test_df['company_age'] = CURRENT_YEAR - input_feature_test_df['yr_of_estab']
                logging.info("Added company_age feature to testing data")                   
                logging.info(f"Dropping columns: {drop_cols} from testing data")
                input_feature_test_df = drop_columns(dataframe=input_feature_test_df,columns=drop_cols )
                target_feature_test_df = target_feature_test_df.replace(TargetValueMapping().to_dict()) 
                logging.info("Replaced target feature values with numerical mapping for testing data")  

                logging.info("Applying preprocessing object on training and testing dataframe")
                input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
                logging.info("Applied preprocessing object on training dataframe")
                input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)  
                logging.info("Applied preprocessing object on testing dataframe")

                logging.info("Applying SMOTETomek on training and testing data")
                smt = SMOTETomek(sampling_strategy='minority')


                logging.info("Applying SMOTETomek on training")
               
                input_feature_train_res, target_feature_train_res = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )
                logging.info("Applied SMOTETomek on training data")

                logging.info("Applying SMOTETomek on testing")
                input_feature_test_res, target_feature_test_res = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )
                logging.info("Applied SMOTETomek on testing data")
                logging.info("Obtained resampled input and target features for training and testing data")  

                logging.info("Created training and testing arrays by concatenating input and target features")
                train_arr = np.c_[input_feature_train_res, np.array(target_feature_train_res)]
                test_arr = np.c_[input_feature_test_res, np.array(target_feature_test_res)]


                logging.info("Saving transformed training and testing arrays and preprocessing object")

                save_numpy_array_data(
                    file_path=self.data_transformation_config.transformed_train_file_path,
                    array=train_arr
                )

                save_numpy_array_data(
                    file_path=self.data_transformation_config.transformed_test_file_path,
                    array=test_arr
                )

                save_object(
                    file_path=self.data_transformation_config.transformed_object_file_path,
                    obj=preprocessor_obj
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )

                logging.info("Data transformation completed successfully")
                return data_transformation_artifact
        except Exception as e:
            raise USVisaException(e, sys)
    