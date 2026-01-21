import sys
from us_visa.exception import USVisaException
from us_visa.logger import logging

from pandas import DataFrame
from sklearn.pipeline import Pipeline

class TargetValueMapping:
    def __init__(self):
        self.Certified: int = 0
        self.Denied: int = 1
    def to_dict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))
    
class USvisaModel:
    def __init__(self,preprocessiong_object: Pipeline,trained_model_object: object):
        """

        :param preprocessiong_object: Input object of processor
        :param trained_model_object:  Output object of processor
    
        """
        self.preprocessing_object = preprocessiong_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe :DataFrame) -> DataFrame:
        """
        Functions accepts raw data and transforms uisng the preprocessiong object ensuring its the format
        as training data.
        Performs predictions on transformed fetures
        """
        logging.info ("Entered Predict Method of USvisaModel class:")
        try:
            logging.info("using trained model to get predictions")
            transformed_features = self.preprocessing_object.transform(dataframe)
            logging.info("Used the rained model to get predictions")
            return self.trained_model_object.predict(transformed_features)
        
        except Exception as e:
            raise USVisaException(e,sys) from e
    

    def __repr__(self):
        return f"{self.trained_model_object.__name__}()"
    
    def __str__(self):
        return f"{self.trained_model_object.__name__}()"


        
        


    
    