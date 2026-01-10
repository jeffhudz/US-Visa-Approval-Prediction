from us_visa.configuration.mongo_db_connection import MongoDBClient     
from us_visa.constants import DATABASE_NAME
from us_visa.exception import USVisaException
import sys
import pandas as pd 
from typing import Optional
import numpy as np  



class USVisaData:
    """
    Class Name: USVisaData
    Description: This class is used to fetch the data from MongoDB database and convert it into a pandas DataFrame.
    Output: DataFrame
    On Failure: Raise Exception
    """

    def __init__(self):
        """
        Method Name: __init__
        Description: This method initializes the MongoDBClient object.
        Output: None
        On Failure: Raise Exception
        """
        try:
           
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise USVisaException(e, sys)

    def get_collection_as_dataframe(self, collection_name: str,database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Method Name: get_collection_as_dataframe
        Description: This method fetches the data from MongoDB collection and converts it into a pandas DataFrame.
        Output: DataFrame
        On Failure: Raise Exception
        """
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.client[database_name][collection_name]
            
            df = pd.DataFrame(list(collection.find()))
            
            if '_id' in df.columns:
                df.drop(columns=['_id'], inplace=True)
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise USVisaException(e, sys)