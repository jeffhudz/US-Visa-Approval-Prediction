import sys
from us_visa.exception import USVisaException
from us_visa.logger import logging


import certifi
import os
import pymongo
from us_visa.constants import DATABASE_NAME, MONGO_DB_URL_KEY


ca =certifi.where() 

class MongoDBClient:
    """
    Class Name: MongoDBClient
    Description: This class is used to create a MongoDB client connection.
    Outputs: Connection to MongoDB database.
    On failure: Raise Exception 
    """

    client = None
    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:

                mongo_db_url = os.getenv(MONGO_DB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGO_DB_URL_KEY} is not set.")
                    mongo_db_url = connection_string
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
                self.client = MongoDBClient.client
                self.database= self.client[database_name]
                self.database_name = database_name
                logging.info(f"MongoDB Client connected to database: {database_name} successfully.")


        except Exception as e:
            raise USVisaException(e, sys)
        