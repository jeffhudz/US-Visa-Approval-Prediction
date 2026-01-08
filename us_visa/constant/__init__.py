import os
from datetime import date


DATABASE_NAME = "US_Visa"

COLLECTION_NAME = "visa_data"   

MONGO_DB_URL = "mongodb+srv://usvisa_db_user:manchester1@cluster0.i9bofez.mongodb.net/?appName=Cluster0"

PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME: str = "model.pkl"

TARGET_COLUMN: str = "case_status"
CURRENT_YEAR = date.today().year
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILENAME: str = "us_visa.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"    

"""Data Ingestion related constant start with DATA_INGESTION_VAR_NAME"""
DATA_INGESTION_COLLECTION_NAME: str = "visa_data"
DATA_INGESTION_COLLECTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"     
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2





