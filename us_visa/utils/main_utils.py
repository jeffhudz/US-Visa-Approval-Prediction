import os
import sys

import dill
import yaml
from pandas import DataFrame    
import numpy as np
from us_visa.exception import USVisaException
import us_visa.logger as logging


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML file. 
    Returns:
        dict: The contents of the YAML file as a dictionary.
        """
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise USVisaException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes a dictionary to a YAML file.

    Args:
        file_path (str): The path to the YAML file.
        data (dict): The data to write to the YAML file.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)
    except Exception as e:
        raise USVisaException(e, sys)

def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to a file using dill.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The Python object to save.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise USVisaException(e, sys)
def load_object(file_path: str) -> object:
    """
    Loads a Python object from a file using dill.

    Args:
        file_path (str): The path to the file from which the object will be loaded.

    Returns:
        object: The loaded Python object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise USVisaException(e, sys)
def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Saves a numpy array to a file.

    Args:
        file_path (str): The path to the file where the array will be saved.
        array (np.ndarray): The numpy array to save.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise USVisaException(e, sys)
def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a numpy array from a file.

    Args:
        file_path (str): The path to the file from which the array will be loaded.

    Returns:
        np.ndarray: The loaded numpy array.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise USVisaException(e, sys)
def drop_columns(dataframe: DataFrame, columns: list) -> DataFrame:
    """
    Drops specified columns from a pandas DataFrame.

    Args:
        dataframe (DataFrame): The input pandas DataFrame.
        columns (list): A list of column names to drop from the DataFrame.

    Returns:
        DataFrame: The DataFrame after dropping the specified columns.
    """
    try:
        return dataframe.drop(columns=columns, axis=1)
    except Exception as e:
        raise USVisaException(e, sys)   

        