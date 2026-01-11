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