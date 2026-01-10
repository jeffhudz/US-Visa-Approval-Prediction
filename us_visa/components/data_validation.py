import json
import sys

import pandas as pd
from evidently.report import Report 
from evidently.model_profile.sections import DataDriftProfileSection

from pandas import DataFrame

from us_visa.constants import DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
from us_visa.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.exception import USVisaException   
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa.constants import sch
