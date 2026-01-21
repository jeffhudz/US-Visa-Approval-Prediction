# US Visa Approval Prediction - AI Agent Instructions

## Architecture Overview
This is a machine learning pipeline for predicting US visa approvals using a modular component-based architecture. Data flows from MongoDB through ingestion, validation, transformation, and model training components orchestrated by `TrainingPipeline` in `us_visa/pipeline/training_pipeline.py`.

Key components:
- **DataIngestion**: Fetches from MongoDB collection "visa_data" and splits train/test (80/20)
- **DataValidation**: Uses Evidently for data drift detection against schema in `config/schema.yaml`
- **DataTransformation**: Applies preprocessing (ordinal/one-hot encoding) based on schema columns
- **ModelTrainer**: Uses neuro_mf library for automated model selection via grid search on configured models

## Critical Workflows
- **Training**: Run `python demo.py` to execute the pipeline; artifacts saved to timestamped `artifact/` directories
- **Environment**: Create conda env "visa" with Python 3.8, install from `requirements.txt`
- **Data Access**: MongoDB connection via `us_visa/configuration/mongo_db_connection.py` using env var `MONGO_DB_URL`
- **Logging**: All components log to timestamped files in `logs/` directory with structured format

## Project Conventions
- **Configuration**: Use dataclasses in `us_visa/entity/config_entity.py` for all configs; paths auto-generated with timestamps
- **Serialization**: Save/load models/preprocessors with `dill` via `us_visa/utils/main_utils.py` functions
- **Error Handling**: Raise `USVisaException` from `us_visa/exception/` with error details
- **Schema Management**: Define column types, transformations in `config/schema.yaml`; drop `case_id` and `yr_of_estab`
- **Model Selection**: Configure models in `config/model.yaml` for neuro_mf; expects accuracy >0.6

## Integration Points
- **Database**: MongoDB with collection "visa_data"; remove `_id` field, replace "na" with NaN
- **External Libs**: neuro_mf for model factory, Evidently for drift reports, CatBoost/XGBoost for models
- **Serving**: FastAPI setup in `app.py` (currently empty) for prediction endpoints

## Key Files
- `us_visa/pipeline/training_pipeline.py`: Main orchestration
- `config/schema.yaml`: Data schema and transformation rules
- `config/model.yaml`: Model configurations for selection
- `us_visa/data_access/usvisa_data.py`: MongoDB data fetching
- `us_visa/components/model_trainer.py`: Model training with neuro_mf</content>
<parameter name="filePath">d:\DataScience Projects\Visa_App\US-Visa-Approval-Prediction\.github\copilot-instructions.md