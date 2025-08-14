from textSummarizer.logger import logger
from textSummarizer.pipeline.stage1_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage2_data_validation import DataValidationTrainingPipeline

STAGE_NAME = "Data Ingestion"

try:
    logger.info(f"stage {STAGE_NAME} started...")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f"stage {STAGE_NAME} completed successfully.")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation"

try:
    logger.info(f"stage {STAGE_NAME} started...")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f"stage {STAGE_NAME} completed successfully.")

except Exception as e:
    logger.exception(e)
    raise e