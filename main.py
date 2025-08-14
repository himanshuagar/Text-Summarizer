from textSummarizer.logger import logger
from textSummarizer.pipeline.stage1_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion"

try:
    logger.info(f"stage {STAGE_NAME} started...")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f"stage {STAGE_NAME} completed successfully.")

except Exception as e:
    logger.exception(e)
    raise e