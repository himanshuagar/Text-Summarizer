from textSummarizer.logger import logger
from textSummarizer.pipeline.stage1_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage2_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipeline.stage3_data_transforamtion import DataTransformationTrainingPipeline
from textSummarizer.pipeline.stage4_model_trainer import ModelTrainerTrainingPipeline
from textSummarizer.pipeline.stage5_model_evaluation import ModelEvaluationTrainingPipeline



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


STAGE_NAME = "Data Transforamtionn"

try:
    logger.info(f"stage {STAGE_NAME} started...")
    data_transforamtion = DataTransformationTrainingPipeline()
    data_transforamtion.main()
    logger.info(f"stage {STAGE_NAME} completed successfully.")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer stage"

try:
    logger.info(f"stage {STAGE_NAME} started...")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logger.info(f"stage {STAGE_NAME} completed successfully.")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Evaluate stage"

try:
    logger.info(f"stage {STAGE_NAME} started...")
    model_evaluate = ModelEvaluationTrainingPipeline()
    model_evaluate.main()
    logger.info(f"stage {STAGE_NAME} completed successfully.")
except Exception as e:
    logger.exception(e)
    raise e