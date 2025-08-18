from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.model_trainer import ModelTrainer
from textSummarizer.logger import logger
from textSummarizer.conponents.model_evaluation import ModelEvaluation

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.evaluate()
        