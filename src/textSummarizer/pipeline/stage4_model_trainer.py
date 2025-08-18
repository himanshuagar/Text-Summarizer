from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.model_trainer import ModelTrainer
from textSummarizer.logger import logger

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        print("🔍 Initializing configuration...")
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()
            
        print("🧠 Initializing model trainer...")
        model_trainer = ModelTrainer(config=model_trainer_config)
            
        print("🚀 Starting training process...")
        model_trainer.train()
        