# End to End text summarizer 

## Overview  

This project is an **end-to-end NLP pipeline** for **text summarization**, built using **Hugging Face Transformers, PyTorch, and FastAPI**.  
It demonstrates how to implement **MLOps concepts** such as modular pipeline design, model training, evaluation, and deployment.  

The system allows users to input raw text and get a concise, meaningful summary.  

---

## Workflows 

1. Update config.yaml
2. Update params.yaml
3. Update entity
4. Update the configuration manager in src config
5. update the conponents
6. update the pipeline
7. update the main.py
8. update the app.py

## Project Structure

Text-Summarizer/
│── artifacts/ # Stores intermediate data & models
│── config/ # Config files (YAML)
│── src/textSummarizer/ # Core package
│ ├── components/ # Data ingestion, validation, transformation, training, evaluation
│ ├── pipeline/ # Training pipelines
│ ├── utils/ # Helper functions
│ ├── config/ # Configuration management
│ ├── logger.py # Logging
│ ├── exception.py # Custom exception handling
│── app.py # FastAPI app for deployment
│── main.py # Orchestrates pipeline execution
│── requirements.txt # Python dependencies
│── README.md # Project documentation





