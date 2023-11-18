# main.py will be my endpoint, always

# how will the CNN_Classifier/__init__.py work?
# check this code here

# import the logger created here CNN_Classifier/__init__.py
from CNN_Classifier import logger
# from src.CNN_Classifier import logger
# could also omit the src. part since we installed it from the setup.py file (I installed CNN_Classifier as a local package!)

# logger.info("Welcome to our custom log.")
# git bash -> python main.py
# [2023-11-16 18:14:55,724: INFO: main: Welcome to our custom log.]


#############################################################################################################################
#############################################################################################################################
#############################################################################################################################


# Data Ingestion
from CNN_Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
 

STAGE_NAME = "Data Ingestion stage" 
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=========x")
except Exception as e:
    logger.exception(e)
    raise e



# Prepare Base Model
from CNN_Classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline


STAGE_NAME = "Prepare Base Model"
if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=========x")
    except Exception as e:
        logger.exception(e)
        raise e



# Model training
from CNN_Classifier.pipeline.stage_03_model_training import ModelTrainingPipeline
STAGE_NAME = "Model Training"
if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=========x")
    except Exception as e:
        logger.exception(e)
        raise e
    


# Model evaluation
from CNN_Classifier.pipeline.stage_04_model_evaluation import EvaluationPipeline
STAGE_NAME = "Model Evaluation"
if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=========x")
    except Exception as e:
        logger.exception(e)
        raise e