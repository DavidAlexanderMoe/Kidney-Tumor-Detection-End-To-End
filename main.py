# how will the CNN_Classifier/__init__.py work?
# check this code here

# import the logger created here CNN_Classifier/__init__.py
from CNN_Classifier import logger
# from src.CNN_Classifier import logger
# could also omit the src. part since we installed it from the setup.py file (I installed CNN_Classifier as a local package!)

logger.info("Welcome to our custom log.")
# git bash -> python main.py
# [2023-11-16 18:14:55,724: INFO: main: Welcome to our custom log.]