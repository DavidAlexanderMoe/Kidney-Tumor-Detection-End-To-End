import os
import sys
import logging      # to create custom logging

# logging string
# ASCII time - log label name (kind of log) - module name in which i'm running the file on - message
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

#create log folder
log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),          # get the logged file from the it's path
        logging.StreamHandler(sys.stdout)           # to print ou the logging
    ]
)

logger = logging.getLogger("cnnClassifierLogger")