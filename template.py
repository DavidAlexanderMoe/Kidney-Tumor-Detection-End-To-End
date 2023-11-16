# Template to setup and automate folder structure
import os
from pathlib import Path
import logging

# logging string creation to prevent writing many print() statements
# Same string in every project
# what info do i want and in which format(time + message)
logging.basicConfig(level = logging.INFO, format = '[%(asctime)s]: %(message)s:')

project_name = 'CNN_Classifier'

# list of files and folders
list_of_files = [
    ".github/workflows/.gitkeep",                   # important Path, also for CI/CD pipeline
    # create files inside the folder 'project_name' which is src=source
    f"src/{project_name}/__init__.py",              
    f"src/{project_name}/components/__init__.py",   # components
    f"src/{project_name}/utils/__init__.py",        # utility files, ecc..
    f"src/{project_name}/config/__init__.py",       
    f"src/{project_name}/config/configuration.py",  
    f"src/{project_name}/pipeline/__init__.py",     
    f"src/{project_name}/entity/__init__.py",       
    f"src/{project_name}/constants/__init__.py",    
    "config/config.yaml",                     
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",                        # notebook experiments
    "templates/index.html",                         # for Flask webapp
    # If needed add other files here
    # "test.py"

]


# for loop 
for filepath in list_of_files:
    filepath = Path(filepath)                       # use Path() class to convert from \ to / as readable by python (windows OS problems)
    filedir, filename = os.path.split(filepath)     # separating folders and files in those folders


    if filedir !="":
        # make directories and add logging
        os.makedirs(filedir, exist_ok=True)         
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    # if the file is not present in my directory/ files size is 0, then create the file
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")