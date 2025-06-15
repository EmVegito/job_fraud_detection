import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)



list_of_files=[
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/data_loader.py",
    "src/data/preprocessor.py",
    "src/models/__init__.py",
    "src/models/model_trainer.py",
    "src/logger.py",
    "src/exception.py",
    "src/utils.py",
    "main.py",
    "app.py",
    ".gitignore",
    "requirements.txt",
    "setup.py",
    "README.md"
    "LICENSE"
]

for filepath in list_of_files:
    filepath= Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exist.")