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
    "src/models/fraud_detector.py",
    "src/models/model_trainer.py",
    "src/visualization/__init__.py",
    "src/visualization/dashboard_components.py",
    "src/api/__init__.py",
    "src/api/main.py",
    "src/api/endpoints.py",
    "src/utils/__init__.py",
    "src/utils/text_processing.py",
    "src/utils/model_utils.py",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    "README.md"
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