import gdown
import os
import sys
from src.logger import logging
from src.exception import CustomException

def download_google_drive_csv(file_id, output_path):
    """
    Downloads a CSV file from Google Drive using its file ID.

    Args:
        file_id (str): The Google Drive file ID of the CSV.
        output_path (str): The local path where the CSV file will be saved.
    """
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
        print(f"CSV file downloaded successfully to: {output_path}")
    except Exception as e:
        print(f"Error downloading the file: {e}")