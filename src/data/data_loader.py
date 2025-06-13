import gdown
import os

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

if __name__ == "__main__":
    # --- IMPORTANT: Replace with your actual File ID and desired output path ---
    google_drive_file_id = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
    output_filename = "downloaded_data.csv"
    output_directory = "./data"  # You can change this to your desired directory

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    full_output_path = os.path.join(output_directory, output_filename)

    if google_drive_file_id == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
        print("Please replace 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE' with your actual Google Drive file ID.")
    else:
        download_google_drive_csv(google_drive_file_id, full_output_path)

    # You can now read the CSV file using pandas, for example:
    try:
        import pandas as pd
        df = pd.read_csv(full_output_path)
        print("\nFirst 5 rows of the downloaded CSV:")
        print(df.head())
    except ImportError:
        print("\nInstall pandas (pip install pandas) to easily read the CSV file.")
    except FileNotFoundError:
        print(f"\nCould not find the downloaded file at {full_output_path}. Check for download errors.")