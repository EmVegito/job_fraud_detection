import pandas as pd

def load_dataframe_from_public_google_drive_csv(google_drive_share_link):
    """
    Loads a Pandas DataFrame from a publicly shared CSV file on Google Drive.

    Args:
        google_drive_share_link (str): The shareable link from Google Drive       This ID
                                       (e.g., 'https://drive.google.com/file/d/{YOUR_FILE_ID}/view?usp=sharing').

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    try:
        # Extract the file ID from the shareable link
        file_id = google_drive_share_link
        
        # Construct the direct download URL
        download_url = f'https://drive.google.com/uc?id={file_id}'
        
        df = pd.read_csv(download_url)
        print("DataFrame loaded successfully from public Google Drive link.")
        return df
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        print("Please ensure the Google Drive file is publicly accessible ('Anyone with the link' can view).")
        return None