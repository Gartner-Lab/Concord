
import anndata as ad
import pandas as pd
import io

# Save the object to a file
def save_object(obj, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

# Load the object from a file
def load_object(filename):
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)

def sanitize_filename(filename):
    import re
    # Replace invalid characters with an underscore or any other desired character
    sanitized_filename = re.sub(r'[\/:*?"<>|]', '_', filename)
    return sanitized_filename



def is_url(file_path_or_url: str) -> bool:
    """
    Check if the given string is a URL.

    Parameters:
    file_path_or_url (str): The string to check.

    Returns:
    bool: True if the string is a URL, False otherwise.
    """
    return file_path_or_url.startswith(('http://', 'https://'))



def read_csv(file_path_or_url: str) -> pd.DataFrame:
    """
    Read a CSV file from a local path or a URL.

    Parameters:
    file_path_or_url (str): The file path or URL to the CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the CSV data.
    """
    if is_url(file_path_or_url):
        # Download the file
        import requests
        response = requests.get(file_path_or_url)
        response.raise_for_status()  # Ensure the request was successful

        # Decompress the gzip file if necessary
        if file_path_or_url.endswith('.gz'):
            import gzip
            with gzip.open(io.BytesIO(response.content), 'rt') as f:
                df = pd.read_csv(f, index_col=0)
        else:
            df = pd.read_csv(io.StringIO(response.text), index_col=0)
    else:
        # Read from local file system
        if file_path_or_url.endswith('.gz'):
            import gzip
            with gzip.open(file_path_or_url, 'rt') as f:
                df = pd.read_csv(f, index_col=0)
        else:
            df = pd.read_csv(file_path_or_url, index_col=0)

    return df


def csv_to_anndata(file_path_or_url: str) -> ad.AnnData:
    """
    Read a raw count matrix from a CSV file (local or online) and convert it to an AnnData object.

    Parameters:
    file_path_or_url (str): Path or URL to the CSV file containing the raw count matrix.

    Returns:
    ad.AnnData: AnnData object containing the raw count matrix.
    """
    # Read the CSV file into a pandas DataFrame
    df = read_csv(file_path_or_url)

    # Convert the DataFrame to an AnnData object
    adata = ad.AnnData(X=df)

    return adata

def load_json(file_path_or_url: str) -> dict:
    """
    Load a JSON file from a local path or a URL.

    Parameters:
    file_path_or_url (str): The file path or URL to the JSON file.

    Returns:
    dict: Dictionary containing the JSON data.
    """
    import json
    if is_url(file_path_or_url):
        # Download the file
        import requests
        response = requests.get(file_path_or_url)
        response.raise_for_status()  # Ensure the request was successful

        # Load the JSON data
        data = response.json()
    else:
        # Read from local file system
        with open(file_path_or_url, 'r') as f:
            data = json.load(f)

    return data 

