from typing import List, Tuple, Dict, Optional, Union
from sklearn.preprocessing import LabelEncoder
import logging
import pandas as pd
import kagglehub
import os
from src.config import *

# configure logging
logging.basicConfig(level=logging.INFO, format= "%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# download the dataset 
def download_data() -> Optional[pd.DataFrame]:
    """Function for downloading data directly from kaggle. This function
    gets the mental health data and converts it to a dataframe. It does not
    take any parameter.

    Returns:
        pd.DataFrame: The returned mental health dataset.
    """
    data = None
    try:
        # download and prepare the data path
        path = kagglehub.dataset_download(DATASET)
        data_path: List[str] = os.listdir(path)
        data_path: str = os.path.join(path, data_path[0])

        # load data as a pandas dataframe
        data = pd.read_csv(data_path, usecols = ['statement','status'])
        logger.info(msg=f"Dataframe created successfully with shape: {data.shape}")

    except Exception as err:
        logger.error(msg= f"Encountered error {err} while setting up download")

    return data


def data_cleaner(data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[LabelEncoder]]:
    """
    This function performs data cleaning on the dataset. The cleaning process includes
    the following:
    - dropping missing values.
    - fixing inconsistent index.
    - convert the datatype to string across board
    - encode the label column to numeric values using LabelEncoder

    Args:
        data(pd.DataFrame): The dataset to be cleaned.
    
    Returns:
        Tuple: the cleaned dataset and the encoder object.
    """

    encoder = None
    try:
        # drop missing values
        data.dropna(inplace = True)
        data.reset_index(drop=True, inplace = True)

        # convert the columns to the right data types
        data = data.astype({'statement': 'str','status': 'str'})

        # change the column names to text and labels
        data.rename(columns={'statement':'text','status': 'label'}, inplace=True)

        # perform encoding on label columnn.
        encoder = LabelEncoder()
        data['label'] = encoder.fit_transform(data['label'])

        logger.info(msg = f"""Cleaning completed:\n 
        final shape of data: {data.shape}.
        final column names: {data.columns}.
        final data types: {data['text'].dtypes, data['label'].dtypes}
        """)

    except Exception as err:
        logger.error(f"Encountered error {err} while cleaning")
    
    return data, encoder

if __name__ == "__main__":
    data = data_cleaner(data= download_data())
    print(data)