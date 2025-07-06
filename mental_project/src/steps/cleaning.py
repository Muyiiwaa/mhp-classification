from zenml import step
from src.utils.data_cleaning import data_cleaner, download_data
from typing import Optional,Tuple
from typing_extensions import Annotated
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from zenml.logger import get_logger


# configure logging
logger = get_logger(__name__)



@step
def data_download() -> Annotated[Optional[pd.DataFrame], "The Full dataset"]:
    data = None
    try:
        data= download_data()
        logger.info('downloaded data successfully.')
    except Exception as err:
        logger.error(f"Encountered error {err} while downloading the data")
    
    return data


@step
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "Cleaned Data"], 
    Annotated[Optional[LabelEncoder], "Encoder Object"]]:
    encoder = None
    try:
        final_data, encoder = data_cleaner(data)
        logger.info("Data Cleaning completed successfully!")
    except Exception as err:
        logger.error(f"Encountered error {err} while cleaning")
    
    return final_data, encoder



if __name__ == "__main__":
    data_cleaner(data_download())