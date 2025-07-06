from zenml import step
from src.utils.data_preparation import split_dataset, final_dataloader, MentalHealthData
from typing import Optional,Tuple, Dict
from typing_extensions import Annotated
import pandas as pd
from zenml.logger import get_logger
from torch.utils.data import DataLoader

# configure logging
logger = get_logger(__name__)

# define function
@step
def data_split(data: pd.DataFrame) -> Tuple[
    Annotated[Optional[pd.DataFrame], "Train DataSet"],
    Annotated[Optional[pd.DataFrame],"Test DataSet"]]:
    """Contains the logic for the data splitting step. Uses the function split_dataset
    under the hood."""
    train_df, test_df = None, None
    try:
        train_df, test_df = split_dataset(data=data)
        logger.info("Completed data splitting successfully")
    except Exception as err:
        logger.error("Encountered error {err} while splitting the dataset.")

    return train_df, test_df


@step
def setup_dataloader(train_data:pd.DataFrame, test_data:pd.DataFrame) ->Tuple[
                         Annotated[Optional[DataLoader], "Train DataLoader"],
                         Annotated[Optional[DataLoader], "Test DataLoader"]]:
    train_loader, test_loader = None, None
    try:
        train_loader, test_loader = final_dataloader(train_data=train_data, test_data=test_data)
        logger.info(msg="DataLoading completed successfully!")
    except Exception as err:
        logger.error(msg=f"Encountered error {err} while setup dataloader")
    
    return train_loader, test_loader

    
if __name__ == "__main__":
    train_df, test_df = split_dataset(data = data)
    train_loader, test_loader = final_dataloader(train_data=train_df,
                                                    test_data=test_df)

