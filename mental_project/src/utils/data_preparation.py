from ast import Dict
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
from data_cleaning import data_cleaner, download_data


# configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# define a function that splits the dataset

def split_dataset(data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame],Optional[pd.DataFrame]]:
    train_data, test_data = None, None
    try:
        # split the data 
        train_data, test_data = train_test_split(data, test_size=0.2, random_state = 23,
        stratify=data['label'])

        logger.info(f'dataset splitted successfully! ')
    
    except Exception as err:
        logger.error(f"Encountered error {err} while splitting the dataset")
    
    return train_data, test_data



# define the pytorch dataset class
class MentalHealthData(Dataset):

    def __init__(self, data:pd.DataFrame,tokenizer:AutoTokenizer,max_length: int = 128) -> None:
        self.texts = data['text'].to_list()
        self.labels = data['label'].to_list()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> Dict:
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt',
            add_special_tokens = True,
            max_length = self.max_length
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(data=label, dtype = torch.long)}


# setup dataloader function
def final_dataloader(train_data: pd.DataFrame, test_data, MentalHealthData=MentalHealthData) -> Tuple[Optional[DataLoader],Optional[DataLoader]]:
    """_This function takes the train and test dataframe, converts them to Pytorch's dataset
    and returns train and test dataloader for training and validation._

    Args:
        train_data (pd.DataFrame): _train dataset_
        test_data (_type_): _test dataset_
        MentalHealthData (_type_, optional): _dataset class that defines the blueprint_. Defaults to MentalHealthData.

    Returns:
        Tuple[Optional[DataLoader],Optional[DataLoader]]: _description_
    """
    train_loader, test_loader = None, None
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    try:
        # create pytorch dataset instance for train and test.
        train_dataset = MentalHealthData(train_data, tokenizer=tokenizer)
        test_dataset = MentalHealthData(test_data, tokenizer=tokenizer)

        # create a dataloder object for train and test
        train_loader = DataLoader(dataset= train_dataset, shuffle=True, batch_size=12)
        test_loader = DataLoader(dataset = test_dataset, shuffle=True, batch_size = 12)

        logger.info(msg = 'train and test data loader created successfully.')
    except Exception as err:
        logger.error(msg = f"encountered error {err} while processing data loading.")
    
    return train_loader, test_loader



if __name__ == "__main__":
    data, encoder = data_cleaner(data= download_data())
    train_df, test_df = split_dataset(data = data)
    train_loader, test_loader = final_dataloader(train_data=train_df,
                                                 test_data=test_df)