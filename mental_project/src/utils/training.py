import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from torch import optim
import torch
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from typing import Optional, Tuple, Union, Dict
import logging
from torch.utils.data import DataLoader

# configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# define model hyperparameters
def get_params(model: PreTrainedModel) -> Tuple:

    learning_rate = 2e-5
    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate,weight_decay=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, step_size=2, gamma=0.1)

    return learning_rate, optimizer, scheduler

def load_model(model_uri: str) -> Tuple[Optional[PreTrainedModel], Optional[str]]:
    """_This function loads the pretrained model from huggingface, moves the model
    to available device and returns both the device and model object._

    Args:
        model_uri (str): _the model id on huggingface._

    Returns:
        Tuple[Optional[PreTrainedModel], Optional[str]]: _MODEL, DEVICE_
    """
    model = None
    device = None
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoModelForSequenceClassification.from_pretrained(model_uri, num_labels = 7)
        model = model.to(device)
        logger.info(f"loaded model {model_uri} on {device} successfully")
    
    except Exception as err:
        logger.error(f"Encountered error {err} while setting up the model")
    
    return model, device


def train_model(epoch: int, model:PreTrainedModel, data:DataLoader, device: str) -> Optional[Dict]:
    metrics = None
    learning_rate, optimizer, scheduler = get_params(model=model)
    try:
        train_batch = tqdm(data, desc= f'Training Epoch: {epoch}')
        for batch in train_batch:
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss

            # back propagation




