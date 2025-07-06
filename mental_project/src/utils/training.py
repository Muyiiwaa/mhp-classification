import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from torch import optim
import torch
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from typing import Optional, Tuple, Union, Dict, List, Literal
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from src.config import *
from src.utils.data_preparation import *
from src.utils.data_cleaning import *

# configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# define model hyperparameters
def get_params(model: PreTrainedModel) -> Tuple:

    learning_rate = LEARNING_RATE
    optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE,weight_decay=OPTIMIZER["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, 
                                          step_size=SCHEDULER["step_size"], gamma=SCHEDULER["gamma"])

    return learning_rate, optimizer, scheduler

def load_model(model_uri: str) -> Tuple[Optional[PreTrainedModel], Optional[str]]:
    """_This function loads the pretrained model from huggingface, moves the model
    to available device and returns both the device and model object._

    Args:
        model_uri (str): _the model id on huggingface._

    Returns:
        Tuple[Optional[PreTrainedModel], Optional[str]]: _MODEL, DEVICE._
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


def train_model(epoch: int, model, 
                data:DataLoader, device: Literal['cpu','cuda']) -> Tuple[List[float], List[float], float]:
    """_This function performs the training/fine-tuning of the model on our mental health dataset.
    It only contains the logic for training. It performs this operation on only one epoch and
    returns the predictions, labels and value of the loss._

    Args:
        epoch (int): _The current epoch value._
        model (PreTrainedModel): _The huggingface model object that we are finetuning on._
        data (DataLoader): _the training dataset in form of a dataloader._
        device (str): _the device type._

    Returns:
        Tuple[List[float], List[float], float]: _LABELS, PREDICTIONS, FINAL_LOSS._
    """
    _, optimizer, scheduler = get_params(model=model)

    # track labels and predictions.
    labels, predictions, loss_list = [], [], []
    final_loss = 0.0
    epoch = epoch + 1
    try:
        model.train()
        train_batch = tqdm(data, desc= f'Training Epoch: {epoch}')
        for batch in train_batch:
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # populate predictions and labels lists with the training output.
            _,preds = torch.max(outputs.logits, 1) # get the actual model prediction.
            if device == 'cpu':
                labels.extend(label.detach().numpy())
                predictions.extend(preds.detach().numpy())
            else:
                labels.extend(label.cpu().detach().numpy())
                predictions.extend(preds.cpu().detach().numpy())
            
            # update loss list
            loss_list.append(loss.item())

            train_batch.set_postfix(loss=loss.item())

        scheduler.step()

        # get the final loss.
        final_loss = sum(loss_list)/len(loss_list)
        logger.info(msg=f'Completed training really epoch: {epoch} with loss: {final_loss}')
    except Exception as err:
        logger.error(msg=f'Encountered error {err} while training epoch {epoch}')

    return labels, predictions, final_loss


# setup the validation loop
def validate_model(epoch: int, model, 
                data:DataLoader, device: Literal['cpu','cuda']) -> Tuple[List[float], List[float], float]:
    """_This function evaluates the trained.
    It only contains the logic for evaluating the model on the test set. It performs this operation on only one epoch and
    returns the predictions, labels and value of the loss._

    Args:
        epoch (int): _The current epoch value._
        model (PreTrainedModel): _The huggingface model object that we are finetuning on._
        data (DataLoader): _the test dataset in form of a dataloader._
        device (str): _the device type._

    Returns:
        Tuple[List[float], List[float], float]: _LABELS, PREDICTIONS, FINAL_LOSS._
    """
    # track labels and predictions.
    labels, predictions, loss_list = [], [], []
    final_loss = 0.0
    epoch = epoch + 1
    try:
        model.eval()
        with torch.no_grad():
            test_batch = tqdm(data, desc= f'testing Epoch: {epoch}')
            for batch in test_batch:
                input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                label = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
                loss = outputs.loss

                # populate predictions and labels lists with the testing output.
                _,preds = torch.max(outputs.logits, 1) # get the actual model prediction.
                if device == 'cpu':
                    labels.extend(label.detach().numpy())
                    predictions.extend(preds.detach().numpy())
                else:
                    labels.extend(label.cpu().detach().numpy())
                    predictions.extend(preds.cpu().detach().numpy())

                # update loss list
                loss_list.append(loss.item())
                test_batch.set_postfix(loss=loss.item())
            
            final_loss = sum(loss_list)/len(loss_list)
        logger.info(msg=f"Completed testing epoch: {epoch} with loss: {final_loss}")

    except Exception as err:
        logger.error(msg= f"Encountered error {err} while testing epoch {epoch}")

    return labels, predictions, final_loss


def compute_metrics(label:List[float], predictions: List[float],
                    final_loss:float, epoch: int,suffix:Literal['train','test']) -> Optional[Dict]:
    """_Helper function for computing metrics. It takes labels and 
    predictions and returns key outputs._

    Args:
        label (List[float]): _The true labels._
        predictions (List[float]): _The model predictions._
        final_loss (float): _The loss from that training run._
        epoch (int): _The current epoch being trained._

    Returns:
        Optional[Dict]: _the dictionary containing the metrics._
    """
    
    f1 = f1_score(y_true=label, y_pred=predictions,average="weighted")
    precision = precision_score(y_true=label, y_pred=predictions,average="weighted")
    recall = recall_score(y_true=label, y_pred=predictions,average="weighted")

    return {
        f'{suffix}_f1_score': f1,
        f'{suffix}_precision_score': precision,
        f'{suffix}_recall_score': recall,
        f'{suffix}_loss': final_loss,
        f'epoch': epoch +  1
    }


if __name__ == "__main__":
    data, encoder = data_cleaner(data= download_data())
    train_df, test_df = split_dataset(data = data)
    train_loader, test_loader = final_dataloader(train_data=train_df,
                                                 test_data=test_df)
    model, device = load_model(model_uri=MODEL_URI)
    for epoch in range(5):
        train_labels, train_preds, train_loss = train_model(epoch=epoch, model=model,
                                                            data=train_loader,device=device)
        test_labels, test_preds, test_loss = validate_model(epoch=epoch, model=model,
                                                            data=test_loader,device=device)    
