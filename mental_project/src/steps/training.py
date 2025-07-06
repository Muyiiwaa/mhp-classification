from zenml import step
from src.utils.training import train_model, validate_model, compute_metrics
from typing import Optional,Dict
from typing_extensions import Annotated
from transformers import PreTrainedModel
import pandas as pd
from zenml.logger import get_logger
from torch.utils.data import DataLoader
import wandb
from src.config import *
from src.utils.training import load_model

logger = get_logger(__name__)


@step
def model_training(epochs: int, train_loader:DataLoader, test_loader:DataLoader) -> Annotated[
                       Optional[Dict],"Final Metrics"]:
    try:
        logger.info('Starting model training!')
        run = wandb.init(
            project = "MHP-Text-Classification",
            config = {
                "learning rate": LEARNING_RATE,
                "architecture": MODEL_URI,
                "dataset": DATASET,
                "optimizer": OPTIMIZER,
                "scheduler": SCHEDULER,
                "epochs": EPOCHS
            }
        )
        model, device = load_model(MODEL_URI)
        for epoch in range(epochs):
            train_labels, train_preds, train_loss = train_model(epoch=epoch, model=model,
                                                                data=train_loader,device=device)
            train_metrics = compute_metrics(train_labels, train_preds, train_loss, epoch=epoch)
            test_labels, test_preds, test_loss = validate_model(epoch=epoch, model=model,
                                                                data=test_loader,device=device)
            test_metrics = compute_metrics(test_labels, test_preds, test_loss, epoch=epoch)

            # combine train and test metrics
            train_metrics.update(test_metrics)
            # log the metrics for this epoch in weight and bias.
            run.log(train_metrics)

        logger.info(f"""Completed Training with metrics:\n
                    {train_metrics}""")

    except Exception as err:
        logger.error(f"Encountered error {err} while training.")
    else:
        # finish run.    
        run.finish()

    return train_metrics


if __name__ == "__main__":
    pass
  