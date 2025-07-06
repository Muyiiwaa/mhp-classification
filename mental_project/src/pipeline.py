from src.config import *
from src.steps.cleaning import data_download,clean_data
from src.steps.preparation import data_split, setup_dataloader
from src.steps.training import model_training
from src.utils.training import load_model
from zenml import pipeline




@pipeline(enable_cache=False)
def run_pipeline():
    data = data_download()
    cleaned_data, encoder = clean_data(data = data)
    train_data, test_data = data_split(data=cleaned_data)
    train_loader, test_loader = setup_dataloader(train_data=train_data,
                                                  test_data=test_data)
    final_metrics = model_training(epochs=EPOCHS,train_loader=train_loader,
                                   test_loader=test_loader)


if __name__ == "__main__":
    run_pipeline()
