LEARNING_RATE = 2e-5
MODEL_URI = "distilbert-base-uncased"
DATASET = "suchintikasarkar/sentiment-analysis-for-mental-health"
OPTIMIZER = {
    "name": "AdamW",
    "weight_decay": 0.01}
SCHEDULER = {
    "name": "StepLR",
    "gamma": 0.1,
    "step_size": 2}
EPOCHS = 10