from .data_ingestion import xlsx_to_df
from .data_transformation import transform_data
from .model_training import train_and_evaluate


def trainpipe(dataset_path: str, models: dict):
    "Returns the best model and its evaluation scores."
    
    df = xlsx_to_df(dataset_path)
    X, Y = transform_data(df)
    return train_and_evaluate(X, Y, models)