import pandas as pd

from ..logger import log

def xlsx_to_df(excel_path: str):
    "Reads the excel data file and returns the dataframe after dropping unnecessary columns"
    
    df = pd.read_excel(excel_path)
    df = df.drop(columns=["Name", "CustomerID"], errors="ignore")
    
    log.info("Data ingested!")
    return df