from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

from ..logger import log

if TYPE_CHECKING:
    from pandas import DataFrame

def transform_data(df: 'DataFrame'):
    "Encodes the data into numeric form and returns it as numpy arrays."
    
    df.Gender = df.Gender.apply(lambda x: 1 if x=="Male" else 0)
    df = df.rename(columns={"Gender": "is_male"})
    
    df.Location = df.Location.apply(lambda x: x.replace(" ", "_").lower())
    df = pd.get_dummies(df, "", "", columns=["Location"], dtype=int)
    
    df = df[["Age", "Subscription_Length_Months", "Monthly_Bill", "Total_Usage_GB",
         "is_male", "chicago", "houston", "los_angeles", "miami", "new_york",
         "Churn"]]
    
    df_arr = df.to_numpy(np.float32)
    X = df_arr[:, :-1]
    Y = df_arr[:, -1]
    
    log.info("Data Transformation complete!")
    return X, Y