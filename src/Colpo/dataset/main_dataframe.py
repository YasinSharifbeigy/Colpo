import pandas as pd
import numpy as np
from pandas import DataFrame
import os
from collections import Counter


extra_features_columns = ["Age", "Smoking", "Drink", "SD", "Married", "#Partner", "Pop"]
def extract_extra_features(ex_file: DataFrame):
    extra_df = pd.DataFrame(data= np.array([[np.nan]*len(extra_features_columns)]), columns=extra_features_columns)
    age_idx = 3
    
    if "Age" not in ex_file.iloc[age_idx,0]:
        i=0
        while "Age" not in ex_file.iloc[i,0]:
          i += 1
        age_idx = i
    smoke_idx = 22
    if "Smoking" not in ex_file.iloc[smoke_idx,0]:
        i=0
        while "Smoking" not in ex_file.iloc[i,0]:
          i += 1
        smoke_idx = i

    extra_df['Age'] = ex_file.iloc[age_idx+1,0]
    extra_df['Smoking'] = ex_file.iloc[smoke_idx+1,0]
    extra_df['Drink'] = ex_file.iloc[smoke_idx+1,1]
    extra_df['SD'] = ex_file.iloc[smoke_idx+1,2]
    extra_df['Married'] = ex_file.iloc[smoke_idx+1,3]
    extra_df['#Partner'] = ex_file.iloc[smoke_idx+1,4]

    return extra_df

