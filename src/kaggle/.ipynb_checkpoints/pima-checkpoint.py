import os
import numpy as np
import pandas as pd

pima_data = pd.read_csv(os.path.abspath("src/kaggle/datasets/diabetes.csv"))
print(pima_data.head())
print(pima_data.info())

for c1 in pima_data.columns:
    for c2 in pima_data.columns:
        print(c1, c2, pima_data[c1].corr(pima_data[c2]))

