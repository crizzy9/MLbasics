import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import sklearn
import os

style.use('ggplot')

plt.rcParams["patch.force_edgecolor"] = True

housing_path = '../datasets/housing'

housing = pd.read_csv(os.path.join(housing_path, 'housing.csv'))

print(housing.head())
print(housing[housing['ocean_proximity'] == 'ISLAND'])
ocp = housing['ocean_proximity'].value_counts()
plt.pie(ocp.values, labels=ocp.index, autopct='%1.3f%%')
# housing.hist(bins=50, figsize=(20,15))
plt.show()


# reference: hands on ml
