import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, neighbors

df = pd.read_csv('breast-cancer-wisconsin.csv')
df.replace('?', -99999, inplace=True)
# df.dropna(inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,10,2,4,6,8,9,11,4]])


prediction = clf.predict(example_measures)

print(df.head())
print(accuracy)
print(prediction)

plt.scatter(df['mitosis'], df['class'])
plt.show()
