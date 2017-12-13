import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


# Configuring quandl api key
quandl.ApiConfig.api_key = 'R8zsfTz84zryodGdWj5N'


# df = quandl.get('WIKI/GOOGL')
df = pd.read_csv('Googl.csv')

df.set_index('Date', inplace=True)

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]


forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)

# with regression you 'generally' (dont have to) forcast out cols you want
# 0.1 is to get data for past 10 days into the future
forcast_out = int(math.ceil(0.01*len(df)))
print(forcast_out)

# the negative shift will set Adj.Close according to forcast_out i.e 10 days into the future
# and hence we can learn from that data what the data after 10 days looked like 
df['label'] = df[forecast_col].shift(-forcast_out)

df.dropna(inplace=True)

# features
X = np.array(df.drop(['label'], 1))
# labels
y = np.array(df['label'])

#scaling X
X = preprocessing.scale(X)


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=20)
# clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)


print(accuracy)
