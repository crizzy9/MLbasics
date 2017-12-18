import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import time
import datetime
import pickle

# multivariate linear regression

style.use('ggplot')

# Configuring quandl api key
quandl.ApiConfig.api_key = 'R8zsfTz84zryodGdWj5N'

# df = quandl.get('WIKI/GOOGL')
df = pd.read_csv('Googl.csv')

df.set_index('Date', inplace=True)

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)

# with regression you 'generally' (dont have to) forcast out cols you want
# 0.1 is to get data for past 10 days into the future
forcast_out = int(math.ceil(0.1 * len(df)))
print(forcast_out)

# the negative shift will set Adj.Close according to forcast_out i.e 10 days into the future
# and hence we can learn from that data what the data after 10 days looked like 
df['label'] = df[forecast_col].shift(-forcast_out)

# features
# X = np.array(df.drop(['label'], 1))
X = np.array(df[['Adj. Close']])
# scaling X avoid doing this
X = preprocessing.scale(X)
# data we are trying to predict
X_lately = X[-forcast_out:]
X = X[:-forcast_out]

# labels
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
print(X_test)

clf = LinearRegression(n_jobs=20)
# clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)

# saving trained model in pickle so dont have train full data again and again
# with open('linear_reg.pickle', 'wb') as f:
# 	pickle.dump(clf, f)

# pickle_in = open('linear_reg.pickle', 'rb')
# clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)


t0, t1 = clf.intercept_, clf.coef_


print(forecast_set, accuracy, forcast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
print(last_date)
last_unix = int(time.mktime(datetime.datetime.strptime(last_date, "%Y-%m-%d").timetuple()))
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

print(df.head())
print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.plot(X_train, t0 + t1*X_train, 'g')
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# reference: sentdex
