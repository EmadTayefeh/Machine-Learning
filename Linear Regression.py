import pandas as pd
import Quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


style.use('ggplot')

df = Quandl.get('WIKI/GOOGL')

#make a list
df = df[['Adj. open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. high'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'

#using nan Data
df.fillna(-99999, inplace=True)

#Predict prices for next week base of ten percent of data entry
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

#Shifting a colum up to show prices from ten days ago
df['label'] = df[forecast_col].shift(-forecast_out)


#predict again for last 30 days, Scaling and Labeling
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]


df.dropna(inplace=True)
y = np.array(df['label'])


#Shuffling all data and save it as Train and test for X and y
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

#seperate the data and clear the number of jobs to use for the computation
accuracy = clf.score(X_test,y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

#Last date info
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

#Create or replace (predict) new data for a specific range of next days
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for ... in range(len(df.columns)-1)] + [i]

print(df.tail())

df['Adj. close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


