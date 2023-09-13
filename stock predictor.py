import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
dataset = pd.read_csv(r'C:\Users\ARNAB BANDYOPADHYAY\Downloads\Stock predictor\GOOGL.csv',index_col="Date",parse_dates=True)
dataset.head()
dataset.isna().any()
dataset.info()
dataset['open'].plot(figsize=(16,6))
dataset.isna().any()
dataset.info()
dataset['open'].plot(figsize=(16,6))
dataset.rolling(7).mean().head(20)
dataset['open'].plot(figsize=(16,6))
dataset.rolling(window=30).mean()['close'].plot()
dataset['close:30 Day Mean']=dataset['close'].rolling(window=30).mean()
dataset['close'].expanding(min_periods=1).mean().plot(figsize=(16,6))
training_set=dataset['open']
training_set=pd.DataFrame(training_set)
dataset.isna().any()
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
x_train= []
y_train= []
for i in range(60,1258):
    x_train.append(training_set[i-60:i,0])
    y_train.append(training_set[i,0])
x_train,y_train =np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train.shape[0],x_train.shape[1],1)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
regressor= Sequential()
regressor.add(LSTM(units=50,return_sequences = True,input_shape = (x_train.shape[1],1)))
regressor.add(Dropout(0,2))
regressor.add(LSTM(units=50,return_sequences=True)) 
regressor.add(Dropout(0,2))
regressor.add(LSTM(units=50,return_sequences=True)) 
regressor.add(Dropout(0,2))
regressor.add(LSTM(units=50)) 
regressor.add(Dropout(0,2))
regressor.add(Dense(units =1))
regressor.compile(optimizer = 'adam',loss='mean_squared_error')
regressor.fit(x_train,epochs=100,batch_size = 32)
dataset_test = pd.read_csv('')
real_stock_price = dataset_test.iloc[:,1:2].values
dataset_test.head()
dataset_test.info()
dataset_test["Volume"] = dataset_test["Volume"].str.replace(',','').astype(float)
test_set=dataset_test['open']
test_set=pd.DataFrame(test_set)
test_set.info()
dataset_total = pd.contact((dataset['open']),axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs =sc.transform(inputs)
x_test=[]
for i in range(60,80):
    x_test.append(input[i-60:i,0])
x_test = np.array(x_test,(x_test.shape[1],1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price=pd.DataFrame(predicted_stock_price)
predicted_stock_price.info()
plt.plot(real_stock_price,color = 'red',label = 'Real Google stock Price')
plt.plot(predicted_stock_price, color = 'blue', label='predicted Google stock price')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()




 







 
