import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
dataset =
dataset.head()
dataset.isna().any()
dataset.info()
dataset['Open'].plot(figsize=(16,6))
dataset[“"Close"] = dataset["Close"].str.replace(’,’, '').astype(float)
 dataset["Volume"] = dataset["Volume"].str.replace(’,’, '').astype(float)
dataset.isna().any()
dataset.info()
dataset['open'].plot(figsize-(16,6))
dataset["close"]=dataset["close"].str.replace(',','')  .astype(float)
dataset["Volume"]=dataset["Volume"].str.replace(',','').astype(float)
dataset.rolling(7).mean().head(20)
dataset['open'].plot(figsize-(16,6))
dataset.rolling(windows=30).mean()['close'].plot()
dataset['close:30 Day Mean']=dataset['close'].rolling(windows=30).mean()
dataset['close'].expanding(min_periods=1).mean().plot(figsize=(16,6))
training_set=dataset['open']
training_set=pd.DataFrame(training_set)
dataset.isna().any()
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set scaled = sc.fit_transform(training_set)
x_train= []
y_train= []
for i in range(60,1258):






 
