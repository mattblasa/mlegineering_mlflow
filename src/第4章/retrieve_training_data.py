# Databricks notebook source
#Acquire data

# COMMAND ----------

import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import requests
from matplotlib import pyplot as plt

 #Workaround to handle issue https://github.com/pydata/pandas-datareader/issues/868
USER_AGENT = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                ' Chrome/91.0.4472.124 Safari/537.36')
    }
sesh = requests.Session()
sesh.headers.update(USER_AGENT)


start = datetime.datetime(2014, 1, 1)
end = datetime.datetime(2020, 12, 31)

btc_df = web.DataReader("BTC-USD", 'yahoo', start, end,  session=sesh)
btc_df

# COMMAND ----------


btc_df['Open'].plot()
resolution_value = 1200
plt.savefig("myImage.png", format="png", dpi=resolution_value)


# COMMAND ----------



# COMMAND ----------

btc_df['delta_pct'] = (btc_df['Close'] - btc_df['Open'])/btc_df['Open']

# COMMAND ----------

def rolling_window(a, window):
    """
        Takes np.array 'a' and size 'window' as parameters
        Outputs an np.array with all the ordered sequences of values of 'a' of size 'window'
        e.g. Input: ( np.array([1, 2, 3, 4, 5, 6]), 4 )
             Output: 
                     array([[1, 2, 3, 4],
                           [2, 3, 4, 5],
                           [3, 4, 5, 6]])
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)




# COMMAND ----------

    
btc_df['going_up'] = btc_df['delta_pct'].apply(lambda d: 1 if d>0.00001 else 0).to_numpy()

# COMMAND ----------

element=btc_df['going_up'].to_numpy()

# COMMAND ----------

WINDOW_SIZE=15

# COMMAND ----------

training_data = rolling_window(element, WINDOW_SIZE)

# COMMAND ----------

training_data

# COMMAND ----------

pd.DataFrame(training_data).to_csv("training_data.csv", index=False)



# COMMAND ----------

Y=training_data[:,-1]

# COMMAND ----------

X=training_data[:,:-1]

# COMMAND ----------

X

# COMMAND ----------



# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=4284, stratify=Y)


# COMMAND ----------

X_train

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

import mlflow
mlflow.sklearn.autolog()

# COMMAND ----------

lr = LogisticRegression()
    lr.fit(X,Y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


# COMMAND ----------


