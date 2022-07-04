# Databricks notebook source
# MAGIC %md
# MAGIC # Retrieve Data

# COMMAND ----------

import pandas as pd

import mlflow

import xgboost as xgb

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

# COMMAND ----------


pandas_df  = pd.read_csv('/dbfs/FileStore/shared_uploads/blasa.matthew@yahoo.com/training_data.csv')

X=pandas_df.iloc[:,:-1]
Y=pandas_df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=4284, stratify=Y)

# COMMAND ----------

# MAGIC %md
# MAGIC # Set Experiment

# COMMAND ----------



# COMMAND ----------

experiment_name = "/Experiments/ml_flow_run_xgboost"
mlflow.set_experiment(experiment_name)
mlflow.xgboost.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC # Run the  experiment

# COMMAND ----------

dtrain=xgb.DMatrix(X_train, y_train)
dtest=xgb.DMatrix(X_test, y_test)
threshold = 0.5
with mlflow.start_run(run_name='xgboost_model_baseline') as run:
    model=xgb.train(dtrain=dtrain,params={})
    preds = model.predict(dtest)
    y_bin = [1. if y_cont > threshold else 0. for y_cont in preds]
    f1= f1_score(y_test,y_bin)
    mlflow.log_metric(key="f1_experiment_score", value=f1)

    
    

    
    

# COMMAND ----------



# COMMAND ----------


