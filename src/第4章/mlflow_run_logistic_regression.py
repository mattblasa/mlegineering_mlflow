# Databricks notebook source
# MAGIC %md
# MAGIC # Import dependencies

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# COMMAND ----------

# MAGIC %md
# MAGIC # Retrieve Data

# COMMAND ----------


pandas_df  = pd.read_csv('/dbfs/FileStore/shared_uploads/blasa.matthew@yahoo.com/training_data.csv')
X=pandas_df.iloc[:,:-1]
Y=pandas_df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=4284, stratify=Y)

# COMMAND ----------

# MAGIC %md
# MAGIC # Set Experiment

# COMMAND ----------


experiment_name = "/Experiments/ml_flow_run_xgboost"
mlflow.set_experiment(experiment_name)


# COMMAND ----------

# MAGIC %md
# MAGIC # Create model

# COMMAND ----------

mlflow.sklearn.autolog()
model = LogisticRegression()

# COMMAND ----------

# MAGIC %md
# MAGIC # Run the model

# COMMAND ----------

with mlflow.start_run(run_name='logistic_regression_model_baseline') as run:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    y_pred = np.where(preds>0.5,1,0)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric(key="f1_experiment_score", value=f1)

# COMMAND ----------

f1

# COMMAND ----------


