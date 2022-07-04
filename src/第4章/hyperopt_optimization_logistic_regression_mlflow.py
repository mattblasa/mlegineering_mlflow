# Databricks notebook source
# MAGIC %md
# MAGIC # Import variables

# COMMAND ----------

!pip install hyperopt

# COMMAND ----------

from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data

# COMMAND ----------

pandas_df = pandas.read_csv("training_data.csv")
X=pandas_df.iloc[:,:-1]
y=pandas_df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4284, stratify=y)

# COMMAND ----------

# MAGIC %md
# MAGIC # Define objective function

# COMMAND ----------

N_FOLDS = 4
MAX_EVALS = 10

def objective(params, n_folds = N_FOLDS):
    """Objective function for Logistic Regression Hyperparameter Tuning"""

    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evaluate based on ROC AUC
    mlflow.sklearn.autolog()
    with mlflow.start_run(nested=True):
        clf = LogisticRegression(**params,random_state=0,verbose =0)
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')

        # Extract the best score
        best_score = max(scores)

        # Loss must be minimized
        loss = 1 - best_score

        # Dictionary with information for evaluation
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC # Define parameter space

# COMMAND ----------


space = {
    'warm_start' : hp.choice('warm_start', [True, False]),
    'fit_intercept' : hp.choice('fit_intercept', [True, False]),
    'tol' : hp.uniform('tol', 0.00001, 0.0001),
    'C' : hp.uniform('C', 0.05, 3),
    'solver' : hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
    'max_iter' : hp.choice('max_iter', range(5,1000))
}

# COMMAND ----------

# MAGIC %md
# MAGIC # Create experiment

# COMMAND ----------

mlflow.set_experiment("Hyperopt_Optimization")

# COMMAND ----------

# MAGIC %md
# MAGIC # Define Optimization Trials

# COMMAND ----------

# Algorithm
tpe_algorithm = tpe.suggest

# Trials object to track progress
bayes_trials = Trials()




with mlflow.start_run():
    best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)

# COMMAND ----------

best

# COMMAND ----------


