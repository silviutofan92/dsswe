# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Getting production model from registry

# COMMAND ----------

### Get the model that has been promoted to "production" stage by the DS team

import mlflow.pyfunc

model_name = "a-wine-model"
stage = "Production"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

# COMMAND ----------

# MAGIC %sh wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

# COMMAND ----------

### Get some data and make sure the model can predict stuff

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

wine_data_path = "/databricks/driver/winequality-red.csv"

data = pd.read_csv(wine_data_path, sep=None)
train, _ = train_test_split(data)
train_x = train.drop(["quality"], axis=1)
sample = train_x.iloc[[0]]

print(model.predict(sample))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

