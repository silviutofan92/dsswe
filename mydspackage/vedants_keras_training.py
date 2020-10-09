# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/themes/databricks/assets/images/databricks-logo.png" alt="logo" width="240"/> 
# MAGIC ## MNIST with Keras, distributed Hyperopt with
# MAGIC 
# MAGIC ## <img width="100px" src="https://spark.apache.org/docs/latest/img/spark-logo-hd.png"> + automated <img width="100px" src="https://mlflow.org/docs/0.7.0/_static/MLflow-logo-final-black.png"> tracking
# MAGIC 
# MAGIC 
# MAGIC Hyperparameter tuning using the Bayesian method Tree of Parzen Estimators (TPE)
# MAGIC - An adaptive method that iteratively searches the hyperparameter space, trading off exploration vs exploitation
# MAGIC - offers improved compute efficiency compared to a brute force approach such as grid search
# MAGIC - helps overcome the curse of dimensionality
# MAGIC 
# MAGIC Distributed out across a cluster using Apache Spark, with each tuning run automatically tracked with mlflow
# MAGIC 
# MAGIC 
# MAGIC To replicate this example notebook you will need: 
# MAGIC 1. A cluster with the [Databricks Runtime for Machine Learning](https://databricks.com/blog/2018/06/05/distributed-deep-learning-made-simple.html) 6.5 or greater (CPU or GPU)
# MAGIC 
# MAGIC To learn more about deep learning on Databricks, please see the [documentation](https://docs.databricks.com/applications/deep-learning/index.html)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Load libraries

# COMMAND ----------

import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

import math
import time
import tempfile

import tensorflow as tf

import mlflow
import mlflow.keras
import os
print(f"MLflow Version: {mlflow.__version__}.")

from datetime import datetime
RUN_NAME = "mnist"

from hyperopt import fmin, hp, tpe, STATUS_OK
from hyperopt import SparkTrials

import matplotlib.pyplot as plt

import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download data, transform it and write out
# MAGIC 
# MAGIC The Databricks file system ([dbfs](https://docs.databricks.com/data/databricks-file-system.html)) allows you to access objects in cloud storage as if they were on the local file system.

# COMMAND ----------

temp_path = "/dbfs/ml/tmp/KnowledgeRepo/distributed_hyperopt/"
dbutils.fs.mkdirs("dbfs:/ml/tmp/KnowledgeRepo/distributed_hyperopt")

x_train_path = os.path.join(temp_path, 'x_train.npy')
y_train_path = os.path.join(temp_path, 'y_train.npy')
x_val_path = os.path.join(temp_path, 'x_val.npy')
y_val_path = os.path.join(temp_path, 'y_val.npy')

log_dir_local = os.path.join(temp_path, "tensorboard")
log_dir_cloud = "dbfs:/ml/tmp/KnowledgeRepo/distributed_hyperopt/tensorboard"

parallelism = sc.defaultParallelism #set to number of workers on the cluster

# COMMAND ----------

# MAGIC %fs ls /ml/tmp/KnowledgeRepo/distributed_hyperopt/

# COMMAND ----------

# MAGIC %md #### Load, split, and pre-process train/validation datasets

# COMMAND ----------

img_rows, img_cols = 28, 28
num_classes = 10

(x_train, y_train), (x_val, y_val) = mnist.load_data()

#subset mnist for faster training
x_train = x_train[:5000]
y_train = y_train[:5000]
x_val = x_val[:500]
y_val = y_val[:500]

# COMMAND ----------

plt.imshow(x_val[111])
display()

# COMMAND ----------

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'validation samples')

# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_val = tensorflow.keras.utils.to_categorical(y_val, num_classes)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write numpy arrays
# MAGIC 
# MAGIC Files are written to `dbfs` will be re-read in later on Spark workers during distributed hyperparameter tuning

# COMMAND ----------

np.save(x_train_path, x_train)
np.save(y_train_path, y_train)
np.save(x_val_path, x_val)
np.save(y_val_path, y_val)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train convolutional neural network

# COMMAND ----------

# MAGIC %md
# MAGIC #### Start TensorBoard to visualize run
# MAGIC 
# MAGIC [TensorBoard](https://docs.databricks.com/applications/deep-learning/single-node-training/tensorflow.html#tensorboard) is TensorFlow’s suite of visualization tools for debugging, optimizing, and understanding TensorFlow programs.

# COMMAND ----------

dbutils.fs.mkdirs(log_dir_local)
dbutils.tensorboard.start(log_dir_local)

# COMMAND ----------

# MAGIC %md #### Define CNN network

# COMMAND ----------

def create_model(hpo):
  """
  Feed in hyperparameters (note only stride length optimized here, but you can also add kernal_size, dropout etc.)
  :param hpo: dict of parameters for hyperparameter optimization. 
  :return: model object
  """
  
  model = Sequential()
  
  # Convolution Layer
  model.add(Conv2D(32, kernel_size=(5,5),
                 activation='relu',
                 input_shape=input_shape)) 
  
  # Convolution layer
  model.add(Conv2D(64, (5,5), activation='relu'))
  
  # Pooling with stride
  model.add(MaxPooling2D(pool_size=(int(hpo['stride']), int(hpo['stride']))))
  
  # Delete neuron randomly while training
  # Regularization technique to avoid overfitting
  model.add(Dropout(0.5))
  
  # Flatten layer 
  model.add(Flatten())
  
  # Fully connected Layer
  model.add(Dense(128, activation='relu'))
  
  # Delete neuron randomly while training 
  # Regularization technique to avoid overfitting
  model.add(Dropout(0.5))
  
  # Apply Softmax
  model.add(Dense(num_classes, activation='softmax'))

  return model

# COMMAND ----------

# MAGIC %md #### Train and validate model 
# MAGIC 
# MAGIC hyperparameters & loss metrics are automatically tracked in mlflow

# COMMAND ----------

def setup_tensorboard(hpo):
  """
  setup tensorboard for each run, creating a log dir named based on the hyperparams
  :param hpo: dict of hyperparameters to be tuned
  """
  params = "_".join(sorted(str(x) for x in list(hpo.values())))
  run_log_dir = log_dir_local + "/" + params
  tensorboard = TensorBoard(log_dir=run_log_dir)
  return tensorboard

def runCNN(hpo):
  import tensorflow.keras.optimizers
  import tensorflow.keras.losses
  import tensorflow.keras
  import mlflow.keras

  """
  run convolutional neural network with the following steps: 
  1. create model
  2. compile model
  3. setup tensorboard and logging directory
  4. fit model
  5. return loss (metric to be minimzed during training)
  :param hpo: dict of hyperparameters to be tuned, in this case optimizer, learning rate, & batch_size
  :return: validation loss and status of run
  """

  model = create_model(hpo)
  
  optimizer = tensorflow.keras.optimizers.get({'class_name': hpo['optimizer'], 'config': {'learning_rate': math.pow(10, hpo['learning_rate'])}})
  
  model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])
  
  tensorboard = setup_tensorboard(hpo)
  
  history = model.fit(np.load(x_train_path), np.load(y_train_path),
                      batch_size=int(hpo['batch_size']),
                      callbacks=[tensorboard],
                      epochs=single_node_epochs,
                      verbose=1,
                      validation_data=(np.load(x_val_path), np.load(y_val_path))
                     )
  
  mlflow.keras.autolog()
  
  obj_metric = history.history["val_loss"][-1]

  
#   modelPath = "/dbfs/mnt/ved-demo/mlmodels/mnist/model-%f" % (obj_metric)
  modelPath = "/dbfs/mnt/ved-demo/mlmodels/mnist/model"
  mlflow.keras.log_model(model, artifact_path="mnist-model")
  
#   mlflow.keras.save_model(model, modelpath)

  return {'loss': obj_metric, 'status': STATUS_OK}


# COMMAND ----------

# MAGIC %md #### Setup hyperparameter space and training config, and then invoke model training via HyperOpt
# MAGIC 
# MAGIC To learn more about Hyperopt & SparkTrials see [here](https://docs.databricks.com/spark/latest/mllib/hyperopt-spark-mlflow-integration.html#how-to-use-hyperopt-with-sparktrials)

# COMMAND ----------

single_node_epochs = 20
num_classes = 10

#search space for hyperparameter tuning
space = {
  'stride': hp.quniform('stride', 2, 4, 1),
  'batch_size': hp.uniform('batch_size', 32, 128),
  'learning_rate': hp.uniform('learning_rate', -10, 0),
  'optimizer': hp.choice('optimizer', ['adadelta', 'adam', 'rmsprop'])
}
dbutils.fs.rm('/mnt/ved-demo/mlmodels/mnist', True)
dbutils.fs.mkdirs('/mnt/ved-demo/mlmodels/mnist')
  
spark_trials = SparkTrials(parallelism=parallelism)
with mlflow.start_run():
  argmin = fmin(
    fn=runCNN, 
    space=space, 
    algo=tpe.suggest, 
    max_evals=32, 
    show_progressbar=False, 
    trials=spark_trials
)
  
#install keras separetly 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Return the set of hyperparams that minimized the loss 

# COMMAND ----------

argmin

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review results on <img width="90px" src="https://mlflow.org/docs/0.7.0/_static/MLflow-logo-final-black.png"> & <img width="40px" src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/957px-Tensorflow_logo.svg.png"> Tensorboard

# COMMAND ----------

# MAGIC %md
# MAGIC #### Compare each hyperparameter tuning run with mlflow experiments
# MAGIC 
# MAGIC Hyperparameters and loss metrics are automatically tracked during each training run

# COMMAND ----------

# MAGIC %md
# MAGIC #### Parallel Coordinates Plot
# MAGIC 
# MAGIC When comparing mlflow runs, this plot enables visualization of each training run to see how each hyperparameter affected the training
# MAGIC 
# MAGIC In this case a high learning rate has led to loss divergence during training, resulting in the network overshooting the ideal weights and never finding the optimum

# COMMAND ----------

# MAGIC %md
# MAGIC #### Screenshot of tensorboard
# MAGIC 
# MAGIC Track the model training during each epoch for each set of hyperparameters

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean up

# COMMAND ----------

dbutils.tensorboard.stop()
dbutils.fs.rm(temp_path, True)

# COMMAND ----------

