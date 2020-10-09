# Databricks notebook source
# MAGIC %sh wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

# COMMAND ----------

def get_uri_and_model(model_name, stage):
  import mlflow.pyfunc
  #Get a model and a model_uri from model registry
  model_uri = f"models:/{model_name}/{stage}"
  model = mlflow.pyfunc.load_model(model_uri = model_uri)
  return model_uri, model

def read_dataset(wine_data_path="/databricks/driver/winequality-red.csv"):
  import pandas as pd
  #Read some data for our model
  data = pd.read_csv(wine_data_path, sep=None)
  return data

def test_model(data, model):
  from sklearn.model_selection import train_test_split
  #Get 1 row of data and predict it
  train, _ = train_test_split(data)
  train_x = train.drop(["quality"], axis=1)
  sample = train_x.iloc[[0]]
  print("Testing the model on a sample")
  print(model.predict(sample))
  return sample
  
def fixed_data_test(model):
  import pandas as pd
  #Get 1 fixed row of data and predict it
  test_data = {"row_1": [7.2,0.52,0.07,1.4,0.074,5.0,20.0,0.9973,3.32,0.81,9.6]}
  df_to_score = pd.DataFrame.from_dict(test_data, orient="index", columns=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"])
  print("Testing the model on a static sample")
  print(model.predict(df_to_score))
  return df_to_score
  
def auth_to_aml(): 
  from azureml.core.authentication import ServicePrincipalAuthentication
  from azureml.core import Workspace
  #Authenticate to Azure ML using a Service Principal
  print("Authenticating to AML...")
  svc_pr = ServicePrincipalAuthentication(
     tenant_id="9f37a392-f0ae-4280-9796-f1864a10effc",
     service_principal_id="c8023ad4-8d7f-4c06-adae-36054275b392",
     service_principal_password=str(dbutils.secrets.get(scope="silviuscope", key="dsswesp")))
  workspace = Workspace(
     subscription_id="3f2e4d32-8e8d-46d6-82bc-5bb8d962328b",
     resource_group="silv-tech-summit-2020",
     workspace_name="silv-ts-aml",
     auth=svc_pr
     )
  print("I have authenticated to AML")
  return workspace


def build_aml_image(model_uri, workspace):
  import mlflow.azureml
  #Build an Azure ML model image
  print("Creating model image...")
  model_image, azure_model = mlflow.azureml.build_image(model_uri=model_uri, 
                                                      workspace=workspace, 
                                                      model_name="dsswe-wine-testmodel",
                                                      image_name="dsswe-wine-testcontainerimage",
                                                      description="Sklearn ElasticNet image for rating wines", 
                                                      tags={
                                                        "stage": str("Prod")
                                                      },
                                                      synchronous=True)
  model_image.wait_for_creation(show_output=True)
  print("Model image has been created successfully")  
  return model_image, azure_model

def deploy_to_aci(model_image, workspace, dev_webservice_name="dsswe-wine-devwebservice5"):
  from azureml.core.webservice import AciWebservice, Webservice
  #Deploy a model image to ACI
  print("Deploying to ACI...")
  # make sure this dev_webservice_name is unique and doesnt already exist, else need to replace
  dev_webservice_deployment_config = AciWebservice.deploy_configuration()
  dev_webservice = Webservice.deploy_from_image(name=dev_webservice_name, image=model_image, deployment_config=dev_webservice_deployment_config, workspace=workspace)
  dev_webservice.wait_for_deployment()
  print("Deployment to ACI successfully complete")
  return dev_webservice

def query_endpoint_example(scoring_uri, inputs, service_key=None): 
  import requests
  import json
  #Send a request to an endpoint to score some data
  headers = {
    "Content-Type": "application/json",
  }
  if service_key is not None:
    headers["Authorization"] = "Bearer {service_key}".format(service_key=service_key)

  print("Sending batch prediction request with inputs: {}".format(inputs))
  response = requests.post(scoring_uri, data=inputs, headers=headers)
  print("Response: {}".format(response.text))
  preds = json.loads(response.text)
  print("Received response: {}".format(preds))
  return preds
  
def create_aks(workspace, aks_cluster_name="dsswe-wineprod"):
  from azureml.core.compute import AksCompute, ComputeTarget
  #Create an AKS cluster
  print("Creating AKS cluster...")
  # Use the default configuration (you can also provide parameters to customize this)
  prov_config = AksCompute.provisioning_configuration()
  # Create the cluster
  aks_target = ComputeTarget.create(workspace = workspace, 
                                    name = aks_cluster_name, 
                                    provisioning_configuration = prov_config)
  # Wait for the create process to complete
  aks_target.wait_for_completion(show_output = True)
  print(aks_target.provisioning_state)
  print(aks_target.provisioning_errors)
  print("AKS cluster creation completed")
  return aks_target

def deploy_to_aks(workspace, model_image, aks_target, prod_webservice_name="dsswe-wprodm"):
  from azureml.core.webservice import Webservice, AksWebservice
  #Deploy a model image to AKS
  print("Deploying to AKS...")
  # Set configuration and service name
  prod_webservice_deployment_config = AksWebservice.deploy_configuration()
  # Deploy from image
  prod_webservice = Webservice.deploy_from_image(workspace = workspace, 
                                                 name = prod_webservice_name,
                                                 image = model_image,
                                                 deployment_config = prod_webservice_deployment_config,
                                                 deployment_target = aks_target)
  # Wait for the deployment to complete
  prod_webservice.wait_for_deployment(show_output = True)
  print("Deployment to AKS completed sucessfully")
  return prod_webservice

# COMMAND ----------

###My actual code

version = "9"

#Get the Model from MLflow model registry
model_uri, model = get_uri_and_model("a-wine-model", "Production")

#Read some dataset to score and test the model
data = read_dataset()
df_to_score = test_model(data, model)
fixed_data_test(model)

#Authenticate to Azure ML using a Service Principal
workspace = auth_to_aml()

#Create a model image - this takes about 6 mins
model_image, azure_model = build_aml_image(model_uri, workspace)

#Deploy to ACI - this takes about 15 mins
dev_webservice = deploy_to_aci(model_image, workspace, "dsswe-wine-devwebservice"+str(version))

#Test ACI
print("Testing ACI")
sample_json = df_to_score.to_json(orient="split")
dev_scoring_uri = dev_webservice.scoring_uri
dev_prediction = query_endpoint_example(scoring_uri=dev_scoring_uri, inputs=sample_json)
print(dev_prediction)

#Create AKS
aks_target = create_aks(workspace, aks_cluster_name="dsswe-wineprod"+str(version))

#Use existing AKS
###tofill

#Deploy to AKS - this takes about 15 mins
prod_webservice = deploy_to_aks(workspace, model_image, aks_target, prod_webservice_name="dsswe-wprodm"+str(version))

#Test AKS
print("Testing AKS")
sample_json = df_to_score.to_json(orient="split")
prod_scoring_uri = prod_webservice.scoring_uri
prod_service_key = prod_webservice.get_keys()[0] if len(prod_webservice.get_keys()) > 0 else None
prod_prediction = query_endpoint_example(scoring_uri=prod_scoring_uri, service_key=prod_service_key, inputs=sample_json)
print(prod_prediction)

# COMMAND ----------

###Update AKS with new model - this takes about 20 mins
#print("Updating model in AKS")
#model_uri, model = get_uri_and_model("a-wine-model", "Production")
#model_image_updated, azure_model_updated = build_aml_image(model_uri, workspace)
#prod_webservice.update(image=model_image_updated)
#prod_webservice.wait_for_deployment(show_output = True)
#prod_prediction_updated = query_endpoint_example(scoring_uri=prod_scoring_uri, service_key=prod_service_key, inputs=sample_json)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

