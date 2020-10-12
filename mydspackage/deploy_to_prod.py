import mlflow
import sklearn
mlflow.set_tracking_uri("databricks://AZDO")


def get_uri_and_model(model_name, stage):
    import mlflow.pyfunc
    # Get a model and a model_uri from model registry
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.keras.load_model(model_uri=model_uri)
    return model_uri, model


def fixed_data_test(model):
    import pandas as pd
    # Get 1 fixed row of data and predict it
    test_data = {"row_1": [7.2, 0.52, 0.07, 1.4, 0.074, 5.0, 20.0, 0.9973, 3.32, 0.81, 9.6]}
    df_to_score = pd.DataFrame.from_dict(test_data, orient="index",
                                         columns=["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                                                  "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                                                  "pH", "sulphates", "alcohol"])
    print("Testing the model on a static sample")
    print(model.predict(df_to_score))
    return df_to_score


def auth_to_aml(client_secret):
    from azureml.core.authentication import ServicePrincipalAuthentication
    from azureml.core import Workspace
    #Authenticate to Azure ML using a Service Principal
    print("Authenticating to AML...")
    svc_pr = ServicePrincipalAuthentication(
     tenant_id="9f37a392-f0ae-4280-9796-f1864a10effc",
     service_principal_id="c8023ad4-8d7f-4c06-adae-36054275b392",
     service_principal_password=client_secret)
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
    # Build an Azure ML model image
    print("Creating model image...")
    model_image, azure_model = mlflow.azureml.build_image(model_uri=model_uri,
                                                          workspace=workspace,
                                                          model_name="dsswe-mnist-testmodel",
                                                          image_name="dsswe-mnist-testcontainerimage",
                                                          description="Keras for MNIST",
                                                          tags={
                                                              "stage": str("Prod")
                                                          },
                                                          synchronous=True)
    model_image.wait_for_creation(show_output=True)
    print("Model image has been created successfully")
    return model_image, azure_model


def query_endpoint_example(scoring_uri, inputs, service_key=None):
    import requests
    import json
    # Send a request to an endpoint to score some data
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


def create_aks(workspace, aks_cluster_name="dsswe-mnistp"):
    from azureml.core.compute import AksCompute, ComputeTarget
    # Create an AKS cluster
    print("Creating AKS cluster...")
    # Use the default configuration (you can also provide parameters to customize this)
    prov_config = AksCompute.provisioning_configuration()
    # Create the cluster
    aks_target = ComputeTarget.create(workspace=workspace,
                                      name=aks_cluster_name,
                                      provisioning_configuration=prov_config)
    # Wait for the create process to complete
    aks_target.wait_for_completion(show_output=True)
    print(aks_target.provisioning_state)
    print(aks_target.provisioning_errors)
    print("AKS cluster creation completed")
    return aks_target


def deploy_to_aks(workspace, model_image, aks_target, prod_webservice_name="dsswe-mprodm"):
    from azureml.core.webservice import Webservice, AksWebservice
    # Deploy a model image to AKS
    print("Deploying to AKS...")
    # Set configuration and service name
    prod_webservice_deployment_config = AksWebservice.deploy_configuration()
    # Deploy from image
    prod_webservice = Webservice.deploy_from_image(workspace=workspace,
                                                   name=prod_webservice_name,
                                                   image=model_image,
                                                   deployment_config=prod_webservice_deployment_config,
                                                   deployment_target=aks_target)
    # Wait for the deployment to complete
    prod_webservice.wait_for_deployment(show_output=True)
    print("Deployment to AKS completed sucessfully")
    return prod_webservice

import sys
version = sys.argv[2]

#Get the Model from MLflow model registry
model_uri, model = get_uri_and_model("a-silviu-mnist", "Production")

#Read some dataset to score and test the model
#df_to_score = fixed_data_test(model)

#Authenticate to Azure ML using a Service Principal
client_secret = sys.argv[1]
workspace = auth_to_aml(client_secret)

#Create a model image - this takes about 6 mins
model_image, azure_model = build_aml_image(model_uri, workspace)

#Create AKS
aks_target = create_aks(workspace, aks_cluster_name="dsswe-mnistp"+str(version))

#Use existing AKS
###tofill

#Deploy to AKS - this takes about 15 mins
prod_webservice = deploy_to_aks(workspace, model_image, aks_target, prod_webservice_name="dsswe-mpm"+str(version))

#Test AKS
#print("Testing AKS")
#sample_json = df_to_score.to_json(orient="split")
#prod_scoring_uri = prod_webservice.scoring_uri
#prod_service_key = prod_webservice.get_keys()[0] if len(prod_webservice.get_keys()) > 0 else None
#prod_prediction = query_endpoint_example(scoring_uri=prod_scoring_uri, service_key=prod_service_key, inputs=sample_json)
#print(prod_prediction)


print("Everything works on AKS!")



