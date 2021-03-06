import mlflow
import sklearn
mlflow.set_tracking_uri("databricks://AZDO")

def get_uri_and_model(model_name, stage):
    import mlflow.pyfunc
    # Get a model and a model_uri from model registry
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
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
    #print(client_secret)
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


def build_aml_image(model_uri, workspace, model_name, image_name):
    import mlflow.azureml
    # Build an Azure ML model image
    print("Creating model image...")
    model_image, azure_model = mlflow.azureml.build_image(model_uri=model_uri,
                                                          workspace=workspace,
                                                          model_name=model_name,
                                                          image_name=image_name,
                                                          description="Scikit for Wine Prediction",
                                                          tags={
                                                              "stage": str("qa_for_prod")
                                                          },
                                                          synchronous=True)
    model_image.wait_for_creation(show_output=True)
    print("Model image has been created successfully")
    return model_image, azure_model


def deploy_to_aci(model_image, workspace, dev_webservice_name):
    from azureml.core.webservice import AciWebservice, Webservice
    # Deploy a model image to ACI
    print("Deploying to ACI...")
    # make sure this dev_webservice_name is unique and doesnt already exist, else need to replace
    dev_webservice_deployment_config = AciWebservice.deploy_configuration()
    dev_webservice = Webservice.deploy_from_image(name=dev_webservice_name, image=model_image,
                                                  deployment_config=dev_webservice_deployment_config,
                                                  workspace=workspace)
    dev_webservice.wait_for_deployment()
    print("Deployment to ACI successfully complete")
    return dev_webservice


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

import sys
version = sys.argv[2]

#Get the Model from MLflow model registry
model_uri, model = get_uri_and_model("a-wine-model", "Production")

#Read some dataset to score and test the model
df_to_score = fixed_data_test(model)

#Authenticate to Azure ML using a Service Principal
import sys
client_secret = sys.argv[1]
workspace = auth_to_aml(client_secret)

#Create a model image - this takes about 6 mins
model_name = "dsswe-skw-" + str(version)
image_name = "dsswe-skwimg-" + str(version)
model_image, azure_model = build_aml_image(model_uri, workspace, model_name, image_name)

#Deploy to ACI - this takes about 15 mins
dev_webservice_name = "dsswe-skaci-" + str(version)
dev_webservice = deploy_to_aci(model_image, workspace, dev_webservice_name)

#Test ACI
print("Testing ACI")
sample_json = df_to_score.to_json(orient="split")
dev_scoring_uri = dev_webservice.scoring_uri
dev_prediction = query_endpoint_example(scoring_uri=dev_scoring_uri, inputs=sample_json)
print(dev_prediction)

print("Everything works on ACI! Ready for Prod")





