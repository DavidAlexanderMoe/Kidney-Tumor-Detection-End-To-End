# Kidney-Tumor-Detection-End-To-End
End to End Kidney disease classification using Deep Learning, MLFlow, DVC, and Deployment on AWS


## Workflows

1. Update config.yaml
2. Update params.yaml
3. Update the entity
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline
7. Update the main.py
8. Update the dvc.yaml
9. app.py


# MLFlow Model Monitoring Preview
![Alt text](image.png)


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/DavidAlexanderMoe/Kidney-Tumor-Detection-End-To-End
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n cnnkidney python=3.8 -y
```

```bash
conda activate cnnkidney
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now, open up your local host and port.







## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/DavidAlexanderMoe/Kidney-Tumor-Detection-End-To-End.mlflow

export MLFLOW_TRACKING_USERNAME=DavidAlexanderMoe 

export MLFLOW_TRACKING_PASSWORD=...

```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your experiments
 - Logging & tagging your model


DVC 

 - Its very lightweight for POC only
 - lite weight experiments tracker
 - It can perform Orchestration (Creating Pipelines)



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access: It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 406660890134.dkr.ecr.us-east-1.amazonaws.com/kidney

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optional

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app
