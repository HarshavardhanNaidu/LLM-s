# LLM-s
Large Language Models
import boto3
import sagemaker

# Create a SageMaker session
session = sagemaker.Session()

# Define IAM role
role = sagemaker.get_execution_role()

# Set the bucket name and prefix for storing model artifacts
bucket = session.default_bucket()
prefix = 'hello-world-endpoint'

# Define the model data
model_data = 's3://{}/{}/model.tar.gz'.format(bucket, prefix)

# Create SageMaker model
model = sagemaker.Model(model_data=model_data,
                        role=role,
                        image_uri='174872318107.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3')

# Deploy the model to an endpoint
predictor = model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

# Use the deployed model endpoint to make predictions
response = predictor.predict("Hello, World!")
print(response)

# Delete the endpoint when done
predictor.delete_endpoint()
