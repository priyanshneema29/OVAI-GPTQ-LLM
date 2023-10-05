import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

from configs import *

boto3_session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
sagemaker_session = sagemaker.Session(boto3_session)

# Hub Model configuration
configs = {
	'HF_MODEL_ID':'TheBloke/Pygmalion-2-13B-GPTQ',
	'SM_NUM_GPUS': json.dumps(NUM_GPUS),
    'QUANTIZE': QUANTIZATION,
    'NUM_SHARD':json.dumps(NUM_SHARD),
    'GPTQ_BITS': json.dumps(NUM_BITS),
    'GPTQ_GROUPSIZE': json.dumps(GROUP_SIZE),
    'REVISION': REVISION
}
llm_image = get_huggingface_llm_image_uri("huggingface", version="1.0.3")
print(f"llm image uri: {llm_image}")

# create Hugging Face Model Class
llm_model = HuggingFaceModel(
	image_uri=llm_image,
	env=configs,
	role=SAGEMAKER_ROLE_ARN, 
)

# # deploy model to SageMaker Inference
predictor = llm_model.deploy(
	initial_instance_count=NUM_DEPLOYED_INSTANCE,
	instance_type=INSTANCE_TYPE,
	container_startup_health_check_timeout=300,
    endpoint_name=SAGEMAKER_ENDPOINT_NAME
  )

print(f"Endpoint name: {predictor.endpoint_name}")

test_prompt = """<|system|>Enter RP mode. Pretend to be {{char}} whose persona follows:
{{persona}}

You shall reply to the user while staying in character, and generate long responses."""

# hyperparameters for llm
payload = {
  "inputs": test_prompt.format(char="Pygmalion", persona="I"),
}

# send request to endpoint
response = predictor.predict(payload)

print(response[0]["generated_text"])