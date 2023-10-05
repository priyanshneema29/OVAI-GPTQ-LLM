from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access variables
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Deployment configuration
# Ref: table in https://huggingface.co/TheBloke/Pygmalion-2-13B-GPTQ
NUM_GPUS = 1
QUANTIZATION = "gptq"
NUM_SHARD = 1   
NUM_BITS = 4
GROUP_SIZE = 128
REVISION = "main" # branch of repo, related with HF_MODEL_ID 

INSTANCE_TYPE = "ml.g5.2xlarge"
NUM_DEPLOYED_INSTANCE = 1
SAGEMAKER_ENDPOINT_NAME = "pygmalion-2-13b-gptq"

## Define inference parameters for sagemaker endpoint
# temperature: Controls randomness in the model. Lower values will make the model more deterministic and higher values will make the model more random. Default value is 1.0.
# max_new_tokens: The maximum number of tokens to generate. Default value is 20, max value is 512.
# repetition_penalty: Controls the likelihood of repetition, defaults to null.
# seed: The seed to use for random generation, default is null.
# stop: A list of tokens to stop the generation. The generation will stop when one of the tokens is generated.
# top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering. Default value is null, which disables top-k-filtering.
# top_p: The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling, default to null
# do_sample: Whether or not to use sampling ; use greedy decoding otherwise. Default value is false.
# best_of: Generate best_of sequences and return the one if the highest token logprobs, default to null.
# details: Whether or not to return details about the generation. Default value is false.
# return_full_text: Whether or not to return the full text or only the generated part. Default value is false.
# truncate: Whether or not to truncate the input to the maximum length of the model. Default value is true.
# typical_p: The typical probability of a token. Default value is null.
# watermark: The watermark to use for the generation. Default value is false.
INFERENCE_PARAMETERS = {
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.8,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.03,
    "return_full_text": False,
    "stop": ["\n<|user|>", "<|endoftext|>", " <|user|>", "<|user|>"],
}

