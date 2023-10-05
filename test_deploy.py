import boto3
import json
import io
from sagemaker.base_deserializers import StreamDeserializer

from configs import SAGEMAKER_ENDPOINT_NAME
from utils import LineIterator

body = {
    "inputs":"""Enter RP mode. You shall reply to {{user}} while staying in character. Your responses must be detailed, creative, immersive, and drive the scenario forward. 
                What is AWS Sagemaker? Explain in detail.""",
    "parameters":{
        "return_full_text": False
    },
    "stream": True
}

smr = boto3.client('sagemaker-runtime')
# ref https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpointWithResponseStream.html
resp = smr.invoke_endpoint_with_response_stream(EndpointName=SAGEMAKER_ENDPOINT_NAME,
                                                Body=json.dumps(body), ContentType="application/json")

event_stream = resp['Body']
start_json = b'{'
for line in LineIterator(event_stream):
    if line != b'' and start_json in line:
        data = json.loads(line[line.find(start_json):].decode('utf-8'))
        if data['token']['text'] != "<|user|>":
            print(data['token']['text'],end='')