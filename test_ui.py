import gradio as gr
import boto3
import json
from utils import LineIterator
from configs import SAGEMAKER_ENDPOINT_NAME, INFERENCE_PARAMETERS

# ref https://pygmalionai.github.io/blog/posts/introducing_pygmalion_2/
user = "Son"
char = "Captain America"
system_prompt = """Enter RP mode. You shall reply to {{user}} while staying in character. 
                    Your responses must be detailed, creative, immersive, and drive the scenario forward.
                    You will follow {{char}}'s persona.""".format(user=user, char=char)
# system_prompt = """Enter RP mode. Your responses must be detailed, creative, immersive, and drive the scenario forward."""

# helper method to format prompt
def format_prompt(message, system_prompt):
    prompt = ""
    if system_prompt:
        prompt += f"{system_prompt}\n"

    prompt += f"""<|user|> {message}"""
    return prompt

def create_gradio_app(session=boto3, 
                      parameters=INFERENCE_PARAMETERS,
                      system_prompt=system_prompt,
                      concurrency_count=4,
                      share=True
                      ):
    
    smr = session.client("sagemaker-runtime")

    def generate(
        prompt,
        history
    ):
        formatted_prompt = format_prompt(prompt, system_prompt)

        request = {"inputs": formatted_prompt, "parameters": parameters, "stream": True}
        resp = smr.invoke_endpoint_with_response_stream(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            Body=json.dumps(request),
            ContentType="application/json",
        )

        output = ""
        event_stream = resp['Body']
        start_json = b'{'
        for line in LineIterator(event_stream):
            if line != b'' and start_json in line:
                data = json.loads(line[line.find(start_json):].decode('utf-8'))
                if data['token']['text'] not in parameters["stop"]:
                    output += data['token']['text']

                yield output
        return output

    with gr.Blocks() as demo:
        gr.Markdown("## Chat with Amazon SageMaker")
        gr.ChatInterface(
            generate,
        )

    demo.queue(concurrency_count=concurrency_count).launch(share=share)
    
create_gradio_app(share=False)