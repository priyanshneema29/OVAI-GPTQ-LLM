# Prerequisite
- Python >= 3.8

# Setup Environment
## Setup AWS Env
- Install aws cli based on this link: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
- Setup the aws credential in your env, run:
    ```
    aws configure
    ```
- Get the AWS Access Key and Secret access key from .env file for the configuration

## Setup develop Env
- **Notes**: Recommend setup with conda environment
- Install necessary library:
    ```
    pip install -r requirements.txt
    ```
- Install autogptq from source, this one is compatible with pytorch 2 with cuda 11.7
    ```
    pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/
    ```

**Notes**: Check configs.py for every configuration parameters
# Training
## Prepare dataset
- Get dataset from hugging face hub, run:
    ```
    python get_dataset.py
    ```
- Check the preprocess_data.py and change the prompt template that you want
## Train
- Run with default configuration:
    ```
    python train.py
    ```
- Run with customize parameters
    ```
    python train.py <flags>
    ```
    - FLAGS:
    ```
    -b, --base_model=BASE_MODEL
        Type: str
        Default: 'TheBloke/Pygmalion-...
    --data_path=DATA_PATH
        Type: str
        Default: 'data'
    --dataset_name=DATASET_NAME
        Type: str
        Default: 'lonestar108_enlightene...
    -o, --output_dir=OUTPUT_DIR
        Type: str
        Default: 'outputs'
    -m, --micro_batch_size=MICRO_BATCH_SIZE
        Type: int
        Default: 2
    -g, --gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
        Type: int
        Default: 2
    -n, --num_epochs=NUM_EPOCHS
        Type: int
        Default: 1
    --learning_rate=LEARNING_RATE
        Type: float
        Default: 0.0002
    --lora_r=LORA_R
        Type: int
        Default: 8
    --lora_alpha=LORA_ALPHA
        Type: int
        Default: 16
    --lora_dropout=LORA_DROPOUT
        Type: float
        Default: 0.05
    --lora_target_modules=LORA_TARGET_MODULES
        Type: typing.List[str]
        Default: ['q_proj', 'v_proj']
    ```
## Merge lora adapter
- Run:
    ```
    upython merge_lora_adapter.py [-h] [--base_model_name_or_path BASE_MODEL_NAME_OR_PATH] [--peft_model_path PEFT_MODEL_PATH] [--output_dir OUTPUT_DIR] [--device DEVICE] [--push_to_hub]

    optional arguments:
    -h, --help            show this help message and exit
    --base_model_name_or_path BASE_MODEL_NAME_OR_PATH
    --peft_model_path PEFT_MODEL_PATH
    --output_dir OUTPUT_DIR
    --device DEVICE
    --push_to_hub
    ```

# Deployment
## Run deployment
- Check and modify the 'HF_MODEL_ID' in configs from deploy.py to the hugging face model id you want
- **Notes**: The script right now only support for GPTQ model
- Run:
    ```
    python deploy.py
    ```
## Test deployment
- You can quickly check the deployed endpoint with:
    ```
    python test_deploy.py
    ```
- Then, you can test it further with the gradio ui:
    ```
    gradio test_ui.py
    ```
