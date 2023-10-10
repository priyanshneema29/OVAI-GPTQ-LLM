import argparse
import os
import time
import torch
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, GPTQConfig, Trainer,
                          TrainingArguments, pipeline)
from preprocess_data import format_prompts_inference

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()

def main():
    args = get_args()

    if args.device == 'auto':
        device_arg = { 'device_map': 'auto' }
    else:
        device_arg = { 'device_map': { "": args.device} }

    print(f"Loading base model: {args.base_model_name_or_path}")
    quantization_config_loading = GPTQConfig(bits=4, disable_exllama=False)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        **device_arg,
        quantization_config=quantization_config_loading
    )

    print(f"Loading PEFT: {args.peft_model_path}")
    lora_model = PeftModel.from_pretrained(base_model, args.peft_model_path, **device_arg)
    print(f"Running merge_and_unload")
    # lora_model = lora_model.merge_and_unload()
    lora_model.train(False)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    
    # Run text generation pipeline with our next model
    prompt = ["What is the difference between Nirvana and the world?", "What is Nirvana?"]
    prompt_formatted = format_prompts_inference(prompt)
    
    print("Start warmup...")
    start = time.time()
    pipe = pipeline(task="text-generation", model=lora_model, tokenizer=tokenizer, max_length=200)
    results = pipe(prompt_formatted)
    print()
    print("Output: ")
    for result in results:
        print(result)
        print()
    print("Process warmup time: ", time.time() - start)
    
    start_time = time.time()
    results = pipe(prompt_formatted)
    output_time = time.time() - start_time
    print("Output time: ", output_time)
    
if __name__ == "__main__" :
    main()