import os
from typing import List

import fire
import pandas as pd
import requests
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, GPTQConfig, Trainer,
                          TrainingArguments)
from trl import SFTTrainer

from preprocess_data import formatting_prompts_pygmalion2

# saving callback to automatically save the adapters only
# class PeftSavingCallback(TrainerCallback):
#     def on_save(self, args, state, control, **kwargs):
#         checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
#         kwargs["model"].save_pretrained(checkpoint_path)

#         if "pytorch_model.bin" in os.listdir(checkpoint_path):
#             os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))
            
def train(
    # model/data params
    base_model: str = "TheBloke/Pygmalion-2-13B-GPTQ", 
    data_path: str = "data",
    dataset_name: str = "lonestar108_enlightenedllm",
    output_dir: str = "outputs",
    micro_batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    num_epochs: int = 1,
    learning_rate: float = 2e-4,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ]
):
    device_map = "auto"
    # Step 1: Load the model and tokenizer
    quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                 quantization_config=quantization_config_loading,
                                                 device_map=device_map)
    
    print("Quantization config:\n")
    print(model.config.quantization_config.to_dict())
    # prepare model for training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Add this for training LoRA
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    print(model.print_trainable_parameters())
    
    # Step 2: Load the dataset
    train_data = Dataset.from_csv(os.path.join(data_path, f"{dataset_name}_train_subset.csv"))
    validation_data = Dataset.from_csv(os.path.join(data_path, f"{dataset_name}_validation.csv"))
    # test_data = Dataset.from_csv(os.path.join(data_path, f"{dataset_name}_test.csv"))

    # Step 3: Initiate the trainer parameters
    print("Configure training parameters...")
    training_params = TrainingArguments(
                        per_device_train_batch_size=micro_batch_size,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        warmup_steps=10,
                        num_train_epochs=num_epochs,
                        learning_rate=learning_rate,
                        fp16=True,
                        logging_steps=10,
                        optim="adamw_hf",
                        evaluation_strategy="steps",
                        save_strategy="steps",
                        eval_steps=100,
                        save_steps=100,
                        output_dir=output_dir,
                        save_total_limit=3
                        # report_to="tensorboard"
                    )
    
    # Step 4: Initiate the trainer 
    print("Initiate the trainer...")
    trainer = SFTTrainer(
                model=model,
                args=training_params,
                train_dataset=train_data,
                eval_dataset=validation_data,
                peft_config=config,
                tokenizer=tokenizer,
                formatting_func=formatting_prompts_pygmalion2,
                packing=False,
                max_seq_length=1024
                )
    print("Start training...")
    trainer.train()

    # Step 5: save the model
    print("Save the model...")
    # Save trained model
    trainer.model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)


