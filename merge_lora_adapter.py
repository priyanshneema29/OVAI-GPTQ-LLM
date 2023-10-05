from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--push_to_hub", action="store_true")

    return parser.parse_args()

def main():
    args = get_args()

    if args.device == 'auto':
        device_arg = { 'device_map': 'auto' }
    else:
        device_arg = { 'device_map': { "": args.device} }

    print(f"Loading base model: {args.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        **device_arg
    )
    
    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    print(f"Loading PEFT: {args.peft_model_path}")
    lora_model = PeftModel.from_pretrained(base_model, args.peft_model_path, **device_arg)
    
    lora_weight = lora_model.base_model.model.model.layers[
        0
    ].self_attn.q_proj.weight
    # Check the q_proj matrix is the same
    assert torch.allclose(first_weight_old, first_weight)

    print(f"Running merge_and_unload")
    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)

    # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    # Save to huggingface hub
    if args.push_to_hub:
        print(f"Saving to hub ...")
        lora_model.push_to_hub(f"{args.output_dir}", use_temp_dir=False)
        tokenizer.push_to_hub(f"{args.output_dir}", use_temp_dir=False)
    else:
        lora_model.save_pretrained(f"{args.output_dir}")
        tokenizer.save_pretrained(f"{args.output_dir}")
        print(f"Model saved to {args.output_dir}")

if __name__ == "__main__" :
    main()