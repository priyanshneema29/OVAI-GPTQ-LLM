SYSTEM_TOKEN = "<|system|>"
USER = "student"
SYSTEM_PROMPT_TEMPLATE = f"""Enter RP mode. Pretend to be an Enlightened teacher. You shall reply to {USER} while staying in character. 
                            Your responses must be detailed, creative, immersive, and drive the scenario forward"""
INPUT_TOKEN = "<|user|>"
OUTPUT_TOKEN = "<|model|>"

def formatting_prompts_pygmalion2(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        response = examples["output"][i]
        text = f"{SYSTEM_TOKEN} {SYSTEM_PROMPT_TEMPLATE} {INPUT_TOKEN} {instruction} {OUTPUT_TOKEN} {response}"
        
        output_text.append(text)
        
    print("Sample prompt:")
    print(output_text[0])
    
    return output_text