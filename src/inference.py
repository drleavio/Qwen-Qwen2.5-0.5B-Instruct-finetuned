import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import os

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED_LORA_PATH = os.path.join(BASE_DIR, "models", "qwen-0.5b-finetuned")

def main():
    print(f"Loading base model: {BASE_MODEL_ID}")
    # Load the base model in half-precision for faster inference and lower memory usage
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print(f"Loading tokenizer...")
    # Load the tokenizer from the fine-tuned directory
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_LORA_PATH, trust_remote_code=True)
    
    print(f"Loading LoRA adapters from {FINETUNED_LORA_PATH}...")
    # Wrap the base model with the fine-tuned LoRA adapters
    model = PeftModel.from_pretrained(base_model, FINETUNED_LORA_PATH)
    model.eval()
    
    print("\n" + "="*50)
    print("Model is ready! Type your prompt below (or 'quit' to exit).")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input.strip():
                continue
                
            messages = [
                {"role": "system", "content": "You are a helpful assistant classifying AI systems under the EU AI Act."},
                {"role": "user", "content": user_input}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            print("\n" + "="*50)
            print("MODEL RESPONSE:")
            print(response)
            print("="*50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
