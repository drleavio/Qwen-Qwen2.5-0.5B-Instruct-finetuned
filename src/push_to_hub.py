from huggingface_hub import HfApi, login
from transformers import AutoTokenizer

import os

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNED_LORA_PATH = os.path.join(BASE_DIR, "models", "qwen-0.5b-finetuned")

# TODO: Replace with your actual Hugging Face username!
HF_USERNAME = "your-hf-username"
MODEL_NAME = "qwen-0.5b-eu-ai-act-classifier"
# TODO: Replace with your Hugging Face Write Token!
HF_TOKEN = "your_hf_token_here"

def main():
    repo_id = f"{HF_USERNAME}/{MODEL_NAME}"
    
    print("Logging into Hugging Face...")
    login(token=HF_TOKEN)
    
    print(f"Loading tokenizer from {FINETUNED_LORA_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_LORA_PATH, trust_remote_code=True)
    
    print(f"Pushing tokenizer to {repo_id}...")
    tokenizer.push_to_hub(repo_id)
    
    print(f"Pushing LoRA adapters to {repo_id}...")
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    
    api.upload_folder(
        folder_path=FINETUNED_LORA_PATH,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Successfully pushed model and tokenizer to https://huggingface.co/{repo_id}")
    print(f"Successfully pushed model and tokenizer to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
