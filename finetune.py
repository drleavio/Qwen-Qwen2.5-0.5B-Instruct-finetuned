import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# Configuration
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "dataset.jsonl"
OUTPUT_DIR = "./qwen-0.5b-finetuned"

def format_instruction(example):
    """
    Formats the dataset row into the prompt structure expected by the model.
    Adjust the keys below if you want to include more information (e.g., 'Unnamed: 3').
    """
    # The first line of the JSONL appears to be a header, and subsequent lines are data.
    # 'Unnamed: 1' contains the Use case description (Input)
    # 'Unnamed: 2' contains the Risk Classification (Target)
    # 'Unnamed: 3' contains the Relevant Articles (Target)
    # 'Unnamed: 4' contains the Target Annex / Clause
    # 'Unnamed: 5' contains the Target Rationale
    # 'Unnamed: 7' contains the Source URL(s)
    
    input_text = example.get("Unnamed: 1", "")
    classification = example.get("Unnamed: 2", "")
    articles = example.get("Unnamed: 3", "")
    annex = example.get("Unnamed: 4", "")
    rationale = example.get("Unnamed: 5", "")
    urls = example.get("Unnamed: 7", "")
    
    target_text = (
        f"Risk Classification: {classification}\n"
        f"Relevant Articles: {articles}\n"
        f"Annex / Clause: {annex}\n"
        f"Rationale: {rationale}\n"
        f"Source URLs: {urls}"
    )
    
    # Qwen ChatML format
    text = f"<|im_start|>system\nYou are a helpful assistant classifying AI systems under the EU AI Act.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{target_text}<|im_end|>"
    return {"text": text}

def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    # Load the JSONL dataset
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # Filter out the header row if present
    dataset = dataset.filter(lambda x: x.get("Unnamed: 1") != "Input Use Case Description")
    
    # Map the dataset to the required format
    dataset = dataset.map(format_instruction)
    
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Configuring QLoRA...")
    # Using 4-bit quantization since it's a 14B model (requires ~10-14GB VRAM for training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config targeting linear layers
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Training arguments (SFTConfig extends TrainingArguments with SFT-specific fields)
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,          # Adjust depending on VRAM
        gradient_accumulation_steps=4,          # Adjust depending on VRAM
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,                              # Use bf16 if supported by your GPU (Ampere+), else use fp16=True
        max_grad_norm=0.3,
        num_train_epochs=3,                     # Number of epochs
        warmup_steps=10,
        lr_scheduler_type="cosine",
        report_to="none",                       # Change to "wandb" to track metrics
        dataset_text_field="text",
        max_length=2048,
    )
    
    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
