import torch
import faiss
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer

# Configuration
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED_LORA_PATH = "./qwen-0.5b-finetuned"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index.bin"
MAPPING_PATH = "faiss_mapping.pkl"

model = None
tokenizer = None
embedder = None
faiss_index = None
faiss_mapping = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global model, tokenizer, embedder, faiss_index, faiss_mapping
    
    try:
        print("Loading RAG components (FAISS index & embedding model)...")
        faiss_index = faiss.read_index(INDEX_PATH)
        with open(MAPPING_PATH, 'rb') as f:
            faiss_mapping = pickle.load(f)
        embedder = SentenceTransformer(EMBEDDING_MODEL_ID)
    except Exception as e:
        print(f"Warning: Failed to load RAG components. Make sure you ran build_index.py. Error: {e}")
        
    print(f"Loading base model: {BASE_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_LORA_PATH, trust_remote_code=True)
    
    print(f"Loading LoRA adapters from {FINETUNED_LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, FINETUNED_LORA_PATH)
    model.eval()
    print("Models loaded successfully and API is ready to receive requests!")
    yield
    # Shutdown logic (not needed here)

app = FastAPI(title="EU AI Act Classifier API", lifespan=lifespan)

class QueryRequest(BaseModel):
    use_case: str
    temperature: float = 0.1
    max_tokens: int = 256

class QueryResponse(BaseModel):
    response: str

@app.post("/classify", response_model=QueryResponse)
def classify_use_case(req: QueryRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Models are not loaded yet.")
    
    rag_context = ""
    if embedder and faiss_index and faiss_mapping:
        # 1. Embed the user's query
        query_emb = embedder.encode([req.use_case], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        # 2. Search FAISS for top 3 matches
        k = 3
        distances, indices = faiss_index.search(query_emb, k)
        
        # 3. Construct RAG context string
        retrieved_examples = []
        for idx in indices[0]:
            if idx != -1:
                match = faiss_mapping[idx]
                retrieved_examples.append(f"Similar Use Case: {match['use_case']}\nClassification:\n{match['target']}")
                
        if retrieved_examples:
            rag_context = "Here are some similar examples from the EU AI Act:\n\n" + "\n\n---\n\n".join(retrieved_examples) + "\n\n=====\n\n"
            
    system_prompt = "You are a helpful assistant classifying AI systems under the EU AI Act."
    user_prompt = f"{rag_context}Now classify this new use case:\n{req.use_case}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Apply the ChatML template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=True if req.temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the generated tokens
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return QueryResponse(response=response_text)

if __name__ == "__main__":
    print("Starting API Server...")
    # Hosts on 0.0.0.0 (all interfaces) on port 8080
    uvicorn.run("api:app", host="0.0.0.0", port=3001, reload=False)
