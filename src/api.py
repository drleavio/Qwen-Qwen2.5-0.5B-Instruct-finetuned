import torch
import faiss
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from pymongo import MongoClient

load_dotenv()
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer

import os

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED_LORA_PATH = os.path.join(BASE_DIR, "models", "qwen-0.5b-finetuned")
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index.bin")
MAPPING_PATH = os.path.join(BASE_DIR, "data", "faiss_mapping.pkl")

model = None
tokenizer = None
embedder = None
faiss_index = None
faiss_mapping = None
mongo_client = None
db_collection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global model, tokenizer, embedder, faiss_index, faiss_mapping, mongo_client, db_collection
    
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        db_collection = mongo_client["eu_ai_act_db"]["queries"]
        print("Connected to MongoDB.")
    except Exception as e:
        print(f"Warning: Failed to connect to MongoDB: {e}")

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
    name: str
    email: str
    questionnaire: str
    requirements: str
    domain: str = ""
    use_case_description: str = ""
    temperature: float = 0.1
    max_tokens: int = 256

class QueryResponse(BaseModel):
    response: str

def send_email_report(to_address: str, report_content: str):
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    sender_email = os.getenv("SENDER_EMAIL", "your_email@gmail.com")
    sender_password = os.getenv("SENDER_PASSWORD", "your_password")
    
    if sender_password == "your_password" or not sender_password:
        print("Skipping email sending: Please update SENDER_PASSWORD in .env")
        return
        
    msg = MIMEText(report_content)
    msg['Subject'] = 'Your EU AI Act Classification Report'
    msg['From'] = sender_email
    msg['To'] = to_address
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(f"Report emailed to {to_address}")
    except Exception as e:
        print(f"Failed to send email to {to_address}: {e}")

@app.post("/classify", response_model=QueryResponse)
def classify_use_case(req: QueryRequest, background_tasks: BackgroundTasks):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Models are not loaded yet.")
    
    full_use_case = f"Name: {req.name}\nEmail: {req.email}\nQuestionnaire: {req.questionnaire}\nRequirements: {req.requirements}"
    if req.domain or req.use_case_description:
        full_use_case += f"\nDomain: {req.domain}\nDescription: {req.use_case_description}"
    
    rag_context = ""
    if embedder and faiss_index and faiss_mapping:
        # 1. Embed the user's query
        query_emb = embedder.encode([full_use_case], convert_to_numpy=True)
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
    user_prompt = f"{rag_context}Now classify this new use case:\n{full_use_case}"
    
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
    
    # Save to MongoDB
    if db_collection is not None:
        try:
            record = req.dict()
            record["response"] = response_text
            db_collection.insert_one(record)
        except Exception as e:
            print(f"Failed to insert record to MongoDB: {e}")
            
    # Send email in background
    background_tasks.add_task(send_email_report, req.email, response_text)
    
    return QueryResponse(response=response_text)

if __name__ == "__main__":
    print("Starting API Server...")
    # Hosts on 0.0.0.0 (all interfaces) on port 8080
    uvicorn.run("api:app", host="0.0.0.0", port=3001, reload=False)
