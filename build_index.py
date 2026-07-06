import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

DATASET_PATH = "dataset.jsonl"
INDEX_PATH = "faiss_index.bin"
MAPPING_PATH = "faiss_mapping.pkl"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL_ID}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_ID)
    
    texts = []
    metadata = []
    
    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue # Skip header
            try:
                data = json.loads(line)
                use_case = data.get("Unnamed: 1", "")
                if use_case == "Input Use Case Description" or not use_case:
                    continue
                
                # Store the use case string to embed
                texts.append(use_case)
                
                # Store the target data as metadata
                classification = data.get("Unnamed: 2", "")
                articles = data.get("Unnamed: 3", "")
                rationale = data.get("Unnamed: 5", "")
                
                target_text = (
                    f"Risk Classification: {classification}\n"
                    f"Relevant Articles: {articles}\n"
                    f"Rationale: {rationale}"
                )
                
                metadata.append({
                    "use_case": use_case,
                    "target": target_text
                })
            except Exception as e:
                pass

    print(f"Generating embeddings for {len(texts)} use cases (this may take a minute)...")
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    print("Normalizing embeddings for cosine similarity search...")
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    # Use Inner Product (which equals cosine similarity since vectors are normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    print(f"Saving FAISS index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)
    
    print(f"Saving metadata mapping to {MAPPING_PATH}...")
    with open(MAPPING_PATH, 'wb') as f:
        pickle.dump(metadata, f)
        
    print("Done! RAG index built successfully.")

if __name__ == "__main__":
    main()
