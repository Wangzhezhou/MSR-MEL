import os
import json
import numpy as np
import torch
import clip
from tqdm import tqdm
from openai import OpenAI
import time
import random


MY_API_KEY =

MODEL_NAME = "gpt-4o-mini" 

DATASET_DIR = "LLM_res_Rich"
os.makedirs(DATASET_DIR, exist_ok=True)


OUTPUT_TEXT_FILE = os.path.join(DATASET_DIR, "llm_entity_desc_4k.json")
OUTPUT_EMB_FILE = os.path.join(DATASET_DIR, "llm_entity_emb_4k.npy")

INDICES_FILE = os.path.join(DATASET_DIR, "target_entity_indices_4k.json")  
SUCCESS_LOG = os.path.join(DATASET_DIR, "success_entity_indices.json")        
FAIL_LOG = os.path.join(DATASET_DIR, "failed_entity_indices.json")          


TARGET_COUNT = 4000 

class LLMWorker:
    def __init__(self):
        self.client = OpenAI(api_key=MY_API_KEY)

    def generate_entity_enhancement(self, entity_data, retries=3):
       
        name = entity_data.get("entity_name", "Unknown")
        instance_type = entity_data.get("instance", "Unknown")
        attributes = entity_data.get("attr", "")
        
        attr_str = attributes if attributes and len(attributes) > 2 else "No specific attributes provided."

        prompt_content = (
            f"Task: Entity Description Generation for Knowledge Graph.\n"
            f"Target Entity: \"{name}\"\n"
            f"Entity Type: {instance_type}\n"
            f"Known Attributes: \"{attr_str}\"\n\n"
            f"Instructions:\n"
            f"1. You are an expert encyclopedia editor. Use your internal knowledge to identify this specific entity based on the name and attributes.\n"
            f"2. Write a concise, comprehensive definition (30-50 words).\n"
            f"3. [CRITICAL] Do NOT simply list the attributes. Combine them with specific facts (e.g., famous works, historical significance) to create a dense semantic representation.\n" 
            f"4. If the entity is ambiguous, use the attributes to pick the correct one.\n"
            f"5. Output Format: [Canonical Name]: [Definition]\n"
        )

        for i in range(retries):
            try:
                completion = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for Knowledge Representation."},
                        {"role": "user", "content": prompt_content}
                    ],
                    max_tokens=100,
                    temperature=0.3 
                )
                
                content = completion.choices[0].message.content
                if content:
                    return content.strip()
                
            except Exception as e:
                print(f"\n[Error] API failed: {e}. Retrying ({i+1}/{retries})...")
                time.sleep(2)
        
        return None

def load_clip_model(device):
    print("Loading CLIP model (ViT-B/32)...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model

def encode_text_batch(model, text_list, device, batch_size=256):
    model.eval()
    all_features = []
    processed_texts = [t[:300] for t in text_list]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="Encoding Entity Texts"):
            batch_texts = processed_texts[i : i + batch_size]
            tokens = clip.tokenize(batch_texts, truncate=True).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=1, keepdim=True)
            all_features.append(features.cpu().numpy())
            
    if all_features:
        return np.vstack(all_features)
    else:
        return np.empty((0, 512))

def get_target_indices(total_count, count, save_path):
    if os.path.exists(save_path):
        print(f"Loading existing target indices from {save_path}")
        with open(save_path, 'r') as f:
            indices = json.load(f)
        return indices
    else:
        print(f"Generating NEW random indices ({count} samples)...")
        # 防止 count 大于 total_count
        actual_count = min(count, total_count)
        indices = random.sample(range(total_count), actual_count)
        with open(save_path, 'w') as f:
            json.dump(indices, f)
        return indices

def main():
    kb_path = os.path.join("../RichpediaMEL", "kb_entity.json")
    if not os.path.exists(kb_path):
        print(f"Error: KB file not found at {kb_path}")
        return

    print(f"Loading entities from {kb_path}...")
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    
    total_kb_count = len(kb)
    print(f"Total entities in KB: {total_kb_count}")

    target_indices = get_target_indices(total_kb_count, TARGET_COUNT, INDICES_FILE)
    
    generated_cache = {} 
    if os.path.exists(OUTPUT_TEXT_FILE):
        with open(OUTPUT_TEXT_FILE, 'r') as f:
            generated_cache = json.load(f)
        print(f"Loaded {len(generated_cache)} cached descriptions.")

    todo_indices = [idx for idx in target_indices if str(idx) not in generated_cache]
    print(f"Need to generate: {len(todo_indices)}")

    if len(todo_indices) > 0:
        llm = LLMWorker()
        dirty_flag = False
        
        print("Starting LLM Generation...")
        for idx in tqdm(todo_indices):
            key = str(idx)
            item = kb[idx]
            
            desc = llm.generate_entity_enhancement(item)
            
            if desc:
                generated_cache[key] = desc
                dirty_flag = True
            else:
                print(f"[Warning] Generation failed for Entity ID {idx}")
            
            if dirty_flag and (len(generated_cache) % 50 == 0):
                with open(OUTPUT_TEXT_FILE, 'w') as f:
                    json.dump(generated_cache, f, ensure_ascii=False)
                dirty_flag = False
        
        with open(OUTPUT_TEXT_FILE, 'w') as f:
            json.dump(generated_cache, f, ensure_ascii=False)

    print("LLM Generation Step Finished.")


    print("Constructing Pure LLM Entity Embeddings...")
    
    texts_to_encode = []
    indices_to_update = [] 
    
    success_indices = []
    failed_indices = []

    for idx in target_indices:
        text = generated_cache.get(str(idx), "")
        if text and len(text) > 5:
            texts_to_encode.append(text)
            indices_to_update.append(idx)
            success_indices.append(idx)
        else:
            failed_indices.append(idx)
            
    print(f"Ready to encode: {len(texts_to_encode)} entities.")
    print(f"Failed/Skipped: {len(failed_indices)} entities.")

    with open(SUCCESS_LOG, 'w') as f:
        json.dump(success_indices, f)
    with open(FAIL_LOG, 'w') as f:
        json.dump(failed_indices, f)
    print(f"Logs saved to {SUCCESS_LOG} and {FAIL_LOG}")


    final_emb = np.zeros((total_kb_count, 1, 512), dtype=np.float32)

    if len(texts_to_encode) > 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = load_clip_model(device)
        
        llm_vectors = encode_text_batch(clip_model, texts_to_encode, device)
        
        llm_vectors_expanded = llm_vectors[:, np.newaxis, :]
        
        final_emb[indices_to_update] = llm_vectors_expanded
        print(f"Populated {len(indices_to_update)} rows in the final matrix.")

    np.save(OUTPUT_EMB_FILE, final_emb)
    print(f"Done! Saved Pure LLM Entity Embeddings to {OUTPUT_EMB_FILE}")
    print(f"Matrix Shape: {final_emb.shape} (Sparse-like, mostly zeros)")

if __name__ == "__main__":
    main()