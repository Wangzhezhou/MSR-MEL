import os
import json
import numpy as np
import torch
import clip
from tqdm import tqdm
from openai import OpenAI
import time


MY_API_KEY = "" 

MODEL_NAME = "gpt-4o-mini" 

DATASET_DIR = "LLM_res_Rich" 
os.makedirs(DATASET_DIR, exist_ok=True)

OUTPUT_TEXT_FILE = os.path.join(DATASET_DIR, "mention_llm_desc.json")
OUTPUT_EMB_FILE = os.path.join(DATASET_DIR, "mention_llm_emb.npy")
OUTPUT_FAIL_LOG = os.path.join(DATASET_DIR, "mention_llm_failed_indices.json")

# ===========================================

class LLMWorker:
    def __init__(self):
        self.client = OpenAI(api_key=MY_API_KEY)

    def generate_positive_pair_content(self, mention, context, retries=3):
      
        prompt_content = (
            f"Task: Entity Linking Positive Pair Generation.\n"
            f"Context: \"{context}\"\n"
            f"Target Mention: \"{mention}\"\n\n"
            f"Instructions:\n"
            f"1. Identify the unique, canonical entity this mention refers to.\n"
            f"2. Provide the Canonical Name of the entity.\n"
            f"3. Provide a concise but comprehensive definition (30-50 words) that uniquely identifies this entity globally, independent of the current context.\n"
            f"4. Format: [Canonical Name]: [Definition]\n"
        )

        for i in range(retries):
            try:
                completion = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an expert in Entity Linking and Knowledge Representation."},
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
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="Encoding LLM Texts"):
            batch_texts = processed_texts[i : i + batch_size]

            tokens = clip.tokenize(batch_texts, truncate=True).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=1, keepdim=True)
            all_features.append(features.cpu().numpy())
            
    if all_features:
        return np.vstack(all_features)
    else:
        return np.empty((0, 512))

def main():
    test_path = os.path.join("../RichpediaMEL", "RichpediaMEL_test.json")
    if not os.path.exists(test_path):
        print(f"Error: Dataset not found at {test_path}")
        return

    print(f"Loading mentions from {test_path}...")
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    total_count = len(test_data)
    print(f"Total mentions to process: {total_count}")

    generated_cache = {} 
    if os.path.exists(OUTPUT_TEXT_FILE):
        with open(OUTPUT_TEXT_FILE, 'r') as f:
            generated_cache = json.load(f)
        print(f"Loaded {len(generated_cache)} cached descriptions.")

    todo_indices = [i for i in range(total_count) if str(i) not in generated_cache]
    print(f"Need to generate: {len(todo_indices)}")

    if len(todo_indices) > 0:
        llm = LLMWorker()
        dirty_flag = False
        
        print("Starting LLM Generation for Positive Pairs...")
        for idx in tqdm(todo_indices):
            key = str(idx)
            item = test_data[idx]
            
            mention_text = item.get('mentions', '')
            context_text = item.get('sentence', '')
            
            desc = llm.generate_positive_pair_content(mention_text, context_text)
            
            if desc:
                generated_cache[key] = desc
                dirty_flag = True
            else:
                print(f"\n[Warning] Generation failed for index {idx}")
            
            if dirty_flag and (len(generated_cache) % 50 == 0):
                with open(OUTPUT_TEXT_FILE, 'w') as f:
                    json.dump(generated_cache, f, ensure_ascii=False)
                dirty_flag = False

        with open(OUTPUT_TEXT_FILE, 'w') as f:
            json.dump(generated_cache, f, ensure_ascii=False)

    print("LLM Generation Step Finished.")

    print("Constructing Pure LLM Embeddings...")
    

    texts_to_encode = []
    indices_to_update = []
    failed_indices = []

    for idx in range(total_count):
        text = generated_cache.get(str(idx), "")

        if text and isinstance(text, str) and len(text) > 5:
            texts_to_encode.append(text)
            indices_to_update.append(idx)
        else:
            failed_indices.append(idx)
    
    print(f"Ready to encode: {len(texts_to_encode)} items.")
    print(f"Failed/Missing: {len(failed_indices)} items (These will be Zero Vectors).")
    
    if failed_indices:
        with open(OUTPUT_FAIL_LOG, 'w') as f:
            json.dump(failed_indices, f)
        print(f"Failed indices saved to {OUTPUT_FAIL_LOG}")


    final_emb = np.zeros((total_count, 1, 512), dtype=np.float32)

    if len(texts_to_encode) > 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = load_clip_model(device)
        
        llm_vectors = encode_text_batch(clip_model, texts_to_encode, device)
        
        llm_vectors_expanded = llm_vectors[:, np.newaxis, :]

        final_emb[indices_to_update] = llm_vectors_expanded
        print(f"Successfully populated {len(indices_to_update)} embeddings.")

    np.save(OUTPUT_EMB_FILE, final_emb)
    print(f"Done! Saved Pure LLM Mention Embeddings to {OUTPUT_EMB_FILE}")
    print(f"Output Shape: {final_emb.shape}")

if __name__ == "__main__":
    main()