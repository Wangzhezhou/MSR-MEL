import numpy as np
import json
from tqdm import tqdm

import json
import numpy as np
import faiss
from tqdm import tqdm
import difflib
import os


class SimpleReranker:

    def __init__(self):
        self.weights = np.array([
            0.20, 0.05, 0.05, 0.10,  # CLIP
            0.10, 0.02, 0.02, 0.05,  # GNN
            0.30,                    # String
            0.05, 0.05, 0.01,        # Stats
            0.00, 0.00               # Bias
        ], dtype=np.float32)
        
    def predict(self, features):
        return np.dot(features, self.weights)

class NeuroSymbolicCorrector:
    def __init__(self):
        # 8: Str (String Similarity)
        pass

    def apply(self, score, feats):
        feats = np.array(feats)
        triggered = []
        
        # Rule 1: Name Match Boosting
        if feats[8] >= 0.95:
            bonus = 0.05
            score += bonus
            triggered.append("name_match_boost")

        return score, triggered

    def solve_tie(self, candidates):
        """
        handle Tie-Breaking
        candidates: list of (idx, score, feats)
        """
        if len(candidates) < 2: return candidates, None
        
        top1 = candidates[0]
        top2 = candidates[1]
        
        # Margin < 0.005 
        if (top1[1] - top2[1]) < 0.005:
            if top2[2][8] > top1[2][8] + 0.2: 
                new_list = [top2, top1] + candidates[2:]
                return new_list, "tie_break_str"
                
        return candidates, None

def safe_dot(v1, v2):
    if v1 is None or v2 is None: return 0.0
    return float(np.dot(v1, v2))

def calculate_string_similarity(s1, s2):
    if not s1 or not s2: return 0.0
    return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


print("Loading data...")


GNN_ENTITY_TEXT_PATH = 'res_Wikimel/rebuild_res/entity_text_gnn.npy'    
GNN_ENTITY_IMG_PATH = 'res_Wikimel/rebuild_res/entity_image_gnn.npy'
GNN_MENTION_TEXT_PATH = 'res_Wikimel/rebuild_res/mention_text_gnn.npy' 
GNN_MENTION_IMG_PATH = 'res_Wikimel/rebuild_res/mention_image_gnn.npy'
parsed_text, mention_clip, kb_text, kb_img, Tm, kb = [], {}, [], {}, [], []
kb_gnn, kb_gnn_img_matrix, mention_gnn_text, mention_gnn_img = None, None, None, None

try:
    with open('../WikiMEL/WIKIMEL_test.json', 'r') as f: parsed_text = json.load(f)
    with open('../WikiMEL/mention_image_clip.json', 'r') as f: mention_clip = json.load(f)
    with open('../WikiMEL/entity_text_clip.json', 'r') as f: kb_text = json.load(f)
    with open('../WikiMEL/entity_image_clip.json', 'r') as f: kb_img = json.load(f)
    with open('../WikiMEL/mention_add_cap_clip.json', 'r') as f: Tm = json.load(f)
    with open("../WikiMEL/kb_entity.json", "r") as f: kb = json.load(f)
    
    if os.path.exists(GNN_ENTITY_TEXT_PATH):
        print(f"Loading GNN embeddings...")
        kb_gnn = np.load(GNN_ENTITY_TEXT_PATH).astype(np.float32)
        if os.path.exists(GNN_ENTITY_IMG_PATH):
            kb_gnn_img_matrix = np.load(GNN_ENTITY_IMG_PATH).astype(np.float32)
            if kb_gnn_img_matrix.ndim > 2:
                kb_gnn_img_matrix = kb_gnn_img_matrix.reshape(kb_gnn_img_matrix.shape[0], -1)
            faiss.normalize_L2(kb_gnn_img_matrix)
        mention_gnn_text = np.load(GNN_MENTION_TEXT_PATH).astype(np.float32)
        if os.path.exists(GNN_MENTION_IMG_PATH):
            mention_gnn_img = np.load(GNN_MENTION_IMG_PATH).astype(np.float32)
    else:
        raise FileNotFoundError("GNN files missing")

except Exception as e:
    print(f"Warning: Data loading failed ({e}), using Mock Data.")
    pass 

print("Building Indices...")
kb_text_flat = [np.array(item).astype(np.float32).flatten() for item in kb_text]
clip_matrix = np.array(kb_text_flat).astype(np.float32)
faiss.normalize_L2(clip_matrix)
index_clip_text = faiss.IndexFlatIP(clip_matrix.shape[1])
index_clip_text.add(clip_matrix)

if kb_gnn.ndim > 2: kb_gnn = kb_gnn.reshape(kb_gnn.shape[0], -1)
gnn_matrix = kb_gnn 
faiss.normalize_L2(gnn_matrix) 
index_gnn = faiss.IndexFlatIP(gnn_matrix.shape[1])
index_gnn.add(gnn_matrix)
print("Indices built.")


def extract_features(
    q_clip_t, q_clip_i, q_gnn_t, q_gnn_i,
    cand_clip_t, cand_clip_i, cand_gnn_t, cand_gnn_i,
    str_score, has_m_img, has_e_img
):
    s_c_tt = safe_dot(q_clip_t, cand_clip_t)
    s_c_tv = safe_dot(q_clip_t, cand_clip_i)
    s_c_vt = safe_dot(q_clip_i, cand_clip_t)
    s_c_vv = safe_dot(q_clip_i, cand_clip_i)
    
    s_g_tt = safe_dot(q_gnn_t, cand_gnn_t)
    s_g_tv = safe_dot(q_gnn_t, cand_gnn_i)
    s_g_vt = safe_dot(q_gnn_i, cand_gnn_t)
    s_g_vv = safe_dot(q_gnn_i, cand_gnn_i)
    
    sims = [s_c_tt, s_c_tv, s_c_vt, s_c_vv, s_g_tt, s_g_tv, s_g_vt, s_g_vv]
    sim_mean = np.mean(sims)
    sim_max = np.max(sims)
    
    sorted_sims = sorted(sims, reverse=True)
    sim_gap = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else 0.0
    
    feats = [
        s_c_tt, s_c_tv, s_c_vt, s_c_vv,
        s_g_tt, s_g_tv, s_g_vt, s_g_vv,
        str_score,
        sim_mean, sim_max, sim_gap,
        float(has_m_img), float(has_e_img)
    ]
    return np.array(feats, dtype=np.float32)


reranker = SimpleReranker()
corrector = NeuroSymbolicCorrector() 

acc_1, acc_5, acc_10 = 0, 0, 0

print("Starting Feature Extraction & Neuro-Symbolic Reranking...")

for idx in tqdm(range(len(parsed_text))):
    mention_data = parsed_text[idx]
    gt_answer = mention_data["answer"]
    mention_str = mention_data["mentions"]

    q_clip_text = np.array(Tm[idx]).astype(np.float32).flatten()
    faiss.normalize_L2(q_clip_text.reshape(1, -1))
    
    mention_img_path = mention_data["imgPath"]
    if not mention_img_path.endswith(".jpg"): mention_img_path = mention_img_path.rsplit('.', 1)[0] + ".jpg"
    
    has_image = False
    q_clip_img = None
    if len(mention_img_path) > 1 and mention_img_path in mention_clip:
        q_clip_img = np.array(mention_clip[mention_img_path][0]).astype(np.float32).flatten()
        faiss.normalize_L2(q_clip_img.reshape(1, -1))
        has_image = True
        
    q_gnn_text = mention_gnn_text[idx].flatten() 
    faiss.normalize_L2(q_gnn_text.reshape(1, -1))
    
    q_gnn_img = None
    if mention_gnn_img is not None and has_image:
        q_gnn_img = mention_gnn_img[idx].flatten()
        faiss.normalize_L2(q_gnn_img.reshape(1, -1))

    candidate_indices_set = set()
    _, I_c_t = index_clip_text.search(q_clip_text.reshape(1, -1), 250)
    candidate_indices_set.update(I_c_t[0])
    _, I_g_t = index_gnn.search(q_gnn_text.reshape(1, -1), 250)
    candidate_indices_set.update(I_g_t[0])
    if has_image and q_clip_img is not None:
        _, I_c_i = index_clip_text.search(q_clip_img.reshape(1, -1), 250)
        candidate_indices_set.update(I_c_i[0])
    if q_gnn_img is not None:
        _, I_g_i = index_gnn.search(q_gnn_img.reshape(1, -1), 250)
        candidate_indices_set.update(I_g_i[0])

    candidates = list(candidate_indices_set)
    
    # --- C. Construct Features ---
    batch_features = []
    valid_candidates = []
    
    for k_idx in candidates:
        entity_data = kb[k_idx]
        e_clip_text = clip_matrix[k_idx]
        e_gnn_text = gnn_matrix[k_idx]
        
        entity_img_list = entity_data.get("image_list", [])
        e_img_path = entity_img_list[0] if len(entity_img_list) > 0 else None
        e_has_img = False
        e_clip_img = None
        if e_img_path and e_img_path in kb_img:
            raw_vec = np.array(kb_img[e_img_path]).astype(np.float32).flatten()
            norm = np.linalg.norm(raw_vec)
            if norm > 0: e_clip_img = raw_vec / norm
            e_has_img = True
        e_gnn_img = None
        if kb_gnn_img_matrix is not None and e_img_path:
            e_gnn_img = kb_gnn_img_matrix[k_idx]
            
        str_sim = calculate_string_similarity(mention_str, entity_data["entity_name"])
        
        feats = extract_features(
            q_clip_text, q_clip_img, q_gnn_text, q_gnn_img,
            e_clip_text, e_clip_img, e_gnn_text, e_gnn_img,
            str_sim, has_image, e_has_img
        )
        batch_features.append(feats)
        valid_candidates.append(k_idx)
        

    if not batch_features:
        continue
        
    X = np.vstack(batch_features) 
    base_scores = reranker.predict(X)
    
    # ========================================
    # D+. Neuro-Symbolic Correction 
    # ========================================
    final_scores = []
    for i in range(len(base_scores)):
        s_new, _ = corrector.apply(base_scores[i], X[i])
        final_scores.append(s_new)
    
    # --- E. Ranking ---
    combined = list(zip(valid_candidates, final_scores, batch_features))
    combined.sort(key=lambda x: x[1], reverse=True)
    
    # ========================================
    # E+. Tie-Breaking 
    # ========================================
    combined_refined, _ = corrector.solve_tie(combined)
    
    if combined_refined:
        combined = combined_refined 
    
    final_rank_indices = [x[0] for x in combined]
    
    if len(final_rank_indices) > 0:
        top1_qid = kb[final_rank_indices[0]]["qid"]
        if top1_qid == gt_answer: acc_1 += 1  
        
        top5_qids = [kb[i]["qid"] for i in final_rank_indices[:5]]
        if gt_answer in top5_qids: acc_5 += 1    
        
        top10_qids = [kb[i]["qid"] for i in final_rank_indices[:10]]
        if gt_answer in top10_qids: acc_10 += 1

print(f"\nFinal Results with Neuro-Symbolic Correction:")
print(f"Total: {len(parsed_text)}")
print(f"Hit@1:  {acc_1 / len(parsed_text):.4f}")
print(f"Hit@5:  {acc_5 / len(parsed_text):.4f}")
print(f"Hit@10: {acc_10 / len(parsed_text):.4f}")
