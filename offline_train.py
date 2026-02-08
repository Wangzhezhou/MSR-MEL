import json
import os
import argparse
import time
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import Ridge
from tqdm import tqdm
from scipy import sparse as sp
from sklearn.preprocessing import normalize

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool


# ==================== 0. Utils ====================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== 1. PPR & Subgraph Extraction ====================

class PPR:
    def __init__(self, adj_mat, maxsize=20, n_order=3, alpha=0.85):
        self.n_order = n_order
        self.maxsize = maxsize
        self.adj_mat = adj_mat
        self.P = normalize(adj_mat, norm='l1', axis=0)
        self.d = np.array(adj_mat.sum(1)).squeeze()
        self.alpha = alpha

    def search(self, seed):
        x = sp.csc_matrix((np.ones(1), ([seed], np.zeros(1, dtype=int))),
                          shape=[self.P.shape[0], 1])
        r = x.copy()
        for _ in range(self.n_order):
            x = (1 - self.alpha) * r + self.alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9)
        idx = scores.argsort()[::-1][:self.maxsize]
        neighbor = np.array(x.indices[idx])

        seed_idx = np.where(neighbor == seed)[0]
        if seed_idx.size == 0:
            neighbor = np.append(np.array([seed]), neighbor)
        else:
            seed_idx = seed_idx[0]
            neighbor[seed_idx], neighbor[0] = neighbor[0], neighbor[seed_idx]
        return neighbor


class OptimizedSubgraphExtractor:
    def __init__(self, x, edge_index, maxsize=20, n_order=10, cache_path=None):
        """
        x: [N, d] 
        edge_index: [2, E]
        """
        self.x = x
        self.edge_index = edge_index
        self.node_num = x.shape[0]
        self.maxsize = maxsize
        self.cache_path = cache_path

        edge_num = edge_index.shape[1]
        rows = edge_index[0].cpu().numpy()
        cols = edge_index[1].cpu().numpy()
        data = np.ones(edge_num)

        self.sp_adj = sp.csc_matrix(
            (data, (rows, cols)),
            shape=[self.node_num, self.node_num]
        )
        self.ppr = PPR(self.sp_adj, maxsize=maxsize, n_order=n_order)

        self.adj_list = {i: set() for i in range(self.node_num)}
        for u, v in zip(rows, cols):
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)

        self.subgraph_cache = {}

    def adjust_edge(self, idx):
        idx = [int(i) for i in idx]
        dic = {idx[i]: i for i in range(len(idx))}
        new_index = [[], []]
        nodes = set(idx)

        for i in idx:
            if i not in self.adj_list:
                continue
            edge = list(self.adj_list[i] & nodes)
            edge = [dic[_] for _ in edge]
            new_index[0] += len(edge) * [dic[i]]
            new_index[1] += edge
        return torch.LongTensor(new_index)

    def extract_subgraph(self, node_idx):
        neighbors = self.ppr.search(node_idx)[:self.maxsize]
        x = torch.FloatTensor(self.x[neighbors])
        edge = self.adjust_edge(neighbors)
        return Data(x=x, edge_index=edge)

    def precompute_all_subgraphs(self, node_list=None):
        if node_list is None:
            node_list = list(range(self.node_num))
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"Loading subgraphs from {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                self.subgraph_cache = pickle.load(f)
            return
        print(f"Pre-computing subgraphs for {len(node_list)} nodes...")
        for node_idx in tqdm(node_list, desc="Pre-computing"):
            self.subgraph_cache[node_idx] = self.extract_subgraph(node_idx)
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.subgraph_cache, f)

    def extract_batch(self, node_list):
        batch_data = []
        indices = []
        size = 0
        for node in node_list:
            if isinstance(node, torch.Tensor):
                node = node.item()
            subgraph = self.subgraph_cache.get(node)
            if subgraph is None:
                subgraph = self.extract_subgraph(node)
            batch_data.append(subgraph)
            indices.append(size)
            size += subgraph.x.size(0)
        return Batch.from_data_list(batch_data), torch.tensor(indices)


# ==================== 2. GNN Encoder & Model ====================

class EncoderWithSkip(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.1):
        super(EncoderWithSkip, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.prelu1 = nn.PReLU(hidden_channels)
        self.dropout_layer = nn.Dropout(dropout)

        if num_layers > 1:
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.bn2 = nn.BatchNorm1d(hidden_channels)
            self.prelu2 = nn.PReLU(hidden_channels)

        self.skip_proj = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        identity = x
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.dropout_layer(x)
        if self.skip_proj is not None:
            identity = self.skip_proj(identity)
        x = x + identity

        if self.num_layers > 1:
            identity = x
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = self.prelu2(x)
            x = self.dropout_layer(x)
            x = x + identity
        return x


class Pool(nn.Module):
    def __init__(self, in_channels):
        super(Pool, self).__init__()

    def forward(self, x, batch):
        return global_mean_pool(x, batch)


class SubgraphGCL(nn.Module):
    """
    一个统一的 GCL 模型：
    - loss_subcon: 基于 node/subgraph 的结构自监督
    - loss_llm_infoNCE: 节点 embedding vs LLM embedding 的 InfoNCE
    - loss_xmodal_infoNCE: 文本 vs 图像（同一个节点）的 cross-modal InfoNCE
    """
    def __init__(self, hidden_channels, encoder, pool):
        super(SubgraphGCL, self).__init__()
        self.encoder = encoder
        self.pool = pool
        self.marginloss = nn.MarginRankingLoss(0.5)

    def forward(self, x, edge_index, batch=None, index=None):
        hidden = self.encoder(x, edge_index)
        if index is None:
            return hidden
        z = hidden[index]
        summary = self.pool(hidden, batch)
        return z, summary

    def loss_subcon(self, hidden1, summary1):
        shuf_index = torch.randperm(summary1.size(0), device=summary1.device)
        hidden2 = hidden1[shuf_index]
        summary2 = summary1[shuf_index]
        logits_aa = torch.sigmoid(torch.sum(hidden1 * summary1, dim=-1))
        logits_bb = torch.sigmoid(torch.sum(hidden2 * summary2, dim=-1))
        logits_ab = torch.sigmoid(torch.sum(hidden1 * summary2, dim=-1))
        logits_ba = torch.sigmoid(torch.sum(hidden2 * summary1, dim=-1))
        ones = torch.ones(logits_aa.size(0)).to(logits_aa.device)
        return self.marginloss(logits_aa, logits_ba, ones) + self.marginloss(logits_bb, logits_ab, ones)

    def loss_llm_infoNCE(self, z, llm_emb, mask, tau=0.1):
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return torch.tensor(0.0, device=z.device)

        z_pos = z[idx]          # [M, d]
        h_pos = llm_emb[idx]    # [M, d]

        z_pos = F.normalize(z_pos, p=2, dim=-1)
        h_pos = F.normalize(h_pos, p=2, dim=-1)

        logits = torch.matmul(z_pos, h_pos.t()) / tau  # [M, M]
        labels = torch.arange(logits.size(0), device=z.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def loss_xmodal_infoNCE(self, z1, z2, mask, tau=0.1):
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return torch.tensor(0.0, device=z1.device)

        a = z1[idx]
        b = z2[idx]

        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)

        logits = torch.matmul(a, b.t()) / tau
        labels = torch.arange(logits.size(0), device=z1.device)
        loss = F.cross_entropy(logits, labels)
        return loss


# ==================== 3. Graph Building (Fixed & Optimized) ====================

def align_space(X_src, X_tgt, alpha=1.0):
    """
    使用 Ridge Regression 将 X_src 映射到 X_tgt 的空间。
    """
    print(f"Aligning embeddings (Ridge Regression): Source {X_src.shape} -> Target {X_tgt.shape}...")
    # 转为 numpy 进行 sklearn 训练
    if isinstance(X_src, torch.Tensor): X_src = X_src.cpu().numpy()
    if isinstance(X_tgt, torch.Tensor): X_tgt = X_tgt.cpu().numpy()
    
    # 确保输入是 2D
    if X_src.ndim > 2: X_src = X_src.reshape(X_src.shape[0], -1)
    if X_tgt.ndim > 2: X_tgt = X_tgt.reshape(X_tgt.shape[0], -1)

    reg = Ridge(alpha=alpha)
    reg.fit(X_src, X_tgt)
    
    score = reg.score(X_src, X_tgt)
    print(f"Alignment R^2 Score: {score:.4f}")
    return reg

def build_multimodal_graph(
    e_text_feat, e_img_feat, m_text_feat, m_img_feat,       # CLIP features (numpy)
    m_llm_feat, e_llm_feat_2k, target_entity_indices_2k,    # LLM features (numpy)
    e_has_img, m_has_img,
    w1=1.0, w2=0.5, w3=1.0,  # Fusion Weights
    k_neighbors=50,          # Final Top-K
    chunk_size=1000
):
    print(f"\n{'='*20} Building Advanced Graph {'='*20}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_ent = e_text_feat.shape[0]
    n_men = m_text_feat.shape[0]
    
    # ================= 1. Pre-process & Normalize =================
    
    # CLIP Features -> Tensor & Normalize
    e_txt = torch.tensor(e_text_feat, dtype=torch.float32).to(device)
    m_txt = torch.tensor(m_text_feat, dtype=torch.float32).to(device)
    e_txt = F.normalize(e_txt, dim=1)
    m_txt = F.normalize(m_txt, dim=1)
    
    e_img = torch.tensor(e_img_feat, dtype=torch.float32).to(device)
    m_img = torch.tensor(m_img_feat, dtype=torch.float32).to(device)
    e_img = F.normalize(e_img, dim=1)
    m_img = F.normalize(m_img, dim=1)

    # ================= 2. Alignment Logic =================

    if m_llm_feat.ndim == 3: m_llm_feat = m_llm_feat.squeeze(1)
    if e_llm_feat_2k.ndim == 3: e_llm_feat_2k = e_llm_feat_2k.squeeze(1)

    projector = align_space(m_llm_feat, m_text_feat)
    
    m_llm_aligned_np = projector.predict(m_llm_feat)
    m_llm_aligned = torch.tensor(m_llm_aligned_np, dtype=torch.float32).to(device)
    m_llm_aligned = F.normalize(m_llm_aligned, dim=1) # [Fix 1: Normalize]

    valid_indices = target_entity_indices_2k # list of global IDs
    e_llm_valid_np = e_llm_feat_2k[valid_indices] # (2000, D)
    
    e_llm_aligned_2k_np = projector.predict(e_llm_valid_np)
    e_llm_aligned_2k = torch.tensor(e_llm_aligned_2k_np, dtype=torch.float32).to(device)
    e_llm_aligned_2k = F.normalize(e_llm_aligned_2k, dim=1) # [Fix 1: Normalize]

    print(f"Projected 2k Entity LLM shape: {e_llm_aligned_2k.shape}")

    # ================= 3. M-E Linear Fusion & Retrieval =================
    print(f"Constructing Mention-Entity Edges (Fusion: w1={w1}, w2={w2}, w3={w3})...")
    
    edge_list = []
    
    for i in tqdm(range(0, n_men, chunk_size), desc="M-E Fusion"):
        end = min(i + chunk_size, n_men)
        
        b_m_txt = m_txt[i:end]
        b_m_img = m_img[i:end]
        b_m_llm = m_llm_aligned[i:end]
        
        # Sim 1: Text (CLIP)
        sim_txt = torch.matmul(b_m_txt, e_txt.t())
        
        # Sim 2: Image (CLIP)
        sim_img = torch.matmul(b_m_img, e_img.t())
        
        # Sim 3: LLM (Aligned) -> Entity Text (CLIP)
        sim_llm = torch.matmul(b_m_llm, e_txt.t())
        
        # Linear Fusion
        sim_final = w1 * sim_txt + w2 * sim_img + w3 * sim_llm
        
        # Top-K
        vals, indices = torch.topk(sim_final, k=k_neighbors, dim=1)
        
        # Build Edges
        # Source: Mention Global ID (n_ent + i + local_idx)
        src = torch.arange(i, end, device=device).unsqueeze(1).expand(-1, k_neighbors) + n_ent
        tgt = indices
        
        edge_list.append(torch.stack([src.reshape(-1), tgt.reshape(-1)], dim=0))

    me_edges = torch.cat(edge_list, dim=1)
    print(f"Basic M-E edges (Top-{k_neighbors}): {me_edges.shape[1]}")

    # ================= 4. Mutual Verification (2k Entities) =================
    if len(target_entity_indices_2k) > 0:
        print("Applying Mutual Verification for 2k LLM Entities...")
        
        # 计算: Entity LLM (2k) vs Mention Text (All)
        # e_llm_aligned_2k: [2000, D] (Normalized)
        # m_txt: [N_men, D] (Normalized)
        sim_mutual = torch.matmul(e_llm_aligned_2k, m_txt.t()) # [2000, N_men]
        
        mutual_k = 10 
        _, mutual_indices = torch.topk(sim_mutual, k=mutual_k, dim=1) # [2000, 10]
        
        # [Fix 2: Correct Index Mapping]
        # Source: Entity Global IDs (need to expand)
        src_ent_ids = torch.tensor(target_entity_indices_2k, device=device).unsqueeze(1).expand(-1, mutual_k)
        
        # Target: Mention Global IDs (need to add offset n_ent)
        tgt_men_ids = mutual_indices + n_ent 
        
        # Shape Check:
        # src_ent_ids: [2000, 10] -> reshape(-1) -> [20000]
        # tgt_men_ids: [2000, 10] -> reshape(-1) -> [20000]
        # Now they match!
        
        mutual_edges_src = src_ent_ids.reshape(-1)
        mutual_edges_tgt = tgt_men_ids.reshape(-1)
        
        # Make edges M->E direction to match me_edges (Source=Mention, Target=Entity)
        mutual_edges = torch.stack([mutual_edges_tgt, mutual_edges_src], dim=0)
        
        # Union Strategy
        me_edges = torch.cat([me_edges, mutual_edges], dim=1)
        me_edges = torch.unique(me_edges, dim=1) # Deduplicate
        
        print(f"Edges after mutual verification: {me_edges.shape[1]}")

    # ================= 5. E-E & M-M Edges (Keep Previous Logic) =================
    print("Building E-E and M-M edges...")
    
    # E-E: Top-10
    ee_edges_list = []
    k_ee = 10
    for i in range(0, n_ent, chunk_size):
        end = min(i + chunk_size, n_ent)
        sim = torch.matmul(e_txt[i:end], e_txt.t())
        self_mask = torch.arange(i, end, device=device).unsqueeze(1)
        sim.scatter_(1, self_mask, -1e9)
        _, idx = torch.topk(sim, k=k_ee, dim=1)
        src = torch.arange(i, end, device=device).unsqueeze(1).expand(-1, k_ee)
        ee_edges_list.append(torch.stack([src.reshape(-1), idx.reshape(-1)], dim=0))
    ee_edges = torch.cat(ee_edges_list, dim=1)
    
    # M-M: Top-5
    mm_edges_list = []
    k_mm = 5
    for i in range(0, n_men, chunk_size):
        end = min(i + chunk_size, n_men)
        sim = torch.matmul(m_txt[i:end], m_txt.t())
        self_mask = torch.arange(i, end, device=device).unsqueeze(1)
        sim.scatter_(1, self_mask, -1e9)
        _, idx = torch.topk(sim, k=k_mm, dim=1)
        src = torch.arange(i, end, device=device).unsqueeze(1).expand(-1, k_mm) + n_ent
        tgt = idx + n_ent
        mm_edges_list.append(torch.stack([src.reshape(-1), tgt.reshape(-1)], dim=0))
    mm_edges = torch.cat(mm_edges_list, dim=1)

    # ================= 6. Final Assemble =================
    # me_edges: M->E
    # ee_edges: E->E
    # mm_edges: M->M
    all_edges = torch.cat([me_edges, ee_edges, mm_edges], dim=1)
    
    # Symmetrize (To Undirected)
    rev_edges = torch.stack([all_edges[1], all_edges[0]], dim=0)
    final_edges = torch.cat([all_edges, rev_edges], dim=1)
    final_edges = torch.unique(final_edges, dim=1)
    
    print(f"Final Graph: {final_edges.shape[1]} edges, Nodes: {n_ent + n_men}")
    return final_edges.cpu()


# ==================== 4. Embedding Extraction ====================

def get_embeddings_in_batches(model, extractor, total_count, batch_size, device, desc="Extracting"):
    model.eval()
    embeddings = []
    all_indices = list(range(total_count))
    with torch.no_grad():
        for i in tqdm(range(0, total_count, batch_size), desc=desc):
            batch_nodes = all_indices[i: min(i + batch_size, total_count)]
            batch_data, index = extractor.extract_batch(batch_nodes)
            batch_data = batch_data.to(device)
            index = index.to(device)
            z, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch, index)
            embeddings.append(z.cpu().numpy())
            del batch_data, index, z
    return np.vstack(embeddings)


# ==================== 5. Main Training Pipeline ====================

def main():
    parser = argparse.ArgumentParser(description='LLM-guided Unsupervised GNN for Entity Linking')
    parser.add_argument('--dataset', type=str, default='../RichpediaMEL')
    parser.add_argument('--llm_dir', type=str, default='LLM_res_Rich')
    parser.add_argument('--epochs_text', type=int, default=20)
    parser.add_argument('--epochs_img', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--subgraph_size', type=int, default=10)
    parser.add_argument('--k_neighbors', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lambda_llm', type=float, default=1.0)
    parser.add_argument('--lambda_graph', type=float, default=1.0)
    parser.add_argument('--lambda_xmodal', type=float, default=1.0)
    parser.add_argument('--cache_dir', type=str, default='cache_rich')
    parser.add_argument('--output_dir', type=str, default='res_Rich/rebuild')
    args = parser.parse_args()

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- 1. Load Base Data (WikiMEL) ----------------
    print("Loading base multimodal data...")
    with open(f'{args.dataset}/entity_text_clip.json', 'r') as f:
        kb_text = json.load(f)
    with open(f'{args.dataset}/mention_add_caption_clip.json', 'r') as f:
        m_text = json.load(f)
    with open(f'{args.dataset}/entity_image_clip.json', 'r') as f:
        kb_img = json.load(f)
    with open(f'{args.dataset}/mention_image_clip.json', 'r') as f:
        m_img_raw = json.load(f)
    with open(f"{args.dataset}/kb_entity.json", "r") as f:
        kb = json.load(f)
    with open(f'{args.dataset}/RichpediaMEL_test.json', 'r') as f:
        test_data = json.load(f)

    # Text features
    e_text_feat = np.array([item[0] for item in kb_text], dtype=np.float32)
    m_text_feat = np.array([item[0] for item in m_text], dtype=np.float32)
    n_entities = len(e_text_feat)
    n_mentions = len(m_text_feat)
    print(f"#Entities={n_entities}, #Mentions={n_mentions}")

    # Image features
    img_dim = 512
    e_img_feat = np.zeros((n_entities, img_dim), dtype=np.float32)
    e_has_img = np.zeros(n_entities, dtype=bool)
    for idx, entry in enumerate(kb):
        if entry.get("image_list") and len(entry["image_list"]) > 0:
            path = entry["image_list"][0]
            if path in kb_img:
                e_img_feat[idx] = np.array(kb_img[path][0])
                e_has_img[idx] = True

    m_img_feat = np.zeros((n_mentions, img_dim), dtype=np.float32)
    m_has_img = np.zeros(n_mentions, dtype=bool)
    for idx in range(min(len(test_data), len(m_img_feat))):
        path = test_data[idx]["imgPath"]
        if not path.endswith(".jpg"):
            path = path.rsplit('.', 1)[0] + ".jpg"
        if len(path) > 1 and path in m_img_raw:
            m_img_feat[idx] = np.array(m_img_raw[path][0])
            m_has_img[idx] = True

    n_total = n_entities + n_mentions

    # ---------------- 2. Load LLM Embeddings ----------------
    print("\nLoading LLM enhanced embeddings...")

    def load_llm_emb(path):
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr.squeeze(1)
        elif arr.ndim == 2:
            pass 
        else:
            raise ValueError(f"Unexpected LLM embedding shape {arr.shape} for {path}")
        return arr.astype(np.float32)

    print("\nLoading LLM enhanced embeddings...")

    # 2.1 Entity LLM (only 4k)
    llm_entity_emb_4k = load_llm_emb(os.path.join(args.llm_dir, 'llm_entity_emb_4k.npy'))  # [4000, d_llm]
    with open(os.path.join(args.llm_dir, 'target_entity_indices_4k.json'), 'r') as f:
        target_entity_indices_2k = json.load(f)  # list of length 4000, global entity idx

    d_llm = llm_entity_emb_4k.shape[1]
    all_entity_llm = np.zeros((n_entities, d_llm), dtype=np.float32)
    entity_llm_mask = np.zeros(n_entities, dtype=bool)
    for i, eid in enumerate(target_entity_indices_2k):
        all_entity_llm[eid] = llm_entity_emb_4k[i]
        entity_llm_mask[eid] = True

    # 2.2 Mention LLM (assume order aligned with mentions)
    mention_llm_emb = load_llm_emb(os.path.join(args.llm_dir, 'mention_llm_emb.npy'))  # [n_mentions, d_llm]
    assert mention_llm_emb.shape[0] == n_mentions, "mention_llm_emb rows != n_mentions"
    mention_llm_mask = np.ones(n_mentions, dtype=bool)


    # 2.3 Stack entity + mention -> all nodes
    all_llm_text_feat = np.vstack([all_entity_llm, mention_llm_emb])       # [N, d]
    all_llm_text_mask = np.concatenate([entity_llm_mask, mention_llm_mask])  # [N]

    all_llm_tensor = torch.FloatTensor(all_llm_text_feat).to(device)
    all_llm_mask_tensor = torch.BoolTensor(all_llm_text_mask).to(device)

    print(f"LLM text feat shape: {all_llm_text_feat.shape}, valid nodes: {all_llm_text_mask.sum()}")

    # ---------------- 3. Build Graph ----------------
    w_text = 1.0
    w_img = 0.5 
    w_llm = 1.0  
    
    unified_edge_index = build_multimodal_graph(
        e_text_feat=e_text_feat, 
        e_img_feat=e_img_feat, 
        m_text_feat=m_text_feat, 
        m_img_feat=m_img_feat,
        m_llm_feat=mention_llm_emb, 
        e_llm_feat_2k=llm_entity_emb_4k, 
        target_entity_indices_2k=target_entity_indices_2k,
        e_has_img=e_has_img, 
        m_has_img=m_has_img,
        w1=w_text, w2=w_img, w3=w_llm,
        k_neighbors=50,  
        chunk_size=1000
    )

    # ---------------- 4. Train TEXT GNN (LLM-guided) ----------------
    print(f"\n{'=' * 40}\nTraining TEXT GNN with LLM Guidance\n{'=' * 40}")

    all_text_feat_orig = np.vstack([e_text_feat, m_text_feat])  # [N, d_text]
    all_text_tensor_orig = torch.FloatTensor(all_text_feat_orig)

    extractor_txt = OptimizedSubgraphExtractor(
        all_text_tensor_orig,
        unified_edge_index,
        maxsize=args.subgraph_size,
        cache_path=os.path.join(args.cache_dir, 'text_subgraph.pkl')
    )
    extractor_txt.precompute_all_subgraphs()

    model_txt = SubgraphGCL(
        args.hidden_size,
        EncoderWithSkip(all_text_feat_orig.shape[1], args.hidden_size),
        Pool(args.hidden_size)
    ).to(device)

    optimizer_txt = torch.optim.Adam(model_txt.parameters(), lr=args.lr)

    indices_all = torch.arange(n_total)

    for epoch in range(args.epochs_text):
        model_txt.train()
        total_loss = 0.0
        total_graph_loss = 0.0
        total_llm_loss = 0.0

        perm = torch.randperm(n_total)
        pbar = tqdm(range(0, n_total, args.batch_size), desc=f"Text Epoch {epoch}")

        for it in pbar:
            batch_idx = perm[it: it + args.batch_size]
            batch_nodes = batch_idx.tolist()

            batch_data, idx_in_batch = extractor_txt.extract_batch(batch_nodes)
            batch_data = batch_data.to(device)
            idx_in_batch = idx_in_batch.to(device)

            batch_llm_emb = all_llm_tensor[batch_idx].to(device)
            batch_llm_mask = all_llm_mask_tensor[batch_idx].to(device)

            z, summary = model_txt(batch_data.x, batch_data.edge_index,
                                   batch_data.batch, idx_in_batch)

            loss_graph = model_txt.loss_subcon(z, summary)
            loss_llm = model_txt.loss_llm_infoNCE(z, batch_llm_emb, batch_llm_mask)

            loss = args.lambda_graph * loss_graph + args.lambda_llm * loss_llm

            optimizer_txt.zero_grad()
            loss.backward()
            optimizer_txt.step()

            total_loss += loss.item()
            total_graph_loss += loss_graph.item()
            total_llm_loss += loss_llm.item()

            pbar.set_postfix({
                "loss": f"{total_loss / ((it // args.batch_size) + 1):.4f}",
                "graph": f"{total_graph_loss / ((it // args.batch_size) + 1):.4f}",
                "llm": f"{total_llm_loss / ((it // args.batch_size) + 1):.4f}"
            })

    print("\nExtracting final Text GNN embeddings...")
    z_txt_all = get_embeddings_in_batches(
        model_txt,
        extractor_txt,
        n_total,
        args.batch_size * 2,
        device,
        desc="Text Embeddings"
    )
    entity_text_gnn = z_txt_all[:n_entities]
    mention_text_gnn = z_txt_all[n_entities:]
    np.save(os.path.join(args.output_dir, 'entity_text_gnn.npy'), entity_text_gnn)
    np.save(os.path.join(args.output_dir, 'mention_text_gnn.npy'), mention_text_gnn)
    print(f"Text GNN embeddings saved to {args.output_dir}")

    text_teacher = torch.FloatTensor(z_txt_all).to(device)

    # ---------------- 5. Train IMAGE GNN (Text-guided + Graph) ----------------
    print(f"\n{'=' * 40}\nTraining IMAGE GNN with Text Guidance\n{'=' * 40}")

    all_img_feat_orig = np.vstack([e_img_feat, m_img_feat])  # [N, d_img]
    all_img_tensor_orig = torch.FloatTensor(all_img_feat_orig)

    all_has_img = np.concatenate([e_has_img, m_has_img])
    all_has_img_tensor = torch.BoolTensor(all_has_img).to(device)

    extractor_img = OptimizedSubgraphExtractor(
        all_img_tensor_orig,
        unified_edge_index,
        maxsize=args.subgraph_size,
        cache_path=os.path.join(args.cache_dir, 'img_subgraph.pkl')
    )
    extractor_img.precompute_all_subgraphs()

    model_img = SubgraphGCL(
        args.hidden_size,
        EncoderWithSkip(all_img_feat_orig.shape[1], args.hidden_size),
        Pool(args.hidden_size)
    ).to(device)

    optimizer_img = torch.optim.Adam(model_img.parameters(), lr=args.lr)

    for epoch in range(args.epochs_img):
        model_img.train()
        total_loss = 0.0
        total_graph_loss = 0.0
        total_xmodal_loss = 0.0

        perm = torch.randperm(n_total)
        pbar = tqdm(range(0, n_total, args.batch_size), desc=f"Image Epoch {epoch}")

        for it in pbar:
            batch_idx = perm[it: it + args.batch_size]
            batch_nodes = batch_idx.tolist()

            batch_data, idx_in_batch = extractor_img.extract_batch(batch_nodes)
            batch_data = batch_data.to(device)
            idx_in_batch = idx_in_batch.to(device)

            z_img, summary_img = model_img(batch_data.x, batch_data.edge_index,
                                           batch_data.batch, idx_in_batch)

            batch_text_teacher = text_teacher[batch_idx]
            batch_has_img = all_has_img_tensor[batch_idx]

            loss_graph = model_img.loss_subcon(z_img, summary_img)
            loss_xmodal = model_img.loss_xmodal_infoNCE(z_img, batch_text_teacher, batch_has_img)

            loss = args.lambda_graph * loss_graph + args.lambda_xmodal * loss_xmodal

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            total_loss += loss.item()
            total_graph_loss += loss_graph.item()
            total_xmodal_loss += loss_xmodal.item()

            pbar.set_postfix({
                "loss": f"{total_loss / ((it // args.batch_size) + 1):.4f}",
                "graph": f"{total_graph_loss / ((it // args.batch_size) + 1):.4f}",
                "xmodal": f"{total_xmodal_loss / ((it // args.batch_size) + 1):.4f}"
            })

    print("\nExtracting final Image GNN embeddings...")
    z_img_all = get_embeddings_in_batches(
        model_img,
        extractor_img,
        n_total,
        args.batch_size * 2,
        device,
        desc="Image Embeddings"
    )
    entity_image_gnn = z_img_all[:n_entities]
    mention_image_gnn = z_img_all[n_entities:]
    np.save(os.path.join(args.output_dir, 'entity_image_gnn.npy'), entity_image_gnn)
    np.save(os.path.join(args.output_dir, 'mention_image_gnn.npy'), mention_image_gnn)
    print(f"Image GNN embeddings saved to {args.output_dir}")

    print("\nAll done.")


if __name__ == '__main__':
    main()
