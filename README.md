# Multi-Perspective Evidence Synthesis and Reasoning for Unsupervised Multimodal Entity Linking

This repository contains code and experimental assets for Multimodal Entity Linking (MEL). The main pipeline is organized around four core scripts:

- `llm_entity.py`: generates LLM descriptions and embeddings for candidate entities.
- `llm_mention.py`: generates LLM descriptions and embeddings for input mentions.
- `offline_train.py`: performs offline graph construction and representation learning.
- `online_inference.py`: performs online entity linking and final prediction.

The repository also includes MEL datasets, pre-extracted CLIP features, cached intermediate files, and saved experiment outputs.

## Repository Structure

```text
entity_linking/
├── LLM/                # Main LLM-enhanced MEL pipeline
│   ├── llm_entity.py
│   ├── llm_mention.py
│   ├── offline_train.py
│   └── online_inference.py
├── MSRMEL/             # Original/reference OpenMEL/MSRMEL resources
├── RichpediaMEL/       # RichpediaMEL data and pre-extracted features
├── WikiMEL/            # WikiMEL data and pre-extracted features
├── WikiDiverse/        # WikiDiverse data and features
├── WeiboNewsMEL/       # WeiboNewsMEL data
├── ablation/           # Ablation scripts and logs
├── draw/               # Plotting scripts and figures
├── requirements.txt    # Root dependency notes
└── *.zip               # Archived data or result files
```

## Environment Setup

Using a dedicated Python environment is recommended:

```bash
conda create -n entity_linking python=3.8 -y
conda activate entity_linking
```

Install the main dependencies:

```bash
pip install numpy scipy scikit-learn pandas tqdm pillow requests lxml
pip install torch torchvision
pip install torch_geometric
pip install faiss-cpu
pip install openai transformers clip
```

You can also refer to `requirements.txt` and `MSRMEL/requirements.txt`. If you use a GPU, install PyTorch and PyG versions that match your local CUDA version.

## Data Layout

The scripts assume that datasets are placed directly under the repository root. Example layouts:

```text
RichpediaMEL/
├── RichpediaMEL_test.json
├── kb_entity.json
├── qid2id.json
├── entity_text_clip.json
├── entity_image_clip.json
├── mention_add_caption_clip.json
└── mention_image_clip.json

WikiMEL/
├── WIKIMEL_test.json
├── kb_entity.json
├── qid2id.json
├── entity_text_clip.json
├── entity_image_clip.json
├── mention_add_cap_clip.json
└── mention_image_clip.json
```

Feature filenames differ slightly across datasets. For example, RichpediaMEL uses `mention_add_caption_clip.json`, while WikiMEL uses `mention_add_cap_clip.json`. Many scripts use hard-coded dataset-specific paths, so check the path constants before switching datasets.

## Main Code

The main implementation is reduced to four scripts. The first two scripts prepare LLM-based semantic features, and the last two scripts implement the offline/online MEL pipeline.

| File | Stage | Description |
| --- | --- | --- |
| `llm_entity.py` | LLM feature generation | Generates descriptions and embeddings for candidate entities in the knowledge base. |
| `llm_mention.py` | LLM feature generation | Generates descriptions and embeddings for mentions in the test set. |
| `offline_train.py` | Offline training | Builds multimodal graphs, aligns CLIP and LLM features, trains graph representations, and saves reusable embeddings/caches. |
| `online_inference.py` | Online inference | Loads offline artifacts, retrieves candidate entities, applies multimodal scoring/reranking, and reports final entity linking metrics. |

### LLM Feature Generation

Run the entity and mention LLM scripts before offline training if the required LLM feature files are not already available.

```bash
python llm_entity.py
python llm_mention.py
```

Typical outputs include JSON description files and NumPy embedding files, for example:

```text
LLM_res/
├── mention_llm_desc.json
├── mention_llm_emb.npy
├── llm_entity_desc_*.json
├── llm_entity_emb_*.npy
└── target_entity_indices_*.json
```

### offline_train.py

`offline_train.py` is the offline stage. It prepares graph-based representations used later by online inference.

Main responsibilities:

1. Load dataset features, including CLIP text/image features and LLM embeddings.
2. Align LLM embeddings with the CLIP text feature space.
3. Construct multimodal graphs over mentions and entities.
4. Train graph encoders or contrastive representation models.
5. Save entity and mention embeddings for online inference.

Example command:

```bash
python offline_train.py \
  --dataset ../RichpediaMEL \
  --llm_dir LLM_res_Rich \
  --epochs_text 20 \
  --epochs_img 20 \
  --batch_size 256 \
  --hidden_size 512 \
  --subgraph_size 10 \
  --k_neighbors 10 \
  --lr 0.001 \
  --lambda_llm 1.0 \
  --lambda_graph 1.0 \
  --lambda_xmodal 1.0 \
  --cache_dir cache_rich \
  --output_dir res_Rich/rebuild
```

The offline stage usually saves files such as:

```text
entity_text_gnn.npy
mention_text_gnn.npy
entity_image_gnn.npy
mention_image_gnn.npy
```

### online_inference.py

`online_inference.py` is the online stage. It loads the trained offline artifacts and performs final entity linking.

Main responsibilities:

1. Load test mentions, knowledge-base entities, CLIP features, LLM features, and offline GNN embeddings.
2. Retrieve candidate entities for each mention.
3. Combine text, image, LLM, graph, and string-matching signals.
4. Apply final reranking or global coherence logic.
5. Report standard entity linking metrics.

The current `online_inference.py` script uses path constants inside the file instead of command-line arguments. By default, it loads WikiMEL data and GNN embeddings from:

```python
GNN_ENTITY_TEXT_PATH = 'res_Wikimel/rebuild_res/entity_text_gnn.npy'
GNN_ENTITY_IMG_PATH = 'res_Wikimel/rebuild_res/entity_image_gnn.npy'
GNN_MENTION_TEXT_PATH = 'res_Wikimel/rebuild_res/mention_text_gnn.npy'
GNN_MENTION_IMG_PATH = 'res_Wikimel/rebuild_res/mention_image_gnn.npy'
```

It also reads the dataset files from `../WikiMEL/`, including `WIKIMEL_test.json`, `mention_image_clip.json`, `entity_text_clip.json`, `entity_image_clip.json`, `mention_add_cap_clip.json`, and `kb_entity.json`.

Run online inference:

```bash
python online_inference.py
```

The inference stage reports metrics such as:

```text
HIT@1
HIT@5
HIT@10
```

## Recommended Experiment Order

```text
1. Run `llm_entity.py` to generate or refresh entity-side LLM descriptions and embeddings.
2. Run `llm_mention.py` to generate or refresh mention-side LLM descriptions and embeddings.
3. Run `offline_train.py` to build graph representations and save reusable offline artifacts.
4. Run `online_inference.py` to perform final entity linking and report evaluation metrics.
```
