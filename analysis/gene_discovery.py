# Manuscript-linked analysis: Original gene encoder, discovery and signatures
# Source lineage and packaging changes: provenance/source_lineage.json
import os as _release_os
try:
    from IPython.display import display
except ImportError:

    def display(value):
        print(value)
pass
pass
import os
import sys
import gc
import json
import math
import random
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_geometric
    from torch_geometric.nn import RGCNConv
    from torch_geometric.utils import negative_sampling
except ImportError:
    raise RuntimeError('Install the recorded analysis dependencies before running this script; see docs/ENVIRONMENT.md.')
    import torch_geometric
    from torch_geometric.nn import RGCNConv
    from torch_geometric.utils import negative_sampling
try:
    import igraph as ig
    import leidenalg
except ImportError:
    raise RuntimeError('Install the recorded analysis dependencies before running this script; see docs/ENVIRONMENT.md.')
    import igraph as ig
    import leidenalg
from scipy import sparse
from scipy.stats import ttest_ind, norm
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score
SEED = 20261004
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if str(torch_geometric.__version__) != '2.8.0':
    raise RuntimeError(f'torch_geometric must be 2.8.0; observed={torch_geometric.__version__}')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable. Select Tesla T4.')
GPU_NAME = torch.cuda.get_device_name(0)
DEVICE = torch.device('cuda')
print('GPU:', GPU_NAME)
print('torch:', torch.__version__)
print('torch_geometric:', torch_geometric.__version__)
PROJECT = Path(_release_os.environ.get('AML_GRAPHRX_ROOT', '/content/drive/MyDrive/AML_GraphRx'))
AUDIT = PROJECT / 'results' / 'audit'
STEP5 = PROJECT / 'data' / 'patient_graphs' / '20260823T123040Z'
R5G_ROOT = PROJECT / 'results' / 'R5_gene_first_state_discovery' / '20261004'
R5G_ROOT.mkdir(parents=True, exist_ok=True)
RANK_FILE = STEP5 / '20260823T123040Z_patient_rank_raw_float64.npy'
PRESENCE_FILE = STEP5 / '20260823T123040Z_gene_presence_mask_bool.npy'
TOPOLOGY_FILE = STEP5 / '20260823T123040Z_shared_locked_topology.pt'
NODE_FILE = STEP5 / '20260823T123040Z_node_index.csv'
DISCOVERY_LEDGER_FILE = AUDIT / '452_R5DICE_discovery_ledger.csv'
SPLIT_FILE = AUDIT / '453_R5DICE_internal_train_validation_split.csv'
R5S_SUMMARY = AUDIT / '478_R5BOTTLENECK_final_summary.csv'
R5S_HARD_GATES = AUDIT / '479_R5BOTTLENECK_hard_gates.csv'
R5S_MANIFEST = AUDIT / '480_R5BOTTLENECK_manifest.csv'
F481 = AUDIT / '481_R5G_predeclared_protocol.json'
F482 = AUDIT / '482_R5G_gene_RGAE_training_history.csv'
F483 = AUDIT / '483_R5G_gene_embeddings.npy'
F484 = AUDIT / '484_R5G_gene_embedding_ledger.csv'
F485 = AUDIT / '485_R5G_gene_RGAE_audit.csv'
F486 = AUDIT / '486_R5G_patient_projection.npy'
F487 = AUDIT / '487_R5G_patient_projection_ledger.csv'
F488 = AUDIT / '488_R5G_cross_cohort_MNN_edges.csv'
F489 = AUDIT / '489_R5G_Leiden_raw_assignments.csv'
F490 = AUDIT / '490_R5G_bootstrap_state_stability.csv'
F491 = AUDIT / '491_R5G_recurrent_state_summary.csv'
F492 = AUDIT / '492_R5G_recurrent_state_patient_assignments.csv'
F493 = AUDIT / '493_R5G_state_gene_meta_statistics.csv'
F494 = AUDIT / '494_R5G_state_graph_coherence.csv'
F495 = AUDIT / '495_R5G_final_summary.csv'
F496 = AUDIT / '496_R5G_hard_gates.csv'
F497 = AUDIT / '497_R5G_manifest.csv'
GENE_MODEL_FILE = R5G_ROOT / 'gene_RGAE_final_epoch_030.pt'
EXPECTED_SHA = {RANK_FILE: '8c2a914c11562073fb3e6bfbc36f8e2c0fc5d76ca88237a938761e4ca79f7a10', PRESENCE_FILE: '97f404ededfc48f6484584a8564710db294ab79f809563f20e0bfaa69917c49b', TOPOLOGY_FILE: '97e214d5eea1b2e04125d4fcd0f5a0703f1f67c6fde02316eb51d2b3f0f7f9e7', NODE_FILE: 'c0596efbded29dcb848c1620c25bb00344012c72cfda2ca4ac5efc256c1baf20', DISCOVERY_LEDGER_FILE: 'c33172332ef0ab385c0701472338895acce5d993355f57ec2d13e83a35be4778', SPLIT_FILE: '7cf26469d181b780befeed6c6e01430ab0d91f3c0bdb73dfed15cc684c388db9', R5S_SUMMARY: '6bb7d5d2973f3db4e7c10f401f9b2bfd4b59180535506d2614ed2df122008216', R5S_HARD_GATES: 'fb8961042f499658b62b8878ba8d17c575b505eada2022b7302ea6e22afeccf2', R5S_MANIFEST: '66427be3a1861e86375802dd278db64564a121e94dd7c5fa2c2f2a312e285dfd'}
DISCOVERY_GSES = ['GSE15434', 'GSE161532', 'GSE1729', 'GSE34860', 'GSE37021', 'GSE37642', 'GSE6891', 'GSE71014']
FORBIDDEN_GSES = {'GSE114868', 'GSE13159', 'GSE9476', 'GSE201492'}
EXPECTED_COUNTS = {'GSE15434': 251, 'GSE161532': 61, 'GSE1729': 43, 'GSE34860': 78, 'GSE37021': 97, 'GSE37642': 562, 'GSE6891': 537, 'GSE71014': 104}
N_NODES = 9554
N_RELATIONS = 4
GENE_INPUT_DIM = 32
GENE_HIDDEN_DIM = 64
GENE_LATENT_DIM = 32
RELATION_ORDER = ['signaling', 'PPI', 'TF_regulation', 'kinase_substrate']
RGAE_EPOCHS = 30
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-05
MAX_POS_PER_RELATION_PER_EPOCH = 30000
MNN_K = 10
LEIDEN_RESOLUTION = 1.0
BOOTSTRAPS = 30
BOOTSTRAP_EDGE_KEEP_FRACTION = 0.8
MIN_STATE_PATIENTS = 40
MIN_STATE_COHORTS = 4
MAX_SINGLE_COHORT_FRACTION = 0.4
MIN_BOOTSTRAP_JACCARD = 0.6
MIN_MNN_NODE_COVERAGE = 0.75
MIN_RECURRENT_STATE_COVERAGE = 0.5
STATE_META_FDR = 0.05
STATE_DIRECTION_CONSISTENCY = 0.8
GRAPH_PERMUTATIONS = 200

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    path = Path(path)
    require(path.is_file(), f'Missing file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(chunk_size), b''):
            h.update(block)
    return h.hexdigest()

def json_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')

def csv_bytes(df, sort_cols=None):
    x = df.copy()
    if sort_cols:
        x = x.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)
    return x.to_csv(index=False, lineterminator='\n').encode('utf-8')

def atomic_write(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)

def write_once_or_verify(path, data):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == data, f'Existing locked artifact differs: {path}')
        print('PASS existing:', path.name)
    else:
        atomic_write(path, data)
        print('CREATED:', path.name)

def atomic_numpy_save(array, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.save(f, array, allow_pickle=False)
    os.replace(tmp, path)

def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p, kind='mergesort')
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1, dtype=float)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    result = np.empty_like(q)
    result[order] = q
    return result
print()
print('=' * 105)
print('AML-GraphRx — R5-2G GENE-FIRST STATE DISCOVERY')
print('=' * 105)
for path, expected in EXPECTED_SHA.items():
    observed = sha256_file(path)
    require(observed == expected, f'SHA mismatch:\n{path}\nexpected={expected}\nobserved={observed}')
    print('SHA256 PASS:', path.name)
protocol = {'protocol_id': 'AML_GRAPHRX_R5_2G_GENE_FIRST_STATE_DISCOVERY_V1', 'analysis_type': 'GENE_FIRST_GRAPH_INFORMED_PATIENT_STATE_DISCOVERY', 'gene_GNN_training_data': 'locked biological gene graph only', 'patient_data_used_in_gene_GNN_training': False, 'discovery_cohorts': DISCOVERY_GSES, 'discovery_AML_n': 1733, 'forbidden_cohorts': sorted(FORBIDDEN_GSES), 'gene_encoder': '2-layer relation-aware graph autoencoder', 'gene_latent_dimension': GENE_LATENT_DIM, 'gene_decoder': 'DistMult relation-aware edge reconstruction', 'gene_GAE_epochs': RGAE_EPOCHS, 'patient_projection': 'within-cohort gene-median-centered percentile rank matrix multiplied by standardized gene graph embeddings', 'patient_projection_postprocessing': 'within-cohort median/IQR standardization followed by patient L2 normalization', 'clustering_graph': 'cross-cohort-only mutual nearest neighbor', 'within_GSE_edges_allowed': False, 'MNN_k': MNN_K, 'community_detection': 'Leiden RBConfiguration', 'Leiden_resolution': LEIDEN_RESOLUTION, 'bootstrap_replicates': BOOTSTRAPS, 'bootstrap': '80% edge-retention perturbation', 'recurrent_state_rule': {'minimum_patients': MIN_STATE_PATIENTS, 'minimum_independent_cohorts': MIN_STATE_COHORTS, 'maximum_single_cohort_fraction': MAX_SINGLE_COHORT_FRACTION, 'minimum_bootstrap_median_Jaccard': MIN_BOOTSTRAP_JACCARD}, 'state_gene_rule': {'meta_BH_FDR': STATE_META_FDR, 'direction_consistency': STATE_DIRECTION_CONSISTENCY}, 'supervised_labels_used': False, 'patient_GNN_used': False, 'feature_selection': False, 'synthetic_samples': False, 'primary_R5_2G_acceptance': {'MNN_node_coverage_min': MIN_MNN_NODE_COVERAGE, 'recurrent_states_min': 2, 'recurrent_state_patient_coverage_min': MIN_RECURRENT_STATE_COVERAGE}}
write_once_or_verify(F481, json_bytes(protocol))
discovery = pd.read_csv(DISCOVERY_LEDGER_FILE, dtype={'analysis_unit_id': str, 'analysis_unit_index_0based': int, 'gse': str})
require(len(discovery) == 1733, 'Discovery ledger must contain 1733 AML samples.')
require(set(discovery['gse']) == set(DISCOVERY_GSES), 'Discovery cohort set changed.')
require(len(set(discovery['gse']) & FORBIDDEN_GSES) == 0, 'Forbidden GSE entered R5-2G.')
for gse, expected_n in EXPECTED_COUNTS.items():
    observed = int(discovery['gse'].eq(gse).sum())
    require(observed == expected_n, f'{gse}: expected {expected_n}; observed {observed}')
rank_mm = np.load(RANK_FILE, mmap_mode='r', allow_pickle=False)
presence_mm = np.load(PRESENCE_FILE, mmap_mode='r', allow_pickle=False)
nodes = pd.read_csv(NODE_FILE, dtype=str, keep_default_na=False)
require(rank_mm.shape == (4554, 9554), 'Rank matrix changed.')
require(presence_mm.shape == (4554, 9554), 'Presence matrix changed.')
require(len(nodes) == N_NODES, 'Node index changed.')
require('canonical_NCBI_Gene_ID' in nodes.columns, 'Node index lacks canonical_NCBI_Gene_ID.')
topology = torch.load(TOPOLOGY_FILE, map_location='cpu', weights_only=False)
require(int(topology['node_count']) == N_NODES, 'Graph node count changed.')
message_edges = []
message_types = []
reconstruction_edges_cpu = {}
for relation_id, relation in enumerate(RELATION_ORDER):
    edge = topology['edge_index_by_relation'][relation].long().contiguous()
    message_edges.append(edge)
    message_types.append(torch.full((edge.shape[1],), relation_id, dtype=torch.long))
    nonself = edge[0] != edge[1]
    reconstruction_edges_cpu[relation_id] = edge[:, nonself].long().contiguous()
ppi = topology['edge_index_by_relation']['PPI'].long()
ppi_nonself = ppi[0] != ppi[1]
ppi_reverse = ppi[:, ppi_nonself].flip(0).contiguous()
message_edges.append(ppi_reverse)
message_types.append(torch.full((ppi_reverse.shape[1],), RELATION_ORDER.index('PPI'), dtype=torch.long))
EDGE_INDEX = torch.cat(message_edges, dim=1)
EDGE_TYPE = torch.cat(message_types, dim=0)
sort_order = torch.argsort(EDGE_TYPE, stable=True)
EDGE_TYPE = EDGE_TYPE[sort_order]
EDGE_INDEX = EDGE_INDEX[:, sort_order]
EDGE_INDEX_GPU = EDGE_INDEX.to(DEVICE)
EDGE_TYPE_GPU = EDGE_TYPE.to(DEVICE)
print()
print('Message-passing arcs:', EDGE_INDEX.shape[1])

class GeneRGAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.node_embedding = nn.Embedding(N_NODES, GENE_INPUT_DIM)
        self.input_projection = nn.Linear(GENE_INPUT_DIM, GENE_HIDDEN_DIM)
        self.conv1 = RGCNConv(GENE_HIDDEN_DIM, GENE_HIDDEN_DIM, num_relations=N_RELATIONS, aggr='mean', root_weight=True, bias=True, is_sorted=True)
        self.conv2 = RGCNConv(GENE_HIDDEN_DIM, GENE_HIDDEN_DIM, num_relations=N_RELATIONS, aggr='mean', root_weight=True, bias=True, is_sorted=True)
        self.norm1 = nn.LayerNorm(GENE_HIDDEN_DIM)
        self.norm2 = nn.LayerNorm(GENE_HIDDEN_DIM)
        self.latent_head = nn.Linear(GENE_HIDDEN_DIM, GENE_LATENT_DIM)
        self.relation_embedding = nn.Parameter(torch.empty(N_RELATIONS, GENE_LATENT_DIM))
        nn.init.xavier_uniform_(self.relation_embedding)

    def encode(self):
        node_ids = torch.arange(N_NODES, dtype=torch.long, device=DEVICE)
        x = self.node_embedding(node_ids)
        h = F.gelu(self.input_projection(x))
        h1 = self.conv1(h, EDGE_INDEX_GPU, EDGE_TYPE_GPU)
        h = self.norm1(h + F.gelu(h1))
        h2 = self.conv2(h, EDGE_INDEX_GPU, EDGE_TYPE_GPU)
        h = self.norm2(h + F.gelu(h2))
        z = self.latent_head(h)
        return z

    def score(self, z, edge_index, relation_id):
        source = z[edge_index[0]]
        target = z[edge_index[1]]
        relation = self.relation_embedding[relation_id]
        return torch.sum(source * relation[None, :] * target, dim=1)
gene_model = GeneRGAE().to(DEVICE)
optimizer = torch.optim.AdamW(gene_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
training_rows = []
print()
print('=' * 100)
print('TRAINING GENE R-GAE')
print('=' * 100)
for epoch in range(1, RGAE_EPOCHS + 1):
    gene_model.train()
    optimizer.zero_grad(set_to_none=True)
    z = gene_model.encode()
    relation_losses = []
    for relation_id in range(N_RELATIONS):
        pos_all = reconstruction_edges_cpu[relation_id]
        n_pos_all = pos_all.shape[1]
        n_use = min(n_pos_all, MAX_POS_PER_RELATION_PER_EPOCH)
        generator = torch.Generator()
        generator.manual_seed(SEED + epoch * 100 + relation_id)
        if n_use < n_pos_all:
            perm = torch.randperm(n_pos_all, generator=generator)[:n_use]
            pos_cpu = pos_all[:, perm]
        else:
            pos_cpu = pos_all
        torch.manual_seed(SEED + epoch * 1000 + relation_id)
        neg_cpu = negative_sampling(edge_index=pos_all, num_nodes=N_NODES, num_neg_samples=pos_cpu.shape[1], method='sparse')
        pos = pos_cpu.to(DEVICE)
        neg = neg_cpu.to(DEVICE)
        pos_score = gene_model.score(z, pos, relation_id)
        neg_score = gene_model.score(z, neg, relation_id)
        pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        relation_loss = (pos_loss + neg_loss) / 2.0
        relation_losses.append(relation_loss)
    loss = torch.stack(relation_losses).mean()
    require(torch.isfinite(loss).item(), 'Non-finite gene R-GAE loss.')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gene_model.parameters(), max_norm=5.0)
    optimizer.step()
    training_rows.append({'epoch': epoch, 'mean_relation_BCE': float(loss.detach().cpu())})
    if epoch == 1 or epoch % 5 == 0 or epoch == RGAE_EPOCHS:
        print(f'epoch={epoch:02d}/{RGAE_EPOCHS} loss={float(loss.detach().cpu()):.6f}')
training_history = pd.DataFrame(training_rows)
write_once_or_verify(F482, csv_bytes(training_history, ['epoch']))
torch.save({'protocol_id': protocol['protocol_id'], 'epoch': RGAE_EPOCHS, 'model_state_dict': gene_model.state_dict()}, GENE_MODEL_FILE)
gene_model.eval()
with torch.no_grad():
    gene_embeddings_raw = gene_model.encode().float().cpu().numpy().astype(np.float32)
require(gene_embeddings_raw.shape == (N_NODES, GENE_LATENT_DIM), 'Gene embedding shape changed.')
require(np.isfinite(gene_embeddings_raw).all(), 'Non-finite gene embedding.')
gene_centered = gene_embeddings_raw - gene_embeddings_raw.mean(axis=0, keepdims=True)
gene_cov = gene_centered.T @ gene_centered / float(N_NODES - 1)
gene_eig = np.linalg.eigvalsh(gene_cov)
gene_eig = np.clip(gene_eig, 0, None)
gene_prob = gene_eig / gene_eig.sum()
gene_prob = gene_prob[gene_prob > 0]
GENE_EFFECTIVE_RANK = float(np.exp(-np.sum(gene_prob * np.log(gene_prob))))
relation_audit_rows = []
gene_model.eval()
with torch.no_grad():
    z_gpu = gene_model.encode()
    for relation_id, relation in enumerate(RELATION_ORDER):
        pos_all = reconstruction_edges_cpu[relation_id]
        n_eval = min(20000, pos_all.shape[1])
        rng = np.random.default_rng(SEED + relation_id)
        selected = rng.choice(pos_all.shape[1], size=n_eval, replace=False)
        pos_cpu = pos_all[:, selected]
        torch.manual_seed(SEED + 5000 + relation_id)
        neg_cpu = negative_sampling(pos_all, num_nodes=N_NODES, num_neg_samples=n_eval, method='sparse')
        pos_score = gene_model.score(z_gpu, pos_cpu.to(DEVICE), relation_id).cpu().numpy()
        neg_score = gene_model.score(z_gpu, neg_cpu.to(DEVICE), relation_id).cpu().numpy()
        y = np.concatenate([np.ones(len(pos_score)), np.zeros(len(neg_score))])
        score = np.concatenate([pos_score, neg_score])
        auc = float(roc_auc_score(y, score))
        relation_audit_rows.append({'relation': relation, 'evaluation_positive_edges_n': len(pos_score), 'AUROC_training_fit_descriptive': auc, 'positive_score_mean': float(np.mean(pos_score)), 'negative_score_mean': float(np.mean(neg_score))})
gene_audit = pd.DataFrame(relation_audit_rows)
MEAN_GENE_LINK_AUC = float(gene_audit['AUROC_training_fit_descriptive'].mean())
gene_audit = pd.concat([gene_audit, pd.DataFrame([{'relation': 'ALL_RELATIONS_SUMMARY', 'evaluation_positive_edges_n': gene_audit['evaluation_positive_edges_n'].sum(), 'AUROC_training_fit_descriptive': MEAN_GENE_LINK_AUC, 'positive_score_mean': np.nan, 'negative_score_mean': np.nan}])], ignore_index=True)
write_once_or_verify(F485, csv_bytes(gene_audit, ['relation']))
embedding_mean = gene_embeddings_raw.mean(axis=0, keepdims=True)
embedding_std = gene_embeddings_raw.std(axis=0, keepdims=True)
embedding_std[embedding_std < 1e-08] = 1.0
gene_embeddings = (gene_embeddings_raw - embedding_mean) / embedding_std
row_norm = np.linalg.norm(gene_embeddings, axis=1, keepdims=True)
gene_embeddings = (gene_embeddings / np.maximum(row_norm, 1e-12)).astype(np.float32)
atomic_numpy_save(gene_embeddings, F483)
gene_embedding_ledger = nodes[['canonical_NCBI_Gene_ID']].copy()
gene_embedding_ledger['gene_index_0based'] = np.arange(N_NODES)
for d in range(GENE_LATENT_DIM):
    gene_embedding_ledger[f'GZ{d + 1:02d}'] = gene_embeddings[:, d]
write_once_or_verify(F484, csv_bytes(gene_embedding_ledger, ['gene_index_0based']))
discovery_indices = discovery['analysis_unit_index_0based'].to_numpy(dtype=int)
X = np.asarray(rank_mm[discovery_indices, :], dtype=np.float32)
require(X.shape == (1733, N_NODES), 'Discovery expression shape changed.')
require(np.isfinite(X).all(), 'Non-finite discovery expression.')
P = np.zeros((len(discovery), GENE_LATENT_DIM), dtype=np.float32)
for gse in DISCOVERY_GSES:
    cohort_mask = discovery['gse'].eq(gse).to_numpy()
    Xg = X[cohort_mask]
    cohort_gene_median = np.median(Xg, axis=0, keepdims=True)
    residual = Xg - cohort_gene_median
    projected = residual @ gene_embeddings / math.sqrt(N_NODES)
    med = np.median(projected, axis=0, keepdims=True)
    q25 = np.quantile(projected, 0.25, axis=0, keepdims=True)
    q75 = np.quantile(projected, 0.75, axis=0, keepdims=True)
    iqr = q75 - q25
    iqr[iqr < 1e-06] = 1.0
    projected = (projected - med) / iqr
    projected = np.clip(projected, -8.0, 8.0)
    norm_rows = np.linalg.norm(projected, axis=1, keepdims=True)
    projected = projected / np.maximum(norm_rows, 1e-12)
    P[cohort_mask] = projected.astype(np.float32)
require(np.isfinite(P).all(), 'Patient projection contains non-finite values.')
atomic_numpy_save(P, F486)
projection_ledger = discovery.copy()
for d in range(GENE_LATENT_DIM):
    projection_ledger[f'P{d + 1:02d}'] = P[:, d]
write_once_or_verify(F487, csv_bytes(projection_ledger, ['analysis_unit_index_0based']))
edge_rows = []
for i in range(len(DISCOVERY_GSES)):
    for j in range(i + 1, len(DISCOVERY_GSES)):
        gse_a = DISCOVERY_GSES[i]
        gse_b = DISCOVERY_GSES[j]
        idx_a = np.where(discovery['gse'].eq(gse_a).to_numpy())[0]
        idx_b = np.where(discovery['gse'].eq(gse_b).to_numpy())[0]
        A = P[idx_a]
        B = P[idx_b]
        similarity = A @ B.T
        k_ab = min(MNN_K, len(idx_b))
        k_ba = min(MNN_K, len(idx_a))
        top_b_for_a = np.argpartition(-similarity, kth=k_ab - 1, axis=1)[:, :k_ab]
        top_a_for_b = np.argpartition(-similarity.T, kth=k_ba - 1, axis=1)[:, :k_ba]
        reverse_sets = [set(row.tolist()) for row in top_a_for_b]
        for local_a in range(len(idx_a)):
            for local_b in top_b_for_a[local_a]:
                local_b = int(local_b)
                if local_a not in reverse_sets[local_b]:
                    continue
                weight = float(similarity[local_a, local_b])
                if weight <= 0:
                    continue
                global_a = int(idx_a[local_a])
                global_b = int(idx_b[local_b])
                require(discovery.iloc[global_a]['gse'] != discovery.iloc[global_b]['gse'], 'CRITICAL: within-GSE MNN edge created.')
                edge_rows.append({'patient_node_u': global_a, 'patient_node_v': global_b, 'gse_u': gse_a, 'gse_v': gse_b, 'cosine_similarity': weight})
MNN = pd.DataFrame(edge_rows)
require(len(MNN) > 0, 'Cross-cohort MNN graph has zero edges.')
write_once_or_verify(F488, csv_bytes(MNN, ['patient_node_u', 'patient_node_v']))
connected_nodes = set(MNN['patient_node_u'].tolist() + MNN['patient_node_v'].tolist())
MNN_NODE_COVERAGE = len(connected_nodes) / len(discovery)
print()
print('MNN edges:', len(MNN))
print('MNN node coverage:', f'{MNN_NODE_COVERAGE:.4f}')
edges_for_igraph = list(zip(MNN['patient_node_u'].astype(int), MNN['patient_node_v'].astype(int)))
graph = ig.Graph(n=len(discovery), edges=edges_for_igraph, directed=False)
graph.es['weight'] = MNN['cosine_similarity'].astype(float).tolist()
partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=LEIDEN_RESOLUTION, seed=SEED)
raw_membership = np.asarray(partition.membership, dtype=int)
require(len(raw_membership) == 1733, 'Leiden membership length mismatch.')
raw_assignments = discovery.copy()
raw_assignments['raw_community'] = raw_membership
write_once_or_verify(F489, csv_bytes(raw_assignments, ['analysis_unit_index_0based']))
print('Raw Leiden communities:', len(np.unique(raw_membership)))
full_communities = {}
for community_id in np.unique(raw_membership):
    full_communities[int(community_id)] = set(np.where(raw_membership == community_id)[0].tolist())
stability_values = {community_id: [] for community_id in full_communities}
rng = np.random.default_rng(SEED + 9000)
edge_array = np.asarray(edges_for_igraph, dtype=int)
edge_weights = MNN['cosine_similarity'].to_numpy(dtype=float)
n_keep = int(round(BOOTSTRAP_EDGE_KEEP_FRACTION * len(edge_array)))
for bootstrap_id in range(BOOTSTRAPS):
    keep = np.sort(rng.choice(len(edge_array), size=n_keep, replace=False))
    boot_graph = ig.Graph(n=len(discovery), edges=[tuple(x) for x in edge_array[keep]], directed=False)
    boot_graph.es['weight'] = edge_weights[keep].tolist()
    boot_partition = leidenalg.find_partition(boot_graph, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=LEIDEN_RESOLUTION, seed=SEED + bootstrap_id + 1)
    boot_membership = np.asarray(boot_partition.membership, dtype=int)
    boot_communities = {}
    for boot_id in np.unique(boot_membership):
        boot_communities[int(boot_id)] = set(np.where(boot_membership == boot_id)[0].tolist())
    for full_id, full_nodes in full_communities.items():
        best_jaccard = 0.0
        for boot_nodes in boot_communities.values():
            intersection = len(full_nodes & boot_nodes)
            union = len(full_nodes | boot_nodes)
            if union == 0:
                continue
            jaccard = intersection / union
            if jaccard > best_jaccard:
                best_jaccard = jaccard
        stability_values[full_id].append(best_jaccard)
bootstrap_rows = []
for community_id, values in stability_values.items():
    bootstrap_rows.append({'raw_community': community_id, 'bootstrap_replicates': BOOTSTRAPS, 'median_best_Jaccard': float(np.median(values)), 'mean_best_Jaccard': float(np.mean(values)), 'minimum_best_Jaccard': float(np.min(values))})
bootstrap_stability = pd.DataFrame(bootstrap_rows)
write_once_or_verify(F490, csv_bytes(bootstrap_stability, ['raw_community']))
state_rows = []
for community_id in sorted(np.unique(raw_membership)):
    mask = raw_membership == community_id
    sub = discovery.loc[mask]
    n = len(sub)
    cohort_counts = sub['gse'].value_counts()
    cohorts_n = len(cohort_counts)
    max_cohort_fraction = float(cohort_counts.max() / n) if n > 0 else 1.0
    stability = float(bootstrap_stability.loc[bootstrap_stability['raw_community'].eq(community_id), 'median_best_Jaccard'].iloc[0])
    recurrent = bool(n >= MIN_STATE_PATIENTS and cohorts_n >= MIN_STATE_COHORTS and (max_cohort_fraction <= MAX_SINGLE_COHORT_FRACTION) and (stability >= MIN_BOOTSTRAP_JACCARD))
    state_rows.append({'raw_community': int(community_id), 'patients_n': int(n), 'independent_cohorts_n': int(cohorts_n), 'max_single_cohort_fraction': max_cohort_fraction, 'bootstrap_median_Jaccard': stability, 'RECURRENT_AML_STATE': recurrent})
state_summary = pd.DataFrame(state_rows)
valid_states_table = state_summary.loc[state_summary['RECURRENT_AML_STATE'].eq(True)].sort_values(['patients_n', 'independent_cohorts_n', 'raw_community'], ascending=[False, False, True]).reset_index(drop=True)
state_name_map = {int(row.raw_community): f'AML_STATE_{i:02d}' for i, row in enumerate(valid_states_table.itertuples(index=False), start=1)}
state_summary['state_id'] = state_summary['raw_community'].map(state_name_map).fillna('NON_RECURRENT')
write_once_or_verify(F491, csv_bytes(state_summary, ['raw_community']))
patient_states = raw_assignments.copy()
patient_states['state_id'] = patient_states['raw_community'].map(state_name_map).fillna('UNRESOLVED')
write_once_or_verify(F492, csv_bytes(patient_states, ['analysis_unit_index_0based']))
VALID_STATES = sorted([x for x in patient_states['state_id'].unique() if x != 'UNRESOLVED'])
RECURRENT_ASSIGNED_N = int(patient_states['state_id'].ne('UNRESOLVED').sum())
RECURRENT_COVERAGE = RECURRENT_ASSIGNED_N / 1733.0
assigned_only = patient_states.loc[patient_states['state_id'].ne('UNRESOLVED')]
if len(assigned_only) > 0 and assigned_only['state_id'].nunique() > 1:
    STATE_COHORT_NMI = float(normalized_mutual_info_score(assigned_only['gse'], assigned_only['state_id']))
else:
    STATE_COHORT_NMI = np.nan
print()
print('Recurrent AML states:', len(VALID_STATES))
print('Recurrent-state coverage:', f'{RECURRENT_COVERAGE:.4f}')
print('State/cohort NMI descriptive:', STATE_COHORT_NMI)
state_gene_rows = []
for state_id in VALID_STATES:
    print()
    print('Meta-analyzing state:', state_id)
    cohort_effects = []
    cohort_z_values = []
    cohort_weights = []
    eligible_cohorts = []
    for gse in DISCOVERY_GSES:
        cohort_mask = discovery['gse'].eq(gse).to_numpy()
        state_mask = patient_states['state_id'].eq(state_id).to_numpy() & cohort_mask
        other_mask = patient_states['state_id'].ne(state_id).to_numpy() & cohort_mask
        n_state = int(state_mask.sum())
        n_other = int(other_mask.sum())
        if n_state < 5 or n_other < 10:
            continue
        A = X[state_mask]
        B = X[other_mask]
        effect = A.mean(axis=0) - B.mean(axis=0)
        test = ttest_ind(A, B, axis=0, equal_var=False, nan_policy='omit')
        p = np.asarray(test.pvalue, dtype=float)
        p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
        p = np.clip(p, 1e-300, 1.0)
        signed_z = np.sign(effect) * norm.isf(p / 2.0)
        weight = math.sqrt(n_state * n_other / (n_state + n_other))
        cohort_effects.append(effect.astype(np.float64))
        cohort_z_values.append(signed_z.astype(np.float64))
        cohort_weights.append(float(weight))
        eligible_cohorts.append(gse)
    if len(eligible_cohorts) < 3:
        continue
    E = np.stack(cohort_effects, axis=0)
    Z = np.stack(cohort_z_values, axis=0)
    W = np.asarray(cohort_weights, dtype=float)
    meta_z = (Z * W[:, None]).sum(axis=0) / math.sqrt(float(np.sum(W ** 2)))
    meta_p = 2.0 * norm.sf(np.abs(meta_z))
    meta_fdr = bh_fdr(meta_p)
    weighted_effect = (E * W[:, None]).sum(axis=0) / W.sum()
    meta_direction = np.sign(meta_z)
    consistency = np.mean(np.sign(E) == meta_direction[None, :], axis=0)
    signature_member = (meta_fdr < STATE_META_FDR) & (consistency >= STATE_DIRECTION_CONSISTENCY) & (meta_direction != 0)
    for gene_index in range(N_NODES):
        state_gene_rows.append({'state_id': state_id, 'gene_index_0based': gene_index, 'canonical_NCBI_Gene_ID': nodes.iloc[gene_index]['canonical_NCBI_Gene_ID'], 'eligible_cohorts_n': len(eligible_cohorts), 'eligible_cohorts': ';'.join(eligible_cohorts), 'meta_effect': float(weighted_effect[gene_index]), 'meta_Z': float(meta_z[gene_index]), 'meta_p': float(meta_p[gene_index]), 'meta_FDR_BH': float(meta_fdr[gene_index]), 'direction_consistency_fraction': float(consistency[gene_index]), 'direction': 'UP_IN_STATE' if meta_z[gene_index] > 0 else 'DOWN_IN_STATE', 'STATE_SIGNATURE_GENE': bool(signature_member[gene_index])})
state_gene_meta = pd.DataFrame(state_gene_rows)
write_once_or_verify(F493, csv_bytes(state_gene_meta, ['state_id', 'gene_index_0based'] if len(state_gene_meta) else None))
undirected_pairs = set()
for relation in RELATION_ORDER:
    edge = topology['edge_index_by_relation'][relation].cpu().numpy()
    for u, v in zip(edge[0], edge[1]):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        if u > v:
            u, v = (v, u)
        undirected_pairs.add((u, v))
rows = []
cols = []
for u, v in undirected_pairs:
    rows.extend([u, v])
    cols.extend([v, u])
BIO_ADJ = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(N_NODES, N_NODES))
graph_coherence_rows = []
graph_rng = np.random.default_rng(SEED + 17000)
for state_id in VALID_STATES:
    if len(state_gene_meta) == 0:
        continue
    genes = state_gene_meta.loc[state_gene_meta['state_id'].eq(state_id) & state_gene_meta['STATE_SIGNATURE_GENE'].eq(True), 'gene_index_0based'].to_numpy(dtype=int)
    n_genes = len(genes)
    if n_genes < 10:
        graph_coherence_rows.append({'state_id': state_id, 'signature_genes_n': n_genes, 'observed_internal_edges': np.nan, 'random_mean_internal_edges': np.nan, 'edge_enrichment_ratio': np.nan, 'empirical_p': 1.0})
        continue
    observed_edges = BIO_ADJ[genes, :][:, genes].nnz / 2.0
    random_edges = []
    for permutation in range(GRAPH_PERMUTATIONS):
        random_genes = graph_rng.choice(N_NODES, size=n_genes, replace=False)
        random_edge_count = BIO_ADJ[random_genes, :][:, random_genes].nnz / 2.0
        random_edges.append(random_edge_count)
    random_edges = np.asarray(random_edges, dtype=float)
    random_mean = float(random_edges.mean())
    empirical_p = float((1 + np.sum(random_edges >= observed_edges)) / (GRAPH_PERMUTATIONS + 1))
    enrichment = float(observed_edges / random_mean) if random_mean > 0 else np.nan
    graph_coherence_rows.append({'state_id': state_id, 'signature_genes_n': n_genes, 'observed_internal_edges': float(observed_edges), 'random_mean_internal_edges': random_mean, 'edge_enrichment_ratio': enrichment, 'empirical_p': empirical_p})
graph_coherence = pd.DataFrame(graph_coherence_rows)
if len(graph_coherence) > 0:
    graph_coherence['FDR_BH'] = bh_fdr(graph_coherence['empirical_p'].to_numpy(dtype=float))
write_once_or_verify(F494, csv_bytes(graph_coherence, ['state_id'] if len(graph_coherence) else None))
GENE_GRAPH_ACCEPTED = bool(GENE_EFFECTIVE_RANK >= 8.0 and MEAN_GENE_LINK_AUC >= 0.7)
MNN_ACCEPTED = bool(MNN_NODE_COVERAGE >= MIN_MNN_NODE_COVERAGE)
STATE_DISCOVERY_ACCEPTED = bool(len(VALID_STATES) >= 2 and RECURRENT_COVERAGE >= MIN_RECURRENT_STATE_COVERAGE)
R5_2G_ACCEPTED = bool(GENE_GRAPH_ACCEPTED and MNN_ACCEPTED and STATE_DISCOVERY_ACCEPTED)
if len(state_gene_meta) > 0:
    signature_counts = state_gene_meta.loc[state_gene_meta['STATE_SIGNATURE_GENE'].eq(True)].groupby('state_id').size().to_dict()
else:
    signature_counts = {}
summary = pd.DataFrame([('protocol_id', protocol['protocol_id']), ('gene_graph_nodes', N_NODES), ('gene_GNN_epochs', RGAE_EPOCHS), ('gene_embedding_dimension', GENE_LATENT_DIM), ('gene_embedding_effective_rank', GENE_EFFECTIVE_RANK), ('mean_gene_link_AUROC_descriptive', MEAN_GENE_LINK_AUC), ('discovery_AML_n', 1733), ('discovery_cohorts_n', 8), ('patient_GNN_used', False), ('forbidden_GSEs_used', False), ('cross_cohort_MNN_edges_n', len(MNN)), ('MNN_node_coverage', MNN_NODE_COVERAGE), ('raw_Leiden_communities_n', int(len(np.unique(raw_membership)))), ('recurrent_AML_states_n', len(VALID_STATES)), ('patients_in_recurrent_states_n', RECURRENT_ASSIGNED_N), ('recurrent_state_patient_coverage', RECURRENT_COVERAGE), ('state_cohort_NMI_descriptive', STATE_COHORT_NMI), ('state_signature_gene_counts', json.dumps(signature_counts, sort_keys=True)), ('GENE_GRAPH_ACCEPTED', GENE_GRAPH_ACCEPTED), ('MNN_ACCEPTED', MNN_ACCEPTED), ('STATE_DISCOVERY_ACCEPTED', STATE_DISCOVERY_ACCEPTED), ('R5_2G_ACCEPTED', R5_2G_ACCEPTED), ('next_stage', 'R5_3_HELDOUT_AML_CONTROL_VALIDATION' if R5_2G_ACCEPTED else 'STOP_AND_INTERPRET_NEGATIVE_STATE_DISCOVERY')], columns=['metric', 'value'])
write_once_or_verify(F495, csv_bytes(summary))
gate_rows = []

def add_gate(gate, passed, detail):
    gate_rows.append({'gate': gate, 'status': 'PASS' if bool(passed) else 'FAIL', 'detail': str(detail)})
add_gate('exact_8_discovery_GSEs', set(discovery['gse']) == set(DISCOVERY_GSES), sorted(discovery['gse'].unique()))
add_gate('exact_1733_AML', len(discovery) == 1733, len(discovery))
add_gate('forbidden_GSEs_absent', len(set(discovery['gse']) & FORBIDDEN_GSES) == 0, sorted(FORBIDDEN_GSES))
add_gate('gene_GNN_patient_data_used', True, False)
add_gate('patient_GNN_used', True, False)
add_gate('within_GSE_MNN_edges_absent', bool((MNN['gse_u'] != MNN['gse_v']).all()), True)
add_gate('gene_embedding_effective_rank', GENE_EFFECTIVE_RANK >= 8.0, GENE_EFFECTIVE_RANK)
add_gate('gene_link_reconstruction', MEAN_GENE_LINK_AUC >= 0.7, MEAN_GENE_LINK_AUC)
add_gate('MNN_node_coverage', MNN_ACCEPTED, MNN_NODE_COVERAGE)
add_gate('minimum_two_recurrent_states', len(VALID_STATES) >= 2, len(VALID_STATES))
add_gate('recurrent_state_patient_coverage', RECURRENT_COVERAGE >= MIN_RECURRENT_STATE_COVERAGE, RECURRENT_COVERAGE)
hard_gates = pd.DataFrame(gate_rows)
write_once_or_verify(F496, csv_bytes(hard_gates, ['gate']))
manifest_rows = []

def add_manifest(role, path):
    path = Path(path)
    require(path.exists(), f'Missing manifest artifact: {path}')
    manifest_rows.append({'role': role, 'path': str(path), 'size_bytes': int(path.stat().st_size), 'sha256': sha256_file(path), 'protocol_id': protocol['protocol_id']})
for role, path in [('rank_matrix', RANK_FILE), ('presence_mask', PRESENCE_FILE), ('graph_topology', TOPOLOGY_FILE), ('discovery_ledger', DISCOVERY_LEDGER_FILE), ('protocol', F481), ('gene_training_history', F482), ('gene_embeddings', F483), ('gene_embedding_ledger', F484), ('gene_RGAE_audit', F485), ('patient_projection', F486), ('patient_projection_ledger', F487), ('MNN_edges', F488), ('raw_Leiden_assignments', F489), ('bootstrap_stability', F490), ('recurrent_state_summary', F491), ('patient_state_assignments', F492), ('state_gene_meta', F493), ('state_graph_coherence', F494), ('final_summary', F495), ('hard_gates', F496), ('gene_model', GENE_MODEL_FILE)]:
    add_manifest(role, path)
manifest = pd.DataFrame(manifest_rows)
write_once_or_verify(F497, csv_bytes(manifest, ['role']))
print()
print('=' * 105)
print('R5-2G — FINAL SUMMARY')
print('=' * 105)
print(summary.to_string(index=False))
print()
print('=' * 105)
print('R5-2G — RECURRENT STATE SUMMARY')
print('=' * 105)
print(state_summary.to_string(index=False))
print()
print('=' * 105)
print('R5-2G — GENE R-GAE AUDIT')
print('=' * 105)
print(gene_audit.to_string(index=False))
if len(graph_coherence) > 0:
    print()
    print('=' * 105)
    print('R5-2G — STATE GRAPH COHERENCE')
    print('=' * 105)
    print(graph_coherence.to_string(index=False))
print()
print('=' * 105)
print('R5-2G — HARD GATES')
print('=' * 105)
print(hard_gates.to_string(index=False))
print()
if R5_2G_ACCEPTED:
    print('R5-2G ACCEPTED ✅')
    print('RECURRENT AML STATES DISCOVERED ✅')
    print('NO MORE DISCOVERY TRAINING.')
    print('NEXT = R5-3 HELD-OUT AML/CONTROL VALIDATION')
else:
    print('R5-2G COMPLETE')
    print('PREDECLARED STATE-DISCOVERY RULE NOT SATISFIED ❌')
    print('DO NOT MODIFY K / RESOLUTION / STATE RULES POST-HOC.')
