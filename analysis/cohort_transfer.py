# Manuscript-linked analysis: Corrective leave-one-cohort-out assessment
# Source lineage and packaging changes: provenance/source_lineage.json
"""
AML-GraphRx R5X — Corrective state-transfer analysis
=====================================================
Purpose
-------
Test whether the poor transportability of R5 AML states is partly caused by a
mismatch between the representation used for discovery and the representation
used for scoring new samples.

Design
------
1) Freeze the topology-only gene embedding from R5-2G (no patient data were
   used to train the gene R-GAE).
2) Use ONE patient transform in both discovery and transfer:
      (rank - 0.5) -> gene-embedding projection -> training-only robust
      median/IQR scaling -> clipping -> row L2 normalization.
3) Outer leave-one-GSE-out (LOCO) across the eight discovery cohorts.
4) In every outer fold, discover recurrent states only in the seven training
   cohorts using cross-cohort MNN + Leiden + the original recurrence rules.
5) Fit fold-local state centroids only on the seven training cohorts.
6) Calibrate a state-specific uncertainty threshold only from training data,
   using the 5th percentile of leave-one-out same-state cosine similarity.
7) Assign the held-out GSE by nearest centroid; reject as UNCERTAIN when the
   winning similarity is below its training-derived state threshold.
8) Build fold-local gene signatures only from the seven training cohorts.
9) Do NOT use AML/control AUROC as a state-recovery endpoint.
10) After LOCO is complete, build an all-eight-cohort *candidate* transfer model
    with the same predeclared method. This candidate is not called externally
    validated. A truly new cohort must be processed later with the frozen
    candidate model.

This script never uses original 8-cohort state labels to train, tune, or score
an outer held-out fold. Original R5-2G labels are used only after LOCO for a
clearly labeled descriptive alignment of the final candidate states.
"""
from __future__ import annotations
import os as _release_os
try:
    from IPython.display import display
except ImportError:

    def display(value):
        print(value)
import os
import sys
import json
import math
import time
import hashlib
import subprocess
import importlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

def ensure_import(import_name: str, pip_name: Optional[str]=None):
    try:
        return importlib.import_module(import_name)
    except ImportError:
        pkg = pip_name or import_name
        raise RuntimeError('Install the recorded analysis dependencies before running this script; see docs/ENVIRONMENT.md.')
        return importlib.import_module(import_name)
np = ensure_import('numpy')
pd = ensure_import('pandas')
scipy = ensure_import('scipy')
sklearn = ensure_import('sklearn')
ig = ensure_import('igraph', 'python-igraph')
leidenalg = ensure_import('leidenalg')
from scipy.stats import ttest_ind, norm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
try:
    pass
    pass
except Exception:
    pass
PROJECT = Path(_release_os.environ.get('AML_GRAPHRX_ROOT', '/content/drive/MyDrive/AML_GraphRx'))
AUDIT = PROJECT / 'results' / 'audit'
STEP5 = PROJECT / 'data' / 'patient_graphs' / '20260823T123040Z'
AUDIT.mkdir(parents=True, exist_ok=True)
RANK_FILE = STEP5 / '20260823T123040Z_patient_rank_raw_float64.npy'
NODE_FILE = STEP5 / '20260823T123040Z_node_index.csv'
DISCOVERY_LEDGER_FILE = AUDIT / '452_R5DICE_discovery_ledger.csv'
GENE_EMBED_FILE = AUDIT / '483_R5G_gene_embeddings.npy'
GENE_EMBED_LEDGER_FILE = AUDIT / '484_R5G_gene_embedding_ledger.csv'
ORIGINAL_STATE_FILE = AUDIT / '492_R5G_recurrent_state_patient_assignments.csv'
R5G_MANIFEST_FILE = AUDIT / '497_R5G_manifest.csv'
F556 = AUDIT / '556_R5X_predeclared_corrective_transfer_protocol.json'
F557 = AUDIT / '557_R5X_LOCO_fold_summary.csv'
F558 = AUDIT / '558_R5X_LOCO_training_state_summary.csv'
F559 = AUDIT / '559_R5X_LOCO_heldout_assignments.csv'
F560 = AUDIT / '560_R5X_LOCO_training_centroid_diagnostics.csv'
F561 = AUDIT / '561_R5X_LOCO_state_signature_genes.csv'
F562 = AUDIT / '562_R5X_final_candidate_state_summary.csv'
F563 = AUDIT / '563_R5X_final_candidate_assignments.csv'
F564 = AUDIT / '564_R5X_corrected_vs_original_alignment.csv'
F565 = AUDIT / '565_R5X_final_candidate_centroids.npz'
F566 = AUDIT / '566_R5X_final_candidate_transform.npz'
F567 = AUDIT / '567_R5X_final_candidate_thresholds.csv'
F568 = AUDIT / '568_R5X_final_candidate_signature_genes.csv'
F569 = AUDIT / '569_R5X_final_summary.csv'
F570 = AUDIT / '570_R5X_manifest.csv'
SEED = 20261009
N_GENES = 9554
EMBED_DIM = 32
RANK_NEUTRAL_CENTER = 0.5
PROJECTION_CLIP = 8.0
SCALE_FLOOR = 1e-06
DISCOVERY_GSES = ['GSE15434', 'GSE161532', 'GSE1729', 'GSE34860', 'GSE37021', 'GSE37642', 'GSE6891', 'GSE71014']
EXPECTED_COUNTS = {'GSE15434': 251, 'GSE161532': 61, 'GSE1729': 43, 'GSE34860': 78, 'GSE37021': 97, 'GSE37642': 562, 'GSE6891': 537, 'GSE71014': 104}
MNN_K = 10
LEIDEN_RESOLUTION = 1.0
BOOTSTRAPS = 30
BOOTSTRAP_EDGE_KEEP_FRACTION = 0.8
MIN_STATE_PATIENTS = 40
MIN_STATE_COHORTS = 4
MAX_SINGLE_COHORT_FRACTION = 0.4
MIN_BOOTSTRAP_JACCARD = 0.6
MIN_MNN_NODE_COVERAGE = 0.75
STATE_META_FDR = 0.05
STATE_DIRECTION_CONSISTENCY = 0.8
MIN_STATE_IN_COHORT_FOR_SIGNATURE = 5
MIN_OTHER_IN_COHORT_FOR_SIGNATURE = 10
MIN_ELIGIBLE_COHORTS_FOR_SIGNATURE = 3
UNCERTAINTY_TRAIN_QUANTILE = 0.05
HELDOUT_RECURRENCE_FRACTION = 0.05
HELDOUT_RECURRENCE_MIN_N = 3
EXPECTED_SHA = {RANK_FILE: '8c2a914c11562073fb3e6bfbc36f8e2c0fc5d76ca88237a938761e4ca79f7a10', NODE_FILE: 'c0596efbded29dcb848c1620c25bb00344012c72cfda2ca4ac5efc256c1baf20', DISCOVERY_LEDGER_FILE: 'c33172332ef0ab385c0701472338895acce5d993355f57ec2d13e83a35be4778', ORIGINAL_STATE_FILE: '765facf2bdc0e81f7eefaa7561411f694eac95c0b482d7177f77356bf7cf7876'}
KNOWN_PROJECT_GSES = {'GSE114868', 'GSE13159', 'GSE15434', 'GSE161532', 'GSE1729', 'GSE201492', 'GSE34860', 'GSE37021', 'GSE37642', 'GSE6891', 'GSE71014', 'GSE9476', 'GSE17855', 'GSE110087', 'GSE108003', 'GSE14468'}

def require(condition: bool, message: str):
    if not bool(condition):
        raise RuntimeError(message)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out

def canonical_json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + '\n').encode('utf-8')

def canonical_csv_bytes(df: pd.DataFrame, sort_cols: Optional[List[str]]=None) -> bytes:
    x = df.copy()
    if sort_cols:
        existing = [c for c in sort_cols if c in x.columns]
        if existing:
            x = x.sort_values(existing, kind='mergesort').reset_index(drop=True)
    return x.to_csv(index=False, lineterminator='\n').encode('utf-8')

def write_once_or_verify_bytes(path: Path, payload: bytes):
    path = Path(path)
    if path.exists():
        old = path.read_bytes()
        require(old == payload, f'Existing output differs; refusing overwrite: {path}')
        print(f'PASS existing identical: {path.name}')
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(payload)
    os.replace(tmp, path)
    print(f'CREATED: {path.name}')

def write_csv(path: Path, df: pd.DataFrame, sort_cols: Optional[List[str]]=None):
    write_once_or_verify_bytes(path, canonical_csv_bytes(df, sort_cols))

def write_json(path: Path, obj):
    write_once_or_verify_bytes(path, canonical_json_bytes(obj))

def save_npz_once(path: Path, **arrays):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp.npz')
    np.savez(tmp, **arrays)
    payload = tmp.read_bytes()
    tmp.unlink()
    if path.exists():
        old = np.load(path, allow_pickle=False)
        require(set(old.files) == set(arrays.keys()), f'NPZ keys changed: {path}')
        for k, v in arrays.items():
            require(np.array_equal(old[k], np.asarray(v)), f'NPZ array differs: {path}:{k}')
        print(f'PASS existing numerically identical: {path.name}')
    else:
        path.write_bytes(payload)
        print(f'CREATED: {path.name}')

def l2_normalize_rows(X: np.ndarray, eps: float=1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)

def l2_normalize_vector(x: np.ndarray, eps: float=1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / max(n, eps)

def shannon_entropy_normalized(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    h = -float(np.sum(p * np.log(p)))
    return h / math.log(len(counts))
for p in [RANK_FILE, NODE_FILE, DISCOVERY_LEDGER_FILE, GENE_EMBED_FILE, GENE_EMBED_LEDGER_FILE, ORIGINAL_STATE_FILE, R5G_MANIFEST_FILE]:
    require(p.exists(), f'Missing authoritative input: {p}')
for p, expected in EXPECTED_SHA.items():
    observed = sha256_file(p)
    require(observed == expected, f'SHA256 mismatch: {p.name}\nexpected={expected}\nobserved={observed}')
    print(f'SHA256 PASS: {p.name}')
r5_manifest = pd.read_csv(R5G_MANIFEST_FILE, dtype=str, keep_default_na=False)
require({'role', 'sha256'}.issubset(r5_manifest.columns), '497 manifest schema changed.')
row = r5_manifest.loc[r5_manifest['role'].eq('gene_embeddings')]
require(len(row) == 1, '497 manifest must contain exactly one gene_embeddings row.')
expected_embed_sha = row.iloc[0]['sha256']
observed_embed_sha = sha256_file(GENE_EMBED_FILE)
require(observed_embed_sha == expected_embed_sha, 'Frozen gene embedding SHA differs from R5G manifest.')
print('SHA256 PASS: frozen topology-only gene embedding')
discovery = pd.read_csv(DISCOVERY_LEDGER_FILE, dtype={'analysis_unit_id': str, 'analysis_unit_index_0based': int, 'gse': str})
require(len(discovery) == 1733, 'Discovery ledger must contain 1733 AML profiles.')
require(set(discovery['gse']) == set(DISCOVERY_GSES), 'Discovery cohort set changed.')
require(discovery['analysis_unit_id'].is_unique, 'Discovery analysis_unit_id is not unique.')
require(discovery['analysis_unit_index_0based'].is_unique, 'Discovery matrix index is not unique.')
for gse, n in EXPECTED_COUNTS.items():
    require(int(discovery['gse'].eq(gse).sum()) == n, f'{gse}: unexpected sample count.')
nodes = pd.read_csv(NODE_FILE, dtype=str, keep_default_na=False)
require(len(nodes) == N_GENES, 'Node count changed.')
require('canonical_NCBI_Gene_ID' in nodes.columns, 'Node file lacks canonical_NCBI_Gene_ID.')
rank_mm = np.load(RANK_FILE, mmap_mode='r', allow_pickle=False)
require(rank_mm.shape == (4554, N_GENES), f'Unexpected rank matrix shape: {rank_mm.shape}')
discovery_idx = discovery['analysis_unit_index_0based'].to_numpy(dtype=int)
X = np.asarray(rank_mm[discovery_idx, :], dtype=np.float32)
require(X.shape == (1733, N_GENES), 'Discovery rank matrix extraction failed.')
require(np.isfinite(X).all(), 'Discovery rank matrix contains non-finite values.')
G = np.load(GENE_EMBED_FILE, allow_pickle=False).astype(np.float32)
require(G.shape == (N_GENES, EMBED_DIM), f'Unexpected gene embedding shape: {G.shape}')
require(np.isfinite(G).all(), 'Gene embeddings contain non-finite values.')
row_norms = np.linalg.norm(G, axis=1)
require(float(np.max(np.abs(row_norms - 1.0))) < 0.0001, 'Gene embeddings are not row-L2-normalized as expected.')
original_states = pd.read_csv(ORIGINAL_STATE_FILE, dtype=str, keep_default_na=False)
require(len(original_states) == 1733, 'Original R5G state file length changed.')
protocol = {'protocol_id': 'AML_GRAPHRX_R5X_CORRECTIVE_LOCO_STATE_TRANSFER_V1', 'analysis_role': 'post-validation corrective methodological analysis', 'scientific_question': 'Does using the same train-fitted patient representation for discovery and transfer improve reproducibility of within-AML recurrent states?', 'frozen_gene_embedding': str(GENE_EMBED_FILE), 'gene_embedding_patient_data_used': False, 'outer_validation': 'leave-one-GSE-out across the 8 R5 discovery cohorts', 'outer_folds': DISCOVERY_GSES, 'heldout_labels_used_for_training_or_tuning': False, 'patient_transform': {'fixed_rank_center': RANK_NEUTRAL_CENTER, 'projection': '(rank-0.5) @ frozen_gene_embedding / sqrt(9554)', 'robust_scaler': 'median/IQR fitted on outer-training patients only', 'IQR_floor': SCALE_FLOOR, 'clip': [-PROJECTION_CLIP, PROJECTION_CLIP], 'row_L2_normalization': True, 'same_transform_train_and_test': True}, 'state_discovery': {'cross_cohort_MNN_k': MNN_K, 'within_GSE_edges': False, 'Leiden_resolution': LEIDEN_RESOLUTION, 'bootstrap_replicates': BOOTSTRAPS, 'bootstrap_edge_keep_fraction': BOOTSTRAP_EDGE_KEEP_FRACTION, 'min_state_patients': MIN_STATE_PATIENTS, 'min_state_cohorts': MIN_STATE_COHORTS, 'max_single_cohort_fraction': MAX_SINGLE_COHORT_FRACTION, 'min_bootstrap_median_Jaccard': MIN_BOOTSTRAP_JACCARD}, 'assignment': {'method': 'nearest cosine centroid in the identical corrected representation', 'centroids_fit_on': 'outer training only', 'uncertainty_rule': 'reject if winning cosine similarity < state-specific 5th percentile of leave-one-out same-state training similarity', 'uncertainty_quantile': UNCERTAINTY_TRAIN_QUANTILE, 'heldout_recurrence_rule': 'state has >= max(3, ceil(0.05*n_heldout)) confident assignments', 'threshold_tuning_on_heldout': False}, 'gene_signatures': {'built_with_outer_training_only': True, 'per_cohort_Welch_t': True, 'weighted_Stouffer_meta_Z': True, 'BH_FDR': STATE_META_FDR, 'direction_consistency': STATE_DIRECTION_CONSISTENCY}, 'primary_state_recovery_metrics': ['training leave-one-out centroid macro-F1/ARI/NMI versus fold-discovered labels', 'heldout confident assignment coverage', 'heldout recurrent states recovered under the predeclared recurrence rule'], 'not_success_criteria': ['AML/control AUROC alone', 'mere diversification of heldout assignments'], 'claim_boundary': 'corrective methodological analysis; does not prove state-specific treatment response or causality', 'seed': SEED}
write_json(F556, protocol)

@dataclass
class TransformParams:
    median: np.ndarray
    iqr: np.ndarray

def raw_projection(X_rank: np.ndarray) -> np.ndarray:
    Z = (X_rank.astype(np.float32) - np.float32(RANK_NEUTRAL_CENTER)) @ G / np.float32(math.sqrt(N_GENES))
    require(np.isfinite(Z).all(), 'Raw patient projection contains non-finite values.')
    return Z.astype(np.float32)

def fit_transform_params(Z_train: np.ndarray) -> TransformParams:
    med = np.median(Z_train, axis=0).astype(np.float32)
    q25 = np.quantile(Z_train, 0.25, axis=0).astype(np.float32)
    q75 = np.quantile(Z_train, 0.75, axis=0).astype(np.float32)
    iqr = (q75 - q25).astype(np.float32)
    iqr[iqr < SCALE_FLOOR] = 1.0
    return TransformParams(median=med, iqr=iqr)

def apply_transform(Z: np.ndarray, params: TransformParams) -> np.ndarray:
    P = (Z - params.median[None, :]) / params.iqr[None, :]
    P = np.clip(P, -PROJECTION_CLIP, PROJECTION_CLIP)
    P = l2_normalize_rows(P.astype(np.float64)).astype(np.float32)
    require(np.isfinite(P).all(), 'Transformed patient representation contains non-finite values.')
    return P
Z_ALL = raw_projection(X)

def build_cross_cohort_mnn(P: np.ndarray, ledger: pd.DataFrame) -> pd.DataFrame:
    edge_rows = []
    gses = sorted(ledger['gse'].unique().tolist())
    for i in range(len(gses)):
        for j in range(i + 1, len(gses)):
            ga, gb = (gses[i], gses[j])
            ia = np.where(ledger['gse'].eq(ga).to_numpy())[0]
            ib = np.where(ledger['gse'].eq(gb).to_numpy())[0]
            A, B = (P[ia], P[ib])
            sim = A @ B.T
            k_ab = min(MNN_K, len(ib))
            k_ba = min(MNN_K, len(ia))
            if k_ab < 1 or k_ba < 1:
                continue
            top_b_for_a = np.argpartition(-sim, kth=k_ab - 1, axis=1)[:, :k_ab]
            top_a_for_b = np.argpartition(-sim.T, kth=k_ba - 1, axis=1)[:, :k_ba]
            reverse_sets = [set(row.tolist()) for row in top_a_for_b]
            for la in range(len(ia)):
                for lb0 in top_b_for_a[la]:
                    lb = int(lb0)
                    if la not in reverse_sets[lb]:
                        continue
                    w = float(sim[la, lb])
                    if w <= 0:
                        continue
                    u, v = (int(ia[la]), int(ib[lb]))
                    require(ledger.iloc[u]['gse'] != ledger.iloc[v]['gse'], 'Within-GSE MNN edge created.')
                    edge_rows.append({'patient_node_u': u, 'patient_node_v': v, 'gse_u': ga, 'gse_v': gb, 'cosine_similarity': w})
    return pd.DataFrame(edge_rows)

def leiden_and_recurrence(P: np.ndarray, ledger: pd.DataFrame, seed: int):
    MNN = build_cross_cohort_mnn(P, ledger)
    if len(MNN) == 0:
        return None
    connected = set(MNN['patient_node_u'].astype(int).tolist() + MNN['patient_node_v'].astype(int).tolist())
    coverage = len(connected) / len(ledger)
    edges = list(zip(MNN['patient_node_u'].astype(int), MNN['patient_node_v'].astype(int)))
    graph = ig.Graph(n=len(ledger), edges=edges, directed=False)
    graph.es['weight'] = MNN['cosine_similarity'].astype(float).tolist()
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=LEIDEN_RESOLUTION, seed=seed)
    membership = np.asarray(partition.membership, dtype=int)
    full_communities = {int(cid): set(np.where(membership == cid)[0].tolist()) for cid in np.unique(membership)}
    stability_values = {cid: [] for cid in full_communities}
    rng = np.random.default_rng(seed + 9000)
    edge_array = np.asarray(edges, dtype=int)
    edge_weights = MNN['cosine_similarity'].to_numpy(dtype=float)
    n_keep = max(1, int(round(BOOTSTRAP_EDGE_KEEP_FRACTION * len(edge_array))))
    for b in range(BOOTSTRAPS):
        keep = np.sort(rng.choice(len(edge_array), size=n_keep, replace=False))
        bg = ig.Graph(n=len(ledger), edges=[tuple(x) for x in edge_array[keep]], directed=False)
        bg.es['weight'] = edge_weights[keep].tolist()
        bp = leidenalg.find_partition(bg, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=LEIDEN_RESOLUTION, seed=seed + b + 1)
        bm = np.asarray(bp.membership, dtype=int)
        boot_comms = {int(cid): set(np.where(bm == cid)[0].tolist()) for cid in np.unique(bm)}
        for full_id, full_nodes in full_communities.items():
            best = 0.0
            for boot_nodes in boot_comms.values():
                union = len(full_nodes | boot_nodes)
                if union:
                    best = max(best, len(full_nodes & boot_nodes) / union)
            stability_values[full_id].append(best)
    stab = {cid: float(np.median(vals)) for cid, vals in stability_values.items()}
    rows = []
    for cid in sorted(np.unique(membership)):
        mask = membership == cid
        sub = ledger.loc[mask]
        n = len(sub)
        counts = sub['gse'].value_counts()
        ncoh = len(counts)
        maxfrac = float(counts.max() / n) if n else 1.0
        recurrent = bool(n >= MIN_STATE_PATIENTS and ncoh >= MIN_STATE_COHORTS and (maxfrac <= MAX_SINGLE_COHORT_FRACTION) and (stab[int(cid)] >= MIN_BOOTSTRAP_JACCARD))
        rows.append({'raw_community': int(cid), 'patients_n': int(n), 'independent_cohorts_n': int(ncoh), 'max_single_cohort_fraction': maxfrac, 'bootstrap_median_Jaccard': stab[int(cid)], 'RECURRENT_AML_STATE': recurrent})
    state_summary = pd.DataFrame(rows)
    valid = state_summary.loc[state_summary['RECURRENT_AML_STATE']].sort_values(['patients_n', 'independent_cohorts_n', 'raw_community'], ascending=[False, False, True]).reset_index(drop=True)
    state_map = {int(r.raw_community): f'AML_STATE_{i:02d}' for i, r in enumerate(valid.itertuples(index=False), start=1)}
    state_summary['state_id'] = state_summary['raw_community'].map(state_map).fillna('NON_RECURRENT')
    assignments = ledger.copy().reset_index(drop=True)
    assignments['raw_community'] = membership
    assignments['state_id'] = assignments['raw_community'].map(state_map).fillna('UNRESOLVED')
    return {'MNN': MNN, 'mnn_coverage': float(coverage), 'membership': membership, 'state_summary': state_summary, 'assignments': assignments}

def fit_centroids_and_thresholds(P_train: np.ndarray, assignments: pd.DataFrame):
    valid_mask = assignments['state_id'].ne('UNRESOLVED').to_numpy()
    state_ids = sorted(assignments.loc[valid_mask, 'state_id'].unique().tolist())
    if not state_ids:
        return None
    centroids = []
    state_members = {}
    state_sums = {}
    for s in state_ids:
        idx = np.where(assignments['state_id'].eq(s).to_numpy())[0]
        state_members[s] = idx
        state_sums[s] = P_train[idx].astype(np.float64).sum(axis=0)
        centroids.append(l2_normalize_vector(state_sums[s]))
    C = np.vstack(centroids).astype(np.float64)
    s_to_j = {s: j for j, s in enumerate(state_ids)}
    diag_rows = []
    own_sims_for_threshold = {s: [] for s in state_ids}
    y_true, y_pred = ([], [])
    full_sims = P_train.astype(np.float64) @ C.T
    for i in np.where(valid_mask)[0]:
        true_s = assignments.iloc[i]['state_id']
        sims = full_sims[i].copy()
        members = state_members[true_s]
        n_s = len(members)
        require(n_s >= 2, f'Recurrent state {true_s} has <2 members.')
        loo_vec = (state_sums[true_s] - P_train[i].astype(np.float64)) / (n_s - 1)
        loo_centroid = l2_normalize_vector(loo_vec)
        own_j = s_to_j[true_s]
        own_sim = float(P_train[i].astype(np.float64) @ loo_centroid)
        sims[own_j] = own_sim
        pred_j = int(np.argmax(sims))
        pred_s = state_ids[pred_j]
        sorted_sim = np.sort(sims)
        margin = float(sorted_sim[-1] - sorted_sim[-2]) if len(sorted_sim) > 1 else float('nan')
        own_sims_for_threshold[true_s].append(own_sim)
        y_true.append(true_s)
        y_pred.append(pred_s)
        diag_rows.append({'training_row_local': int(i), 'analysis_unit_id': assignments.iloc[i]['analysis_unit_id'], 'gse': assignments.iloc[i]['gse'], 'true_discovered_state': true_s, 'LOO_predicted_state': pred_s, 'LOO_own_state_similarity': own_sim, 'LOO_winner_similarity': float(sims[pred_j]), 'LOO_margin': margin, 'LOO_correct': bool(pred_s == true_s)})
    thresholds = {s: float(np.quantile(np.asarray(vals, dtype=float), UNCERTAINTY_TRAIN_QUANTILE)) for s, vals in own_sims_for_threshold.items()}
    diag = pd.DataFrame(diag_rows)
    metrics = {'training_LOO_macro_F1': float(f1_score(y_true, y_pred, average='macro')), 'training_LOO_ARI': float(adjusted_rand_score(y_true, y_pred)), 'training_LOO_NMI': float(normalized_mutual_info_score(y_true, y_pred)), 'training_LOO_accuracy': float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))}
    return {'state_ids': state_ids, 'centroids': C, 'thresholds': thresholds, 'diagnostics': diag, 'metrics': metrics}

def assign_by_centroid(P_test: np.ndarray, fit_obj) -> pd.DataFrame:
    state_ids = fit_obj['state_ids']
    C = fit_obj['centroids']
    thresholds = fit_obj['thresholds']
    sims = P_test.astype(np.float64) @ C.T
    winner = np.argmax(sims, axis=1)
    sorted_sims = np.sort(sims, axis=1)
    max_sim = sims[np.arange(len(P_test)), winner]
    margin = sorted_sims[:, -1] - sorted_sims[:, -2] if sims.shape[1] > 1 else np.full(len(P_test), np.nan)
    pred = [state_ids[int(j)] for j in winner]
    threshold = np.asarray([thresholds[s] for s in pred], dtype=float)
    confident = max_sim >= threshold
    assigned = np.where(confident, np.asarray(pred, dtype=object), 'UNCERTAIN')
    return pd.DataFrame({'nearest_state': pred, 'assigned_state': assigned, 'winner_similarity': max_sim, 'winner_cosine_distance': 1.0 - max_sim, 'winner_threshold': threshold, 'margin_to_second': margin, 'confident_assignment': confident})

def build_state_signatures(X_train: np.ndarray, ledger_train: pd.DataFrame, assignments: pd.DataFrame, fold_id: str) -> pd.DataFrame:
    rows = []
    state_ids = sorted([s for s in assignments['state_id'].unique() if s != 'UNRESOLVED'])
    for state_id in state_ids:
        cohort_effects, cohort_z, cohort_weights, eligible = ([], [], [], [])
        for gse in sorted(ledger_train['gse'].unique()):
            cohort_mask = ledger_train['gse'].eq(gse).to_numpy()
            state_mask = assignments['state_id'].eq(state_id).to_numpy() & cohort_mask
            other_mask = assignments['state_id'].ne(state_id).to_numpy() & cohort_mask
            n_state, n_other = (int(state_mask.sum()), int(other_mask.sum()))
            if n_state < MIN_STATE_IN_COHORT_FOR_SIGNATURE or n_other < MIN_OTHER_IN_COHORT_FOR_SIGNATURE:
                continue
            A, B = (X_train[state_mask], X_train[other_mask])
            effect = A.mean(axis=0) - B.mean(axis=0)
            test = ttest_ind(A, B, axis=0, equal_var=False, nan_policy='omit')
            p = np.asarray(test.pvalue, dtype=float)
            p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
            p = np.clip(p, 1e-300, 1.0)
            signed_z = np.sign(effect) * norm.isf(p / 2.0)
            w = math.sqrt(n_state * n_other / (n_state + n_other))
            cohort_effects.append(effect.astype(np.float64))
            cohort_z.append(signed_z.astype(np.float64))
            cohort_weights.append(float(w))
            eligible.append(gse)
        if len(eligible) < MIN_ELIGIBLE_COHORTS_FOR_SIGNATURE:
            continue
        E = np.stack(cohort_effects)
        Z = np.stack(cohort_z)
        W = np.asarray(cohort_weights, dtype=float)
        meta_z = (Z * W[:, None]).sum(axis=0) / math.sqrt(float(np.sum(W ** 2)))
        meta_p = 2.0 * norm.sf(np.abs(meta_z))
        meta_fdr = bh_fdr(meta_p)
        weighted_effect = (E * W[:, None]).sum(axis=0) / W.sum()
        meta_direction = np.sign(meta_z)
        consistency = np.mean(np.sign(E) == meta_direction[None, :], axis=0)
        signature = (meta_fdr < STATE_META_FDR) & (consistency >= STATE_DIRECTION_CONSISTENCY) & (meta_direction != 0)
        for gi in np.where(signature)[0]:
            rows.append({'fold_id': fold_id, 'state_id': state_id, 'gene_index_0based': int(gi), 'canonical_NCBI_Gene_ID': nodes.iloc[int(gi)]['canonical_NCBI_Gene_ID'], 'eligible_cohorts_n': int(len(eligible)), 'eligible_cohorts': ';'.join(eligible), 'meta_effect': float(weighted_effect[gi]), 'meta_Z': float(meta_z[gi]), 'meta_p': float(meta_p[gi]), 'meta_FDR_BH': float(meta_fdr[gi]), 'direction_consistency_fraction': float(consistency[gi]), 'direction': 'UP_IN_STATE' if meta_z[gi] > 0 else 'DOWN_IN_STATE', 'STATE_SIGNATURE_GENE': True})
    return pd.DataFrame(rows)
fold_summary_rows = []
training_state_rows = []
heldout_assignment_frames = []
training_diag_frames = []
signature_frames = []
fold_internal_assignments = {}
print('\n' + '=' * 120)
print('R5X CORRECTIVE OUTER LOCO')
print('=' * 120)
for fold_no, heldout_gse in enumerate(DISCOVERY_GSES, start=1):
    fold_id = f'LOCO_{fold_no:02d}_{heldout_gse}'
    print('\n' + '-' * 120)
    print(f'{fold_id}')
    print('-' * 120)
    test_mask = discovery['gse'].eq(heldout_gse).to_numpy()
    train_mask = ~test_mask
    train_idx_local = np.where(train_mask)[0]
    test_idx_local = np.where(test_mask)[0]
    ledger_train = discovery.loc[train_mask].reset_index(drop=True).copy()
    ledger_test = discovery.loc[test_mask].reset_index(drop=True).copy()
    X_train, X_test = (X[train_mask], X[test_mask])
    Z_train, Z_test = (Z_ALL[train_mask], Z_ALL[test_mask])
    params = fit_transform_params(Z_train)
    P_train = apply_transform(Z_train, params)
    P_test = apply_transform(Z_test, params)
    disc = leiden_and_recurrence(P_train, ledger_train, seed=SEED + fold_no * 1000)
    if disc is None:
        fold_summary_rows.append({'fold_id': fold_id, 'heldout_gse': heldout_gse, 'train_n': len(ledger_train), 'heldout_n': len(ledger_test), 'mnn_node_coverage': 0.0, 'training_recurrent_states_n': 0, 'training_recurrent_patient_coverage': 0.0, 'training_LOO_macro_F1': np.nan, 'training_LOO_ARI': np.nan, 'training_LOO_NMI': np.nan, 'training_LOO_accuracy': np.nan, 'heldout_confident_coverage': 0.0, 'heldout_recurrence_min_n': max(HELDOUT_RECURRENCE_MIN_N, int(math.ceil(HELDOUT_RECURRENCE_FRACTION * len(ledger_test)))), 'heldout_states_recovered_n': 0, 'heldout_state_recovery_fraction': 0.0, 'heldout_median_winner_similarity': np.nan, 'heldout_median_margin': np.nan, 'heldout_assignment_entropy_normalized': 0.0, 'status': 'NO_MNN_EDGES'})
        continue
    assignments_train = disc['assignments'].copy()
    n_states = int(assignments_train['state_id'].ne('UNRESOLVED').groupby(assignments_train['state_id']).ngroups - (1 if 'UNRESOLVED' in assignments_train['state_id'].values else 0))
    state_ids = sorted([s for s in assignments_train['state_id'].unique() if s != 'UNRESOLVED'])
    n_states = len(state_ids)
    recurrent_cov = float(assignments_train['state_id'].ne('UNRESOLVED').mean())
    ss = disc['state_summary'].copy()
    ss.insert(0, 'fold_id', fold_id)
    ss.insert(1, 'heldout_gse', heldout_gse)
    training_state_rows.append(ss)
    fold_internal_assignments[fold_id] = assignments_train[['analysis_unit_id', 'gse', 'state_id']].copy()
    if n_states == 0:
        fold_summary_rows.append({'fold_id': fold_id, 'heldout_gse': heldout_gse, 'train_n': len(ledger_train), 'heldout_n': len(ledger_test), 'mnn_node_coverage': disc['mnn_coverage'], 'training_recurrent_states_n': 0, 'training_recurrent_patient_coverage': recurrent_cov, 'training_LOO_macro_F1': np.nan, 'training_LOO_ARI': np.nan, 'training_LOO_NMI': np.nan, 'training_LOO_accuracy': np.nan, 'heldout_confident_coverage': 0.0, 'heldout_recurrence_min_n': max(HELDOUT_RECURRENCE_MIN_N, int(math.ceil(HELDOUT_RECURRENCE_FRACTION * len(ledger_test)))), 'heldout_states_recovered_n': 0, 'heldout_state_recovery_fraction': 0.0, 'heldout_median_winner_similarity': np.nan, 'heldout_median_margin': np.nan, 'heldout_assignment_entropy_normalized': 0.0, 'status': 'NO_RECURRENT_TRAINING_STATES'})
        continue
    fit_obj = fit_centroids_and_thresholds(P_train, assignments_train)
    require(fit_obj is not None, 'Centroid fit unexpectedly failed.')
    diag = fit_obj['diagnostics'].copy()
    diag.insert(0, 'fold_id', fold_id)
    diag.insert(1, 'heldout_gse', heldout_gse)
    training_diag_frames.append(diag)
    pred = assign_by_centroid(P_test, fit_obj)
    held = pd.concat([ledger_test.reset_index(drop=True), pred], axis=1)
    held.insert(0, 'fold_id', fold_id)
    held.insert(1, 'heldout_gse', heldout_gse)
    heldout_assignment_frames.append(held)
    recurrence_min_n = max(HELDOUT_RECURRENCE_MIN_N, int(math.ceil(HELDOUT_RECURRENCE_FRACTION * len(ledger_test))))
    confident = held.loc[held['confident_assignment']].copy()
    counts = confident['assigned_state'].value_counts()
    recovered = [s for s in fit_obj['state_ids'] if int(counts.get(s, 0)) >= recurrence_min_n]
    recovered_fraction = len(recovered) / len(fit_obj['state_ids'])
    conf_coverage = float(held['confident_assignment'].mean())
    entropy = shannon_entropy_normalized(counts.to_numpy()) if len(counts) else 0.0
    sig = build_state_signatures(X_train, ledger_train, assignments_train, fold_id)
    if len(sig):
        signature_frames.append(sig)
    metrics = fit_obj['metrics']
    fold_summary_rows.append({'fold_id': fold_id, 'heldout_gse': heldout_gse, 'train_n': len(ledger_train), 'heldout_n': len(ledger_test), 'mnn_node_coverage': float(disc['mnn_coverage']), 'training_recurrent_states_n': int(len(fit_obj['state_ids'])), 'training_recurrent_patient_coverage': recurrent_cov, **metrics, 'heldout_confident_coverage': conf_coverage, 'heldout_recurrence_min_n': int(recurrence_min_n), 'heldout_states_recovered_n': int(len(recovered)), 'heldout_state_recovery_fraction': float(recovered_fraction), 'heldout_median_winner_similarity': float(np.median(held['winner_similarity'])), 'heldout_median_margin': float(np.median(held['margin_to_second'])) if len(fit_obj['state_ids']) > 1 else np.nan, 'heldout_assignment_entropy_normalized': float(entropy), 'heldout_recovered_state_ids_fold_local': ';'.join(recovered), 'status': 'COMPLETE'})
fold_summary = pd.DataFrame(fold_summary_rows)
training_state_summary = pd.concat(training_state_rows, ignore_index=True) if training_state_rows else pd.DataFrame()
heldout_assignments = pd.concat(heldout_assignment_frames, ignore_index=True) if heldout_assignment_frames else pd.DataFrame()
training_diagnostics = pd.concat(training_diag_frames, ignore_index=True) if training_diag_frames else pd.DataFrame()
fold_signatures = pd.concat(signature_frames, ignore_index=True) if signature_frames else pd.DataFrame()
write_csv(F557, fold_summary, ['fold_id'])
write_csv(F558, training_state_summary, ['fold_id', 'raw_community'] if len(training_state_summary) else None)
write_csv(F559, heldout_assignments, ['fold_id', 'analysis_unit_index_0based'] if len(heldout_assignments) else None)
write_csv(F560, training_diagnostics, ['fold_id', 'analysis_unit_id'] if len(training_diagnostics) else None)
write_csv(F561, fold_signatures, ['fold_id', 'state_id', 'gene_index_0based'] if len(fold_signatures) else None)
print('\n' + '=' * 120)
print('BUILDING ALL-EIGHT-COHORT CANDIDATE TRANSFER MODEL')
print('=' * 120)
params_all = fit_transform_params(Z_ALL)
P_all = apply_transform(Z_ALL, params_all)
final_disc = leiden_and_recurrence(P_all, discovery.reset_index(drop=True), seed=SEED + 99000)
require(final_disc is not None, 'All-eight corrected discovery produced no MNN edges.')
final_assign = final_disc['assignments'].copy()
final_fit = fit_centroids_and_thresholds(P_all, final_assign)
require(final_fit is not None, 'All-eight corrected discovery produced no recurrent states.')
final_state_summary = final_disc['state_summary'].copy()
final_state_summary['analysis_role'] = 'all-eight corrected candidate; not external validation'
write_csv(F562, final_state_summary, ['raw_community'])
final_self_pred = assign_by_centroid(P_all, final_fit)
final_assign_out = pd.concat([final_assign.reset_index(drop=True), final_self_pred], axis=1)
write_csv(F563, final_assign_out, ['analysis_unit_index_0based'])
threshold_df = pd.DataFrame({'state_id': final_fit['state_ids'], 'training_LOO_similarity_q05_threshold': [final_fit['thresholds'][s] for s in final_fit['state_ids']]})
write_csv(F567, threshold_df, ['state_id'])
state_ids_arr = np.asarray(final_fit['state_ids'], dtype='U32')
threshold_arr = np.asarray([final_fit['thresholds'][s] for s in final_fit['state_ids']], dtype=np.float64)
save_npz_once(F565, state_ids=state_ids_arr, centroids=final_fit['centroids'].astype(np.float64), thresholds=threshold_arr)
save_npz_once(F566, rank_neutral_center=np.asarray([RANK_NEUTRAL_CENTER], dtype=np.float64), projection_median=params_all.median.astype(np.float64), projection_iqr=params_all.iqr.astype(np.float64), clip=np.asarray([PROJECTION_CLIP], dtype=np.float64))
final_signatures = build_state_signatures(X, discovery.reset_index(drop=True), final_assign, 'ALL8_CANDIDATE')
write_csv(F568, final_signatures, ['state_id', 'gene_index_0based'] if len(final_signatures) else None)
orig = original_states[['analysis_unit_id', 'state_id']].rename(columns={'state_id': 'original_state'})
corr = final_assign[['analysis_unit_id', 'state_id']].rename(columns={'state_id': 'corrected_state'})
align_df = orig.merge(corr, on='analysis_unit_id', how='inner', validate='one_to_one')
orig_states = sorted([s for s in align_df['original_state'].unique() if s not in {'UNRESOLVED', 'NON_RECURRENT'}])
corr_states = sorted([s for s in align_df['corrected_state'].unique() if s not in {'UNRESOLVED', 'NON_RECURRENT'}])
jmat = np.zeros((len(orig_states), len(corr_states)), dtype=float)
for i, os_ in enumerate(orig_states):
    A = set(align_df.index[align_df['original_state'].eq(os_)].tolist())
    for j, cs_ in enumerate(corr_states):
        B = set(align_df.index[align_df['corrected_state'].eq(cs_)].tolist())
        u = len(A | B)
        jmat[i, j] = len(A & B) / u if u else 0.0
alignment_rows = []
if len(orig_states) and len(corr_states):
    ri, cj = linear_sum_assignment(-jmat)
    matched = {(int(i), int(j)) for i, j in zip(ri, cj)}
else:
    matched = set()
for i, os_ in enumerate(orig_states):
    for j, cs_ in enumerate(corr_states):
        alignment_rows.append({'original_state': os_, 'corrected_candidate_state': cs_, 'patient_Jaccard': float(jmat[i, j]), 'Hungarian_primary_match': bool((i, j) in matched), 'analysis_role': 'post-LOCO descriptive alignment only; not used for training/tuning/scoring'})
alignment = pd.DataFrame(alignment_rows)
write_csv(F564, alignment, ['original_state', 'corrected_candidate_state'] if len(alignment) else None)
mask_both = align_df['original_state'].ne('UNRESOLVED') & align_df['corrected_state'].ne('UNRESOLVED')
if int(mask_both.sum()) > 1:
    original_corrected_ari = float(adjusted_rand_score(align_df.loc[mask_both, 'original_state'], align_df.loc[mask_both, 'corrected_state']))
    original_corrected_nmi = float(normalized_mutual_info_score(align_df.loc[mask_both, 'original_state'], align_df.loc[mask_both, 'corrected_state']))
else:
    original_corrected_ari = np.nan
    original_corrected_nmi = np.nan
complete = fold_summary.loc[fold_summary['status'].eq('COMPLETE')].copy()
summary_rows = [{'metric': 'outer_folds_total', 'value': 8}, {'metric': 'outer_folds_complete', 'value': int(len(complete))}, {'metric': 'median_training_recurrent_states_n', 'value': float(complete['training_recurrent_states_n'].median()) if len(complete) else np.nan}, {'metric': 'median_training_LOO_macro_F1', 'value': float(complete['training_LOO_macro_F1'].median()) if len(complete) else np.nan}, {'metric': 'median_training_LOO_ARI', 'value': float(complete['training_LOO_ARI'].median()) if len(complete) else np.nan}, {'metric': 'median_training_LOO_NMI', 'value': float(complete['training_LOO_NMI'].median()) if len(complete) else np.nan}, {'metric': 'median_heldout_confident_coverage', 'value': float(complete['heldout_confident_coverage'].median()) if len(complete) else np.nan}, {'metric': 'median_heldout_states_recovered_n', 'value': float(complete['heldout_states_recovered_n'].median()) if len(complete) else np.nan}, {'metric': 'median_heldout_state_recovery_fraction', 'value': float(complete['heldout_state_recovery_fraction'].median()) if len(complete) else np.nan}, {'metric': 'all8_corrected_recurrent_states_n', 'value': int(len(final_fit['state_ids']))}, {'metric': 'all8_corrected_mnn_node_coverage', 'value': float(final_disc['mnn_coverage'])}, {'metric': 'all8_training_LOO_macro_F1', 'value': float(final_fit['metrics']['training_LOO_macro_F1'])}, {'metric': 'corrected_vs_original_ARI_descriptive', 'value': original_corrected_ari}, {'metric': 'corrected_vs_original_NMI_descriptive', 'value': original_corrected_nmi}, {'metric': 'automatic_success_claim', 'value': False}, {'metric': 'next_step', 'value': 'review LOCO results; only then freeze candidate and test one truly new cohort'}]
final_summary = pd.DataFrame(summary_rows)
write_csv(F569, final_summary, ['metric'])
manifest_rows = []
for role, p in [('protocol', F556), ('LOCO_fold_summary', F557), ('LOCO_training_state_summary', F558), ('LOCO_heldout_assignments', F559), ('LOCO_training_centroid_diagnostics', F560), ('LOCO_state_signature_genes', F561), ('final_candidate_state_summary', F562), ('final_candidate_assignments', F563), ('corrected_vs_original_alignment', F564), ('final_candidate_centroids', F565), ('final_candidate_transform', F566), ('final_candidate_thresholds', F567), ('final_candidate_signature_genes', F568), ('final_summary', F569)]:
    require(p.exists(), f'Manifest artifact missing: {p}')
    manifest_rows.append({'role': role, 'path': str(p), 'size_bytes': int(p.stat().st_size), 'sha256': sha256_file(p), 'protocol_id': protocol['protocol_id']})
manifest = pd.DataFrame(manifest_rows)
write_csv(F570, manifest, ['role'])
print('\n' + '=' * 120)
print('R5X COMPLETE — REVIEW BEFORE ANY CLAIM OF IMPROVED TRANSPORTABILITY')
print('=' * 120)
cols = ['heldout_gse', 'training_recurrent_states_n', 'training_LOO_macro_F1', 'training_LOO_ARI', 'mnn_node_coverage', 'heldout_confident_coverage', 'heldout_states_recovered_n', 'heldout_state_recovery_fraction', 'heldout_median_winner_similarity', 'heldout_median_margin', 'status']
print('\nOUTER LOCO SUMMARY')
print(fold_summary[cols].to_string(index=False))
print('\nFINAL SUMMARY')
print(final_summary.to_string(index=False))
print('\nOUTPUTS')
for p in [F556, F557, F558, F559, F560, F561, F562, F563, F564, F565, F566, F567, F568, F569, F570]:
    print(p)
print('\nIMPORTANT:')
print('- No held-out GSE labels were used to discover states, fit centroids, fit scaling, or set uncertainty thresholds.')
print('- Original 8-cohort R5-2G labels were used only after LOCO for descriptive alignment.')
print('- AML/control AUROC is intentionally not used as a state-recovery endpoint.')
print('- Do not call the all-eight candidate externally validated until apply_truly_new_cohort() is run on a genuinely new cohort.')
