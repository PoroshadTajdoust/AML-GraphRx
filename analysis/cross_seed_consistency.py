# Manuscript-linked analysis: Amended raw-latent cross-seed comparison
# Source lineage and packaging changes: provenance/source_lineage.json
"""AML-GraphRx R10C corrected: cross-seed state consistency of completed R10B fits.

Run in one Google Colab cell, or as a Python script. No GNN retraining.
State identity is compared with label-invariant statistics and one-to-one
Jaccard matching; all original cohort/recurrence rules remain unchanged.
Exploratory, within-discovery analysis only; NOT an independent validation.
"""
from __future__ import annotations
import os as _release_os
try:
    from IPython.display import display
except ImportError:

    def display(value):
        print(value)
from pathlib import Path
import hashlib, json, os, sys, subprocess
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
try:
    import igraph as ig
    import leidenalg
except ImportError:
    raise RuntimeError('Install the recorded analysis dependencies before running this script; see docs/ENVIRONMENT.md.')
    import igraph as ig
    import leidenalg
try:
    pass
    pass
except ImportError:
    pass
PROJECT = Path(_release_os.environ.get('AML_GRAPHRX_ROOT', '/content/drive/MyDrive/AML_GraphRx'))
AUDIT = PROJECT / 'results' / 'audit'
STEP5 = PROJECT / 'data' / 'patient_graphs' / '20260823T123040Z'
WORK = PROJECT / 'results' / 'r10b_graph_edge_ablation_v1'
RANK = STEP5 / '20260823T123040Z_patient_rank_raw_float64.npy'
DISC = AUDIT / '452_R5DICE_discovery_ledger.csv'
F481 = AUDIT / '481_R5G_predeclared_protocol.json'
F483 = AUDIT / '483_R5G_gene_embeddings.npy'
F486 = AUDIT / '486_R5G_patient_projection.npy'
F488 = AUDIT / '488_R5G_cross_cohort_MNN_edges.csv'
F489 = AUDIT / '489_R5G_Leiden_raw_assignments.csv'
F492 = AUDIT / '492_R5G_recurrent_state_patient_assignments.csv'
F497 = AUDIT / '497_R5G_manifest.csv'
F760 = AUDIT / '760_R10B_protocol.json'
F761 = AUDIT / '761_R10B_parent_integrity.csv'
F762 = AUDIT / '762_R10B_graph_rewiring_audit.csv'
F763 = AUDIT / '763_R10B_training_history.csv'
F764 = AUDIT / '764_R10B_arm_discovery_metrics.csv'
F765 = AUDIT / '765_R10B_paired_descriptive_contrasts.csv'
F766 = AUDIT / '766_R10B_final_summary.csv'
F767 = AUDIT / '767_R10B_manifest.csv'
F768 = AUDIT / '768_R10C_cross_seed_protocol.json'
F769 = AUDIT / '769_R10C_parent_integrity.csv'
F770 = AUDIT / '770_R10C_arm_reproduction_fidelity.csv'
F771 = AUDIT / '771_R10C_cross_seed_pairwise_consistency.csv'
F772 = AUDIT / '772_R10C_hungarian_state_matches.csv'
F773 = AUDIT / '773_R10C_family_summary.csv'
F774 = AUDIT / '774_R10C_final_summary.csv'
F775 = AUDIT / '775_R10C_manifest.csv'
F776PNG = AUDIT / '776_R10C_cross_seed_consistency.png'
F776PDF = AUDIT / '776_R10C_cross_seed_consistency.pdf'
F777 = AUDIT / '777_R10C_implementation_alignment_amendment.json'
SEEDS = [20261031, 20261101, 20261102]
ARMS = ['REAL_GRAPH', 'DEGREE_PRESERVING_REWIRE', 'NO_MESSAGE_PASSING']
N_GENES = 9554
D = 32
N_PATIENTS = 1733
K = 10
N_BOOT = 30
EDGE_KEEP = 0.8
RES = 1.0
MIN_N = 40
MIN_COH = 4
MAX_COH_SHARE = 0.4
MIN_JAC = 0.6
SPLIT_SEED = 20261004
GSES = ['GSE15434', 'GSE161532', 'GSE1729', 'GSE34860', 'GSE37021', 'GSE37642', 'GSE6891', 'GSE71014']
GSE_COUNTS = [251, 61, 43, 78, 97, 562, 537, 104]
AUDIT.mkdir(parents=True, exist_ok=True)

def require(cond, msg):
    if not bool(cond):
        raise RuntimeError(msg)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write_once(path, payload):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == payload, 'Existing output differs, preserved: ' + str(path))
        print('PASS existing:', path.name)
        return
    temp = path.with_name(path.name + '.partial')
    temp.write_bytes(payload)
    os.replace(temp, path)
    print('CREATED:', path.name)

def save_json(path, obj):
    write_once(path, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())

def save_csv(path, df, keys=None):
    df = df.copy()
    if keys:
        df = df.sort_values(keys, kind='mergesort').reset_index(drop=True)
    write_once(path, df.to_csv(index=False, lineterminator='\n', float_format='%.17g').encode())

def normalize_basis(e):
    e = np.asarray(e, dtype=np.float32).copy()
    require(e.shape == (N_GENES, D), 'Bad basis shape')
    e -= e.mean(axis=0, keepdims=True)
    sd = e.std(axis=0, keepdims=True)
    e /= np.where(sd < 1e-08, 1.0, sd)
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    return e.astype(np.float32)

def patient_projection(X, gses, E):
    P = np.zeros((len(gses), D), dtype=np.float32)
    for g in GSES:
        ix = np.where(gses == g)[0]
        x = X[ix]
        med = np.median(x, axis=0, keepdims=True)
        p = (x - med) @ E / np.sqrt(N_GENES)
        m = np.median(p, axis=0, keepdims=True)
        q25 = np.quantile(p, 0.25, axis=0, keepdims=True)
        q75 = np.quantile(p, 0.75, axis=0, keepdims=True)
        scale = q75 - q25
        scale[scale < 1e-06] = 1.0
        p = np.clip((p - m) / scale, -8, 8)
        p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        P[ix] = p.astype(np.float32)
    return P

def create_mnn(P, gses):
    edges = []
    for i, g1 in enumerate(GSES):
        for g2 in GSES[i + 1:]:
            a = np.where(gses == g1)[0]
            b = np.where(gses == g2)[0]
            sim = P[a] @ P[b].T
            ka = min(K, len(b))
            kb = min(K, len(a))
            ab = np.argpartition(-sim, kth=ka - 1, axis=1)[:, :ka]
            ba = np.argpartition(-sim.T, kth=kb - 1, axis=1)[:, :kb]
            back = [set(z.tolist()) for z in ba]
            for ai in range(len(a)):
                for bj in ab[ai]:
                    bj = int(bj)
                    if ai in back[bj] and sim[ai, bj] > 0:
                        edges.append((int(a[ai]), int(b[bj]), float(sim[ai, bj])))
    require(bool(edges), 'No positive cross-cohort MNN edges')
    return edges

def leiden_graph(n, edges, seed):
    g = ig.Graph(n=n, edges=[(int(a), int(b)) for a, b, w in edges], directed=False)
    g.es['weight'] = [float(w) for a, b, w in edges]
    return np.asarray(leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=RES, seed=int(seed)).membership, dtype=int)

def state_metrics(labels, edges, gses):
    ids = np.unique(labels)
    members = {int(c): set(np.flatnonzero(labels == c).tolist()) for c in ids}
    endpoints = np.asarray([(a, b) for a, b, w in edges], dtype=int)
    weights = np.asarray([w for a, b, w in edges], dtype=float)
    n_keep = int(round(EDGE_KEEP * len(edges)))
    rng = np.random.default_rng(SPLIT_SEED + 9000)
    js = {int(c): [] for c in ids}
    for t in range(N_BOOT):
        idx = np.sort(rng.choice(len(edges), n_keep, replace=False))
        e = [(int(endpoints[j, 0]), int(endpoints[j, 1]), float(weights[j])) for j in idx]
        lab = leiden_graph(len(gses), e, SPLIT_SEED + t + 1)
        cms = [set(np.flatnonzero(lab == c).tolist()) for c in np.unique(lab)]
        for c, s in members.items():
            js[c].append(max((len(s & ss) / len(s | ss) for ss in cms)))
    median_j = {c: float(np.median(v)) for c, v in js.items()}
    passed = []
    for c, s in members.items():
        counts = pd.Series(gses[list(s)]).value_counts()
        if len(s) >= MIN_N and len(counts) >= MIN_COH and (float(counts.max() / len(s)) <= MAX_COH_SHARE) and (median_j[c] >= MIN_JAC):
            passed.append(c)
    valid = np.isin(labels, passed)
    nmi = float(normalized_mutual_info_score(gses[valid], labels[valid])) if valid.any() and len(set(labels[valid])) >= 2 else float('nan')
    return dict(states=len(passed), coverage=float(valid.mean()), stability=float(np.median([median_j[c] for c in passed])) if passed else float('nan'), cohort_nmi=nmi, labels=np.where(valid, labels, -1), mnn_coverage=len({i for a, b, w in edges for i in (a, b)}) / len(gses))

def manifest_check(path, file_list):
    m = pd.read_csv(path, dtype=str)
    require({'path', 'sha256'}.issubset(m.columns), 'Manifest schema invalid: ' + path.name)
    for p in file_list:
        require(p.is_file(), 'MISSING input: ' + str(p))
        ss = m.loc[m['path'].map(str).map(lambda x: Path(x).name).eq(p.name)]
        require(len(ss) == 1, 'Nonunique/missing manifest record: ' + p.name)
        require(str(ss.sha256.iloc[0]) == sha(p), 'SHA mismatch: ' + p.name)

def matched_recurrence(A, B, seed_a, seed_b, arm):
    ca = sorted((int(c) for c in np.unique(A['labels']) if c >= 0))
    cb = sorted((int(c) for c in np.unique(B['labels']) if c >= 0))
    a = [set(np.flatnonzero(A['labels'] == c).tolist()) for c in ca]
    b = [set(np.flatnonzero(B['labels'] == c).tolist()) for c in cb]
    m = max(len(a), len(b))
    mat = np.zeros((m, m), dtype=np.float64)
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            mat[i, j] = len(u & v) / len(u | v)
    rr, cc = linear_sum_assignment(-mat)
    matched = []
    details = []
    for i, j in zip(rr, cc):
        if i < len(a) and j < len(b):
            si = len(a[i])
            sj = len(b[j])
            shared = len(a[i] & b[j])
            u = len(a[i] | b[j])
            jj = float(mat[i, j])
            matched.append((si, sj, jj))
            details.append(dict(arm=arm, seed_a=seed_a, seed_b=seed_b, state_a=ca[i], state_b=cb[j], size_a=si, size_b=sj, intersection_n=shared, union_n=u, jaccard=jj, matched=True))
        elif i < len(a):
            details.append(dict(arm=arm, seed_a=seed_a, seed_b=seed_b, state_a=ca[i], state_b=-1, size_a=len(a[i]), size_b=0, intersection_n=0, union_n=len(a[i]), jaccard=0.0, matched=False))
        elif j < len(b):
            details.append(dict(arm=arm, seed_a=seed_a, seed_b=seed_b, state_a=-1, state_b=cb[j], size_a=0, size_b=len(b[j]), intersection_n=0, union_n=len(b[j]), jaccard=0.0, matched=False))
    n_a = int(np.sum(A['labels'] >= 0))
    n_b = int(np.sum(B['labels'] >= 0))
    denom = (n_a + n_b) / 2
    weight_num = sum((0.5 * (p + q) * j for p, q, j in matched))
    weighted_j = float(weight_num / denom) if denom else np.nan
    shared_mask = (A['labels'] >= 0) & (B['labels'] >= 0)
    na = int(np.unique(A['labels'][shared_mask]).size)
    nb = int(np.unique(B['labels'][shared_mask]).size)
    raw_ari = float(adjusted_rand_score(A['raw'], B['raw']))
    raw_nmi = float(normalized_mutual_info_score(A['raw'], B['raw']))
    joint_ari = float(adjusted_rand_score(A['labels'][shared_mask], B['labels'][shared_mask])) if int(shared_mask.sum()) >= 2 and na >= 2 and (nb >= 2) else np.nan
    pair = dict(arm=arm, seed_a=seed_a, seed_b=seed_b, recurrent_states_a=len(a), recurrent_states_b=len(b), n_recurrent_a=n_a, n_recurrent_b=n_b, n_jointly_recurrent=int(shared_mask.sum()), jointly_recurrent_fraction=float(shared_mask.mean()), raw_partition_ARI_all1733=raw_ari, raw_partition_NMI_all1733=raw_nmi, recurrent_partition_ARI_on_intersection=joint_ari, recurrent_intersection_states_a=na, recurrent_intersection_states_b=nb, matched_recurrent_weighted_Jaccard=weighted_j, matched_states_n=len(matched), unmatched_states_a=len(a) - len(matched), unmatched_states_b=len(b) - len(matched))
    return (pair, details)
source = [RANK, DISC, F481, F483, F486, F488, F489, F492, F497, F760, F761, F762, F763, F764, F765, F766, F767]
for p in source:
    require(p.is_file(), 'Missing frozen file: ' + str(p))
manifest_check(F497, [DISC, F481, F483, F486, F488, F489, F492])
manifest_check(F767, [F760, F761, F762, F763, F764, F765, F766])
r10b = json.loads(F760.read_text())
require(r10b['arms'] == ARMS, 'R10B arm definitions differ')
require(r10b['seeds'] == SEEDS, 'R10B seed definitions differ')
require(r10b['parent_sha256'][RANK.name] == sha(RANK), 'Raw input rank SHA differs from R10B')
require(r10b['parent_sha256'][DISC.name] == sha(DISC), 'Discovery ledger SHA differs from R10B')
protocol = dict(protocol_id='AML_GRAPHRX_R10C_CROSS_SEED_STATE_CONSISTENCY_V1', status='POST_HOC_EXPLORATORY_INTERNAL_DISCOVERY', seeds=SEEDS, arms=ARMS, fixed_pipeline='Exact R10B/R10A projection, cross-cohort MNN, Leiden, recurrent-state definition', source_files_sha256={p.name: sha(p) for p in source}, primary_label_invariant_metrics=['raw_partition_ARI_all1733', 'matched_recurrent_weighted_Jaccard'], supplemental=['raw_partition_NMI_all1733', 'jointly_recurrent_fraction', 'recurrent_partition_ARI_on_intersection'], caution='Never interpret ARI on jointly recurrent subjects without reporting their proportion; a single recurrent cluster gives undefined intersection ARI.', test_scope='Same 1733 discovery AML samples only; no new external transport or clinical claim', locked_parent_outputs_changed=False, drug_response_analysis=False)
save_json(F768, protocol)
save_csv(F769, pd.DataFrame([dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in source]), ['path'])
save_json(F777, {'record_type': 'IMPLEMENTATION_ONLY_CORRECTION', 'failed_first_run_stage': 'ORIGINAL_R5G_PROJECTION_FIDELITY', 'original_R10C_script_sha256': '1a6611766ace5d13883b2cd6b717e8ecbbda6ccc251d949dd63e9a0a932c1a2d', 'reason': 'R10C mistakenly re-standardized frozen R5G and R10B embeddings a second time.', 'correction_1': 'Pass locked F483 embedding into patient_projection unchanged, matching R10B.', 'correction_2': 'Pass nine R10B saved embeddings into patient_projection unchanged, matching R10B.', 'correction_3': 'Reproduce R10B descriptive ARI from recurrent-only labels, not raw all-patient labels.', 'prior_cross_seed_results_produced': False, 'frozen_parent_files_modified': False, 'existing_protocol_768_or_parent_integrity_769_modified': False, 'changes_to_training_or_clustering_thresholds': False, 'comparison_remains': 'retrospective internal discovery only'})
ledger = pd.read_csv(DISC)
require({'analysis_unit_index_0based', 'gse'}.issubset(ledger.columns), 'Bad ledger schema')
require(len(ledger) == N_PATIENTS, 'Discovery size changed')
require(ledger.analysis_unit_index_0based.is_unique, 'Duplicate discovery analysis unit')
ix = ledger.analysis_unit_index_0based.to_numpy(int)
gses = ledger.gse.astype(str).to_numpy()
require(set(gses) == set(GSES), 'Discovery cohorts changed')
require([int((gses == g).sum()) for g in GSES] == GSE_COUNTS, 'Discovery cohort sample counts changed')
r = np.load(RANK, mmap_mode='r', allow_pickle=False)
require(r.ndim == 2 and r.shape[1] == N_GENES and (ix.max() < r.shape[0]), 'Rank input shape changed')
X = np.array(r[ix, :], dtype=np.float32, copy=True)
require(np.isfinite(X).all(), 'Nonfinite discovery input')
baseP = patient_projection(X, gses, np.load(F483, allow_pickle=False))
require(float(np.max(np.abs(baseP - np.load(F486, allow_pickle=False)))) < 0.0001, 'Original R5G projection mismatch')
baseEdges = create_mnn(baseP, gses)
oldEdges = pd.read_csv(F488)
require({(a, b) for a, b, w in baseEdges} == set(zip(oldEdges.patient_node_u.astype(int), oldEdges.patient_node_v.astype(int))), 'Original MNN mismatch')
refLabels = pd.read_csv(F489).set_index('analysis_unit_index_0based').loc[ix, 'raw_community'].to_numpy(int)
baseRaw = leiden_graph(N_PATIENTS, baseEdges, SPLIT_SEED)
require(adjusted_rand_score(refLabels, baseRaw) > 0.999999, 'Original R5G Leiden mismatch')
baseMet = state_metrics(baseRaw, baseEdges, gses)
require(baseMet['states'] == 6 and abs(baseMet['coverage'] - 1720 / 1733) < 1e-10, 'Original recurrence mismatch')
print('PASS: R5G exact fidelity (MNN/Leiden/recurrence).', flush=True)
bmet = pd.read_csv(F764)
require(len(bmet) == len(SEEDS) * len(ARMS), 'R10B arm metrics should have exactly nine rows')
require(not bmet[['seed', 'arm']].duplicated().any(), 'Duplicate R10B arms')
require(set(bmet.arm) == set(ARMS) and set(bmet.seed.astype(int)) == set(SEEDS), 'R10B arms or seeds changed')
manifest = pd.read_csv(F767, dtype=str)
all_arms = {}
fidelity = []
for arm in ARMS:
    for seed in SEEDS:
        row = bmet.loc[(bmet.arm == arm) & (bmet.seed == seed)]
        require(len(row) == 1, 'Missing R10B arm-seed combination')
        row = row.iloc[0]
        emb = WORK / f'{seed}_{arm}' / 'gene_embedding32.npy'
        require(str(emb) == str(row.embeddings_path), 'R10B embedding path discrepancy')
        require(emb.is_file(), 'R10B embedding unavailable: ' + str(emb))
        require(sha(emb) == str(row.embedding_sha256), 'R10B embedding SHA mismatch: ' + str(emb))
        ms = manifest.loc[manifest['path'].eq(str(emb))]
        require(len(ms) == 1, 'Embedding missing/ambiguous in R10B manifest: ' + str(emb))
        require(str(ms.sha256.iloc[0]) == sha(emb), 'R10B manifest embedding checksum mismatch: ' + str(emb))
        E = np.load(emb, allow_pickle=False)
        require(E.shape == (N_GENES, D) and np.isfinite(E).all(), 'Invalid R10B embedding')
        P = patient_projection(X, gses, E)
        edges = create_mnn(P, gses)
        raw = leiden_graph(N_PATIENTS, edges, SPLIT_SEED)
        met = state_metrics(raw, edges, gses)
        expected = {'mnn_edges_n': len(edges), 'recurrent_states_n': met['states'], 'recurrent_coverage': met['coverage'], 'median_recurrent_stability': met['stability'], 'state_cohort_NMI_descriptive': met['cohort_nmi']}
        for key, value in expected.items():
            obs = float(row[key])
            expectedval = float(value)
            if np.isnan(expectedval):
                require(np.isnan(obs), 'R10B reproduction failed (NaN): ' + key)
            else:
                require(np.isclose(obs, expectedval, atol=2e-07, rtol=0), 'R10B reproduction failed ' + arm + '/' + str(seed) + ' ' + key + ' original=' + str(obs) + ' new=' + str(expectedval))
        orig_ari = float(adjusted_rand_score(baseMet['labels'], met['labels']))
        require(np.isclose(orig_ari, float(row.ARI_vs_original_state_labels_descriptive), atol=1e-08, rtol=0), 'R10B original-label ARI mismatch')
        all_arms[arm, seed] = {'raw': raw, 'labels': met['labels'], 'metric': met}
        fidelity.append(dict(arm=arm, seed=seed, embedding_sha256=sha(emb), R10B_reproduced=True, recurrent_states_n=met['states'], recurrent_coverage=met['coverage'], mnn_edges_n=len(edges), ARI_vs_original_state_labels=orig_ari))
        print('PASS reproduce', arm, seed, 'states', met['states'], 'coverage', round(met['coverage'], 4), flush=True)
save_csv(F770, pd.DataFrame(fidelity), ['arm', 'seed'])
pairs = []
allmatches = []
from itertools import combinations
for arm in ARMS:
    for sa, sb in combinations(SEEDS, 2):
        row, match = matched_recurrence(all_arms[arm, sa], all_arms[arm, sb], sa, sb, arm)
        pairs.append(row)
        allmatches.extend(match)
        print('PAIR', arm, sa, sb, 'raw ARI', round(row['raw_partition_ARI_all1733'], 4), 'matched Jaccard', round(row['matched_recurrent_weighted_Jaccard'], 4), 'joint coverage', round(row['jointly_recurrent_fraction'], 4), flush=True)
pairs = pd.DataFrame(pairs)
save_csv(F771, pairs, ['arm', 'seed_a', 'seed_b'])
save_csv(F772, pd.DataFrame(allmatches), ['arm', 'seed_a', 'seed_b', 'state_a', 'state_b'])
families = []
for arm, g in pairs.groupby('arm', sort=False):
    families.append(dict(arm=arm, seed_pairs_n=len(g), median_raw_partition_ARI=float(g.raw_partition_ARI_all1733.median()), min_raw_partition_ARI=float(g.raw_partition_ARI_all1733.min()), median_raw_partition_NMI=float(g.raw_partition_NMI_all1733.median()), median_matched_recurrent_weighted_Jaccard=float(g.matched_recurrent_weighted_Jaccard.median()), min_matched_recurrent_weighted_Jaccard=float(g.matched_recurrent_weighted_Jaccard.min()), median_jointly_recurrent_fraction=float(g.jointly_recurrent_fraction.median()), median_intersection_ARI_nonmissing=float(g.recurrent_partition_ARI_on_intersection.median()) if g.recurrent_partition_ARI_on_intersection.notna().any() else np.nan))
families = pd.DataFrame(families)
save_csv(F773, families, ['arm'])
G = families.set_index('arm')
graph = G.loc['REAL_GRAPH']
rank_by_ari = bool(graph.median_raw_partition_ARI > G.drop('REAL_GRAPH').median_raw_partition_ARI.max())
rank_by_j = bool(graph.median_matched_recurrent_weighted_Jaccard > G.drop('REAL_GRAPH').median_matched_recurrent_weighted_Jaccard.max())
summary = pd.DataFrame([('analysis_id', 'AML_GRAPHRX_R10C_CROSS_SEED_STATE_CONSISTENCY_V1'), ('original_R5G_reproduced', True), ('R10B_all_9_arms_reproduced', True), ('same_1733_discovery_samples', True), ('independent_external_validation_performed', False), ('graph_median_pairwise_raw_ARI', graph.median_raw_partition_ARI), ('graph_median_matched_recurrent_Jaccard', graph.median_matched_recurrent_weighted_Jaccard), ('graph_highest_median_raw_ARI_descriptive', rank_by_ari), ('graph_highest_median_Jaccard_descriptive', rank_by_j), ('graph_cross_seed_superiority_established', False), ('reason_for_false_superiority_flag', 'Three known seeds on one discovery collection; stability does not establish external transport or clinical use.'), ('caution', 'Bootstrap within-seed Jaccard is not cross-seed identity consistency; internal recurrence is not external transport.')], columns=['metric', 'value'])
save_csv(F774, summary)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3), layout='constrained')
metric_cols = ['raw_partition_ARI_all1733', 'matched_recurrent_weighted_Jaccard', 'jointly_recurrent_fraction']
titles = ['All-patient partition ARI', 'Matched recurrent-state Jaccard', 'Jointly recurrent sample fraction']
shorts = ['Real graph', 'Degree-preserving rewire', 'No message passing']
cols = ['#265D8E', '#DA9550', '#6A8590']
for ax, metric, title in zip(axs, metric_cols, titles):
    for j, arm in enumerate(ARMS):
        vals = pairs.loc[pairs.arm == arm, metric].to_numpy(float)
        ax.scatter(np.full(3, j), vals, s=45, color=cols[j], alpha=0.75, zorder=3)
        ax.plot([j - 0.15, j + 0.15], [float(np.median(vals))] * 2, color=cols[j], lw=3, zorder=4)
    ax.set_xticks(range(3), shorts, rotation=20, ha='right')
    ax.set_ylim(0, 1.03)
    ax.set_ylabel('Score / fraction')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.grid(axis='y', alpha=0.2)
fig.suptitle('Cross-seed consistency of gene-first AML state discovery (3 pairs per arm)', fontweight='bold', fontsize=13)
from io import BytesIO
for p, fmt in [(F776PNG, 'png'), (F776PDF, 'pdf')]:
    buf = BytesIO()
    metadata = {'Software': 'AML-GraphRx R10C'} if fmt == 'png' else {'Creator': 'AML-GraphRx R10C', 'CreationDate': None, 'ModDate': None}
    fig.savefig(buf, format=fmt, dpi=320, bbox_inches='tight', facecolor='white', metadata=metadata)
    write_once(p, buf.getvalue())
plt.close(fig)
outputs = [F768, F769, F777, F770, F771, F772, F773, F774, F776PNG, F776PDF]
save_csv(F775, pd.DataFrame([dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in outputs]), ['path'])
print('\n' + '=' * 95)
print('R10C COMPLETE — CROSS-SEED STATE CONSISTENCY, INTERNAL DISCOVERY ONLY')
print('=' * 95)
print('\nPAIRWISE RESULTS\n' + pairs.to_string(index=False))
print('\nFAMILY SUMMARY\n' + families.to_string(index=False))
print('\nFINAL SUMMARY\n' + summary.to_string(index=False))
print('\nAll original frozen files remain unchanged. Results saved in:', AUDIT)
