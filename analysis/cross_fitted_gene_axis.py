# Manuscript-linked analysis: Recorded cross-fitted gene-axis assessment
# Source lineage and packaging changes: provenance/source_lineage.json
import os as _release_os
try:
    from IPython.display import display
except ImportError:

    def display(value):
        print(value)
import os
import json
import math
import hashlib
import itertools
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, norm, mannwhitneyu, ks_2samp, spearmanr, kruskal
from sklearn.metrics import roc_auc_score
try:
    pass
    pass
except Exception:
    pass
PROJECT = Path(_release_os.environ.get('AML_GRAPHRX_ROOT', '/content/drive/MyDrive/AML_GraphRx'))
AUDIT = PROJECT / 'results' / 'audit'
FIGDIR = PROJECT / 'figures'
AUDIT.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)
DISCOVERY_GSES = ['GSE15434', 'GSE161532', 'GSE1729', 'GSE34860', 'GSE37021', 'GSE37642', 'GSE6891', 'GSE71014']
EXPECTED_COUNTS = {'GSE15434': 251, 'GSE161532': 61, 'GSE1729': 43, 'GSE34860': 78, 'GSE37021': 97, 'GSE37642': 562, 'GSE6891': 537, 'GSE71014': 104}
N_GENES = 9554
MIN_PER_POLE_PER_COHORT = 3
MIN_ELIGIBLE_COHORTS = 3
AXIS_FDR = 0.05
AXIS_DIRECTION_CONSISTENCY = 0.8
MIN_AXIS_GENES_PER_DIRECTION = 25
SUPPORT_MIN_EVALUABLE_FOLDS = 6
SUPPORT_MIN_MEDIAN_WEIGHT_SPEARMAN = 0.5
SUPPORT_MIN_OOF_VS_ALL8_SPEARMAN = 0.7
SUPPORT_MIN_MEDIAN_HELDOUT_WITHIN_TRAIN_CENTRAL95 = 0.75
EXPECTED_SHA = {'rank': '8c2a914c11562073fb3e6bfbc36f8e2c0fc5d76ca88237a938761e4ca79f7a10', 'node': 'c0596efbded29dcb848c1620c25bb00344012c72cfda2ca4ac5efc256c1baf20', 'discovery': 'c33172332ef0ab385c0701472338895acce5d993355f57ec2d13e83a35be4778', 'original_states': '765facf2bdc0e81f7eefaa7561411f694eac95c0b482d7177f77356bf7cf7876'}
F571 = AUDIT / '571_R6_predeclared_program_axis_protocol.json'
F572 = AUDIT / '572_R6_fold_axis_definitions.csv'
F573 = AUDIT / '573_R6_fold_gene_weights.csv'
F574 = AUDIT / '574_R6_fold_score_transfer_summary.csv'
F575 = AUDIT / '575_R6_fold_scores_long.csv'
F576 = AUDIT / '576_R6_fold_weight_stability.csv'
F577 = AUDIT / '577_R6_out_of_fold_program_scores.csv'
F578 = AUDIT / '578_R6_all8_candidate_axis_genes.csv'
F579 = AUDIT / '579_R6_all8_candidate_scores.csv'
F580 = AUDIT / '580_R6_original_S03_S06_descriptive_alignment.csv'
F581 = AUDIT / '581_R6_final_summary.csv'
F582 = AUDIT / '582_R6_manifest.csv'

def require(cond, msg):
    if not bool(cond):
        raise RuntimeError(msg)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def find_exact(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    require(len(hits) >= 1, f'File not found: {name}')
    hits = sorted(hits, key=lambda p: (len(str(p)), str(p)))
    return hits[0]

def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out

def canonical_json_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=True) + '\n').encode('utf-8')

def strict_write_json(path: Path, obj):
    b = canonical_json_bytes(obj)
    if path.exists():
        old = path.read_bytes()
        require(old == b, f'HARD STOP: existing JSON differs: {path.name}')
        print(f'PASS existing identical: {path.name}')
    else:
        path.write_bytes(b)
        print(f'CREATED: {path.name}')

def strict_write_csv(path: Path, df: pd.DataFrame, sort_cols=None):
    x = df.copy()
    if sort_cols:
        valid = [c for c in sort_cols if c in x.columns]
        if valid:
            x = x.sort_values(valid, kind='mergesort').reset_index(drop=True)
    b = x.to_csv(index=False, lineterminator='\n').encode('utf-8')
    if path.exists():
        old = path.read_bytes()
        require(old == b, f'HARD STOP: existing CSV differs: {path.name}')
        print(f'PASS existing identical: {path.name}')
    else:
        path.write_bytes(b)
        print(f'CREATED: {path.name}')

def jaccard(a, b):
    a, b = (set(a), set(b))
    if not a and (not b):
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def robust_iqr(x):
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, 0.75) - np.quantile(x, 0.25))

def rank_biserial_from_u(U, n1, n2):
    if n1 == 0 or n2 == 0:
        return np.nan
    return float(2.0 * U / (n1 * n2) - 1.0)
RANK_FILE = find_exact(PROJECT, '20260823T123040Z_patient_rank_raw_float64.npy')
NODE_FILE = find_exact(PROJECT, '20260823T123040Z_node_index.csv')
DISCOVERY_FILE = find_exact(PROJECT, '452_R5DICE_discovery_ledger.csv')
ORIGINAL_STATE_FILE = find_exact(PROJECT, '492_R5G_recurrent_state_patient_assignments.csv')
R5X_HELDOUT_FILE = find_exact(PROJECT, '559_R5X_LOCO_heldout_assignments.csv')
R5X_TRAIN_DIAG_FILE = find_exact(PROJECT, '560_R5X_LOCO_training_centroid_diagnostics.csv')
R5X_FINAL_ASSIGN_FILE = find_exact(PROJECT, '563_R5X_final_candidate_assignments.csv')
R5X_FINAL_SUMMARY_FILE = find_exact(PROJECT, '569_R5X_final_summary.csv')
R5X_MANIFEST_FILE = find_exact(PROJECT, '570_R5X_manifest.csv')
print('\nAUTHORITATIVE INPUTS')
print('-' * 110)
for p in [RANK_FILE, NODE_FILE, DISCOVERY_FILE, ORIGINAL_STATE_FILE, R5X_HELDOUT_FILE, R5X_TRAIN_DIAG_FILE, R5X_FINAL_ASSIGN_FILE, R5X_FINAL_SUMMARY_FILE, R5X_MANIFEST_FILE]:
    print(p)
for key, p in [('rank', RANK_FILE), ('node', NODE_FILE), ('discovery', DISCOVERY_FILE), ('original_states', ORIGINAL_STATE_FILE)]:
    obs = sha256_file(p)
    require(obs == EXPECTED_SHA[key], f'SHA256 mismatch for {p.name}')
    print(f'SHA256 PASS: {p.name}')
r5x_input_hashes = {p.name: sha256_file(p) for p in [R5X_HELDOUT_FILE, R5X_TRAIN_DIAG_FILE, R5X_FINAL_ASSIGN_FILE, R5X_FINAL_SUMMARY_FILE, R5X_MANIFEST_FILE]}
discovery = pd.read_csv(DISCOVERY_FILE, dtype={'analysis_unit_id': str, 'analysis_unit_index_0based': int, 'gse': str})
require(len(discovery) == 1733, 'Discovery ledger must contain 1733 AML profiles.')
require(discovery['analysis_unit_id'].is_unique, 'Discovery analysis_unit_id must be unique.')
require(discovery['analysis_unit_index_0based'].is_unique, 'Discovery matrix indices must be unique.')
require(set(discovery['gse']) == set(DISCOVERY_GSES), 'Discovery cohort set changed.')
for gse, n in EXPECTED_COUNTS.items():
    require(int(discovery['gse'].eq(gse).sum()) == n, f'{gse}: unexpected n')
nodes = pd.read_csv(NODE_FILE, dtype=str, keep_default_na=False)
require(len(nodes) == N_GENES, f'Expected {N_GENES} genes.')
require('canonical_NCBI_Gene_ID' in nodes.columns, 'Node index missing canonical_NCBI_Gene_ID.')
SYMBOL_COL = next((c for c in ['gene_symbol', 'symbol', 'GeneSymbol', 'gene_name'] if c in nodes.columns), None)
rank_mm = np.load(RANK_FILE, mmap_mode='r', allow_pickle=False)
require(rank_mm.shape == (4554, N_GENES), f'Unexpected rank matrix shape: {rank_mm.shape}')
idx = discovery['analysis_unit_index_0based'].to_numpy(dtype=int)
X = np.asarray(rank_mm[idx, :], dtype=np.float32)
require(X.shape == (1733, N_GENES), 'Discovery extraction failed.')
require(np.isfinite(X).all(), 'Non-finite rank values found.')
require(float(X.min()) >= -1e-07 and float(X.max()) <= 1.0000001, 'Rank matrix not in expected [0,1] range.')
train_diag = pd.read_csv(R5X_TRAIN_DIAG_FILE, dtype=str, keep_default_na=False)
heldout_r5x = pd.read_csv(R5X_HELDOUT_FILE, dtype=str, keep_default_na=False)
final_assign = pd.read_csv(R5X_FINAL_ASSIGN_FILE, dtype=str, keep_default_na=False)
original_states = pd.read_csv(ORIGINAL_STATE_FILE, dtype=str, keep_default_na=False)
required_diag = {'fold_id', 'heldout_gse', 'analysis_unit_id', 'gse', 'true_discovered_state'}
require(required_diag.issubset(train_diag.columns), f'560 schema missing: {required_diag - set(train_diag.columns)}')
require({'analysis_unit_id', 'gse', 'state_id'}.issubset(final_assign.columns), '563 schema changed.')
require({'analysis_unit_id', 'state_id'}.issubset(original_states.columns), '492 schema changed.')
id_to_local = pd.Series(np.arange(len(discovery)), index=discovery['analysis_unit_id']).to_dict()
require(set(train_diag['analysis_unit_id']).issubset(set(discovery['analysis_unit_id'])), '560 contains unknown analysis units.')
protocol = {'protocol_id': 'AML_GRAPHRX_R6_CROSSFITTED_PROGRAM_AXIS_V1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'analysis_status': 'post-R5/R5X corrective exploratory branch; prospectively fixed for this R6 run', 'primary_goal': 'test whether a continuous single-sample antagonistic program axis is more stable across cohorts than discrete state transport', 'inputs': {'locked_rank_matrix': str(RANK_FILE), 'locked_node_index': str(NODE_FILE), 'locked_discovery_ledger': str(DISCOVERY_FILE), 'original_R5_states_descriptive_only': str(ORIGINAL_STATE_FILE), 'R5X_training_diagnostics': str(R5X_TRAIN_DIAG_FILE), 'R5X_heldout_assignments': str(R5X_HELDOUT_FILE), 'R5X_all8_candidate_assignments': str(R5X_FINAL_ASSIGN_FILE), 'R5X_input_sha256': r5x_input_hashes}, 'fold_logic': 'each R5X LOCO fold uses only its seven-cohort recurrent-state training labels to select the antagonistic pair and genes; held-out cohort is never used for pair selection or gene selection', 'pair_selection': {'all recurrent-state pairs evaluated within training only': True, 'min_per_pole_per_cohort': MIN_PER_POLE_PER_COHORT, 'min_eligible_cohorts': MIN_ELIGIBLE_COHORTS, 'primary_rank_key_1': 'maximum robust differential genes', 'primary_rank_key_2': 'maximum median absolute meta-Z among robust genes', 'primary_rank_key_3': 'maximum pair sample count', 'final_tie_break': 'lexicographic state-pair ID'}, 'gene_rule': {'meta_method': 'sample-size-weighted Stouffer signed Z from per-cohort Welch t-tests', 'BH_FDR_lt': AXIS_FDR, 'direction_consistency_ge': AXIS_DIRECTION_CONSISTENCY, 'min_genes_each_direction': MIN_AXIS_GENES_PER_DIRECTION}, 'single_sample_primary_score': 'mean within-profile percentile rank of UP genes minus mean within-profile percentile rank of DOWN genes; no cohort scaling and no held-out parameter fitting', 'primary_stability_metrics': ['number of evaluable LOCO folds', 'median pairwise Spearman correlation of sign-aligned full fold meta-Z vectors', 'OOF-vs-all8 score Spearman correlation', 'median held-out fraction within training central 95% score range'], 'support_thresholds_fixed_before_result': {'evaluable_folds_ge': SUPPORT_MIN_EVALUABLE_FOLDS, 'median_pairwise_weight_spearman_ge': SUPPORT_MIN_MEDIAN_WEIGHT_SPEARMAN, 'OOF_vs_all8_spearman_ge': SUPPORT_MIN_OOF_VS_ALL8_SPEARMAN, 'median_heldout_within_training_central95_ge': SUPPORT_MIN_MEDIAN_HELDOUT_WITHIN_TRAIN_CENTRAL95}, 'forbidden': ['use held-out cohort to select state pair', 'use held-out cohort to select genes', 'tune score thresholds after seeing held-out results', 'retrain GNN', 'delete a GSE because of poor performance', 'claim external subtype validation from original eight discovery cohorts'], 'original_State03_State06_use': 'descriptive alignment only after all R6 model construction and primary stability calculations'}
strict_write_json(F571, protocol)

def pair_meta_contrast(assign_df: pd.DataFrame, state_a: str, state_b: str):
    """Training-only state A vs state B gene-wise meta contrast."""
    sub = assign_df.loc[assign_df['true_discovered_state'].isin([state_a, state_b])].copy()
    effects, zs, ws, cohort_names = ([], [], [], [])
    n_a_total = int(sub['true_discovered_state'].eq(state_a).sum())
    n_b_total = int(sub['true_discovered_state'].eq(state_b).sum())
    for gse in sorted(sub['gse'].unique()):
        gg = sub.loc[sub['gse'].eq(gse)]
        ids_a = gg.loc[gg['true_discovered_state'].eq(state_a), 'analysis_unit_id'].tolist()
        ids_b = gg.loc[gg['true_discovered_state'].eq(state_b), 'analysis_unit_id'].tolist()
        if len(ids_a) < MIN_PER_POLE_PER_COHORT or len(ids_b) < MIN_PER_POLE_PER_COHORT:
            continue
        ia = np.asarray([id_to_local[x] for x in ids_a], dtype=int)
        ib = np.asarray([id_to_local[x] for x in ids_b], dtype=int)
        A, B = (X[ia], X[ib])
        effect = A.mean(axis=0, dtype=np.float64) - B.mean(axis=0, dtype=np.float64)
        test = ttest_ind(A, B, axis=0, equal_var=False, nan_policy='omit')
        p = np.asarray(test.pvalue, dtype=float)
        p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
        p = np.clip(p, 1e-300, 1.0)
        signed_z = np.sign(effect) * norm.isf(p / 2.0)
        w = math.sqrt(len(ia) * len(ib) / (len(ia) + len(ib)))
        effects.append(effect)
        zs.append(signed_z)
        ws.append(float(w))
        cohort_names.append(gse)
    if len(cohort_names) < MIN_ELIGIBLE_COHORTS:
        return None
    E = np.stack(effects)
    Z = np.stack(zs)
    W = np.asarray(ws, dtype=float)
    meta_z = (Z * W[:, None]).sum(axis=0) / math.sqrt(float(np.sum(W ** 2)))
    meta_p = 2.0 * norm.sf(np.abs(meta_z))
    meta_fdr = bh_fdr(meta_p)
    meta_direction = np.sign(meta_z)
    consistency = np.mean(np.sign(E) == meta_direction[None, :], axis=0)
    robust = (meta_fdr < AXIS_FDR) & (consistency >= AXIS_DIRECTION_CONSISTENCY) & (meta_direction != 0)
    up = robust & (meta_z > 0)
    down = robust & (meta_z < 0)
    med_abs_z = float(np.median(np.abs(meta_z[robust]))) if robust.any() else 0.0
    return {'state_a': state_a, 'state_b': state_b, 'eligible_cohorts': cohort_names, 'n_a_total': n_a_total, 'n_b_total': n_b_total, 'meta_z': meta_z.astype(np.float64), 'meta_p': meta_p.astype(np.float64), 'meta_fdr': meta_fdr.astype(np.float64), 'consistency': consistency.astype(np.float64), 'robust': robust, 'up': up, 'down': down, 'robust_n': int(robust.sum()), 'up_n': int(up.sum()), 'down_n': int(down.sum()), 'median_abs_meta_z_robust': med_abs_z, 'pair_n': int(n_a_total + n_b_total)}

def select_axis_pair(assign_df: pd.DataFrame):
    states = sorted(assign_df['true_discovered_state'].unique().tolist())
    candidates = []
    for a, b in itertools.combinations(states, 2):
        d = pair_meta_contrast(assign_df, a, b)
        if d is None:
            continue
        if d['up_n'] < MIN_AXIS_GENES_PER_DIRECTION or d['down_n'] < MIN_AXIS_GENES_PER_DIRECTION:
            continue
        candidates.append(d)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda d: (-d['robust_n'], -d['median_abs_meta_z_robust'], -d['pair_n'], d['state_a'], d['state_b']))
    return candidates[0]

def score_axis(Xmat, model):
    up_idx = np.where(model['up'])[0]
    down_idx = np.where(model['down'])[0]
    require(len(up_idx) >= MIN_AXIS_GENES_PER_DIRECTION, 'Too few UP genes.')
    require(len(down_idx) >= MIN_AXIS_GENES_PER_DIRECTION, 'Too few DOWN genes.')
    primary = Xmat[:, up_idx].mean(axis=1, dtype=np.float64) - Xmat[:, down_idx].mean(axis=1, dtype=np.float64)
    wu = np.abs(model['meta_z'][up_idx])
    wd = np.abs(model['meta_z'][down_idx])
    wu = wu / wu.sum()
    wd = wd / wd.sum()
    weighted = Xmat[:, up_idx].astype(np.float64) @ wu - Xmat[:, down_idx].astype(np.float64) @ wd
    return (np.asarray(primary, dtype=float), np.asarray(weighted, dtype=float))
fold_defs = []
fold_weight_rows = []
fold_transfer = []
fold_score_frames = []
fold_models = {}
for fold_no, heldout_gse in enumerate(DISCOVERY_GSES, start=1):
    fold_id = f'LOCO_{fold_no:02d}_{heldout_gse}'
    print('\n' + '=' * 110)
    print('R6', fold_id)
    print('=' * 110)
    ad = train_diag.loc[train_diag['fold_id'].eq(fold_id)].copy()
    if len(ad) == 0:
        fold_defs.append({'fold_id': fold_id, 'heldout_gse': heldout_gse, 'status': 'NOT_EVALUABLE_NO_R5X_RECURRENT_STATES'})
        continue
    require(heldout_gse not in set(ad['gse']), f'Leakage: {heldout_gse} present in training labels for {fold_id}')
    require(ad['analysis_unit_id'].is_unique, f'Duplicate training IDs within {fold_id}')
    model = select_axis_pair(ad)
    if model is None:
        fold_defs.append({'fold_id': fold_id, 'heldout_gse': heldout_gse, 'training_states_n': int(ad['true_discovered_state'].nunique()), 'status': 'NOT_EVALUABLE_NO_PAIR_PASSING_AXIS_RULES'})
        continue
    fold_models[fold_id] = model
    fold_defs.append({'fold_id': fold_id, 'heldout_gse': heldout_gse, 'training_states_n': int(ad['true_discovered_state'].nunique()), 'pole_A_state': model['state_a'], 'pole_B_state': model['state_b'], 'eligible_pairwise_cohorts_n': len(model['eligible_cohorts']), 'eligible_pairwise_cohorts': ';'.join(model['eligible_cohorts']), 'pole_A_n': model['n_a_total'], 'pole_B_n': model['n_b_total'], 'robust_axis_genes_n': model['robust_n'], 'axis_UP_genes_n': model['up_n'], 'axis_DOWN_genes_n': model['down_n'], 'median_abs_meta_Z_robust': model['median_abs_meta_z_robust'], 'status': 'EVALUABLE'})
    robust_idx = np.where(model['robust'])[0]
    for gi in robust_idx:
        fold_weight_rows.append({'fold_id': fold_id, 'heldout_gse': heldout_gse, 'pole_A_state': model['state_a'], 'pole_B_state': model['state_b'], 'gene_index_0based': int(gi), 'canonical_NCBI_Gene_ID': nodes.iloc[gi]['canonical_NCBI_Gene_ID'], 'gene_symbol': nodes.iloc[gi][SYMBOL_COL] if SYMBOL_COL else '', 'meta_Z_A_minus_B': float(model['meta_z'][gi]), 'meta_p': float(model['meta_p'][gi]), 'meta_FDR_BH': float(model['meta_fdr'][gi]), 'direction_consistency_fraction': float(model['consistency'][gi]), 'axis_direction': 'UP_POLE_A' if model['meta_z'][gi] > 0 else 'UP_POLE_B'})
    score, score_w = score_axis(X, model)
    train_mask = ~discovery['gse'].eq(heldout_gse).to_numpy()
    test_mask = discovery['gse'].eq(heldout_gse).to_numpy()
    s_train, s_test = (score[train_mask], score[test_mask])
    q025, q975 = np.quantile(s_train, [0.025, 0.975])
    q005, q995 = np.quantile(s_train, [0.005, 0.995])
    train_iqr = robust_iqr(s_train)
    test_iqr = robust_iqr(s_test)
    median_shift_iqr = (np.median(s_test) - np.median(s_train)) / train_iqr if train_iqr > 0 else np.nan
    iqr_ratio = test_iqr / train_iqr if train_iqr > 0 else np.nan
    within95 = float(np.mean((s_test >= q025) & (s_test <= q975)))
    within99 = float(np.mean((s_test >= q005) & (s_test <= q995)))
    ks = ks_2samp(s_test, s_train, alternative='two-sided', mode='auto')
    idsA = ad.loc[ad['true_discovered_state'].eq(model['state_a']), 'analysis_unit_id']
    idsB = ad.loc[ad['true_discovered_state'].eq(model['state_b']), 'analysis_unit_id']
    ia = np.asarray([id_to_local[x] for x in idsA], dtype=int)
    ib = np.asarray([id_to_local[x] for x in idsB], dtype=int)
    y = np.r_[np.ones(len(ia), dtype=int), np.zeros(len(ib), dtype=int)]
    ss = np.r_[score[ia], score[ib]]
    train_auc = float(roc_auc_score(y, ss))
    mw = mannwhitneyu(score[ia], score[ib], alternative='two-sided')
    rb = rank_biserial_from_u(float(mw.statistic), len(ia), len(ib))
    fold_transfer.append({'fold_id': fold_id, 'heldout_gse': heldout_gse, 'train_n': int(train_mask.sum()), 'heldout_n': int(test_mask.sum()), 'pole_A_state': model['state_a'], 'pole_B_state': model['state_b'], 'robust_axis_genes_n': model['robust_n'], 'axis_UP_genes_n': model['up_n'], 'axis_DOWN_genes_n': model['down_n'], 'training_pole_A_vs_B_AUROC_descriptive': train_auc, 'training_pole_rank_biserial_descriptive': rb, 'training_score_median': float(np.median(s_train)), 'heldout_score_median': float(np.median(s_test)), 'training_score_IQR': train_iqr, 'heldout_score_IQR': test_iqr, 'heldout_median_shift_in_training_IQR': float(median_shift_iqr), 'heldout_IQR_ratio': float(iqr_ratio), 'heldout_fraction_within_training_central95': within95, 'heldout_fraction_within_training_central99': within99, 'KS_statistic_heldout_vs_training': float(ks.statistic), 'KS_p_heldout_vs_training': float(ks.pvalue)})
    tmp = discovery[['analysis_unit_id', 'analysis_unit_index_0based', 'gse']].copy()
    tmp.insert(0, 'fold_id', fold_id)
    tmp.insert(1, 'heldout_gse', heldout_gse)
    tmp['analysis_role'] = np.where(tmp['gse'].eq(heldout_gse), 'HELDOUT_OOF', 'TRAINING_DESCRIPTIVE')
    tmp['program_axis_score'] = score
    tmp['program_axis_score_weighted_secondary'] = score_w
    fold_score_frames.append(tmp)
fold_defs_df = pd.DataFrame(fold_defs)
fold_weights_df = pd.DataFrame(fold_weight_rows)
fold_transfer_df = pd.DataFrame(fold_transfer)
fold_scores_df = pd.concat(fold_score_frames, ignore_index=True) if fold_score_frames else pd.DataFrame()
if len(fold_transfer_df):
    fold_transfer_df['KS_FDR_BH_heldout_vs_training'] = bh_fdr(fold_transfer_df['KS_p_heldout_vs_training'].to_numpy(float))
strict_write_csv(F572, fold_defs_df, ['fold_id'])
strict_write_csv(F573, fold_weights_df, ['fold_id', 'gene_index_0based'])
strict_write_csv(F574, fold_transfer_df, ['fold_id'])
strict_write_csv(F575, fold_scores_df, ['fold_id', 'analysis_unit_index_0based'])
evaluable_fold_ids = [f'LOCO_{i:02d}_{g}' for i, g in enumerate(DISCOVERY_GSES, 1) if f'LOCO_{i:02d}_{g}' in fold_models]
require(len(evaluable_fold_ids) >= 1, 'No evaluable R6 folds.')
reference_fold = evaluable_fold_ids[0]
ref_w = fold_models[reference_fold]['meta_z'].copy()
aligned_weights = {}
fold_sign = {}
for fid in evaluable_fold_ids:
    w = fold_models[fid]['meta_z'].copy()
    rho = float(spearmanr(ref_w, w).statistic)
    sign = -1.0 if np.isfinite(rho) and rho < 0 else 1.0
    aligned_weights[fid] = sign * w
    fold_sign[fid] = sign
stability_rows = []
for f1, f2 in itertools.combinations(evaluable_fold_ids, 2):
    w1, w2 = (aligned_weights[f1], aligned_weights[f2])
    rho = float(spearmanr(w1, w2).statistic)
    m1 = fold_models[f1]
    m2 = fold_models[f2]
    pos1 = set(np.where(m1['robust'] & (fold_sign[f1] * m1['meta_z'] > 0))[0])
    neg1 = set(np.where(m1['robust'] & (fold_sign[f1] * m1['meta_z'] < 0))[0])
    pos2 = set(np.where(m2['robust'] & (fold_sign[f2] * m2['meta_z'] > 0))[0])
    neg2 = set(np.where(m2['robust'] & (fold_sign[f2] * m2['meta_z'] < 0))[0])
    stability_rows.append({'fold_1': f1, 'fold_2': f2, 'spearman_full_metaZ_sign_aligned': rho, 'positive_gene_Jaccard': jaccard(pos1, pos2), 'negative_gene_Jaccard': jaccard(neg1, neg2)})
stability_df = pd.DataFrame(stability_rows)
strict_write_csv(F576, stability_df, ['fold_1', 'fold_2'])
oof_parts = []
for fid in evaluable_fold_ids:
    heldout_gse = fid.split('_', 3)[-1]
    z = fold_scores_df.loc[fold_scores_df['fold_id'].eq(fid) & fold_scores_df['analysis_role'].eq('HELDOUT_OOF')].copy()
    z['orientation_sign_to_reference_fold'] = fold_sign[fid]
    z['OOF_program_axis_score_aligned'] = fold_sign[fid] * z['program_axis_score'].astype(float)
    z['OOF_program_axis_score_weighted_secondary_aligned'] = fold_sign[fid] * z['program_axis_score_weighted_secondary'].astype(float)
    oof_parts.append(z)
oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
require(oof['analysis_unit_id'].is_unique, 'OOF score must contain at most one score per discovery patient.')
strict_write_csv(F577, oof, ['analysis_unit_index_0based'])
if len(oof) > 0 and oof['gse'].nunique() >= 2:
    groups = [g['OOF_program_axis_score_aligned'].astype(float).to_numpy() for _, g in oof.groupby('gse')]
    kw = kruskal(*groups)
    n_kw = sum((len(g) for g in groups))
    k_kw = len(groups)
    eps2 = max(0.0, float((kw.statistic - k_kw + 1) / (n_kw - k_kw))) if n_kw > k_kw else np.nan
else:
    kw = None
    eps2 = np.nan
fa = final_assign.loc[final_assign['state_id'].ne('UNRESOLVED'), ['analysis_unit_id', 'gse', 'state_id']].copy()
fa = fa.rename(columns={'state_id': 'true_discovered_state'})
require(fa['analysis_unit_id'].is_unique, 'All8 corrected recurrent assignment IDs are not unique.')
all8_model = select_axis_pair(fa)
require(all8_model is not None, 'All-eight corrected candidate has no state pair passing R6 axis rules.')
all8_score, all8_score_w = score_axis(X, all8_model)
rho_all8_ref_raw = float(spearmanr(ref_w, all8_model['meta_z']).statistic)
all8_sign = -1.0 if np.isfinite(rho_all8_ref_raw) and rho_all8_ref_raw < 0 else 1.0
all8_score_aligned = all8_sign * all8_score
all8_weight_aligned = all8_sign * all8_model['meta_z']
all8_gene_rows = []
for gi in np.where(all8_model['robust'])[0]:
    signed_z = float(all8_sign * all8_model['meta_z'][gi])
    all8_gene_rows.append({'gene_index_0based': int(gi), 'canonical_NCBI_Gene_ID': nodes.iloc[gi]['canonical_NCBI_Gene_ID'], 'gene_symbol': nodes.iloc[gi][SYMBOL_COL] if SYMBOL_COL else '', 'meta_Z_raw_poleA_minus_poleB': float(all8_model['meta_z'][gi]), 'meta_Z_aligned_to_reference_fold': signed_z, 'meta_p': float(all8_model['meta_p'][gi]), 'meta_FDR_BH': float(all8_model['meta_fdr'][gi]), 'direction_consistency_fraction': float(all8_model['consistency'][gi]), 'aligned_direction': 'POSITIVE_AXIS' if signed_z > 0 else 'NEGATIVE_AXIS'})
all8_genes_df = pd.DataFrame(all8_gene_rows)
strict_write_csv(F578, all8_genes_df, ['gene_index_0based'])
all8_scores_df = discovery[['analysis_unit_id', 'analysis_unit_index_0based', 'gse']].copy()
all8_scores_df['all8_program_axis_score_raw'] = all8_score
all8_scores_df['all8_program_axis_score_aligned'] = all8_score_aligned
all8_scores_df['all8_program_axis_score_weighted_secondary_raw'] = all8_score_w
strict_write_csv(F579, all8_scores_df, ['analysis_unit_index_0based'])
score_compare = oof[['analysis_unit_id', 'OOF_program_axis_score_aligned']].merge(all8_scores_df[['analysis_unit_id', 'all8_program_axis_score_aligned']], on='analysis_unit_id', how='inner', validate='one_to_one')
if len(score_compare) >= 3:
    oof_all8_rho = float(spearmanr(score_compare['OOF_program_axis_score_aligned'].astype(float), score_compare['all8_program_axis_score_aligned'].astype(float)).statistic)
else:
    oof_all8_rho = np.nan
align = all8_scores_df.merge(original_states[['analysis_unit_id', 'state_id']], on='analysis_unit_id', how='left', validate='one_to_one')
sub = align.loc[align['state_id'].isin(['AML_STATE_03', 'AML_STATE_06'])].copy()
if len(sub) > 0 and sub['state_id'].nunique() == 2:
    y = sub['state_id'].eq('AML_STATE_06').astype(int).to_numpy()
    s = sub['all8_program_axis_score_aligned'].astype(float).to_numpy()
    auc_directional = float(roc_auc_score(y, s))
    auc_abs = float(max(auc_directional, 1.0 - auc_directional))
    med3 = float(sub.loc[sub['state_id'].eq('AML_STATE_03'), 'all8_program_axis_score_aligned'].median())
    med6 = float(sub.loc[sub['state_id'].eq('AML_STATE_06'), 'all8_program_axis_score_aligned'].median())
    mw = mannwhitneyu(sub.loc[sub['state_id'].eq('AML_STATE_03'), 'all8_program_axis_score_aligned'].astype(float), sub.loc[sub['state_id'].eq('AML_STATE_06'), 'all8_program_axis_score_aligned'].astype(float), alternative='two-sided')
    align_summary = pd.DataFrame([{'analysis_role': 'DESCRIPTIVE_ONLY_NOT_INDEPENDENT_VALIDATION', 'original_State03_n': int(sub['state_id'].eq('AML_STATE_03').sum()), 'original_State06_n': int(sub['state_id'].eq('AML_STATE_06').sum()), 'median_axis_score_State03': med3, 'median_axis_score_State06': med6, 'AUROC_State06_vs_State03_directional': auc_directional, 'AUROC_State06_vs_State03_absolute_separation': auc_abs, 'Mann_Whitney_p': float(mw.pvalue)}])
else:
    align_summary = pd.DataFrame([{'analysis_role': 'NOT_EVALUABLE'}])
strict_write_csv(F580, align_summary)
evaluable_folds_n = len(evaluable_fold_ids)
median_weight_rho = float(stability_df['spearman_full_metaZ_sign_aligned'].median()) if len(stability_df) else np.nan
median_pos_j = float(stability_df['positive_gene_Jaccard'].median()) if len(stability_df) else np.nan
median_neg_j = float(stability_df['negative_gene_Jaccard'].median()) if len(stability_df) else np.nan
median_within95 = float(fold_transfer_df['heldout_fraction_within_training_central95'].median()) if len(fold_transfer_df) else np.nan
median_abs_shift_iqr = float(np.median(np.abs(fold_transfer_df['heldout_median_shift_in_training_IQR'].astype(float)))) if len(fold_transfer_df) else np.nan
gates = {'evaluable_folds_ge_6': evaluable_folds_n >= SUPPORT_MIN_EVALUABLE_FOLDS, 'median_pairwise_weight_spearman_ge_0.50': np.isfinite(median_weight_rho) and median_weight_rho >= SUPPORT_MIN_MEDIAN_WEIGHT_SPEARMAN, 'OOF_vs_all8_spearman_ge_0.70': np.isfinite(oof_all8_rho) and oof_all8_rho >= SUPPORT_MIN_OOF_VS_ALL8_SPEARMAN, 'median_heldout_within_training_central95_ge_0.75': np.isfinite(median_within95) and median_within95 >= SUPPORT_MIN_MEDIAN_HELDOUT_WITHIN_TRAIN_CENTRAL95}
axis_transport_support = bool(all(gates.values()))
summary_rows = [('protocol_id', 'AML_GRAPHRX_R6_CROSSFITTED_PROGRAM_AXIS_V1'), ('discovery_samples_n', 1733), ('outer_folds_total', 8), ('evaluable_LOCO_folds_n', evaluable_folds_n), ('OOF_scored_patients_n', int(len(oof))), ('OOF_scored_fraction', float(len(oof) / 1733)), ('reference_fold_for_sign_alignment', reference_fold), ('median_pairwise_fold_weight_spearman', median_weight_rho), ('median_positive_gene_Jaccard', median_pos_j), ('median_negative_gene_Jaccard', median_neg_j), ('median_heldout_fraction_within_training_central95', median_within95), ('median_absolute_heldout_shift_training_IQR_units', median_abs_shift_iqr), ('OOF_vs_all8_axis_score_spearman', oof_all8_rho), ('OOF_score_cohort_Kruskal_Wallis_H', float(kw.statistic) if kw is not None else np.nan), ('OOF_score_cohort_Kruskal_Wallis_p', float(kw.pvalue) if kw is not None else np.nan), ('OOF_score_cohort_epsilon_squared', eps2), ('all8_axis_pole_A_state', all8_model['state_a']), ('all8_axis_pole_B_state', all8_model['state_b']), ('all8_axis_robust_genes_n', all8_model['robust_n']), ('all8_axis_positive_genes_n_aligned', int(np.sum(all8_model['robust'] & (all8_weight_aligned > 0)))), ('all8_axis_negative_genes_n_aligned', int(np.sum(all8_model['robust'] & (all8_weight_aligned < 0)))), ('all8_weight_spearman_to_reference_before_sign_alignment', rho_all8_ref_raw), ('gate_evaluable_folds', gates['evaluable_folds_ge_6']), ('gate_weight_stability', gates['median_pairwise_weight_spearman_ge_0.50']), ('gate_OOF_all8_concordance', gates['OOF_vs_all8_spearman_ge_0.70']), ('gate_heldout_calibration', gates['median_heldout_within_training_central95_ge_0.75']), ('R6_PROGRAM_AXIS_TRANSPORT_SUPPORT', axis_transport_support), ('claim_boundary', 'continuous program-axis transportability only; not proof of discrete AML subtype validation, drug response, prognosis, or causality'), ('next_step_if_supported', 'lock all8 axis, then test on genuinely external RNA-seq and only then map to DepMap expression/dependency'), ('next_step_if_not_supported', 'stop axis refinement; retain negative result and do not tune on the same eight cohorts')]
final_summary = pd.DataFrame(summary_rows, columns=['metric', 'value'])
strict_write_csv(F581, final_summary)
manifest_rows = []
for role, p in [('R6_protocol', F571), ('fold_axis_definitions', F572), ('fold_gene_weights', F573), ('fold_score_transfer_summary', F574), ('fold_scores_long', F575), ('fold_weight_stability', F576), ('out_of_fold_program_scores', F577), ('all8_candidate_axis_genes', F578), ('all8_candidate_scores', F579), ('original_S03_S06_descriptive_alignment', F580), ('R6_final_summary', F581)]:
    manifest_rows.append({'role': role, 'path': str(p), 'sha256': sha256_file(p)})
manifest = pd.DataFrame(manifest_rows)
strict_write_csv(F582, manifest, ['role'])
try:
    import matplotlib.pyplot as plt
    if len(oof):
        order = [g for g in DISCOVERY_GSES if g in set(oof['gse'])]
        vals = [oof.loc[oof['gse'].eq(g), 'OOF_program_axis_score_aligned'].astype(float).to_numpy() for g in order]
        fig = plt.figure(figsize=(10, 5.5))
        ax = fig.add_subplot(111)
        ax.boxplot(vals, labels=order, showfliers=False)
        ax.set_ylabel('Cross-fitted program-axis score')
        ax.set_xlabel('Held-out GEO cohort')
        ax.set_title('R6 out-of-fold program-axis scores')
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        fig.savefig(FIGDIR / 'R6_OOF_program_axis_by_cohort_300dpi.png', dpi=300, bbox_inches='tight')
        fig.savefig(FIGDIR / 'R6_OOF_program_axis_by_cohort.pdf', bbox_inches='tight')
        plt.close(fig)
    if len(evaluable_fold_ids) >= 2:
        n = len(evaluable_fold_ids)
        M = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                r = float(spearmanr(aligned_weights[evaluable_fold_ids[i]], aligned_weights[evaluable_fold_ids[j]]).statistic)
                M[i, j] = M[j, i] = r
        fig = plt.figure(figsize=(7.5, 6.5))
        ax = fig.add_subplot(111)
        im = ax.imshow(M, vmin=-1, vmax=1)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels([x.split('_')[-1] for x in evaluable_fold_ids], rotation=45, ha='right')
        ax.set_yticklabels([x.split('_')[-1] for x in evaluable_fold_ids])
        ax.set_title('Sign-aligned fold gene-weight stability')
        fig.colorbar(im, ax=ax, label='Spearman correlation')
        fig.tight_layout()
        fig.savefig(FIGDIR / 'R6_fold_weight_stability_300dpi.png', dpi=300, bbox_inches='tight')
        fig.savefig(FIGDIR / 'R6_fold_weight_stability.pdf', bbox_inches='tight')
        plt.close(fig)
except Exception as e:
    print('WARNING: figure generation skipped:', repr(e))
print('\n' + '=' * 120)
print('R6A/R6B COMPLETE — REVIEW BEFORE ANY MANUSCRIPT CLAIM')
print('=' * 120)
print('\nFOLD AXIS DEFINITIONS')
show_cols = ['fold_id', 'heldout_gse', 'training_states_n', 'pole_A_state', 'pole_B_state', 'robust_axis_genes_n', 'axis_UP_genes_n', 'axis_DOWN_genes_n', 'status']
print(fold_defs_df[[c for c in show_cols if c in fold_defs_df.columns]].to_string(index=False))
print('\nTRANSFER SUMMARY')
show_cols2 = ['heldout_gse', 'robust_axis_genes_n', 'heldout_fraction_within_training_central95', 'heldout_median_shift_in_training_IQR', 'heldout_IQR_ratio', 'KS_FDR_BH_heldout_vs_training']
print(fold_transfer_df[[c for c in show_cols2 if c in fold_transfer_df.columns]].to_string(index=False))
print('\nFINAL SUMMARY')
print(final_summary.to_string(index=False))
print('\nIMPORTANT')
print('- No GNN training was performed.')
print('- No held-out GSE was used for state-pair selection or gene selection.')
print('- Original R5 State03/State06 labels were used only for descriptive alignment after R6 construction.')
print('- If R6 support gates fail, do NOT tune them on these same eight cohorts.')
print('- R6C (DepMap expression/dependency) requires a verified DepMap expression matrix; it is intentionally not fabricated here.')
print('- R6D requires a genuinely external RNA-seq cohort; it is intentionally not claimed here.')
