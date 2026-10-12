# Manuscript-linked analysis: Frozen three-cohort assessment, with the executed join repair
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
import json
import math
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm
from sklearn.metrics import roc_auc_score
SEED = 20261005
rng_global = np.random.default_rng(SEED)
PROJECT = Path(_release_os.environ.get('AML_GRAPHRX_ROOT', '/content/drive/MyDrive/AML_GraphRx'))
AUDIT = PROJECT / 'results' / 'audit'
STEP5 = PROJECT / 'data' / 'patient_graphs' / '20260823T123040Z'
RANK_FILE = STEP5 / '20260823T123040Z_patient_rank_raw_float64.npy'
PRESENCE_FILE = STEP5 / '20260823T123040Z_gene_presence_mask_bool.npy'
UNIT_FILE = STEP5 / '20260823T123040Z_analysis_unit_index.csv'
NODE_FILE = STEP5 / '20260823T123040Z_node_index.csv'
LABEL_FILE = AUDIT / '317_PHASE7A4_final_reconciled_analysis_unit_label_ledger.csv'
F492 = AUDIT / '492_R5G_recurrent_state_patient_assignments.csv'
F493 = AUDIT / '493_R5G_state_gene_meta_statistics.csv'
F495 = AUDIT / '495_R5G_final_summary.csv'
F496 = AUDIT / '496_R5G_hard_gates.csv'
F497 = AUDIT / '497_R5G_manifest.csv'
F498 = AUDIT / '498_R5V_predeclared_validation_protocol.json'
F499 = AUDIT / '499_R5V_validation_ledger.csv'
F500 = AUDIT / '500_R5V_frozen_state_score_weights.csv'
F501 = AUDIT / '501_R5V_sample_state_scores.csv'
F502 = AUDIT / '502_R5V_any_state_metrics.csv'
F503 = AUDIT / '503_R5V_state_specific_metrics.csv'
F504 = AUDIT / '504_R5V_AML_state_assignments.csv'
F505 = AUDIT / '505_R5V_state_recurrence.csv'
F506 = AUDIT / '506_R5V_per_cohort_AML_control_gene_statistics.csv'
F507 = AUDIT / '507_R5V_AML_control_gene_meta_statistics.csv'
F508 = AUDIT / '508_R5V_state_disease_linked_genes.csv'
F509 = AUDIT / '509_R5V_state_disease_linked_summary.csv'
F510 = AUDIT / '510_R5V_any_state_bootstrap.csv'
F511 = AUDIT / '511_R5V_final_summary.csv'
F512 = AUDIT / '512_R5V_hard_gates.csv'
F513 = AUDIT / '513_R5V_manifest.csv'
N_GENES = 9554
VALIDATION_GSES = ['GSE114868', 'GSE13159', 'GSE9476']
EXPECTED_COUNTS = {'GSE114868': {1: 194, 0: 20}, 'GSE13159': {1: 542, 0: 74}, 'GSE9476': {1: 26, 0: 38}}
EXPECTED_TOTAL_N = 894
EXPECTED_AML_N = 762
EXPECTED_CONTROL_N = 132
EXPECTED_STATES = ['AML_STATE_01', 'AML_STATE_02', 'AML_STATE_03', 'AML_STATE_04', 'AML_STATE_05', 'AML_STATE_06']
BOOTSTRAPS = 1000
MIN_MEAN_ANY_STATE_AUROC = 0.7
MIN_COHORT_AUROC = 0.65
MIN_COHORTS_ABOVE_AUC = 2
MIN_VALIDATION_RECURRENT_STATES = 4

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    path = Path(path)
    require(path.is_file(), f'Missing file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(chunk_size), b''):
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

def bool_from_cell(x):
    return str(x).strip().lower() in {'true', '1', 'yes'}
for path in [F492, F493, F495, F496, F497]:
    require(path.is_file(), f'Missing frozen R5-2G artifact: {path}')
parent_summary = pd.read_csv(F495)
parent_gates = pd.read_csv(F496)
parent_manifest = pd.read_csv(F497)
accepted_row = parent_summary.loc[parent_summary['metric'].eq('R5_2G_ACCEPTED')]
require(len(accepted_row) == 1, 'R5_2G_ACCEPTED row missing.')
require(bool_from_cell(accepted_row['value'].iloc[0]), 'R5-2G was not accepted.')
require(parent_gates['status'].eq('PASS').all(), 'Parent R5-2G contains failed hard gates.')
for path in [F492, F493, F495, F496]:
    rows = parent_manifest.loc[parent_manifest['path'].map(lambda x: Path(str(x)).name).eq(path.name)]
    require(len(rows) == 1, f'{path.name} missing from R5-2G manifest.')
    expected_sha = str(rows['sha256'].iloc[0])
    observed_sha = sha256_file(path)
    require(observed_sha == expected_sha, f'Frozen parent artifact SHA mismatch: {path.name}')
print('PASS: frozen R5-2G discovery chain verified.')
protocol = {'protocol_id': 'AML_GRAPHRX_R5_3_HELDOUT_STATE_VALIDATION_V1', 'parent_discovery': 'AML_GRAPHRX_R5_2G_GENE_FIRST_STATE_DISCOVERY_V1', 'parent_manifest_sha256': sha256_file(F497), 'validation_cohorts': VALIDATION_GSES, 'expected_validation_n': EXPECTED_TOTAL_N, 'expected_AML_n': EXPECTED_AML_N, 'expected_CONTROL_n': EXPECTED_CONTROL_N, 'frozen_states': EXPECTED_STATES, 'state_score': 'cosine similarity between centered patient percentile ranks and L2-normalized frozen state meta-Z weights restricted to R5-2G signature genes', 'patient_rank_center': 0.5, 'primary_score': 'maximum frozen-state similarity', 'secondary_score': 'top1_minus_top2_state_similarity_margin', 'classifier_training': False, 'GNN_training': False, 'reclustering': False, 'state_redefinition': False, 'threshold_tuning': False, 'validation_state_recurrence_rule': 'state receives at least max(3,ceil(0.05*nAML)) AML assignments in at least 2 of 3 validation cohorts', 'AML_control_gene_analysis': 'per-cohort Mann-Whitney U + rank-biserial effect; directional weighted Stouffer meta-analysis; BH across all 9554 genes', 'robust_AML_gene_rule': 'meta FDR<0.05, direction consistent 3/3, and leave-one-cohort-out direction stable', 'disease_linked_state_gene': 'frozen R5-2G state signature gene AND robust AML-vs-control meta gene', 'concordant_disease_linked_state_gene': 'disease-linked state gene with same sign for state-vs-other-AML and AML-vs-control', 'bootstrap_replicates': BOOTSTRAPS, 'validation_support_rule': {'cohort_equal_mean_any_state_AUROC_min': MIN_MEAN_ANY_STATE_AUROC, 'cohorts_with_AUROC_ge_0.65_min': MIN_COHORTS_ABOVE_AUC, 'validation_recurrent_states_min': MIN_VALIDATION_RECURRENT_STATES}, 'important_claim_boundary': 'cohorts are held out from R5 state discovery; they are not claimed to be globally untouched datasets because they were used in earlier AML-GraphRx analyses'}
write_once_or_verify(F498, json_bytes(protocol))
rank_mm = np.load(RANK_FILE, mmap_mode='r', allow_pickle=False)
presence_mm = np.load(PRESENCE_FILE, mmap_mode='r', allow_pickle=False)
units = pd.read_csv(UNIT_FILE, dtype=str, keep_default_na=False)
labels = pd.read_csv(LABEL_FILE, dtype=str, keep_default_na=False)
nodes = pd.read_csv(NODE_FILE, dtype=str, keep_default_na=False)
require(rank_mm.shape == (4554, 9554), 'Rank matrix shape changed.')
require(presence_mm.shape == (4554, 9554), 'Presence matrix shape changed.')
require(len(nodes) == N_GENES, 'Node universe changed.')
resolved = labels.loc[labels['label_status'].eq('RESOLVED')].copy()
bridge = resolved.merge(units[['analysis_unit_id', 'analysis_unit_index_0based', 'gse']], on='analysis_unit_id', how='left', validate='one_to_one', suffixes=('_label', '_step5'))
require(bridge['analysis_unit_index_0based'].notna().all(), 'Resolved sample missing from Step5.')
bridge['analysis_unit_index_0based'] = pd.to_numeric(bridge['analysis_unit_index_0based'], errors='raise').astype(int)
if 'gse_label' in bridge.columns:
    bridge['gse_final'] = bridge['gse_label']
elif 'gse' in bridge.columns:
    bridge['gse_final'] = bridge['gse']
else:
    bridge['gse_final'] = bridge['gse_step5']
if 'binary_label_int' in bridge.columns:
    numeric_label = pd.to_numeric(bridge['binary_label_int'], errors='coerce')
else:
    numeric_label = pd.Series(np.nan, index=bridge.index)
if numeric_label.notna().sum() == 0:
    require('binary_label' in bridge.columns, 'No usable binary label column.')
    numeric_label = bridge['binary_label'].map({'AML': 1, 'CONTROL': 0})
bridge['binary_label_int_final'] = numeric_label
validation = bridge.loc[bridge['gse_final'].isin(VALIDATION_GSES) & bridge['binary_label_int_final'].isin([0, 1])].copy()
validation['binary_label_int'] = validation['binary_label_int_final'].astype(int)
validation['gse'] = validation['gse_final']
validation = validation[['analysis_unit_id', 'analysis_unit_index_0based', 'gse', 'binary_label_int']].sort_values('analysis_unit_index_0based', kind='mergesort').reset_index(drop=True)
require(len(validation) == EXPECTED_TOTAL_N, f'Expected validation n={EXPECTED_TOTAL_N}; observed={len(validation)}')
require(int(validation['binary_label_int'].eq(1).sum()) == EXPECTED_AML_N, 'AML validation count mismatch.')
require(int(validation['binary_label_int'].eq(0).sum()) == EXPECTED_CONTROL_N, 'CONTROL validation count mismatch.')
for gse in VALIDATION_GSES:
    for label, expected in EXPECTED_COUNTS[gse].items():
        observed = int((validation['gse'].eq(gse) & validation['binary_label_int'].eq(label)).sum())
        require(observed == expected, f'{gse}, label={label}: expected={expected}, observed={observed}')
write_once_or_verify(F499, csv_bytes(validation, ['analysis_unit_index_0based']))
print()
print('PASS validation n =', len(validation))
print('AML =', int(validation['binary_label_int'].sum()))
print('CONTROL =', int(validation['binary_label_int'].eq(0).sum()))
state_meta = pd.read_csv(F493)
require(len(state_meta) > 0, 'Frozen R5-2G state gene table is empty.')
if state_meta['STATE_SIGNATURE_GENE'].dtype != bool:
    state_meta['STATE_SIGNATURE_GENE'] = state_meta['STATE_SIGNATURE_GENE'].astype(str).str.lower().eq('true')
observed_states = sorted(state_meta.loc[state_meta['STATE_SIGNATURE_GENE'].eq(True), 'state_id'].unique())
require(observed_states == EXPECTED_STATES, f'Frozen state identity changed.\nObserved={observed_states}')
state_weights = {}
weight_rows = []
for state_id in EXPECTED_STATES:
    x = state_meta.loc[state_meta['state_id'].eq(state_id) & state_meta['STATE_SIGNATURE_GENE'].eq(True)].copy()
    require(len(x) >= 10, f'{state_id}: too few frozen signature genes.')
    gene_index = x['gene_index_0based'].to_numpy(dtype=int)
    raw_weight = x['meta_Z'].to_numpy(dtype=float)
    require(np.isfinite(raw_weight).all(), f'{state_id}: non-finite meta-Z.')
    weight_norm = float(np.linalg.norm(raw_weight))
    require(weight_norm > 0, f'{state_id}: zero frozen signature vector.')
    normalized_weight = raw_weight / weight_norm
    state_weights[state_id] = (gene_index, normalized_weight)
    for gene_i, w in zip(gene_index, normalized_weight):
        weight_rows.append({'state_id': state_id, 'gene_index_0based': int(gene_i), 'canonical_NCBI_Gene_ID': nodes.iloc[gene_i]['canonical_NCBI_Gene_ID'], 'normalized_metaZ_weight': float(w)})
frozen_weight_table = pd.DataFrame(weight_rows)
write_once_or_verify(F500, csv_bytes(frozen_weight_table, ['state_id', 'gene_index_0based']))
validation_indices = validation['analysis_unit_index_0based'].to_numpy(dtype=int)
X_validation = np.asarray(rank_mm[validation_indices, :], dtype=np.float32)
P_validation = np.asarray(presence_mm[validation_indices, :], dtype=np.bool_)
require(X_validation.shape == (894, 9554), 'Validation rank matrix shape mismatch.')
score_matrix = np.zeros((len(validation), len(EXPECTED_STATES)), dtype=np.float64)
long_score_rows = []
for state_number, state_id in enumerate(EXPECTED_STATES):
    gene_idx, weight = state_weights[state_id]
    Xs = X_validation[:, gene_idx].astype(np.float64)
    Ps = P_validation[:, gene_idx]
    centered = np.where(Ps, Xs - 0.5, 0.0)
    sample_specific_weight = Ps.astype(np.float64) * weight[None, :]
    numerator = np.sum(centered * weight[None, :], axis=1)
    expression_norm = np.linalg.norm(centered, axis=1)
    weight_norm = np.linalg.norm(sample_specific_weight, axis=1)
    denominator = expression_norm * weight_norm
    score = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    require(np.isfinite(score).all(), f'{state_id}: non-finite validation state score.')
    score_matrix[:, state_number] = score
    for sample_i in range(len(validation)):
        long_score_rows.append({'analysis_unit_id': validation.iloc[sample_i]['analysis_unit_id'], 'analysis_unit_index_0based': int(validation.iloc[sample_i]['analysis_unit_index_0based']), 'gse': validation.iloc[sample_i]['gse'], 'binary_label_int': int(validation.iloc[sample_i]['binary_label_int']), 'state_id': state_id, 'state_similarity': float(score[sample_i])})
sample_state_scores = pd.DataFrame(long_score_rows)
write_once_or_verify(F501, csv_bytes(sample_state_scores, ['analysis_unit_index_0based', 'state_id']))
sorted_scores = np.sort(score_matrix, axis=1)
top1 = sorted_scores[:, -1]
top2 = sorted_scores[:, -2]
state_margin = top1 - top2
top_state_index = np.argmax(score_matrix, axis=1)
assigned_state = np.asarray([EXPECTED_STATES[i] for i in top_state_index])
patient_validation = validation.copy()
patient_validation['max_state_similarity'] = top1
patient_validation['state_margin'] = state_margin
patient_validation['assigned_state'] = assigned_state
any_metric_rows = []
for gse in VALIDATION_GSES:
    mask = patient_validation['gse'].eq(gse).to_numpy()
    y = patient_validation.loc[mask, 'binary_label_int'].to_numpy(dtype=int)
    max_score = patient_validation.loc[mask, 'max_state_similarity'].to_numpy(dtype=float)
    margin = patient_validation.loc[mask, 'state_margin'].to_numpy(dtype=float)
    auc_max = float(roc_auc_score(y, max_score))
    auc_margin = float(roc_auc_score(y, margin))
    aml_max = max_score[y == 1]
    control_max = max_score[y == 0]
    mw = mannwhitneyu(aml_max, control_max, alternative='greater', method='asymptotic')
    any_metric_rows.append({'gse': gse, 'AML_n': int(np.sum(y == 1)), 'CONTROL_n': int(np.sum(y == 0)), 'max_state_similarity_AUROC': auc_max, 'state_margin_AUROC': auc_margin, 'AML_median_max_state_similarity': float(np.median(aml_max)), 'CONTROL_median_max_state_similarity': float(np.median(control_max)), 'one_sided_MannWhitney_p': float(mw.pvalue)})
any_metrics = pd.DataFrame(any_metric_rows)
any_metrics['FDR_BH'] = bh_fdr(any_metrics['one_sided_MannWhitney_p'].to_numpy(dtype=float))
write_once_or_verify(F502, csv_bytes(any_metrics, ['gse']))
state_metric_rows = []
for state_number, state_id in enumerate(EXPECTED_STATES):
    for gse in VALIDATION_GSES:
        mask = validation['gse'].eq(gse).to_numpy()
        y = validation.loc[mask, 'binary_label_int'].to_numpy(dtype=int)
        s = score_matrix[mask, state_number]
        auc = float(roc_auc_score(y, s))
        aml_score = s[y == 1]
        control_score = s[y == 0]
        test = mannwhitneyu(aml_score, control_score, alternative='greater', method='asymptotic')
        state_metric_rows.append({'state_id': state_id, 'gse': gse, 'AML_n': len(aml_score), 'CONTROL_n': len(control_score), 'AUROC': auc, 'AML_median_score': float(np.median(aml_score)), 'CONTROL_median_score': float(np.median(control_score)), 'one_sided_MannWhitney_p': float(test.pvalue)})
state_metrics = pd.DataFrame(state_metric_rows)
state_metrics['FDR_BH_across_18_tests'] = bh_fdr(state_metrics['one_sided_MannWhitney_p'].to_numpy(dtype=float))
write_once_or_verify(F503, csv_bytes(state_metrics, ['state_id', 'gse']))
AML_assignments = patient_validation.loc[patient_validation['binary_label_int'].eq(1)].copy()
write_once_or_verify(F504, csv_bytes(AML_assignments, ['gse', 'analysis_unit_index_0based']))
recurrence_rows = []
for state_id in EXPECTED_STATES:
    row = {'state_id': state_id}
    qualifying_cohorts = 0
    for gse in VALIDATION_GSES:
        cohort_aml = AML_assignments.loc[AML_assignments['gse'].eq(gse)]
        n_aml = len(cohort_aml)
        minimum_assignment = max(3, int(math.ceil(0.05 * n_aml)))
        assigned_n = int(cohort_aml['assigned_state'].eq(state_id).sum())
        qualifies = bool(assigned_n >= minimum_assignment)
        if qualifies:
            qualifying_cohorts += 1
        row[f'{gse}_AML_n'] = n_aml
        row[f'{gse}_minimum_required'] = minimum_assignment
        row[f'{gse}_assigned_n'] = assigned_n
        row[f'{gse}_qualifies'] = qualifies
    row['qualifying_validation_cohorts_n'] = qualifying_cohorts
    row['VALIDATION_RECURRENT'] = bool(qualifying_cohorts >= 2)
    recurrence_rows.append(row)
state_recurrence = pd.DataFrame(recurrence_rows)
write_once_or_verify(F505, csv_bytes(state_recurrence, ['state_id']))
per_cohort_gene_rows = []
cohort_effects = []
cohort_zs = []
cohort_weights = []
for gse in VALIDATION_GSES:
    mask = validation['gse'].eq(gse).to_numpy()
    y = validation.loc[mask, 'binary_label_int'].to_numpy(dtype=int)
    Xg = X_validation[mask].astype(np.float64)
    A = Xg[y == 1]
    C = Xg[y == 0]
    n_aml = A.shape[0]
    n_control = C.shape[0]
    print('AML-control gene statistics:', gse, n_aml, n_control)
    test = mannwhitneyu(A, C, axis=0, alternative='two-sided', method='asymptotic')
    U = np.asarray(test.statistic, dtype=np.float64)
    p = np.asarray(test.pvalue, dtype=np.float64)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    p = np.clip(p, 1e-300, 1.0)
    effect = 2.0 * U / (n_aml * n_control) - 1.0
    signed_z = np.sign(effect) * norm.isf(p / 2.0)
    weight = math.sqrt(n_aml * n_control / (n_aml + n_control))
    cohort_effects.append(effect)
    cohort_zs.append(signed_z)
    cohort_weights.append(weight)
    for gene_i in range(N_GENES):
        per_cohort_gene_rows.append({'gse': gse, 'gene_index_0based': gene_i, 'canonical_NCBI_Gene_ID': nodes.iloc[gene_i]['canonical_NCBI_Gene_ID'], 'AML_n': n_aml, 'CONTROL_n': n_control, 'rank_biserial_effect': float(effect[gene_i]), 'MannWhitney_p': float(p[gene_i]), 'signed_Z': float(signed_z[gene_i])})
per_cohort_gene_stats = pd.DataFrame(per_cohort_gene_rows)
write_once_or_verify(F506, csv_bytes(per_cohort_gene_stats, ['gse', 'gene_index_0based']))
E = np.stack(cohort_effects, axis=0)
Z = np.stack(cohort_zs, axis=0)
W = np.asarray(cohort_weights, dtype=np.float64)
meta_Z = (Z * W[:, None]).sum(axis=0) / math.sqrt(float(np.sum(W ** 2)))
meta_p = 2.0 * norm.sf(np.abs(meta_Z))
meta_FDR = bh_fdr(meta_p)
weighted_effect = (E * W[:, None]).sum(axis=0) / W.sum()
meta_sign = np.sign(meta_Z)
direction_consistency = np.mean(np.sign(E) == meta_sign[None, :], axis=0)
loo_stable = np.ones(N_GENES, dtype=bool)
for heldout in range(3):
    keep = [i for i in range(3) if i != heldout]
    Z_loo = Z[keep]
    W_loo = W[keep]
    meta_Z_loo = (Z_loo * W_loo[:, None]).sum(axis=0) / math.sqrt(float(np.sum(W_loo ** 2)))
    loo_stable &= np.sign(meta_Z_loo) == meta_sign
ROBUST_AML_GENE = (meta_FDR < 0.05) & (direction_consistency == 1.0) & loo_stable & (meta_sign != 0)
aml_gene_meta = pd.DataFrame({'gene_index_0based': np.arange(N_GENES), 'canonical_NCBI_Gene_ID': nodes['canonical_NCBI_Gene_ID'].to_numpy(), 'AML_control_meta_effect': weighted_effect, 'AML_control_meta_Z': meta_Z, 'AML_control_meta_p': meta_p, 'AML_control_meta_FDR_BH': meta_FDR, 'direction_consistency_fraction': direction_consistency, 'LOO_direction_stable': loo_stable, 'AML_direction': np.where(meta_Z > 0, 'AML_UP', 'AML_DOWN'), 'ROBUST_AML_GENE': ROBUST_AML_GENE})
write_once_or_verify(F507, csv_bytes(aml_gene_meta, ['gene_index_0based']))
ROBUST_AML_GENE_N = int(ROBUST_AML_GENE.sum())
print()
print('Robust AML-vs-control genes:', ROBUST_AML_GENE_N)
pass
pass
import os
import json
import math
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
PROJECT = Path(_release_os.environ.get('AML_GRAPHRX_ROOT', '/content/drive/MyDrive/AML_GraphRx'))
AUDIT = PROJECT / 'results' / 'audit'
F498 = AUDIT / '498_R5V_predeclared_validation_protocol.json'
F499 = AUDIT / '499_R5V_validation_ledger.csv'
F500 = AUDIT / '500_R5V_frozen_state_score_weights.csv'
F501 = AUDIT / '501_R5V_sample_state_scores.csv'
F502 = AUDIT / '502_R5V_any_state_metrics.csv'
F503 = AUDIT / '503_R5V_state_specific_metrics.csv'
F504 = AUDIT / '504_R5V_AML_state_assignments.csv'
F505 = AUDIT / '505_R5V_state_recurrence.csv'
F506 = AUDIT / '506_R5V_per_cohort_AML_control_gene_statistics.csv'
F507 = AUDIT / '507_R5V_AML_control_gene_meta_statistics.csv'
F492 = AUDIT / '492_R5G_recurrent_state_patient_assignments.csv'
F493 = AUDIT / '493_R5G_state_gene_meta_statistics.csv'
F495 = AUDIT / '495_R5G_final_summary.csv'
F496 = AUDIT / '496_R5G_hard_gates.csv'
F497 = AUDIT / '497_R5G_manifest.csv'
F508 = AUDIT / '508_R5V_state_disease_linked_genes.csv'
F509 = AUDIT / '509_R5V_state_disease_linked_summary.csv'
F510 = AUDIT / '510_R5V_any_state_bootstrap.csv'
F511 = AUDIT / '511_R5V_final_summary.csv'
F512 = AUDIT / '512_R5V_hard_gates.csv'
F513 = AUDIT / '513_R5V_manifest.csv'
SEED = 20261005
BOOTSTRAPS = 1000
VALIDATION_GSES = ['GSE114868', 'GSE13159', 'GSE9476']
EXPECTED_STATES = ['AML_STATE_01', 'AML_STATE_02', 'AML_STATE_03', 'AML_STATE_04', 'AML_STATE_05', 'AML_STATE_06']
MIN_MEAN_ANY_STATE_AUROC = 0.7
MIN_COHORT_AUROC = 0.65
MIN_COHORTS_ABOVE_AUC = 2
MIN_VALIDATION_RECURRENT_STATES = 4

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    path = Path(path)
    require(path.is_file(), f'Missing file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(chunk_size), b''):
            h.update(block)
    return h.hexdigest()

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
        require(path.read_bytes() == data, f'Existing artifact differs: {path}')
        print('PASS existing:', path.name)
    else:
        atomic_write(path, data)
        print('CREATED:', path.name)

def to_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])

def clean_gene_id(series):
    x = series.astype(str).str.strip()
    x = x.str.replace('\\.0$', '', regex=True)
    return x
for path in [F498, F499, F500, F501, F502, F503, F504, F505, F506, F507, F493, F495, F496, F497]:
    require(path.is_file(), f'Required artifact missing: {path}')
print('PASS: R5-3 artifacts through 507 exist.')
protocol = json.loads(F498.read_text())
require(protocol['protocol_id'] == 'AML_GRAPHRX_R5_3_HELDOUT_STATE_VALIDATION_V1', 'Unexpected R5-3 protocol.')
state_meta = pd.read_csv(F493, dtype={'canonical_NCBI_Gene_ID': str})
require(len(state_meta) > 0, 'Frozen R5-2G state table is empty.')
state_meta['gene_index_0based'] = pd.to_numeric(state_meta['gene_index_0based'], errors='raise').astype(int)
state_meta['STATE_SIGNATURE_GENE'] = to_bool(state_meta['STATE_SIGNATURE_GENE'])
state_signature_subset = state_meta.loc[state_meta['STATE_SIGNATURE_GENE']].copy()
observed_states = sorted(state_signature_subset['state_id'].unique())
require(observed_states == EXPECTED_STATES, f'Frozen state identity changed: {observed_states}')
print('PASS frozen states:', observed_states)
aml_gene_meta = pd.read_csv(F507, dtype={'canonical_NCBI_Gene_ID': str})
require(len(aml_gene_meta) == 9554, f'Expected 9554 AML/control meta genes; observed={len(aml_gene_meta)}')
aml_gene_meta['gene_index_0based'] = pd.to_numeric(aml_gene_meta['gene_index_0based'], errors='raise').astype(int)
aml_gene_meta['ROBUST_AML_GENE'] = to_bool(aml_gene_meta['ROBUST_AML_GENE'])
ROBUST_AML_GENE_N = int(aml_gene_meta['ROBUST_AML_GENE'].sum())
require(ROBUST_AML_GENE_N == 3901, f'Unexpected robust AML gene count. Expected 3901; observed {ROBUST_AML_GENE_N}')
print('PASS robust AML-vs-control genes =', ROBUST_AML_GENE_N)
state_for_merge = state_signature_subset.rename(columns={'canonical_NCBI_Gene_ID': 'canonical_NCBI_Gene_ID_state'})
aml_for_merge = aml_gene_meta.rename(columns={'canonical_NCBI_Gene_ID': 'canonical_NCBI_Gene_ID_AML'})
state_disease = state_for_merge.merge(aml_for_merge, on='gene_index_0based', how='left', validate='many_to_one')
require(state_disease['canonical_NCBI_Gene_ID_AML'].notna().all(), 'Some state genes failed gene-index merge.')
state_id_clean = clean_gene_id(state_disease['canonical_NCBI_Gene_ID_state'])
aml_id_clean = clean_gene_id(state_disease['canonical_NCBI_Gene_ID_AML'])
id_match = state_id_clean == aml_id_clean
require(id_match.all(), 'GeneID mismatch after safe gene-index merge.\n' + state_disease.loc[~id_match, ['gene_index_0based', 'canonical_NCBI_Gene_ID_state', 'canonical_NCBI_Gene_ID_AML']].head(20).to_string(index=False))
state_disease['canonical_NCBI_Gene_ID'] = state_id_clean
state_disease = state_disease.drop(columns=['canonical_NCBI_Gene_ID_state', 'canonical_NCBI_Gene_ID_AML'])
print('PASS: gene-index merge and GeneID integrity check.')
state_disease['DISEASE_LINKED_STATE_GENE'] = state_disease['ROBUST_AML_GENE'].astype(bool)
state_disease['state_sign'] = np.sign(pd.to_numeric(state_disease['meta_Z'], errors='raise').to_numpy(dtype=float))
state_disease['AML_sign'] = np.sign(pd.to_numeric(state_disease['AML_control_meta_Z'], errors='raise').to_numpy(dtype=float))
state_disease['CONCORDANT_DISEASE_LINKED_STATE_GENE'] = state_disease['DISEASE_LINKED_STATE_GENE'] & (state_disease['state_sign'] == state_disease['AML_sign'])
write_once_or_verify(F508, csv_bytes(state_disease, ['state_id', 'gene_index_0based']))
disease_summary_rows = []
for state_id in EXPECTED_STATES:
    x = state_disease.loc[state_disease['state_id'].eq(state_id)]
    signature_n = len(x)
    disease_linked_n = int(x['DISEASE_LINKED_STATE_GENE'].sum())
    concordant_n = int(x['CONCORDANT_DISEASE_LINKED_STATE_GENE'].sum())
    disease_summary_rows.append({'state_id': state_id, 'frozen_state_signature_genes_n': signature_n, 'robust_AML_disease_linked_genes_n': disease_linked_n, 'concordant_disease_linked_genes_n': concordant_n, 'disease_linked_fraction': disease_linked_n / signature_n, 'concordant_fraction_of_signature': concordant_n / signature_n})
disease_summary = pd.DataFrame(disease_summary_rows)
write_once_or_verify(F509, csv_bytes(disease_summary, ['state_id']))
validation = pd.read_csv(F499)
scores = pd.read_csv(F501)
any_metrics = pd.read_csv(F502)
state_recurrence = pd.read_csv(F505)
require(len(validation) == 894, 'Validation ledger changed.')
require(len(scores) == 894 * 6, f'Expected 5364 state-score rows; observed={len(scores)}')
wide = scores.pivot_table(index=['analysis_unit_id', 'analysis_unit_index_0based', 'gse', 'binary_label_int'], columns='state_id', values='state_similarity').reset_index()
for state_id in EXPECTED_STATES:
    require(state_id in wide.columns, f'Missing state score: {state_id}')
state_matrix = wide[EXPECTED_STATES].to_numpy(dtype=float)
require(np.isfinite(state_matrix).all(), 'Non-finite state scores.')
wide['max_state_similarity'] = np.max(state_matrix, axis=1)
bootstrap_storage = {}
for gse_number, gse in enumerate(VALIDATION_GSES, start=1):
    x = wide.loc[wide['gse'].eq(gse)].reset_index(drop=True)
    y = x['binary_label_int'].to_numpy(dtype=int)
    score = x['max_state_similarity'].to_numpy(dtype=float)
    i0 = np.where(y == 0)[0]
    i1 = np.where(y == 1)[0]
    require(len(i0) > 0 and len(i1) > 0, f'{gse}: missing validation class.')
    rng = np.random.default_rng(SEED + 1000 * gse_number)
    aucs = np.zeros(BOOTSTRAPS, dtype=float)
    for b in range(BOOTSTRAPS):
        s0 = rng.choice(i0, size=len(i0), replace=True)
        s1 = rng.choice(i1, size=len(i1), replace=True)
        idx = np.concatenate([s0, s1])
        aucs[b] = roc_auc_score(y[idx], score[idx])
    bootstrap_storage[gse] = aucs
bootstrap_rows = []
for gse in VALIDATION_GSES:
    observed = float(any_metrics.loc[any_metrics['gse'].eq(gse), 'max_state_similarity_AUROC'].iloc[0])
    aucs = bootstrap_storage[gse]
    bootstrap_rows.append({'scope': gse, 'observed_AUROC': observed, 'bootstrap_n': BOOTSTRAPS, 'CI95_lower': float(np.quantile(aucs, 0.025)), 'CI95_upper': float(np.quantile(aucs, 0.975))})
cohort_equal_boot = np.mean(np.stack([bootstrap_storage[gse] for gse in VALIDATION_GSES], axis=0), axis=0)
OBSERVED_MEAN_AUC = float(any_metrics['max_state_similarity_AUROC'].mean())
bootstrap_rows.append({'scope': 'COHORT_EQUAL_MEAN', 'observed_AUROC': OBSERVED_MEAN_AUC, 'bootstrap_n': BOOTSTRAPS, 'CI95_lower': float(np.quantile(cohort_equal_boot, 0.025)), 'CI95_upper': float(np.quantile(cohort_equal_boot, 0.975))})
bootstrap_table = pd.DataFrame(bootstrap_rows)
write_once_or_verify(F510, csv_bytes(bootstrap_table, ['scope']))
COHORTS_AUC_GE_065 = int((any_metrics['max_state_similarity_AUROC'] >= MIN_COHORT_AUROC).sum())
state_recurrence['VALIDATION_RECURRENT'] = to_bool(state_recurrence['VALIDATION_RECURRENT'])
VALIDATION_RECURRENT_STATES_N = int(state_recurrence['VALIDATION_RECURRENT'].sum())
STATE_ATLAS_VALIDATION_SUPPORTED = bool(OBSERVED_MEAN_AUC >= MIN_MEAN_ANY_STATE_AUROC and COHORTS_AUC_GE_065 >= MIN_COHORTS_ABOVE_AUC and (VALIDATION_RECURRENT_STATES_N >= MIN_VALIDATION_RECURRENT_STATES))
summary = pd.DataFrame([('protocol_id', protocol['protocol_id']), ('validation_cohorts_n', 3), ('validation_samples_n', 894), ('AML_n', 762), ('CONTROL_n', 132), ('frozen_states_n', 6), ('GNN_training', False), ('classifier_training', False), ('reclustering', False), ('threshold_tuning', False), ('cohort_equal_mean_any_state_AUROC', OBSERVED_MEAN_AUC), ('cohorts_any_state_AUROC_ge_0_65_n', COHORTS_AUC_GE_065), ('validation_recurrent_states_n', VALIDATION_RECURRENT_STATES_N), ('robust_AML_control_meta_genes_n', ROBUST_AML_GENE_N), ('STATE_ATLAS_VALIDATION_SUPPORTED', STATE_ATLAS_VALIDATION_SUPPORTED), ('next_stage', 'R5_4_PATHWAY_AND_DEPMAP' if STATE_ATLAS_VALIDATION_SUPPORTED else 'INTERPRET_VALIDATION_WITHOUT_RETUNING_DISCOVERY')], columns=['metric', 'value'])
write_once_or_verify(F511, csv_bytes(summary))
gate_rows = []

def add_gate(gate, passed, detail):
    gate_rows.append({'gate': gate, 'status': 'PASS' if bool(passed) else 'FAIL', 'detail': str(detail)})
add_gate('resume_started_after_507', True, True)
add_gate('safe_merge_key_gene_index_0based', True, True)
add_gate('canonical_GeneID_integrity_after_merge', id_match.all(), int(id_match.sum()))
add_gate('exact_3901_robust_AML_genes', ROBUST_AML_GENE_N == 3901, ROBUST_AML_GENE_N)
add_gate('exact_six_frozen_states', observed_states == EXPECTED_STATES, observed_states)
add_gate('exact_894_validation_samples', len(validation) == 894, len(validation))
add_gate('no_GNN_training', True, False)
add_gate('no_classifier_training', True, False)
add_gate('no_reclustering', True, False)
add_gate('no_threshold_tuning', True, False)
hard_gates = pd.DataFrame(gate_rows)
require(hard_gates['status'].eq('PASS').all(), 'R5-3 HOTFIX HARD GATE FAILURE:\n' + hard_gates.loc[hard_gates['status'].ne('PASS')].to_string(index=False))
write_once_or_verify(F512, csv_bytes(hard_gates, ['gate']))
manifest_rows = []

def add_manifest(role, path):
    path = Path(path)
    require(path.is_file(), f'Missing manifest file: {path}')
    manifest_rows.append({'role': role, 'path': str(path), 'size_bytes': int(path.stat().st_size), 'sha256': sha256_file(path), 'protocol_id': protocol['protocol_id']})
for role, path in [('parent_R5G_state_assignments', F492), ('parent_R5G_state_gene_meta', F493), ('parent_R5G_summary', F495), ('parent_R5G_hard_gates', F496), ('parent_R5G_manifest', F497), ('protocol', F498), ('validation_ledger', F499), ('state_weights', F500), ('sample_state_scores', F501), ('any_state_metrics', F502), ('state_specific_metrics', F503), ('AML_state_assignments', F504), ('state_recurrence', F505), ('per_cohort_gene_statistics', F506), ('AML_control_gene_meta', F507), ('state_disease_linked_genes', F508), ('state_disease_linked_summary', F509), ('bootstrap', F510), ('final_summary', F511), ('hard_gates', F512)]:
    add_manifest(role, path)
manifest = pd.DataFrame(manifest_rows)
write_once_or_verify(F513, csv_bytes(manifest, ['role']))
print()
print('=' * 105)
print('R5-3 HOTFIX — FINAL SUMMARY')
print('=' * 105)
print(summary.to_string(index=False))
print()
print('=' * 105)
print('R5-3 — ANY-STATE VALIDATION')
print('=' * 105)
print(any_metrics.to_string(index=False))
print()
print('=' * 105)
print('R5-3 — STATE RECURRENCE')
print('=' * 105)
print(state_recurrence.to_string(index=False))
print()
print('=' * 105)
print('R5-3 — DISEASE-LINKED STATE GENES')
print('=' * 105)
print(disease_summary.to_string(index=False))
print()
print('=' * 105)
print('R5-3 — BOOTSTRAP')
print('=' * 105)
print(bootstrap_table.to_string(index=False))
print()
print('=' * 105)
print('R5-3 — HARD GATES')
print('=' * 105)
print(hard_gates.to_string(index=False))
print()
if STATE_ATLAS_VALIDATION_SUPPORTED:
    print('R5-3 VALIDATION SUPPORTED ✅')
    print('FROZEN AML STATE ATLAS SUPPORTED ✅')
    print('NEXT = R5-4 PATHWAY + DEPMAP')
else:
    print('R5-3 COMPLETE')
    print('PREDECLARED VALIDATION SUPPORT RULE NOT SATISFIED.')
    print('DO NOT RETUNE R5-2G.')
