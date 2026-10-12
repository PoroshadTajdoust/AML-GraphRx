# Manuscript-linked analysis: Original/PCA/Gaussian/gene-row-shuffled bases
# Source lineage and packaging changes: provenance/source_lineage.json
"""AML-GraphRx R10A — retrospective gene-first representation ablation.

Runs original R5G patient-projection / cross-cohort MNN / weighted-Leiden /
30*80%-edge-perturbation recurrence unchanged under multiple 32D gene bases.

This compares discovery representations, not clinically validated state transfer.
Do not use it to assert GNN superiority in independent cohorts.
"""
from __future__ import annotations
import os as _release_os
try:
    from IPython.display import display
except ImportError:

    def display(value):
        print(value)
from pathlib import Path
import hashlib
import json
import os
import platform
from io import BytesIO
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
try:
    import igraph as ig
    import leidenalg
except ImportError as exc:
    raise RuntimeError('Install actual Leiden before this run: %pip -q install python-igraph leidenalg. A Louvain fallback is not an equivalent R5G ablation.') from exc
try:
    pass
    pass
except ImportError:
    pass
ROOT = Path(_release_os.environ.get('AML_GRAPHRX_ROOT', '/content/drive/MyDrive/AML_GraphRx'))
AUDIT = ROOT / 'results' / 'audit'
STEP5 = ROOT / 'data' / 'patient_graphs' / '20260823T123040Z'
AUDIT.mkdir(parents=True, exist_ok=True)
RANK = STEP5 / '20260823T123040Z_patient_rank_raw_float64.npy'
DISCOVERY = AUDIT / '452_R5DICE_discovery_ledger.csv'
F483 = AUDIT / '483_R5G_gene_embeddings.npy'
F486 = AUDIT / '486_R5G_patient_projection.npy'
F488 = AUDIT / '488_R5G_cross_cohort_MNN_edges.csv'
F489 = AUDIT / '489_R5G_Leiden_raw_assignments.csv'
F492 = AUDIT / '492_R5G_recurrent_state_patient_assignments.csv'
F497 = AUDIT / '497_R5G_manifest.csv'
F481 = AUDIT / '481_R5G_predeclared_protocol.json'
F490 = AUDIT / '490_R5G_bootstrap_state_stability.csv'
OUTPUT = {'protocol': AUDIT / '750_R10A_preanalysis_representation_ablation_protocol.json', 'integrity': AUDIT / '751_R10A_input_integrity.csv', 'fidelity': AUDIT / '752_R10A_original_R5G_fidelity.csv', 'metrics': AUDIT / '753_R10A_arm_state_discovery_metrics.csv', 'comparisons': AUDIT / '754_R10A_descriptive_arm_comparisons.csv', 'summary': AUDIT / '755_R10A_final_summary.csv', 'manifest': AUDIT / '756_R10A_manifest.csv', 'figure_png': AUDIT / '757_R10A_ablation_comparison.png', 'figure_pdf': AUDIT / '757_R10A_ablation_comparison.pdf'}
SEED = 20261004
N_GENES = 9554
N_AML = 1733
DIM = 32
K = 10
RESOLUTION = 1.0
N_BOOT = 30
EDGE_KEEP = 0.8
MIN_PATIENTS = 40
MIN_COHORTS = 4
MAX_COHORT_FRACTION = 0.4
MIN_MEDIAN_JACCARD = 0.6
RAND_SEEDS = [20261011, 20261012, 20261013, 20261014, 20261015]
SHUFFLE_SEEDS = [20261021, 20261022, 20261023, 20261024, 20261025]
COHORTS = ['GSE15434', 'GSE161532', 'GSE1729', 'GSE34860', 'GSE37021', 'GSE37642', 'GSE6891', 'GSE71014']
EXPECTED_COUNTS = dict(zip(COHORTS, [251, 61, 43, 78, 97, 562, 537, 104]))
EXCLUDED = {'GSE114868', 'GSE13159', 'GSE9476', 'GSE201492'}

def require(ok, msg):
    if not bool(ok):
        raise RuntimeError(msg)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_once(path, payload):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == payload, f'Existing output differs, preserved: {path.name}')
        print('PASS existing:', path.name)
        return
    temp = path.with_name(path.name + '.partial')
    temp.write_bytes(payload)
    os.replace(temp, path)
    print('CREATED:', path.name)

def save_json(path, obj):
    write_once(path, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8'))

def save_csv(path, data, columns=None):
    x = data.copy()
    if columns:
        x = x.sort_values(columns, kind='mergesort').reset_index(drop=True)
    write_once(path, x.to_csv(index=False, lineterminator='\n', float_format='%.17g').encode('utf-8'))

def verify_manifest(manifest_path, parents):
    m = pd.read_csv(manifest_path, dtype=str)
    require({'path', 'sha256'}.issubset(m.columns), f'Manifest format changed: {manifest_path}')
    for p in parents:
        require(p.is_file(), f'Missing parent: {p}')
        rows = m.loc[m['path'].map(lambda z: Path(str(z)).name).eq(p.name)]
        require(len(rows) == 1, f'Expected one manifest entry for {p.name}')
        require(rows['sha256'].iloc[0] == sha256(p), f'SHA mismatch: {p.name}')

def normalized_gene_matrix(gene_matrix):
    e = np.asarray(gene_matrix, dtype=np.float32).copy()
    require(e.shape == (N_GENES, DIM), f'Gene basis dimension changed: {e.shape}')
    e -= e.mean(axis=0, keepdims=True)
    e /= np.where(e.std(axis=0, keepdims=True) < 1e-08, 1.0, e.std(axis=0, keepdims=True))
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    return e.astype(np.float32)

def patient_projection(X, gses, genes):
    n = len(gses)
    P = np.zeros((n, DIM), dtype=np.float32)
    for g in COHORTS:
        ix = np.where(gses == g)[0]
        xg = X[ix]
        med = np.median(xg, axis=0, keepdims=True)
        centered = xg - med
        projected = centered @ genes / np.sqrt(N_GENES)
        pmed = np.median(projected, axis=0, keepdims=True)
        p25 = np.quantile(projected, 0.25, axis=0, keepdims=True)
        p75 = np.quantile(projected, 0.75, axis=0, keepdims=True)
        iqr = p75 - p25
        iqr[iqr < 1e-06] = 1.0
        projected = (projected - pmed) / iqr
        projected = np.clip(projected, -8, 8)
        projected = projected / np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
        P[ix] = projected.astype(np.float32)
    return P

def create_mnn(P, gses):
    rows = []
    for i, ga in enumerate(COHORTS):
        for gb in COHORTS[i + 1:]:
            ia = np.where(gses == ga)[0]
            ib = np.where(gses == gb)[0]
            sim = P[ia] @ P[ib].T
            ka = min(K, len(ib))
            kb = min(K, len(ia))
            top_b = np.argpartition(-sim, kth=ka - 1, axis=1)[:, :ka]
            top_a = np.argpartition(-sim.T, kth=kb - 1, axis=1)[:, :kb]
            reverse = [set(row.tolist()) for row in top_a]
            for la in range(len(ia)):
                for lb in top_b[la]:
                    lb = int(lb)
                    if la in reverse[lb] and sim[la, lb] > 0:
                        rows.append((int(ia[la]), int(ib[lb]), float(sim[la, lb])))
    require(len(rows) > 0, 'No cross-cohort MNN edges')
    return rows

def leiden_graph(n, edges, seed):
    gr = ig.Graph(n=n, edges=[(int(a), int(b)) for a, b, w in edges], directed=False)
    gr.es['weight'] = [float(w) for a, b, w in edges]
    part = leidenalg.find_partition(gr, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=RESOLUTION, seed=int(seed))
    return np.asarray(part.membership, dtype=int)

def recurrence_and_stability(membership, edges, gses):
    n = len(membership)
    comm_ids = np.unique(membership)
    communities = {int(c): set(np.where(membership == c)[0].tolist()) for c in comm_ids}
    e_arr = np.asarray([(a, b) for a, b, w in edges], dtype=int)
    n_keep = int(round(EDGE_KEEP * len(edges)))
    weights = np.asarray([w for a, b, w in edges], dtype=float)
    rng = np.random.default_rng(SEED + 9000)
    stability = {int(c): [] for c in comm_ids}
    for b in range(N_BOOT):
        keep = np.sort(rng.choice(len(e_arr), size=n_keep, replace=False))
        sample_edges = [(int(e_arr[j, 0]), int(e_arr[j, 1]), float(weights[j])) for j in keep]
        lab = leiden_graph(n, sample_edges, SEED + b + 1)
        perturb_comms = [set(np.where(lab == c)[0].tolist()) for c in np.unique(lab)]
        for c, nodes in communities.items():
            stability[c].append(max((len(nodes & other) / len(nodes | other) for other in perturb_comms)))
    stable_medians = {c: float(np.median(v)) for c, v in stability.items()}
    passed = []
    for c, nodes in communities.items():
        gcount = pd.Series(gses[list(nodes)]).value_counts()
        if len(nodes) >= MIN_PATIENTS and len(gcount) >= MIN_COHORTS and (float(gcount.max() / len(nodes)) <= MAX_COHORT_FRACTION) and (stable_medians[c] >= MIN_MEDIAN_JACCARD):
            passed.append(c)
    recovered = np.array([int(c in passed) for c in membership], dtype=int)
    assigned = np.isin(membership, passed)
    nmi = float(normalized_mutual_info_score(gses[assigned], membership[assigned])) if assigned.sum() > 0 and len(set(membership[assigned])) >= 2 else float('nan')
    return {'raw_communities_n': len(comm_ids), 'recurrent_states_n': len(passed), 'recurrently_assigned_n': int(assigned.sum()), 'recurrent_coverage': float(assigned.mean()), 'state_cohort_NMI_descriptive': nmi, 'median_recurrent_stability': float(np.median([stable_medians[c] for c in passed])) if passed else float('nan'), 'median_all_community_stability': float(np.median(list(stable_medians.values()))), 'assigned_mask': assigned, 'recurrent_raw_labels': np.where(assigned, membership, -1)}
for p in [RANK, DISCOVERY, F481, F483, F486, F488, F489, F490, F492, F497]:
    require(p.is_file(), f'Missing file: {p}')
verify_manifest(F497, [F481, F483, F486, F488, F489, F490, F492])
locked_protocol = json.loads(F481.read_text())
require(locked_protocol['protocol_id'] == 'AML_GRAPHRX_R5_2G_GENE_FIRST_STATE_DISCOVERY_V1', 'Unexpected R5G discovery protocol')
protocol = {'id': 'AML_GRAPHRX_R10A_GENE_FIRST_DISCOVERY_REPRESENTATION_ABLATION_V1', 'status': 'RETROSPECTIVE_EXPLORATORY', 'purpose': 'Test internal discovery representation sensitivity to removing gene-graph representation', 'scope': 'Same 1733 discovery AML / 8 cohorts as frozen R5G; no heldout test in this stage', 'arms': ['LOCKED_GENE_GNN32', 'EXPRESSION_PCA32'] + [f'RANDOM_GAUSSIAN32_s{i}' for i in range(len(RAND_SEEDS))] + [f'GNN_GENE_ROW_SHUFFLE32_s{i}' for i in range(len(SHUFFLE_SEEDS))], 'fairness': 'Identical 32D patient projection, within-GSE gene median centering, robust IQR scaling, cosine MNN K10, weighted RB Leiden resolution1, 30x80pct edge perturbation and R5G recurrence criteria', 'pca': 'Randomized SVD of cohort-median-centered discovery expression, 32 gene loadings; column-standardize and gene-row L2-normalize like original GNN basis; no supervised labels', 'random_seeds': RAND_SEEDS, 'shuffle_seeds': SHUFFLE_SEEDS, 'bootstrap_edges': N_BOOT, 'bootstrap_edge_keep': EDGE_KEEP, 'original_GNN_fidelity_required': True, 'metrics': ['recurrent_states_n', 'recurrent_coverage', 'state_cohort_NMI_descriptive', 'median_recurrent_stability', 'ARI_vs_original_discovery_labels_descriptive'], 'prohibit': ['GNN retraining', 'post-result filtering', 'GSE exclusion', 'alter original R5G parameters', 'select superiority from a single internal clustering metric', 'claim independent state portability'], 'parent_files_sha256': {p.name: sha256(p) for p in [RANK, DISCOVERY, F481, F483, F486, F488, F489, F490, F492, F497]}}
save_json(OUTPUT['protocol'], protocol)
save_csv(OUTPUT['integrity'], pd.DataFrame([{'file': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in [RANK, DISCOVERY, F481, F483, F486, F488, F489, F490, F492, F497]]), ['file'])
ledger = pd.read_csv(DISCOVERY, dtype={'analysis_unit_id': str, 'gse': str})
require(len(ledger) == N_AML, 'R5G discovery size changed')
require(ledger['analysis_unit_index_0based'].is_unique, 'Duplicate rank matrix row indices')
require(set(ledger['gse']) == set(COHORTS), 'R5G discovery cohort set changed')
for g, n in EXPECTED_COUNTS.items():
    require((ledger['gse'] == g).sum() == n, f'R5G cohort count changed: {g}')
require(not set(ledger['gse']) & EXCLUDED, 'Forbidden/heldout cohort in discovery')
ix = ledger['analysis_unit_index_0based'].to_numpy(dtype=int)
gses = ledger['gse'].to_numpy(str)
mm = np.load(RANK, mmap_mode='r', allow_pickle=False)
require(mm.shape[1] == N_GENES and ix.max() < len(mm), 'Wrong locked rank matrix shape')
X = np.array(mm[ix, :], dtype=np.float32, copy=True)
require(X.shape == (N_AML, N_GENES) and np.isfinite(X).all(), 'Invalid discovery expression')
E_locked = np.load(F483, allow_pickle=False)
require(E_locked.shape == (N_GENES, DIM), 'Wrong locked GNN basis')
P_locked_recalc = patient_projection(X, gses, E_locked)
P_locked_ref = np.load(F486, allow_pickle=False)
require(P_locked_ref.shape == P_locked_recalc.shape, 'Wrong locked patient projection')
proj_max_diff = float(np.max(np.abs(P_locked_recalc - P_locked_ref)))
edges_locked = create_mnn(P_locked_recalc, gses)
old_edges = pd.read_csv(F488)
old_pairs = {(int(u), int(v)) for u, v in zip(old_edges['patient_node_u'], old_edges['patient_node_v'])}
new_pairs = {(u, v) for u, v, w in edges_locked}
edge_intersection = len(old_pairs & new_pairs)
edge_union = len(old_pairs | new_pairs)
edge_jaccard = edge_intersection / edge_union if edge_union else float('nan')
raw_locked_recalc = leiden_graph(N_AML, edges_locked, SEED)
old_community = pd.read_csv(F489).set_index('analysis_unit_index_0based').loc[ix, 'raw_community'].to_numpy(int)
leiden_ari = float(adjusted_rand_score(old_community, raw_locked_recalc))
fidelity = {'GNN_projection_max_abs_delta': proj_max_diff, 'GNN_MNN_edges_recomputed_n': len(edges_locked), 'GNN_MNN_edges_frozen_n': len(old_pairs), 'GNN_MNN_edge_jaccard': edge_jaccard, 'GNN_recomputed_vs_original_Leiden_ARI': leiden_ari, 'R5G_exact_reproduction_pass': bool(proj_max_diff <= 0.0001 and edge_jaccard == 1.0 and (leiden_ari > 0.999999))}
save_csv(OUTPUT['fidelity'], pd.DataFrame([fidelity]))
print('ORIGINAL R5G FIDELITY:', fidelity)
require(fidelity['R5G_exact_reproduction_pass'], 'TECHNICAL STOP: exact original R5G reconstruction failed. Do NOT compare arms or tune settings. Send 752 fidelity CSV.')
original_state = pd.read_csv(F492).set_index('analysis_unit_index_0based').loc[ix, 'state_id'].astype(str).to_numpy()
ref_metrics = recurrence_and_stability(raw_locked_recalc, edges_locked, gses)
require(ref_metrics['recurrent_states_n'] == 6, 'Recurrent state count does not reproduce 6')
require(abs(ref_metrics['recurrent_coverage'] - 1720 / 1733) < 1e-10, 'Recurrent coverage does not reproduce 1720/1733')
require(abs(ref_metrics['state_cohort_NMI_descriptive'] - 0.006042148552179825) < 1e-08, 'Original cohort NMI changed; check Leiden/recurrence environment')
X_center = X.copy()
for g in COHORTS:
    mask = gses == g
    X_center[mask] -= np.median(X_center[mask], axis=0, keepdims=True)
print('Training label-free PCA32 gene basis (discovery-only).')
U, S, Vt = randomized_svd(X_center, n_components=DIM, n_iter=5, random_state=SEED)
del U, S
E_pca = normalized_gene_matrix(Vt.T)
del Vt, X_center
arms = [('LOCKED_GENE_GNN32', E_locked), ('EXPRESSION_PCA32', E_pca)]
for seed in RAND_SEEDS:
    rng = np.random.default_rng(seed)
    arms.append((f'RANDOM_GAUSSIAN32_seed{seed}', normalized_gene_matrix(rng.standard_normal((N_GENES, DIM)).astype(np.float32))))
for seed in SHUFFLE_SEEDS:
    rng = np.random.default_rng(seed)
    arms.append((f'GNN_GENE_ROW_SHUFFLE32_seed{seed}', E_locked[rng.permutation(N_GENES)].copy()))
rows = []
for j, (arm, basis) in enumerate(arms, 1):
    print(f'ARM {j}/{len(arms)}: {arm}', flush=True)
    if arm == 'LOCKED_GENE_GNN32':
        proj = P_locked_recalc
        edges = edges_locked
        assignment = raw_locked_recalc
        metrics = ref_metrics
    else:
        proj = patient_projection(X, gses, basis)
        edges = create_mnn(proj, gses)
        assignment = leiden_graph(N_AML, edges, SEED)
        metrics = recurrence_and_stability(assignment, edges, gses)
    observed_labels = np.asarray(metrics['recurrent_raw_labels'], dtype=int)
    old_label = np.where(original_state == 'UNRESOLVED', 'UNRESOLVED', original_state)
    comparison_label = np.array([f'C{v}' if v != -1 else 'UNRESOLVED' for v in observed_labels])
    ari = float(adjusted_rand_score(old_label, comparison_label))
    group = 'GNN32' if arm.startswith('LOCKED') else 'PCA32' if arm.startswith('EXPRESSION') else 'RANDOM32' if arm.startswith('RANDOM') else 'GNN_SHUFFLE32'
    rows.append({'arm': arm, 'family': group, 'n_subjects': N_AML, 'gene_dimension': DIM, 'MNN_edges_n': len(edges), 'MNN_node_coverage': len({i for a, b, w in edges for i in (a, b)}) / N_AML, 'raw_communities_n': metrics['raw_communities_n'], 'recurrent_states_n': metrics['recurrent_states_n'], 'recurrently_assigned_n': metrics['recurrently_assigned_n'], 'recurrent_coverage': metrics['recurrent_coverage'], 'state_cohort_NMI_descriptive': metrics['state_cohort_NMI_descriptive'], 'median_recurrent_stability': metrics['median_recurrent_stability'], 'median_all_community_stability': metrics['median_all_community_stability'], 'ARI_vs_original_discovery_labels_descriptive': ari, 'external_transfer_metric_tested': False})
metrics_df = pd.DataFrame(rows)
save_csv(OUTPUT['metrics'], metrics_df, ['arm'])
base = metrics_df.set_index('arm').loc['LOCKED_GENE_GNN32']
comparisons = []
for row in metrics_df.itertuples(index=False):
    if row.arm == 'LOCKED_GENE_GNN32':
        continue
    comparisons.append({'comparison': f'GNN32_minus_{row.arm}', 'GNN_recurrent_states_n': int(base['recurrent_states_n']), 'baseline_recurrent_states_n': int(row.recurrent_states_n), 'delta_recurrent_coverage': float(base['recurrent_coverage'] - row.recurrent_coverage), 'delta_median_recurrent_stability': float(base['median_recurrent_stability'] - row.median_recurrent_stability) if pd.notna(row.median_recurrent_stability) else np.nan, 'delta_cohort_NMI': float(base['state_cohort_NMI_descriptive'] - row.state_cohort_NMI_descriptive) if pd.notna(row.state_cohort_NMI_descriptive) else np.nan, 'not_a_test_of_external_portability': True})
comp_df = pd.DataFrame(comparisons)
save_csv(OUTPUT['comparisons'], comp_df, ['comparison'])
families = metrics_df.groupby('family').agg(arm_replicates_n=('arm', 'size'), median_recurrent_coverage=('recurrent_coverage', 'median'), median_recurrent_states_n=('recurrent_states_n', 'median'), median_recurrent_stability=('median_recurrent_stability', 'median'), median_state_cohort_NMI=('state_cohort_NMI_descriptive', 'median')).reset_index()
summary = pd.DataFrame([('analysis_id', protocol['id']), ('scope', 'Same R5G discovery AML; retrospective representation ablation'), ('original_R5G_fidelity_pass', True), ('GNN32_recurrent_states', int(base['recurrent_states_n'])), ('GNN32_recurrent_coverage', float(base['recurrent_coverage'])), ('GNN32_cohort_NMI', float(base['state_cohort_NMI_descriptive'])), ('GNN32_versus_PCA32_delta_coverage', float(base['recurrent_coverage'] - metrics_df.set_index('arm').loc['EXPRESSION_PCA32', 'recurrent_coverage'])), ('independent_state_transport_comparison_completed', False), ('GNN32_SUPERIORITY_ESTABLISHED', False), ('reason_for_false_superiority_flag', 'No independent external state-transfer endpoint or fair graph-edge causal intervention; discovery clustering metrics alone cannot establish superiority'), ('R5G_R7A2_R7B2_R7C_R8H2_inputs_changed', False)], columns=['metric', 'value'])
save_csv(OUTPUT['summary'], summary)
fig, ax = plt.subplots(figsize=(8.8, 4.6))
colors = {'GNN32': '#185f88', 'PCA32': '#dc873b', 'RANDOM32': '#9aa5b1', 'GNN_SHUFFLE32': '#a15d80'}
order = ['GNN32', 'PCA32', 'RANDOM32', 'GNN_SHUFFLE32']
for i, family in enumerate(order):
    y = metrics_df.loc[metrics_df.family == family, 'recurrent_coverage'].to_numpy(dtype=float)
    ax.scatter(np.full(len(y), i), y, s=38, color=colors[family], alpha=0.85, zorder=3)
    if len(y) > 1:
        ax.plot([i - 0.13, i + 0.13], [np.median(y), np.median(y)], color=colors[family], lw=2.0)
ax.set_xticks(range(4), ['Frozen GNN32', 'Expression PCA32', 'Random gene basis (5)', 'Shuffled GNN gene rows (5)'])
ax.set_ylabel('Fraction of discovery AML in recurrent communities')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.17)
ax.set_title('Representation ablation under frozen R5G discovery rules')
ax.text(0.0, -0.23, 'Descriptive discovery-cohort comparison; not independent state-transfer validation', fontsize=8, transform=ax.transAxes)
fig.tight_layout()
for key, fmt in [('figure_png', 'png'), ('figure_pdf', 'pdf')]:
    buf = BytesIO()
    md = {'Creator': 'AML-GraphRx R10A', 'CreationDate': None, 'ModDate': None} if fmt == 'pdf' else {'Software': 'AML-GraphRx R10A'}
    fig.savefig(buf, format=fmt, dpi=320, bbox_inches='tight', facecolor='white', metadata=md)
    write_once(OUTPUT[key], buf.getvalue())
plt.close(fig)
manifest = []
for key in ['protocol', 'integrity', 'fidelity', 'metrics', 'comparisons', 'summary', 'figure_png', 'figure_pdf']:
    p = OUTPUT[key]
    manifest.append({'role': key, 'path': str(p), 'size_bytes': p.stat().st_size, 'sha256': sha256(p)})
save_csv(OUTPUT['manifest'], pd.DataFrame(manifest), ['role'])
print('\n' + '=' * 100 + '\nR10A COMPLETE — RETROSPECTIVE GENE-FIRST REPRESENTATION ABLATION\n' + '=' * 100)
print('R4B supervised classifier results are NOT used in this gene-first state-discovery ablation.')
print('\nR10A arm metrics:')
print(metrics_df[['arm', 'recurrent_states_n', 'recurrent_coverage', 'state_cohort_NMI_descriptive', 'median_recurrent_stability', 'ARI_vs_original_discovery_labels_descriptive']].to_string(index=False))
print('\nFamily summaries:')
print(families.to_string(index=False))
print('\nR10A final:')
print(summary.to_string(index=False))
print('\nNo superior GNN claim can be made without an independent, same-endpoint state-transport comparison.')
