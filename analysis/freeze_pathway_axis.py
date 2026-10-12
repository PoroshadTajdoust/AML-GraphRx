# Manuscript-linked analysis: Frozen component-balanced pathway definition
# Source lineage and packaging changes: provenance/source_lineage.json
import os as _release_os
try:
    from IPython.display import display
except ImportError:

    def display(value):
        print(value)
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, spearmanr
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
N_GENES = 9554
STATE03 = 'AML_STATE_03'
STATE06 = 'AML_STATE_06'
FDR_MAX = 0.05
MIN_SHARED_LEADING_EDGE_GENES = 10
MIN_MAPPED_GENE_COVERAGE = 0.8
REDUNDANCY_JACCARD = 0.5
DISCOVERY_GSES = ['GSE15434', 'GSE161532', 'GSE1729', 'GSE34860', 'GSE37021', 'GSE37642', 'GSE6891', 'GSE71014']
STRESS_GSES = ['GSE114868', 'GSE13159', 'GSE9476']
F595 = AUDIT / '595_R7A2_component_axis_technical_amendment.json'
F596 = AUDIT / '596_R7A2_pathway_input_audit.csv'
F597 = AUDIT / '597_R7A2_redundancy_components.csv'
F598 = AUDIT / '598_R7A2_component_pathway_membership.csv'
F599 = AUDIT / '599_R7A2_component_gene_membership.csv'
F600 = AUDIT / '600_R7A2_discovery_axis_scores.csv'
F601 = AUDIT / '601_R7A2_R5V_stress_scores.csv'
F602 = AUDIT / '602_R7A2_discovery_state03_state06_summary.csv'
F603 = AUDIT / '603_R7A2_R5V_stress_summary.csv'
F604 = AUDIT / '604_R7A2_component_leaveoneout_stability.csv'
F605 = AUDIT / '605_R7A2_frozen_axis_definition.json'
F606 = AUDIT / '606_R7A2_final_summary.csv'
F607 = AUDIT / '607_R7A2_manifest.csv'

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    path = Path(path)
    require(path.is_file(), f'Missing file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def find_exact(root, basename):
    hits = list(Path(root).rglob(basename))
    require(len(hits) > 0, f'Required file not found: {basename}')
    if len(hits) == 1:
        return hits[0]
    by_hash = defaultdict(list)
    for p in hits:
        by_hash[sha256_file(p)].append(p)
    require(len(by_hash) == 1, f'Multiple non-identical copies found: {basename}')
    hits = sorted(hits, key=lambda p: (len(str(p)), str(p)))
    print(f'PASS: {len(hits)} byte-identical copies of {basename}; using {hits[0]}')
    return hits[0]

def canonical_json_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')

def csv_bytes(df, sort_cols=None):
    x = df.copy()
    if sort_cols:
        valid = [c for c in sort_cols if c in x.columns]
        if valid:
            x = x.sort_values(valid, kind='mergesort').reset_index(drop=True)
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
        print('PASS existing identical:', path.name)
    else:
        atomic_write(path, data)
        print('CREATED:', path.name)

def parse_gene_list(x):
    if pd.isna(x):
        return []
    return sorted({z.strip() for z in str(x).replace(',', ';').split(';') if z.strip()})

def normalize_gene_id(series):
    return series.astype(str).str.strip().str.replace('\\.0$', '', regex=True)

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

def epsilon_squared_kw(groups):
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if len(groups) < 2:
        return (np.nan, np.nan, np.nan)
    res = kruskal(*groups)
    n = sum((len(g) for g in groups))
    k = len(groups)
    eps2 = max(0.0, float((res.statistic - k + 1) / (n - k))) if n > k else np.nan
    return (float(res.statistic), float(res.pvalue), eps2)

class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = (self.find(a), self.find(b))
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = (rb, ra)
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
RANK_FILE = find_exact(PROJECT, '20260823T123040Z_patient_rank_raw_float64.npy')
NODE_FILE = find_exact(PROJECT, '20260823T123040Z_node_index.csv')
MAP_FILE = find_exact(PROJECT, '428_R4A_entrez_to_symbol_mapping.csv')
DISCOVERY_FILE = find_exact(PROJECT, '452_R5DICE_discovery_ledger.csv')
STATE_FILE = find_exact(PROJECT, '492_R5G_recurrent_state_patient_assignments.csv')
VALIDATION_FILE = find_exact(PROJECT, '499_R5V_validation_ledger.csv')
PATHWAY_FILE = find_exact(PROJECT, '528_R5BR_shared_pathway_analysis.csv')
R5BR_MANIFEST = find_exact(PROJECT, '534_R5BR_manifest.csv')
R7A_V1_PROTOCOL = find_exact(PROJECT, '583_R7A_predeclared_frozen_pathway_axis_protocol.json')
inputs = [RANK_FILE, NODE_FILE, MAP_FILE, DISCOVERY_FILE, STATE_FILE, VALIDATION_FILE, PATHWAY_FILE, R5BR_MANIFEST, R7A_V1_PROTOCOL]
print('\nAUTHORITATIVE INPUTS')
print('-' * 120)
for p in inputs:
    print(p.name, '|', sha256_file(p))
m534 = pd.read_csv(R5BR_MANIFEST)
row = m534.loc[m534['path'].astype(str).map(lambda z: Path(z).name).eq(PATHWAY_FILE.name)]
require(len(row) == 1, '528 missing/duplicated in 534 manifest.')
require(str(row['sha256'].iloc[0]) == sha256_file(PATHWAY_FILE), '528 SHA mismatch.')
print('PASS: 528 verified against 534 manifest.')
amendment = {'protocol_id': 'AML_GRAPHRX_R7A2_COMPONENT_BALANCED_PATHWAY_AXIS_V1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'parent_R7A_v1_protocol_sha256': sha256_file(R7A_V1_PROTOCOL), 'reason': 'R7A v1 stopped before any pathway-axis scoring because deterministic redundancy collapse left only two S06-high components, violating an arbitrary minimum-three-representatives gate. No R5V stress-test score was available or used to define this amendment.', 'unchanged_rules': {'source_file': PATHWAY_FILE.name, 'source_file_sha256': sha256_file(PATHWAY_FILE), 'FDR_STATE03_lt': FDR_MAX, 'FDR_STATE06_lt': FDR_MAX, 'minimum_shared_leading_edge_genes': MIN_SHARED_LEADING_EDGE_GENES, 'minimum_gene_mapping_coverage': MIN_MAPPED_GENE_COVERAGE, 'redundancy_gene_Jaccard_ge': REDUNDANCY_JACCARD, 'no_new_pathway_download': True, 'no_GNN_training': True, 'no_reclustering': True, 'no_external_performance_tuning': True}, 'amended_representation': {'old': 'choose one representative pathway per redundancy component', 'new': 'score every eligible pathway; average pathway activities within each redundancy component; then give each component equal weight. The final axis is mean(S06-high component activities) minus mean(S03-high component activities).', 'minimum_components': 'at least one component in each orientation', 'reason_for_equal_component_weighting': 'prevents large hierarchical pathway families from dominating merely because they contain many overlapping pathway terms'}, 'R5V_role': 'post-hoc stress test only; not independent external validation', 'next_valid_stage': 'R7B on a genuinely new RNA-seq AML cohort using the frozen R7A2 definition unchanged', 'forbidden': ['change eligibility/FDR/Jaccard rules after R7A2 results', 'select or remove components based on R5V performance', 'drop unfavorable cohorts', 'claim subtype validation, prognosis, drug response, or causality from R7A2']}
write_once_or_verify(F595, canonical_json_bytes(amendment))
X = np.load(RANK_FILE, mmap_mode='r', allow_pickle=False)
require(X.shape == (4554, N_GENES), f'Unexpected rank matrix shape: {X.shape}')
nodes = pd.read_csv(NODE_FILE, dtype=str, keep_default_na=False)
require(len(nodes) == N_GENES, 'Node count changed.')
require('canonical_NCBI_Gene_ID' in nodes.columns, 'Node gene ID missing.')
disc = pd.read_csv(DISCOVERY_FILE, dtype={'analysis_unit_id': str, 'analysis_unit_index_0based': int, 'gse': str})
states = pd.read_csv(STATE_FILE, dtype={'analysis_unit_id': str, 'state_id': str})
val = pd.read_csv(VALIDATION_FILE, dtype={'analysis_unit_id': str, 'analysis_unit_index_0based': int, 'gse': str})
pathways = pd.read_csv(PATHWAY_FILE)
require(len(disc) == 1733, 'Discovery n changed.')
require(len(val) == 894, 'R5V n changed.')
require(set(disc['gse']) == set(DISCOVERY_GSES), 'Discovery GSE set changed.')
require(set(val['gse']) == set(STRESS_GSES), 'R5V GSE set changed.')
require(disc['analysis_unit_id'].is_unique, 'Discovery IDs duplicated.')
require(val['analysis_unit_id'].is_unique, 'Validation IDs duplicated.')
require(states['analysis_unit_id'].is_unique, 'State IDs duplicated.')
mapping_raw = pd.read_csv(MAP_FILE, dtype=str, keep_default_na=False)
node_ids = normalize_gene_id(nodes['canonical_NCBI_Gene_ID'])
universe_ids = set(node_ids)
candidate_id_cols = []
for c in mapping_raw.columns:
    vals = set(normalize_gene_id(mapping_raw[c]))
    candidate_id_cols.append((len(universe_ids & vals), c))
candidate_id_cols.sort(reverse=True)
coverage, ID_COL = candidate_id_cols[0]
require(coverage == N_GENES, 'Could not identify full-coverage GeneID column in 428.')
mapping_raw['_gid'] = normalize_gene_id(mapping_raw[ID_COL])
m = mapping_raw.loc[mapping_raw['_gid'].isin(universe_ids)].copy()
symbol_candidates = [c for c in m.columns if 'symbol' in c.lower() and c != ID_COL]
require(len(symbol_candidates) >= 1, 'Could not identify symbol column in 428.')
SYMBOL_COL = symbol_candidates[0]
m['_symbol'] = m[SYMBOL_COL].astype(str).str.strip()
conf = m.groupby('_gid')['_symbol'].nunique()
require(int((conf > 1).sum()) == 0, 'Conflicting symbols per GeneID.')
m = m[['_gid', '_symbol']].drop_duplicates('_gid', keep='first')
require(len(m) == N_GENES, 'Mapping coverage changed.')
require(m['_symbol'].nunique() == N_GENES, 'Gene symbols must be unique.')
gid_to_index = {gid: i for i, gid in enumerate(node_ids)}
symbol_to_index = {row['_symbol']: gid_to_index[row['_gid']] for _, row in m.iterrows()}
index_to_symbol = {gid_to_index[row['_gid']]: row['_symbol'] for _, row in m.iterrows()}
required_cols = {'library', 'Term', 'NES_STATE03', 'NES_STATE06', 'FDR_STATE03', 'FDR_STATE06', 'direction_relationship', 'shared_leading_edge_n', 'shared_leading_edge_genes'}
require(required_cols.issubset(pathways.columns), f'528 schema missing: {required_cols - set(pathways.columns)}')
p = pathways.copy()
for c in ['NES_STATE03', 'NES_STATE06', 'FDR_STATE03', 'FDR_STATE06']:
    p[c] = pd.to_numeric(p[c], errors='raise')
p['shared_leading_edge_n'] = pd.to_numeric(p['shared_leading_edge_n'], errors='raise').astype(int)
eligible = p.loc[p['direction_relationship'].eq('OPPOSITE_DIRECTION') & (p['FDR_STATE03'] < FDR_MAX) & (p['FDR_STATE06'] < FDR_MAX) & (np.sign(p['NES_STATE03']) != np.sign(p['NES_STATE06'])) & (p['shared_leading_edge_n'] >= MIN_SHARED_LEADING_EDGE_GENES)].copy()
require(len(eligible) > 0, 'No frozen pathway passed eligibility.')
mapped_indices_list = []
mapped_n = []
original_n = []
coverage_frac = []
orientation = []
for _, row in eligible.iterrows():
    genes = parse_gene_list(row['shared_leading_edge_genes'])
    idxs = sorted({symbol_to_index[g] for g in genes if g in symbol_to_index})
    original_n.append(len(genes))
    mapped_n.append(len(idxs))
    coverage_frac.append(len(idxs) / len(genes) if genes else 0.0)
    mapped_indices_list.append(idxs)
    orientation.append('S06_HIGH' if float(row['NES_STATE06']) > 0 else 'S03_HIGH')
eligible['mapped_gene_indices'] = mapped_indices_list
eligible['original_shared_leading_edge_n'] = original_n
eligible['mapped_shared_leading_edge_n'] = mapped_n
eligible['mapped_gene_coverage'] = coverage_frac
eligible['orientation'] = orientation
eligible['joint_NES_strength'] = np.minimum(np.abs(eligible['NES_STATE03'].to_numpy(float)), np.abs(eligible['NES_STATE06'].to_numpy(float)))
eligible['max_FDR'] = np.maximum(eligible['FDR_STATE03'].to_numpy(float), eligible['FDR_STATE06'].to_numpy(float))
eligible['passes_mapping_gate'] = eligible['mapped_gene_coverage'] >= MIN_MAPPED_GENE_COVERAGE
audit_out = eligible.drop(columns=['mapped_gene_indices']).copy()
write_once_or_verify(F596, csv_bytes(audit_out, ['library', 'orientation', 'Term']))
eligible = eligible.loc[eligible['passes_mapping_gate']].reset_index(drop=True)
require(len(eligible) > 0, 'No pathway passed mapping gate.')
component_rows = []
component_members = []
component_id_to_member_indices = {}
component_counter = 0
for (library, ori), grp0 in eligible.groupby(['library', 'orientation'], sort=True):
    grp = grp0.copy().reset_index()
    uf = UnionFind(len(grp))
    for i in range(len(grp)):
        gi = set(grp.iloc[i]['mapped_gene_indices'])
        for j in range(i + 1, len(grp)):
            gj = set(grp.iloc[j]['mapped_gene_indices'])
            if jaccard(gi, gj) >= REDUNDANCY_JACCARD:
                uf.union(i, j)
    comps = defaultdict(list)
    for i in range(len(grp)):
        comps[uf.find(i)].append(i)
    comp_sets = sorted(comps.values(), key=lambda inds: min((str(grp.iloc[i]['Term']) for i in inds)))
    for local_comp_no, inds in enumerate(comp_sets, start=1):
        component_counter += 1
        cid = f'R7C_{component_counter:03d}'
        eligible_indices = [int(grp.iloc[i]['index']) for i in inds]
        component_id_to_member_indices[cid] = eligible_indices
        sub = grp.iloc[inds].copy()
        sub_rank = sub.sort_values(['joint_NES_strength', 'max_FDR', 'mapped_shared_leading_edge_n', 'Term'], ascending=[False, True, False, True], kind='mergesort')
        display_term = str(sub_rank.iloc[0]['Term'])
        component_rows.append({'component_id': cid, 'library': library, 'orientation': ori, 'member_pathways_n': len(inds), 'display_representative_term': display_term, 'component_scoring_rule': 'mean of all member pathway activities'})
        for i in inds:
            erow = grp.iloc[i]
            component_members.append({'component_id': cid, 'library': library, 'orientation': ori, 'Term': str(erow['Term']), 'NES_STATE03': float(erow['NES_STATE03']), 'NES_STATE06': float(erow['NES_STATE06']), 'FDR_STATE03': float(erow['FDR_STATE03']), 'FDR_STATE06': float(erow['FDR_STATE06']), 'mapped_shared_leading_edge_n': int(erow['mapped_shared_leading_edge_n']), 'joint_NES_strength': float(erow['joint_NES_strength'])})
components_df = pd.DataFrame(component_rows)
members_df = pd.DataFrame(component_members)
n_c03 = int(components_df['orientation'].eq('S03_HIGH').sum())
n_c06 = int(components_df['orientation'].eq('S06_HIGH').sum())
require(n_c03 >= 1, 'No S03-high component.')
require(n_c06 >= 1, 'No S06-high component.')
write_once_or_verify(F597, csv_bytes(components_df, ['orientation', 'library', 'component_id']))
write_once_or_verify(F598, csv_bytes(members_df, ['component_id', 'Term']))
component_gene_rows = []
for cid, elig_idxs in component_id_to_member_indices.items():
    gene_union = sorted(set().union(*[set(eligible.loc[eidx, 'mapped_gene_indices']) for eidx in elig_idxs]))
    cmeta = components_df.loc[components_df['component_id'].eq(cid)].iloc[0]
    for gi in gene_union:
        component_gene_rows.append({'component_id': cid, 'library': cmeta['library'], 'orientation': cmeta['orientation'], 'gene_index_0based': int(gi), 'canonical_NCBI_Gene_ID': str(nodes.iloc[int(gi)]['canonical_NCBI_Gene_ID']), 'gene_symbol': str(index_to_symbol[int(gi)])})
component_genes_df = pd.DataFrame(component_gene_rows)
write_once_or_verify(F599, csv_bytes(component_genes_df, ['component_id', 'gene_index_0based']))
eligible = eligible.reset_index(drop=True)
pathway_gene_indices = [np.asarray(v, dtype=int) for v in eligible['mapped_gene_indices']]
eligible_to_component = {}
for cid, idxs in component_id_to_member_indices.items():
    for eidx in idxs:
        eligible_to_component[int(eidx)] = cid
require(len(eligible_to_component) == len(eligible), 'Not every eligible pathway assigned to a component.')
component_ids = components_df['component_id'].tolist()
component_orientation = dict(zip(components_df['component_id'], components_df['orientation']))

def score_samples(sample_indices):
    sample_indices = np.asarray(sample_indices, dtype=int)
    pathway_activity = {}
    for eidx, gi in enumerate(pathway_gene_indices):
        arr = np.asarray(X[np.ix_(sample_indices, gi)], dtype=np.float64)
        pathway_activity[eidx] = arr.mean(axis=1) - 0.5
    component_activity = {}
    for cid in component_ids:
        member_idxs = component_id_to_member_indices[cid]
        stack = np.vstack([pathway_activity[i] for i in member_idxs])
        component_activity[cid] = stack.mean(axis=0)
    s06_components = [component_activity[cid] for cid in component_ids if component_orientation[cid] == 'S06_HIGH']
    s03_components = [component_activity[cid] for cid in component_ids if component_orientation[cid] == 'S03_HIGH']
    require(len(s06_components) >= 1 and len(s03_components) >= 1, 'Both orientations required.')
    s06_mean = np.vstack(s06_components).mean(axis=0)
    s03_mean = np.vstack(s03_components).mean(axis=0)
    axis = s06_mean - s03_mean
    return (axis, s03_mean, s06_mean, component_activity)
disc_idx = disc['analysis_unit_index_0based'].to_numpy(dtype=int)
disc_axis, disc_s03prog, disc_s06prog, disc_comp = score_samples(disc_idx)
disc_scores = disc[['analysis_unit_id', 'analysis_unit_index_0based', 'gse']].copy()
disc_scores['R7A2_axis_score'] = disc_axis
disc_scores['R7A2_S03_high_component_mean'] = disc_s03prog
disc_scores['R7A2_S06_high_component_mean'] = disc_s06prog
disc_scores = disc_scores.merge(states[['analysis_unit_id', 'state_id']], on='analysis_unit_id', how='left', validate='one_to_one')
write_once_or_verify(F600, csv_bytes(disc_scores, ['analysis_unit_index_0based']))
val_idx = val['analysis_unit_index_0based'].to_numpy(dtype=int)
val_axis, val_s03prog, val_s06prog, val_comp = score_samples(val_idx)
val_cols = ['analysis_unit_id', 'analysis_unit_index_0based', 'gse']
for c in ['binary_label', 'binary_label_int']:
    if c in val.columns:
        val_cols.append(c)
val_scores = val[val_cols].copy()
val_scores['R7A2_axis_score'] = val_axis
val_scores['R7A2_S03_high_component_mean'] = val_s03prog
val_scores['R7A2_S06_high_component_mean'] = val_s06prog
val_scores['analysis_role_R7A2'] = 'POST_HOC_R5V_STRESS_TEST_NOT_INDEPENDENT_VALIDATION'
write_once_or_verify(F601, csv_bytes(val_scores, ['analysis_unit_index_0based']))
d36 = disc_scores.loc[disc_scores['state_id'].isin([STATE03, STATE06])].copy()
require(int(d36['state_id'].eq(STATE03).sum()) == 285, 'State03 n changed.')
require(int(d36['state_id'].eq(STATE06).sum()) == 160, 'State06 n changed.')
s03 = d36.loc[d36['state_id'].eq(STATE03), 'R7A2_axis_score'].to_numpy(float)
s06 = d36.loc[d36['state_id'].eq(STATE06), 'R7A2_axis_score'].to_numpy(float)
y = d36['state_id'].eq(STATE06).astype(int).to_numpy()
auc = float(roc_auc_score(y, d36['R7A2_axis_score'].to_numpy(float)))
mw = mannwhitneyu(s06, s03, alternative='two-sided')
summary_rows = [{'scope': 'ALL_DISCOVERY', 'gse': '', 'State03_n': len(s03), 'State06_n': len(s06), 'State03_median': float(np.median(s03)), 'State06_median': float(np.median(s06)), 'State06_vs_State03_AUROC_descriptive': auc, 'Mann_Whitney_p_descriptive': float(mw.pvalue)}]
for gse, gg in d36.groupby('gse', sort=True):
    n3 = int(gg['state_id'].eq(STATE03).sum())
    n6 = int(gg['state_id'].eq(STATE06).sum())
    if n3 >= 3 and n6 >= 3:
        yy = gg['state_id'].eq(STATE06).astype(int).to_numpy()
        ss = gg['R7A2_axis_score'].to_numpy(float)
        gauc = float(roc_auc_score(yy, ss))
    else:
        gauc = np.nan
    summary_rows.append({'scope': 'PER_COHORT', 'gse': gse, 'State03_n': n3, 'State06_n': n6, 'State03_median': float(gg.loc[gg['state_id'].eq(STATE03), 'R7A2_axis_score'].median()) if n3 else np.nan, 'State06_median': float(gg.loc[gg['state_id'].eq(STATE06), 'R7A2_axis_score'].median()) if n6 else np.nan, 'State06_vs_State03_AUROC_descriptive': gauc, 'Mann_Whitney_p_descriptive': np.nan})
state_summary = pd.DataFrame(summary_rows)
write_once_or_verify(F602, csv_bytes(state_summary, ['scope', 'gse']))
disc_all = disc_scores['R7A2_axis_score'].to_numpy(float)
disc_q025, disc_q975 = np.quantile(disc_all, [0.025, 0.975])
disc_iqr = robust_iqr(disc_all)
disc_med = float(np.median(disc_all))
if 'binary_label_int' in val_scores.columns:
    val_aml = val_scores.loc[pd.to_numeric(val_scores['binary_label_int'], errors='coerce').eq(1)].copy()
elif 'binary_label' in val_scores.columns:
    val_aml = val_scores.loc[val_scores['binary_label'].astype(str).str.upper().eq('AML')].copy()
else:
    val_aml = val_scores.copy()
stress_rows = []
for gse, gg in val_aml.groupby('gse', sort=True):
    ss = gg['R7A2_axis_score'].to_numpy(float)
    stress_rows.append({'gse': gse, 'AML_n': int(len(gg)), 'median_axis_score': float(np.median(ss)), 'IQR_axis_score': robust_iqr(ss), 'fraction_within_discovery_central95': float(np.mean((ss >= disc_q025) & (ss <= disc_q975))), 'median_shift_in_discovery_IQR_units': (float(np.median(ss)) - disc_med) / disc_iqr if disc_iqr > 0 else np.nan, 'interpretation_role': 'POST_HOC_STRESS_TEST_ONLY'})
stress_summary = pd.DataFrame(stress_rows)
write_once_or_verify(F603, csv_bytes(stress_summary, ['gse']))
loo_rows = []
full_axis = disc_axis.copy()
for cid in component_ids:
    remaining_s06 = [c for c in component_ids if c != cid and component_orientation[c] == 'S06_HIGH']
    remaining_s03 = [c for c in component_ids if c != cid and component_orientation[c] == 'S03_HIGH']
    if len(remaining_s06) == 0 or len(remaining_s03) == 0:
        loo_rows.append({'left_out_component': cid, 'orientation': component_orientation[cid], 'evaluable': False, 'spearman_with_full_axis': np.nan})
        continue
    s06_mean_loo = np.vstack([disc_comp[c] for c in remaining_s06]).mean(axis=0)
    s03_mean_loo = np.vstack([disc_comp[c] for c in remaining_s03]).mean(axis=0)
    axis_loo = s06_mean_loo - s03_mean_loo
    rho = float(spearmanr(full_axis, axis_loo).statistic)
    loo_rows.append({'left_out_component': cid, 'orientation': component_orientation[cid], 'evaluable': True, 'spearman_with_full_axis': rho})
loo_df = pd.DataFrame(loo_rows)
write_once_or_verify(F604, csv_bytes(loo_df, ['left_out_component']))
frozen_components = []
for _, crow in components_df.sort_values('component_id').iterrows():
    cid = crow['component_id']
    member_idxs = component_id_to_member_indices[cid]
    pathway_defs = []
    for eidx in member_idxs:
        erow = eligible.loc[eidx]
        genes = []
        for gi in erow['mapped_gene_indices']:
            genes.append({'gene_index_0based': int(gi), 'canonical_NCBI_Gene_ID': str(nodes.iloc[int(gi)]['canonical_NCBI_Gene_ID']), 'gene_symbol': str(index_to_symbol[int(gi)])})
        pathway_defs.append({'library': str(erow['library']), 'Term': str(erow['Term']), 'NES_STATE03': float(erow['NES_STATE03']), 'NES_STATE06': float(erow['NES_STATE06']), 'FDR_STATE03': float(erow['FDR_STATE03']), 'FDR_STATE06': float(erow['FDR_STATE06']), 'genes': genes})
    frozen_components.append({'component_id': cid, 'library': str(crow['library']), 'orientation': str(crow['orientation']), 'display_representative_term': str(crow['display_representative_term']), 'member_pathways_n': int(crow['member_pathways_n']), 'component_activity_formula': 'mean of member pathway activities', 'member_pathways': pathway_defs})
axis_definition = {'axis_id': 'AML_GRAPHRX_R7A2_COMPONENT_BALANCED_PATHWAY_AXIS_V1', 'source_528_sha256': sha256_file(PATHWAY_FILE), 'amendment_595_sha256': sha256_file(F595), 'score_orientation': 'higher = State06-like; lower = State03-like', 'pathway_activity': 'mean frozen leading-edge within-profile percentile ranks minus 0.5', 'component_activity': 'mean of pathway activities within component', 'final_axis': 'mean(S06_HIGH component activities) minus mean(S03_HIGH component activities)', 'sample_or_cohort_scaling': False, 'components_total_n': int(len(components_df)), 'S03_high_components_n': n_c03, 'S06_high_components_n': n_c06, 'components': frozen_components, 'claim_boundary': 'frozen candidate continuous pathway representation; requires unchanged testing in a genuinely new external AML cohort'}
write_once_or_verify(F605, canonical_json_bytes(axis_definition))
disc_groups = [gg['R7A2_axis_score'].to_numpy(float) for _, gg in disc_scores.groupby('gse', sort=True)]
disc_kw_H, disc_kw_p, disc_kw_eps2 = epsilon_squared_kw(disc_groups)
all_groups = list(disc_groups) + [gg['R7A2_axis_score'].to_numpy(float) for _, gg in val_aml.groupby('gse', sort=True)]
all_kw_H, all_kw_p, all_kw_eps2 = epsilon_squared_kw(all_groups)
loo_eval = loo_df.loc[loo_df['evaluable'] & loo_df['spearman_with_full_axis'].notna()]
loo_median = float(loo_eval['spearman_with_full_axis'].median()) if len(loo_eval) else np.nan
loo_min = float(loo_eval['spearman_with_full_axis'].min()) if len(loo_eval) else np.nan
final_summary = pd.DataFrame([('protocol_id', 'AML_GRAPHRX_R7A2_COMPONENT_BALANCED_PATHWAY_AXIS_V1'), ('source_shared_pathways_total_n', int(len(pathways))), ('eligible_pathways_n', int(len(eligible))), ('redundancy_components_total_n', int(len(components_df))), ('S03_high_components_n', n_c03), ('S06_high_components_n', n_c06), ('discovery_State03_n', int(len(s03))), ('discovery_State06_n', int(len(s06))), ('discovery_State03_median_axis', float(np.median(s03))), ('discovery_State06_median_axis', float(np.median(s06))), ('discovery_State06_vs_State03_AUROC_descriptive', auc), ('discovery_State03_vs_State06_MW_p_descriptive', float(mw.pvalue)), ('component_LOO_median_spearman_with_full_axis', loo_median), ('component_LOO_min_spearman_with_full_axis', loo_min), ('discovery_cohort_KW_H', disc_kw_H), ('discovery_cohort_KW_p', disc_kw_p), ('discovery_cohort_epsilon_squared', disc_kw_eps2), ('R5V_stress_median_fraction_within_discovery_central95', float(stress_summary['fraction_within_discovery_central95'].median())), ('R5V_stress_median_abs_shift_discovery_IQR_units', float(np.median(np.abs(stress_summary['median_shift_in_discovery_IQR_units'])))), ('combined_discovery_plus_R5V_AML_cohort_KW_H', all_kw_H), ('combined_discovery_plus_R5V_AML_cohort_KW_p', all_kw_p), ('combined_discovery_plus_R5V_AML_cohort_epsilon_squared', all_kw_eps2), ('R7A2_AXIS_FROZEN_FOR_R7B', True), ('independent_external_validation_completed', False), ('claim_boundary', 'candidate pathway-level continuous representation; old R5V cohorts are post-hoc stress tests only'), ('next_step', 'R7B: apply 605_R7A2_frozen_axis_definition.json unchanged to a genuinely new RNA-seq AML cohort')], columns=['metric', 'value'])
write_once_or_verify(F606, csv_bytes(final_summary))
manifest_rows = []
for role, pth in [('technical_amendment', F595), ('pathway_input_audit', F596), ('redundancy_components', F597), ('component_pathway_membership', F598), ('component_gene_membership', F599), ('discovery_axis_scores', F600), ('R5V_stress_scores', F601), ('discovery_state03_state06_summary', F602), ('R5V_stress_summary', F603), ('component_leaveoneout_stability', F604), ('frozen_axis_definition', F605), ('final_summary', F606)]:
    manifest_rows.append({'role': role, 'path': str(pth), 'sha256': sha256_file(pth)})
manifest = pd.DataFrame(manifest_rows)
write_once_or_verify(F607, csv_bytes(manifest, ['role']))
try:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    counts = [int(components_df['orientation'].eq('S03_HIGH').sum()), int(components_df['orientation'].eq('S06_HIGH').sum())]
    ax.bar(['S03-high', 'S06-high'], counts)
    ax.set_ylabel('Redundancy components')
    ax.set_title('R7A2 component-balanced pathway architecture')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'R7A2_component_counts_300dpi.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR / 'R7A2_component_counts.pdf', bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.boxplot([s03, s06], tick_labels=['State03', 'State06'], showfliers=False)
    ax.set_ylabel('R7A2 pathway-axis score')
    ax.set_title('Discovery-only descriptive S03/S06 separation')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'R7A2_discovery_S03_S06_300dpi.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR / 'R7A2_discovery_S03_S06.pdf', bbox_inches='tight')
    plt.close(fig)
    vals = [val_aml.loc[val_aml['gse'].eq(g), 'R7A2_axis_score'].to_numpy(float) for g in STRESS_GSES]
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.boxplot(vals, tick_labels=STRESS_GSES, showfliers=False)
    ax.axhline(disc_med, linewidth=1)
    ax.set_ylabel('R7A2 pathway-axis score')
    ax.set_title('Post-hoc R5V stress test (not independent validation)')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'R7A2_R5V_stress_300dpi.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR / 'R7A2_R5V_stress.pdf', bbox_inches='tight')
    plt.close(fig)
except Exception as e:
    print('WARNING: figure generation skipped:', repr(e))
print('\n' + '=' * 120)
print('R7A2 COMPLETE — COMPONENT-BALANCED FROZEN PATHWAY AXIS')
print('=' * 120)
print('\nCOMPONENT COUNTS')
print(components_df.groupby(['orientation', 'library']).size().rename('components_n').reset_index().to_string(index=False))
print('\nDISCOVERY S03/S06 DESCRIPTIVE CHECK')
print(state_summary.to_string(index=False))
print('\nR5V POST-HOC STRESS SUMMARY')
print(stress_summary.to_string(index=False))
print('\nCOMPONENT LEAVE-ONE-OUT STABILITY')
print(loo_df.to_string(index=False))
print('\nFINAL SUMMARY')
print(final_summary.to_string(index=False))
print('\nIMPORTANT')
print('- R7A v1 remains preserved as a failed structural gate.')
print('- R7A2 changed representation logic before any R7A pathway-axis performance was observed.')
print('- Do not tune R7A2 using the old R5V cohorts.')
print('- The next valid confirmatory step is R7B on a genuinely new RNA-seq AML cohort.')
