# Manuscript-linked analysis: Frozen DepMap candidate-family analysis
# Source lineage and packaging changes: provenance/source_lineage.json
import os as _release_os
try:
    from IPython.display import display
except ImportError:

    def display(value):
        print(value)
import os
import re
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
F605 = AUDIT / '605_R7A2_frozen_axis_definition.json'
F607 = AUDIT / '607_R7A2_manifest.csv'
F600 = AUDIT / '600_R7A2_discovery_axis_scores.csv'
F428 = AUDIT / '428_R4A_entrez_to_symbol_mapping.csv'
F452 = AUDIT / '452_R5DICE_discovery_ledger.csv'
F492 = AUDIT / '492_R5G_recurrent_state_patient_assignments.csv'
NODE_FILE = PROJECT / 'data' / 'patient_graphs' / '20260823T123040Z' / '20260823T123040Z_node_index.csv'
RANK_FILE = PROJECT / 'data' / 'patient_graphs' / '20260823T123040Z' / '20260823T123040Z_patient_rank_raw_float64.npy'
F536 = AUDIT / '536_R5DM_input_audit.csv'
F539 = AUDIT / '539_R5DM_candidate_gene_dependency_statistics.csv'
F544 = AUDIT / '544_R5DM_manifest.csv'
F546 = AUDIT / '546_R5DM_high_confidence_state_membership_clean.csv'
F639 = AUDIT / '639_R7C_final_summary.csv'
F640 = AUDIT / '640_R7C_manifest.csv'
F672 = AUDIT / '672_R8H2_final_summary.csv'
F673 = AUDIT / '673_R8H2_manifest.csv'
F674_PARENT = AUDIT / '674_R8F_predeclared_functional_dependency_protocol.json'
F675_PARENT = AUDIT / '675_R8F_parent_integrity_audit.csv'
F680_PARENT = AUDIT / '680_R8F_candidate_panel_audit.csv'
F687 = AUDIT / '687_R8F2_DepMap_access_technical_amendment.json'
F688 = AUDIT / '688_R8F2_DepMap26Q3_source_audit.csv'
F689 = AUDIT / '689_R8F2_expression_mapping_audit.csv'
F690 = AUDIT / '690_R8F2_masked_discovery_sensitivity.csv'
F691 = AUDIT / '691_R8F2_DepMap_AML_axis_scores.csv'
F692 = AUDIT / '692_R8F2_full533_axis_dependency_associations.csv'
F693 = AUDIT / '693_R8F2_focused33_axis_dependency_associations.csv'
F694 = AUDIT / '694_R8F2_focused33_bootstrap_stability.csv'
F695 = AUDIT / '695_R8F2_FDR_significant_axis_dependency_associations.csv'
F696 = AUDIT / '696_R8F2_final_summary.csv'
F697 = AUDIT / '697_R8F2_manifest.csv'
DEPMAP_DIR = PROJECT / 'external' / 'depmap_26Q3'
if not DEPMAP_DIR.exists():
    raise RuntimeError(f'DepMap directory not found:\n{DEPMAP_DIR}')
print('DepMap directory:', DEPMAP_DIR)
DEPMAP_RELEASE = '26Q3'
EXPECTED_DEPENDENCY_SHA256 = 'c1177633cba85e460d8c4dad8dd1388b5744629e1e267bc354cb7dfc1794a718'
EXPECTED_MODEL_SHA256 = '65eb1895800e71a963a3ab56e5d08c8d006cb0d9e7aa68f7e34a3810a2a3cbc7'
DEPENDENCY_FILENAME = 'CRISPRGeneDependency.csv'
MODEL_FILENAME = 'Model.csv'
EXPRESSION_FILENAME_CANDIDATES = ['OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv', 'OmicsExpressionProteinCodingGenesTPMLogp1.csv']
MIN_PATHWAY_MAPPING_FRACTION = 0.8
MIN_MASKED_VS_FULL_DISCOVERY_SPEARMAN = 0.9
MAX_MASKED_VS_FULL_MEDIAN_ABS_DIFF_IQR = 0.25
MIN_MASKED_STATE03_STATE06_AUROC = 0.7
EXPECTED_R5DM_AML_MODELS = 32
MIN_AXIS_SCORED_AML_MODELS = 20
MIN_PAIRED_AML_MODELS_PER_GENE = 20
FULL_FAMILY_FDR = 0.05
FOCUSED_BOOTSTRAPS = 1000
FOCUSED_BOOTSTRAP_SEED = 20261010
FOCUSED_SIGN_STABILITY = 0.95

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

def write_json(path, obj):
    b = (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')
    if path.exists():
        require(path.read_bytes() == b, f'Existing JSON differs: {path.name}')
        print('PASS existing identical:', path.name)
    else:
        path.write_bytes(b)
        print('CREATED:', path.name)

def write_csv(path, df, sort_cols=None):
    x = df.copy()
    if sort_cols:
        valid = [c for c in sort_cols if c in x.columns]
        if valid:
            x = x.sort_values(valid, kind='mergesort').reset_index(drop=True)
    b = x.to_csv(index=False, lineterminator='\n').encode('utf-8')
    if path.exists():
        require(path.read_bytes() == b, f'Existing CSV differs: {path.name}')
        print('PASS existing identical:', path.name)
    else:
        path.write_bytes(b)
        print('CREATED:', path.name)

def bool_series(s):
    return s.astype(str).str.strip().str.lower().map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False})

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    n = len(p)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out

def robust_iqr(x):
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, 0.75) - np.quantile(x, 0.25))

def normalize_gene_id(s):
    return s.astype(str).str.strip().str.replace('\\.0$', '', regex=True)

def gene_symbol_from_depmap_column(c):
    s = str(c)
    m = re.match('^(.*?)\\s+\\(\\d+\\)$', s)
    if m:
        return m.group(1).strip()
    return None

def identify_model_id_column(df):
    candidates = []
    for c in df.columns:
        vals = df[c].astype(str).str.strip()
        n = int(vals.str.match('^ACH-\\d+$', na=False).sum())
        if n > 0:
            candidates.append((n, c))
    require(len(candidates) > 0, 'Could not identify DepMap ModelID column.')
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise RuntimeError(f'Ambiguous DepMap ModelID columns: {candidates[:5]}')
    return candidates[0][1]
required_files = [F605, F607, F600, F428, F452, F492, NODE_FILE, RANK_FILE, F539, F544, F639, F640, F672, F673, F674_PARENT, F675_PARENT, F680_PARENT]
for path in required_files:
    require(path.is_file(), f'Missing frozen parent file:\n{path}')
print('PASS: all required frozen parent files exist.')
m607 = pd.read_csv(F607)
x = m607.loc[m607['path'].astype(str).map(lambda z: Path(z).name).eq(F605.name)]
require(len(x) == 1, '605 not uniquely present in 607.')
require(str(x['sha256'].iloc[0]) == sha256_file(F605), '605 SHA mismatch vs 607.')
m544 = pd.read_csv(F544)
x = m544.loc[m544['path'].astype(str).map(lambda z: Path(z).name).eq(F539.name)]
require(len(x) == 1, '539 not uniquely present in 544.')
require(str(x['sha256'].iloc[0]) == sha256_file(F539), '539 SHA mismatch vs 544.')
m640 = pd.read_csv(F640)
x = m640.loc[m640['path'].astype(str).map(lambda z: Path(z).name).eq(F639.name)]
require(len(x) == 1, '639 not uniquely present in 640.')
require(str(x['sha256'].iloc[0]) == sha256_file(F639), '639 SHA mismatch.')
s639_df = pd.read_csv(F639)
s639 = dict(zip(s639_df['metric'].astype(str), s639_df['value'].astype(str)))
require(s639.get('R7C_RANK_COMPOSITION_ROBUSTNESS_SUPPORTED', '').lower() == 'true', 'R7C robustness support is not TRUE.')
m673 = pd.read_csv(F673)
x = m673.loc[m673['path'].astype(str).map(lambda z: Path(z).name).eq(F672.name)]
require(len(x) == 1, '672 not uniquely present in 673.')
require(str(x['sha256'].iloc[0]) == sha256_file(F672), '672 SHA mismatch.')
print('PASS: frozen parent integrity chain verified.')
amendment = {'protocol_id': 'AML_GRAPHRX_R8F2_DEPMAP_LOCAL_INGESTION_V1', 'parent_R8F_protocol_file': F674_PARENT.name, 'parent_R8F_protocol_sha256': sha256_file(F674_PARENT), 'parent_R8F_integrity_file': F675_PARENT.name, 'parent_R8F_candidate_panel_file': F680_PARENT.name, 'reason': 'R8F-v1 stopped at HTTP 403 from the DepMap automated download catalogue before any functional association was calculated.', 'scientific_protocol_change': False, 'data_acquisition_change_only': True, 'DepMap_release': DEPMAP_RELEASE, 'local_directory': str(DEPMAP_DIR), 'dependency_model_rule': 'CRISPRGeneDependency.csv and Model.csv must exactly match the previously locked R5DM 26Q3 SHA256 values.', 'expression_rule': 'Use official DepMap Public 26Q3 protein-coding expression file from normal Downloads interface.', 'allowed_expression_filenames': EXPRESSION_FILENAME_CANDIDATES, 'forbidden': ['change frozen R7A2 pathway definition', 'change 533-gene inferential family', 'select AML models by functional outcome', 'change FDR threshold after observing results', 'substitute another DepMap release', 'interpret CRISPR dependency as drug response']}
write_json(F687, amendment)
DEP_FILE = DEPMAP_DIR / DEPENDENCY_FILENAME
MODEL_FILE = DEPMAP_DIR / MODEL_FILENAME
require(DEP_FILE.is_file(), f'Missing:\n{DEP_FILE}')
require(MODEL_FILE.is_file(), f'Missing:\n{MODEL_FILE}')
expression_hits = [DEPMAP_DIR / name for name in EXPRESSION_FILENAME_CANDIDATES if (DEPMAP_DIR / name).is_file()]
require(len(expression_hits) >= 1, 'No valid expression file found.\n\nExpected one of:\n' + '\n'.join(EXPRESSION_FILENAME_CANDIDATES) + '\n\nFolder:\n' + str(DEPMAP_DIR))
require(len(expression_hits) == 1, 'More than one allowed expression file was found. Keep only the actual 26Q3 file to avoid ambiguity:\n' + '\n'.join(map(str, expression_hits)))
EXPR_FILE = expression_hits[0]
EXPR_FILENAME = EXPR_FILE.name
dep_sha = sha256_file(DEP_FILE)
model_sha = sha256_file(MODEL_FILE)
require(dep_sha == EXPECTED_DEPENDENCY_SHA256, f'CRISPRGeneDependency.csv SHA mismatch.\nObserved: {dep_sha}\nExpected: {EXPECTED_DEPENDENCY_SHA256}\n\nSTOP. Do not use a different DepMap release.')
require(model_sha == EXPECTED_MODEL_SHA256, f'Model.csv SHA mismatch.\nObserved: {model_sha}\nExpected: {EXPECTED_MODEL_SHA256}\n\nSTOP. Do not use a different DepMap release.')
expr_sha = sha256_file(EXPR_FILE)
print('PASS: CRISPRGeneDependency exact locked SHA.')
print('PASS: Model.csv exact locked SHA.')
print('Expression file:', EXPR_FILE)
print('Expression SHA256:', expr_sha)
expr_header = pd.read_csv(EXPR_FILE, nrows=0)
require(len(expr_header.columns) > 10000, f'Expression file has unexpectedly few columns: {len(expr_header.columns)}')
source_audit = pd.DataFrame([{'role': 'dependency', 'filename': DEP_FILE.name, 'path': str(DEP_FILE), 'bytes': DEP_FILE.stat().st_size, 'sha256': dep_sha, 'status': 'LOCKED_R5DM_EXACT_SHA_REUSE', 'DepMap_release': DEPMAP_RELEASE}, {'role': 'model_metadata', 'filename': MODEL_FILE.name, 'path': str(MODEL_FILE), 'bytes': MODEL_FILE.stat().st_size, 'sha256': model_sha, 'status': 'LOCKED_R5DM_EXACT_SHA_REUSE', 'DepMap_release': DEPMAP_RELEASE}, {'role': 'expression', 'filename': EXPR_FILE.name, 'path': str(EXPR_FILE), 'bytes': EXPR_FILE.stat().st_size, 'sha256': expr_sha, 'status': 'OFFICIAL_LOCAL_26Q3_RECORDED_BEFORE_ANALYSIS', 'DepMap_release': DEPMAP_RELEASE}])
write_csv(F688, source_audit, ['role'])
cand = pd.read_csv(F539)
for c in ['gene_symbol', 'PRIMARY_ELIGIBLE', 'DEPMAP_HIGH_CONFIDENCE_SUPPORT']:
    require(c in cand.columns, f'539 missing column: {c}')
primary_flag = bool_series(cand['PRIMARY_ELIGIBLE'])
focused_flag = bool_series(cand['DEPMAP_HIGH_CONFIDENCE_SUPPORT'])
require(primary_flag.notna().all(), 'PRIMARY_ELIGIBLE could not be parsed.')
require(focused_flag.notna().all(), 'DEPMAP_HIGH_CONFIDENCE_SUPPORT could not be parsed.')
full_genes = sorted(cand.loc[primary_flag, 'gene_symbol'].astype(str).str.strip().drop_duplicates())
focused_genes = sorted(cand.loc[focused_flag, 'gene_symbol'].astype(str).str.strip().drop_duplicates())
require(len(full_genes) == 533, f'Locked primary candidate count changed: {len(full_genes)}')
require(len(focused_genes) == 33, f'Locked focused candidate count changed: {len(focused_genes)}')
require(set(focused_genes).issubset(full_genes), 'Focused 33 are not subset of full 533.')
print('PASS: locked full candidate family = 533')
print('PASS: locked focused panel = 33')
print('\nLoading DepMap files...')
dep = pd.read_csv(DEP_FILE, low_memory=False)
model = pd.read_csv(MODEL_FILE, low_memory=False)
expr = pd.read_csv(EXPR_FILE, low_memory=False)
print('Dependency shape:', dep.shape)
print('Model shape:', model.shape)
print('Expression shape:', expr.shape)
dep_id_col = identify_model_id_column(dep)
model_id_col = identify_model_id_column(model)
expr_id_col = identify_model_id_column(expr)
print('Dependency ModelID column:', dep_id_col)
print('Model metadata ID column:', model_id_col)
print('Expression ModelID column:', expr_id_col)
require('OncotreePrimaryDisease' in model.columns, 'Model.csv missing OncotreePrimaryDisease.')
if expr[expr_id_col].astype(str).duplicated().any():
    default_candidates = [c for c in expr.columns if re.sub('[^a-z0-9]+', '', str(c).lower()) in {'isdefaultentryformodel', 'isdefaultentrymodel'}]
    require(len(default_candidates) == 1, f'Expression ModelIDs are duplicated but default-entry field could not be uniquely identified.\nCandidates: {default_candidates}')
    dc = default_candidates[0]
    default_flag = bool_series(expr[dc])
    require(default_flag.notna().all(), f'Could not parse {dc}.')
    expr = expr.loc[default_flag].copy()
    print('Expression default-entry filtering applied:', dc)
require(not dep[dep_id_col].astype(str).duplicated().any(), 'Dependency ModelIDs duplicated.')
require(not expr[expr_id_col].astype(str).duplicated().any(), 'Expression ModelIDs still duplicated.')
dep_ids = set(dep[dep_id_col].astype(str))
aml_metadata_ids = model.loc[model['OncotreePrimaryDisease'].astype(str).eq('Acute Myeloid Leukemia'), model_id_col].astype(str)
aml_models = sorted(set(aml_metadata_ids) & dep_ids)
require(len(aml_models) == EXPECTED_R5DM_AML_MODELS, f'Expected 32 CRISPR-screened AML models but found {len(aml_models)}.')
expr_ids = set(expr[expr_id_col].astype(str))
aml_scored_models = sorted(set(aml_models) & expr_ids)
require(len(aml_scored_models) >= MIN_AXIS_SCORED_AML_MODELS, f'Too few AML models with both expression and dependency data: {len(aml_scored_models)}')
print('\nCRISPR-screened AML models:', len(aml_models))
print('AML models with expression + dependency:', len(aml_scored_models))
dep_symbol_to_col = {}
for c in dep.columns:
    symbol = gene_symbol_from_depmap_column(c)
    if symbol is None:
        continue
    require(symbol not in dep_symbol_to_col, f'Duplicated dependency gene symbol: {symbol}')
    dep_symbol_to_col[symbol] = c
expr_symbol_to_col = {}
for c in expr.columns:
    symbol = gene_symbol_from_depmap_column(c)
    if symbol is None:
        continue
    require(symbol not in expr_symbol_to_col, f'Duplicated expression gene symbol: {symbol}')
    expr_symbol_to_col[symbol] = c
require(len(dep_symbol_to_col) > 10000, f'Unexpected dependency gene count: {len(dep_symbol_to_col)}')
require(len(expr_symbol_to_col) > 10000, f'Unexpected expression gene count: {len(expr_symbol_to_col)}')
print('Dependency genes:', len(dep_symbol_to_col))
print('Expression genes:', len(expr_symbol_to_col))
mapping = pd.read_csv(F428, dtype=str, keep_default_na=False)
symbol_candidates = [c for c in mapping.columns if 'symbol' in c.lower()]
require(len(symbol_candidates) >= 1, 'No symbol column found in 428.')
SYMBOL_COL = symbol_candidates[0]
project_symbols = pd.Index(mapping[SYMBOL_COL].astype(str).str.strip().replace('', np.nan).dropna().drop_duplicates())
require(len(project_symbols) == 9554, f'Expected 9554 project symbols; found {len(project_symbols)}.')
mapped_project_symbols = [s for s in project_symbols if s in expr_symbol_to_col]
whole_universe_fraction = len(mapped_project_symbols) / len(project_symbols)
print('\nProject genes mapped to DepMap expression:', len(mapped_project_symbols), '/ 9554')
print('Whole-universe mapping fraction:', whole_universe_fraction)
axis = json.loads(F605.read_text(encoding='utf-8'))
components = axis['components']
require(len(components) == 114, f'Frozen component count changed: {len(components)}')
s03_cids = [c['component_id'] for c in components if c['orientation'] == 'S03_HIGH']
s06_cids = [c['component_id'] for c in components if c['orientation'] == 'S06_HIGH']
require(len(s03_cids) == 112, 'Expected 112 S03-high components.')
require(len(s06_cids) == 2, 'Expected 2 S06-high components.')
pathway_rows = []
for comp in components:
    for j, pw in enumerate(comp['member_pathways'], start=1):
        genes = sorted({str(g['gene_symbol']).strip() for g in pw['genes'] if str(g['gene_symbol']).strip()})
        require(len(genes) > 0, 'Frozen pathway contains zero genes.')
        mapped = [g for g in genes if g in expr_symbol_to_col]
        fraction = len(mapped) / len(genes)
        pathway_rows.append({'component_id': comp['component_id'], 'pathway_id': f"{comp['component_id']}|P{j:03d}", 'orientation': comp['orientation'], 'Term': pw['Term'], 'frozen_genes_n': len(genes), 'mapped_genes_n': len(mapped), 'mapping_fraction': fraction, 'mapping_gate': bool(fraction >= MIN_PATHWAY_MAPPING_FRACTION)})
pathway_audit = pd.DataFrame(pathway_rows)
require(pathway_audit['mapping_gate'].all(), 'At least one frozen pathway has <80% DepMap expression coverage.\nSTOP. Do not lower the threshold.')
expr_mapping_audit = pd.DataFrame([{'project_gene_universe_n': 9554, 'DepMap_expression_mapped_project_genes_n': len(mapped_project_symbols), 'whole_universe_mapping_fraction_descriptive': whole_universe_fraction, 'frozen_pathways_n': len(pathway_audit), 'minimum_frozen_pathway_mapping_fraction': float(pathway_audit['mapping_fraction'].min()), 'pathway_mapping_gate_all_pass': bool(pathway_audit['mapping_gate'].all()), 'CRISPR_screened_AML_models_n': len(aml_models), 'AML_models_with_expression_and_dependency_n': len(aml_scored_models)}])
write_csv(F689, expr_mapping_audit)
print('\n' + '=' * 110)
print('EXPRESSION / MAPPING AUDIT')
print('=' * 110)
print(expr_mapping_audit.to_string(index=False))
nodes = pd.read_csv(NODE_FILE, dtype=str, keep_default_na=False)
require('canonical_NCBI_Gene_ID' in nodes.columns, 'Node file missing canonical_NCBI_Gene_ID.')
node_ids = normalize_gene_id(nodes['canonical_NCBI_Gene_ID'])
require(len(node_ids) == 9554, f'Node file expected 9554 genes; found {len(node_ids)}.')
universe_ids = set(node_ids)
id_candidates = []
for c in mapping.columns:
    vals = set(normalize_gene_id(mapping[c]))
    id_candidates.append((len(universe_ids & vals), c))
id_candidates.sort(reverse=True)
coverage, ID_COL = id_candidates[0]
require(coverage == 9554, 'Could not identify full-coverage GeneID column in 428.')
map2 = mapping.copy()
map2['_gid'] = normalize_gene_id(map2[ID_COL])
map2['_symbol'] = map2[SYMBOL_COL].astype(str).str.strip()
map2 = map2.loc[map2['_gid'].isin(universe_ids)].drop_duplicates('_gid')
require(len(map2) == 9554, 'Project mapping changed.')
gid_to_idx = {gid: i for i, gid in enumerate(node_ids)}
symbol_to_idx = {row['_symbol']: gid_to_idx[row['_gid']] for _, row in map2.iterrows()}
available_indices = np.array(sorted((symbol_to_idx[s] for s in mapped_project_symbols if s in symbol_to_idx)), dtype=int)
require(len(available_indices) == len(mapped_project_symbols), 'Project-symbol/index mapping mismatch.')
disc = pd.read_csv(F452, dtype={'analysis_unit_id': str, 'analysis_unit_index_0based': int, 'gse': str})
states = pd.read_csv(F492, dtype={'analysis_unit_id': str, 'state_id': str})
full_scores = pd.read_csv(F600)
rank_mm = np.load(RANK_FILE, mmap_mode='r', allow_pickle=False)
require(rank_mm.shape[1] == 9554, f'Rank matrix gene dimension changed: {rank_mm.shape}')
disc_idx = disc['analysis_unit_index_0based'].to_numpy(dtype=int)
require(len(disc_idx) == 1733, f'Discovery ledger expected 1733 rows; found {len(disc_idx)}.')
sub = pd.DataFrame(np.asarray(rank_mm[np.ix_(disc_idx, available_indices)], dtype=np.float64).T)
masked_rank = sub.rank(axis=0, method='average', pct=True).to_numpy(dtype=float)
project_idx_to_local = {project_idx: local_idx for local_idx, project_idx in enumerate(available_indices)}
symbol_to_local = {s: project_idx_to_local[symbol_to_idx[s]] for s in mapped_project_symbols}
component_scores_disc = {}
for comp in components:
    member_scores = []
    for pw in comp['member_pathways']:
        genes = sorted({str(g['gene_symbol']).strip() for g in pw['genes'] if str(g['gene_symbol']).strip()})
        local_indices = [symbol_to_local[g] for g in genes if g in symbol_to_local]
        fraction = len(local_indices) / len(genes)
        require(fraction >= MIN_PATHWAY_MAPPING_FRACTION, 'Masked-discovery pathway coverage fell below 80%.')
        member_scores.append(masked_rank[local_indices, :].mean(axis=0) - 0.5)
    component_scores_disc[comp['component_id']] = np.vstack(member_scores).mean(axis=0)
masked_s03 = np.vstack([component_scores_disc[c] for c in s03_cids]).mean(axis=0)
masked_s06 = np.vstack([component_scores_disc[c] for c in s06_cids]).mean(axis=0)
masked_axis = masked_s06 - masked_s03
require('R7A2_axis_score' in full_scores.columns, '600 missing R7A2_axis_score.')
full_axis = pd.to_numeric(full_scores['R7A2_axis_score'], errors='raise').to_numpy(dtype=float)
require(len(full_axis) == 1733, 'Full discovery axis length changed.')
require(len(masked_axis) == 1733, 'Masked discovery axis length changed.')
rho_mask = spearmanr(full_axis, masked_axis)
full_iqr = robust_iqr(full_axis)
require(full_iqr > 0, 'Full discovery axis IQR is zero.')
median_abs_diff_iqr = float(np.median(np.abs(masked_axis - full_axis))) / full_iqr
d = disc[['analysis_unit_id', 'gse']].copy()
d['masked_axis'] = masked_axis
d = d.merge(states[['analysis_unit_id', 'state_id']], on='analysis_unit_id', how='left', validate='one_to_one')
d36 = d.loc[d['state_id'].isin(['AML_STATE_03', 'AML_STATE_06'])].copy()
require(len(d36) == 445, f'Expected State03 + State06 n=445; found {len(d36)}.')
y36 = d36['state_id'].eq('AML_STATE_06').astype(int).to_numpy()
auc_masked = float(roc_auc_score(y36, d36['masked_axis'].to_numpy(dtype=float)))
gate_mask_rho = bool(float(rho_mask.statistic) >= MIN_MASKED_VS_FULL_DISCOVERY_SPEARMAN)
gate_mask_diff = bool(median_abs_diff_iqr <= MAX_MASKED_VS_FULL_MEDIAN_ABS_DIFF_IQR)
gate_mask_auc = bool(auc_masked >= MIN_MASKED_STATE03_STATE06_AUROC)
masked_gate = bool(gate_mask_rho and gate_mask_diff and gate_mask_auc)
masked_summary = pd.DataFrame([{'DepMap_available_project_genes_n': len(mapped_project_symbols), 'DepMap_available_project_gene_fraction': whole_universe_fraction, 'masked_vs_full_discovery_score_spearman': float(rho_mask.statistic), 'masked_vs_full_discovery_score_spearman_p': float(rho_mask.pvalue), 'median_absolute_score_difference_full_IQR_units': float(median_abs_diff_iqr), 'masked_discovery_State06_vs_State03_AUROC': auc_masked, 'gate_spearman': gate_mask_rho, 'gate_score_difference': gate_mask_diff, 'gate_State03_State06_AUROC': gate_mask_auc, 'MASKED_DISCOVERY_PRESERVATION_GATE': masked_gate}])
write_csv(F690, masked_summary)
print('\n' + '=' * 110)
print('MASKED DISCOVERY SENSITIVITY')
print('=' * 110)
print(masked_summary.to_string(index=False))
require(masked_gate, 'DepMap expression masked-discovery preservation FAILED.\nSTOP before functional analysis.')
expr_indexed = expr.set_index(expr_id_col, drop=False)
expression_cols = [expr_symbol_to_col[s] for s in mapped_project_symbols]
X = pd.DataFrame({model_id: pd.to_numeric(expr_indexed.loc[model_id, expression_cols], errors='coerce').to_numpy(dtype=float) for model_id in aml_scored_models}, index=mapped_project_symbols)
X = X.loc[~X.isna().all(axis=1)].copy()
if X.isna().any().any():
    gene_medians = X.median(axis=1)
    X = X.T.fillna(gene_medians).T
require(np.isfinite(X.to_numpy(dtype=float)).all(), 'Non-finite AML expression values remain.')
for comp in components:
    for pw in comp['member_pathways']:
        genes = sorted({str(g['gene_symbol']).strip() for g in pw['genes'] if str(g['gene_symbol']).strip()})
        mapped = [g for g in genes if g in X.index]
        fraction = len(mapped) / len(genes)
        require(fraction >= MIN_PATHWAY_MAPPING_FRACTION, f"Pathway coverage fell below 80% after missingness handling:\n{pw['Term']}")
rank_ext = X.rank(axis=0, method='average', pct=True)
component_scores_ext = {}
for comp in components:
    member_scores = []
    for pw in comp['member_pathways']:
        genes = sorted({str(g['gene_symbol']).strip() for g in pw['genes'] if str(g['gene_symbol']).strip()})
        mapped = [g for g in genes if g in rank_ext.index]
        member_score = rank_ext.loc[mapped, aml_scored_models].mean(axis=0).to_numpy(dtype=float) - 0.5
        member_scores.append(member_score)
    component_scores_ext[comp['component_id']] = np.vstack(member_scores).mean(axis=0)
ext_s03 = np.vstack([component_scores_ext[c] for c in s03_cids]).mean(axis=0)
ext_s06 = np.vstack([component_scores_ext[c] for c in s06_cids]).mean(axis=0)
ext_axis = ext_s06 - ext_s03
axis_df = pd.DataFrame({'ModelID': aml_scored_models, 'S03_high_component_mean': ext_s03, 'S06_high_component_mean': ext_s06, 'R8F2_frozen_pathway_axis_score': ext_axis})
write_csv(F691, axis_df, ['ModelID'])
print('\nFrozen pathway axis scored in', len(axis_df), 'DepMap AML models.')
dep_indexed = dep.set_index(dep_id_col, drop=False)
axis_map = axis_df.set_index('ModelID')['R8F2_frozen_pathway_axis_score'].astype(float)
association_rows = []
for gene in full_genes:
    dep_col = dep_symbol_to_col.get(gene)
    if dep_col is None:
        association_rows.append({'gene_symbol': gene, 'dependency_column': '', 'paired_AML_n': 0, 'evaluable': False, 'spearman_rho': np.nan, 'p_two_sided': np.nan, 'association_direction': ''})
        continue
    mids = [m for m in aml_scored_models if m in dep_indexed.index]
    dependency_values = pd.to_numeric(dep_indexed.loc[mids, dep_col], errors='coerce').to_numpy(dtype=float)
    pathway_values = axis_map.loc[mids].to_numpy(dtype=float)
    keep = np.isfinite(dependency_values) & np.isfinite(pathway_values)
    dependency_values = dependency_values[keep]
    pathway_values = pathway_values[keep]
    n = len(pathway_values)
    if n < MIN_PAIRED_AML_MODELS_PER_GENE or np.std(dependency_values) == 0 or np.std(pathway_values) == 0:
        rho = np.nan
        p = np.nan
        evaluable = False
        direction = ''
    else:
        test = spearmanr(pathway_values, dependency_values)
        rho = float(test.statistic)
        p = float(test.pvalue)
        evaluable = bool(np.isfinite(rho) and np.isfinite(p))
        if evaluable:
            if rho > 0:
                direction = 'S06_LIKE_HIGHER_DEPENDENCY'
            elif rho < 0:
                direction = 'S03_LIKE_HIGHER_DEPENDENCY'
            else:
                direction = 'NO_MONOTONIC_DIRECTION'
        else:
            direction = ''
    association_rows.append({'gene_symbol': gene, 'dependency_column': dep_col, 'paired_AML_n': n, 'evaluable': evaluable, 'spearman_rho': rho, 'p_two_sided': p, 'association_direction': direction})
assoc = pd.DataFrame(association_rows)
require(len(assoc) == 533, f'Candidate family changed: {len(assoc)}')
eval_idx = assoc.index[assoc['evaluable']].tolist()
require(len(eval_idx) > 0, 'No evaluable candidate genes.')
assoc['BH_FDR_full533_family'] = np.nan
assoc.loc[eval_idx, 'BH_FDR_full533_family'] = bh_fdr(assoc.loc[eval_idx, 'p_two_sided'].to_numpy(dtype=float))
assoc['FDR_significant_full_family'] = assoc['BH_FDR_full533_family'].notna() & (assoc['BH_FDR_full533_family'] <= FULL_FAMILY_FDR)
annotation_cols = [c for c in ['gene_symbol', 'DEPMAP_PRIMARY_FUNCTIONAL_SUPPORT', 'MYELOID_DIRECTION_CONSISTENT', 'DEPMAP_HIGH_CONFIDENCE_SUPPORT', 'bridge_roles_DepMap', 'state03_meta_Z_DepMap', 'state06_meta_Z_DepMap'] if c in cand.columns]
assoc = assoc.merge(cand[annotation_cols].drop_duplicates('gene_symbol'), on='gene_symbol', how='left', validate='one_to_one')
if F546.is_file():
    membership = pd.read_csv(F546)
    if 'gene_symbol' in membership.columns and membership['gene_symbol'].astype(str).nunique() == 33:
        keep_membership = [c for c in ['gene_symbol', 'STATE_MEMBERSHIP', 'STATE03_DIRECTION', 'STATE06_DIRECTION', 'STATE03_STATE06_RELATION'] if c in membership.columns]
        assoc = assoc.merge(membership[keep_membership].drop_duplicates('gene_symbol'), on='gene_symbol', how='left', validate='one_to_one')
write_csv(F692, assoc, ['gene_symbol'])
focused = assoc.loc[assoc['gene_symbol'].isin(focused_genes)].copy()
require(len(focused) == 33, f'Focused panel changed: {len(focused)}')
focused['focused_panel_inference_note'] = 'Full-533-family BH q retained; no separate 33-gene FDR'
write_csv(F693, focused, ['gene_symbol'])
rng = np.random.default_rng(FOCUSED_BOOTSTRAP_SEED)
bootstrap_rows = []
for gene in focused_genes:
    rr = focused.loc[focused['gene_symbol'].eq(gene)]
    require(len(rr) == 1, f'Focused gene row count error: {gene}')
    row = rr.iloc[0]
    if not bool(row['evaluable']):
        bootstrap_rows.append({'gene_symbol': gene, 'observed_rho': row['spearman_rho'], 'bootstrap_evaluable': False, 'bootstrap_n': 0, 'rho_q025': np.nan, 'rho_q975': np.nan, 'same_sign_fraction': np.nan, 'sign_stable_ge_0_95': False})
        continue
    dep_col = dep_symbol_to_col[gene]
    mids = aml_scored_models
    y = pd.to_numeric(dep_indexed.loc[mids, dep_col], errors='coerce').to_numpy(dtype=float)
    x = axis_map.loc[mids].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = len(x)
    boot_rho = []
    for b in range(FOCUSED_BOOTSTRAPS):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        test = spearmanr(xb, yb)
        if np.isfinite(test.statistic):
            boot_rho.append(float(test.statistic))
    boot_rho = np.asarray(boot_rho, dtype=float)
    require(len(boot_rho) >= 950, f'Too many invalid bootstrap replicates for {gene}: {len(boot_rho)}')
    observed_rho = float(row['spearman_rho'])
    same_sign_fraction = float(np.mean(np.sign(boot_rho) == np.sign(observed_rho)))
    bootstrap_rows.append({'gene_symbol': gene, 'observed_rho': observed_rho, 'bootstrap_evaluable': True, 'bootstrap_n': len(boot_rho), 'rho_q025': float(np.quantile(boot_rho, 0.025)), 'rho_q975': float(np.quantile(boot_rho, 0.975)), 'same_sign_fraction': same_sign_fraction, 'sign_stable_ge_0_95': bool(same_sign_fraction >= FOCUSED_SIGN_STABILITY)})
boot_df = pd.DataFrame(bootstrap_rows)
write_csv(F694, boot_df, ['gene_symbol'])
sig = assoc.loc[assoc['FDR_significant_full_family']].copy()
sig = sig.sort_values(['BH_FDR_full533_family', 'spearman_rho'], ascending=[True, False], kind='mergesort').reset_index(drop=True)
sig['display_rank_only'] = np.arange(1, len(sig) + 1)
write_csv(F695, sig, ['display_rank_only'])
n_evaluable = int(assoc['evaluable'].sum())
n_sig = int(assoc['FDR_significant_full_family'].sum())
n_sig_focused = int(focused['FDR_significant_full_family'].fillna(False).astype(bool).sum())
n_stable_focused = int(boot_df['sign_stable_ge_0_95'].fillna(False).astype(bool).sum())
focus_joint = focused[['gene_symbol', 'FDR_significant_full_family']].merge(boot_df[['gene_symbol', 'sign_stable_ge_0_95']], on='gene_symbol', how='left', validate='one_to_one')
n_sig_and_stable_focused = int((focus_joint['FDR_significant_full_family'].fillna(False).astype(bool) & focus_joint['sign_stable_ge_0_95'].fillna(False).astype(bool)).sum())
functional_support = bool(n_sig >= 1)
if n_sig > 0:
    strongest_index = np.argmax(np.abs(sig['spearman_rho'].to_numpy(dtype=float)))
    strongest = sig.iloc[strongest_index]
    strongest_gene = str(strongest['gene_symbol'])
    strongest_rho = float(strongest['spearman_rho'])
    strongest_q = float(strongest['BH_FDR_full533_family'])
else:
    strongest_gene = ''
    strongest_rho = np.nan
    strongest_q = np.nan
final_summary = pd.DataFrame([('protocol_id', 'AML_GRAPHRX_R8F2_DEPMAP_LOCAL_INGESTION_V1'), ('DepMap_release', DEPMAP_RELEASE), ('DepMap_expression_filename', EXPR_FILENAME), ('DepMap_expression_sha256', expr_sha), ('CRISPR_screened_AML_models_n', len(aml_models)), ('AML_models_with_expression_and_dependency_n', len(aml_scored_models)), ('DepMap_expression_mapped_project_genes_n', len(mapped_project_symbols)), ('DepMap_expression_mapped_project_gene_fraction', whole_universe_fraction), ('minimum_frozen_pathway_mapping_fraction', float(pathway_audit['mapping_fraction'].min())), ('masked_vs_full_discovery_axis_spearman', float(rho_mask.statistic)), ('masked_vs_full_discovery_median_abs_diff_IQR_units', float(median_abs_diff_iqr)), ('masked_discovery_State06_vs_State03_AUROC', auc_masked), ('full_R5DM_candidate_family_n', 533), ('full_family_evaluable_genes_n', n_evaluable), ('full_family_FDR_significant_axis_dependency_genes_n', n_sig), ('focused_locked_panel_n', 33), ('focused_panel_FDR_significant_using_full_family_q_n', n_sig_focused), ('focused_panel_bootstrap_sign_stable_n', n_stable_focused), ('focused_panel_FDR_significant_and_sign_stable_n', n_sig_and_stable_focused), ('strongest_FDR_significant_gene_display_only', strongest_gene), ('strongest_FDR_significant_gene_rho_display_only', strongest_rho), ('strongest_FDR_significant_gene_full_family_q_display_only', strongest_q), ('R8F2_AXIS_DEPENDENCY_ASSOCIATION_SUPPORTED', functional_support), ('claim_boundary', 'association between the frozen AML pathway axis and CRISPR gene-dependency probability across DepMap AML cell-line models; not drug response, clinical benefit, causal target validation, or treatment recommendation'), ('next_step_if_supported', 'freeze R8F2 and integrate only full-family FDR-supported functional associations into the manuscript; stop pathway-axis development'), ('next_step_if_not_supported', 'retain the negative result and do not retune the frozen axis, candidate family, thresholds, or AML model set')], columns=['metric', 'value'])
write_csv(F696, final_summary)
manifest_rows = []
for role, path in [('access_technical_amendment', F687), ('parent_R8F_protocol', F674_PARENT), ('parent_R8F_integrity', F675_PARENT), ('parent_candidate_panel', F680_PARENT), ('DepMap_source_audit', F688), ('expression_mapping_audit', F689), ('masked_discovery_sensitivity', F690), ('AML_axis_scores', F691), ('full533_associations', F692), ('focused33_associations', F693), ('focused33_bootstrap', F694), ('significant_associations', F695), ('final_summary', F696)]:
    manifest_rows.append({'role': role, 'path': str(path), 'sha256': sha256_file(path)})
manifest_df = pd.DataFrame(manifest_rows)
write_csv(F697, manifest_df, ['role'])
try:
    import matplotlib.pyplot as plt
    evaluable_df = assoc.loc[assoc['evaluable']].copy()
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.hist(evaluable_df['spearman_rho'].to_numpy(dtype=float), bins=35)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel('Spearman rho: frozen pathway axis vs CRISPR dependency')
    ax.set_ylabel('Locked R5DM bridge candidates')
    ax.set_title('R8F2 functional-dependency association spectrum')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'R8F2_dependency_association_spectrum_300dpi.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR / 'R8F2_dependency_association_spectrum.pdf', bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    qplot = evaluable_df['BH_FDR_full533_family'].to_numpy(dtype=float)
    yplot = -np.log10(np.clip(qplot, 1e-300, 1))
    ax.scatter(evaluable_df['spearman_rho'].to_numpy(dtype=float), yplot, s=18, alpha=0.7)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel('Spearman rho')
    ax.set_ylabel('-log10(full-family BH q)')
    ax.set_title('R8F2 frozen-axis / dependency associations')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'R8F2_dependency_FDR_scatter_300dpi.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR / 'R8F2_dependency_FDR_scatter.pdf', bbox_inches='tight')
    plt.close(fig)
except Exception as e:
    print('WARNING: figure generation skipped:', repr(e))
print('\n' + '=' * 120)
print('R8F2 COMPLETE — DEPMAP FUNCTIONAL GENE-DEPENDENCY VALIDATION')
print('=' * 120)
print('\nDEPMAP SOURCE AUDIT')
print(source_audit.to_string(index=False))
print('\nEXPRESSION / MAPPING AUDIT')
print(expr_mapping_audit.to_string(index=False))
print('\nMASKED DISCOVERY SENSITIVITY')
print(masked_summary.to_string(index=False))
print('\nFULL 533 FAMILY — FDR SIGNIFICANT ASSOCIATIONS')
if len(sig) > 0:
    display_cols = [c for c in ['display_rank_only', 'gene_symbol', 'paired_AML_n', 'spearman_rho', 'p_two_sided', 'BH_FDR_full533_family', 'association_direction', 'DEPMAP_HIGH_CONFIDENCE_SUPPORT', 'STATE_MEMBERSHIP'] if c in sig.columns]
    print(sig[display_cols].to_string(index=False))
else:
    print('No full-family BH-FDR significant axis-dependency associations.')
print('\nFOCUSED LOCKED 33')
focused_display_cols = [c for c in ['gene_symbol', 'paired_AML_n', 'spearman_rho', 'p_two_sided', 'BH_FDR_full533_family', 'FDR_significant_full_family', 'association_direction', 'STATE_MEMBERSHIP'] if c in focused.columns]
print(focused[focused_display_cols].sort_values('BH_FDR_full533_family', na_position='last').to_string(index=False))
print('\nFOCUSED 33 BOOTSTRAP STABILITY')
print(boot_df.sort_values('gene_symbol').to_string(index=False))
print('\nFINAL SUMMARY')
print(final_summary.to_string(index=False))
print('\nIMPORTANT')
print('- Frozen R7A2/R7B2/R7C axis was NOT changed.')
print('- Full inferential family = all 533 locked R5DM candidates.')
print('- Focused 33 reuse the full-family BH q values.')
print('- CRISPRGeneDependency is NOT a drug-response endpoint.')
print('- No candidate/model/pathway may be removed because of R8F2 performance.')
print('- If R8F2 is negative, do NOT retune the axis or candidate list.')
