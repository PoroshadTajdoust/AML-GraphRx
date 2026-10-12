# Manuscript-linked analysis: Frozen Reactome and GO enrichment
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
import json
import math
import hashlib
import subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
try:
    import gseapy as gp
    if str(gp.__version__) != '1.3.0':
        raise ImportError
except Exception:
    raise RuntimeError('Install the recorded analysis dependencies before running this script; see docs/ENVIRONMENT.md.')
    import gseapy as gp
require_gseapy_version = str(gp.__version__)
if require_gseapy_version != '1.3.0':
    raise RuntimeError(f'GSEApy must be exactly 1.3.0; observed={require_gseapy_version}')
print('GSEApy:', gp.__version__)
SEED = 20261005
N_GENES = 9554
EXPECTED_STATES = ['AML_STATE_01', 'AML_STATE_02', 'AML_STATE_03', 'AML_STATE_04', 'AML_STATE_05', 'AML_STATE_06']
GSEA_PERMUTATIONS = 1000
MIN_GENESET_SIZE = 15
MAX_GENESET_SIZE = 500
FDR_THRESHOLD = 0.05
PROJECT = Path(_release_os.environ.get('AML_GRAPHRX_ROOT', '/content/drive/MyDrive/AML_GraphRx'))
AUDIT = PROJECT / 'results' / 'audit'
F493 = AUDIT / '493_R5G_state_gene_meta_statistics.csv'
F494 = AUDIT / '494_R5G_state_graph_coherence.csv'
F508 = AUDIT / '508_R5V_state_disease_linked_genes.csv'
F509 = AUDIT / '509_R5V_state_disease_linked_summary.csv'
F511 = AUDIT / '511_R5V_final_summary.csv'
F512 = AUDIT / '512_R5V_hard_gates.csv'
F513 = AUDIT / '513_R5V_manifest.csv'
MAPPING_FILE = AUDIT / '428_R4A_entrez_to_symbol_mapping.csv'
R4A_LIBRARY_AUDIT = AUDIT / '436_R4A_library_audit.csv'
F514 = AUDIT / '514_R5BIO_predeclared_protocol.json'
F515 = AUDIT / '515_R5BIO_mapping_and_library_audit.csv'
F516 = AUDIT / '516_R5BIO_state_Reactome_GSEA.csv'
F517 = AUDIT / '517_R5BIO_state_GO_BP_GSEA.csv'
F518 = AUDIT / '518_R5BIO_concordant_Reactome_ORA.csv'
F519 = AUDIT / '519_R5BIO_concordant_GO_BP_ORA.csv'
F520 = AUDIT / '520_R5BIO_pathway_summary.csv'
F521 = AUDIT / '521_R5BIO_gene_prioritization.csv'
F522 = AUDIT / '522_R5BIO_gene_priority_summary.csv'
F523 = AUDIT / '523_R5BIO_hard_gates.csv'
F524 = AUDIT / '524_R5BIO_manifest.csv'
EXPECTED_MAPPING_SHA = 'ba1cad3ea4a0532e000bcf62e91b54094fce6f098b03b894efe1db405baa3596'
EXPECTED_R4A_LIBRARY_AUDIT_SHA = '4a6e9e9fe550565c85dbc1128b4f5f5874291dc8415a53fae662c3f3eda04e82'

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

def csv_bytes(dataframe, sort_cols=None):
    x = dataframe.copy()
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

def bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])

def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p, kind='mergesort')
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1, dtype=float)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    result = np.empty_like(q)
    result[order] = q
    return result

def normalize_gene_id(x):
    x = pd.Series(x).astype(str).str.strip().str.replace('\\.0$', '', regex=True)
    return x
for path in [F493, F508, F509, F511, F512, F513, MAPPING_FILE, R4A_LIBRARY_AUDIT]:
    require(path.is_file(), f'Missing required artifact: {path}')
require(sha256_file(MAPPING_FILE) == EXPECTED_MAPPING_SHA, 'R4A gene mapping SHA changed.')
require(sha256_file(R4A_LIBRARY_AUDIT) == EXPECTED_R4A_LIBRARY_AUDIT_SHA, 'R4A library audit SHA changed.')
r5v_manifest = pd.read_csv(F513)
for path in [F508, F509, F511, F512]:
    rows = r5v_manifest.loc[r5v_manifest['path'].astype(str).map(lambda x: Path(x).name).eq(path.name)]
    require(len(rows) == 1, f'{path.name} missing from R5-3 manifest.')
    require(sha256_file(path) == str(rows['sha256'].iloc[0]), f'R5-3 parent artifact SHA mismatch: {path.name}')
print('PASS: frozen R5-3 chain verified.')
r5v_summary = pd.read_csv(F511)
validation_value = str(r5v_summary.loc[r5v_summary['metric'].eq('STATE_ATLAS_VALIDATION_SUPPORTED'), 'value'].iloc[0]).strip().lower()
require(validation_value == 'false', 'This R5-4A protocol is specifically the post-negative-transportability biological consolidation branch.')
print('PASS: R5-3 transportability result preserved as NEGATIVE.')

def locate_unique_snapshot(filename):
    preferred_locations = [PROJECT / 'results' / 'external_snapshots' / 'R4A_20261002' / filename, PROJECT / 'data' / 'external_snapshots' / 'R4A_20261002' / filename, PROJECT / 'external_snapshots' / 'R4A_20261002' / filename]
    for candidate in preferred_locations:
        if candidate.is_file():
            return candidate
    matches = list(PROJECT.rglob(filename))
    require(len(matches) == 1, f'Expected exactly one frozen {filename}; found {len(matches)}:\n' + '\n'.join((str(x) for x in matches)))
    return matches[0]
REACTOME_GMT = locate_unique_snapshot('Reactome_2022.raw.gmt')
GO_BP_GMT = locate_unique_snapshot('GO_Biological_Process_2025.raw.gmt')
print('Reactome snapshot:', REACTOME_GMT)
print('GO BP snapshot:', GO_BP_GMT)
REACTOME_GMT_SHA = sha256_file(REACTOME_GMT)
GO_BP_GMT_SHA = sha256_file(GO_BP_GMT)
state_meta = pd.read_csv(F493, dtype={'canonical_NCBI_Gene_ID': str})
state_disease = pd.read_csv(F508, dtype={'canonical_NCBI_Gene_ID': str})
require(len(state_meta) > 0, 'State meta table empty.')
require(len(state_disease) > 0, 'Disease-linked state table empty.')
state_disease['CONCORDANT_DISEASE_LINKED_STATE_GENE'] = bool_series(state_disease['CONCORDANT_DISEASE_LINKED_STATE_GENE'])
state_disease['DISEASE_LINKED_STATE_GENE'] = bool_series(state_disease['DISEASE_LINKED_STATE_GENE'])
observed_states = sorted(state_meta['state_id'].unique())
require(observed_states == EXPECTED_STATES, f'Unexpected state identities: {observed_states}')
mapping_raw = pd.read_csv(MAPPING_FILE, dtype=str, keep_default_na=False)
universe_gene_ids = set(normalize_gene_id(state_meta.loc[state_meta['state_id'].eq(EXPECTED_STATES[0]), 'canonical_NCBI_Gene_ID']).tolist())
require(len(universe_gene_ids) == N_GENES, 'Expected 9554 unique GeneIDs in state universe.')
id_candidate_scores = []
for column in mapping_raw.columns:
    normalized = normalize_gene_id(mapping_raw[column])
    coverage = len(universe_gene_ids & set(normalized.tolist()))
    id_candidate_scores.append((coverage, column))
id_candidate_scores.sort(reverse=True)
best_id_coverage, ID_COLUMN = id_candidate_scores[0]
require(best_id_coverage == N_GENES, 'Could not identify full-coverage GeneID mapping column.')
mapping_raw['_GENE_ID_NORMALIZED'] = normalize_gene_id(mapping_raw[ID_COLUMN])
mapping_target = mapping_raw.loc[mapping_raw['_GENE_ID_NORMALIZED'].isin(universe_gene_ids)].copy()
symbol_candidates = [c for c in mapping_target.columns if c not in {ID_COLUMN, '_GENE_ID_NORMALIZED'} and 'symbol' in c.lower()]
if len(symbol_candidates) == 0:
    for c in mapping_target.columns:
        if c in {ID_COLUMN, '_GENE_ID_NORMALIZED'}:
            continue
        values = mapping_target[c].astype(str).str.strip()
        if values.ne('').sum() >= N_GENES and values.nunique() >= N_GENES:
            symbol_candidates.append(c)
require(len(symbol_candidates) >= 1, 'Could not identify symbol column.')
SYMBOL_COLUMN = symbol_candidates[0]
mapping_target['_SYMBOL'] = mapping_target[SYMBOL_COLUMN].astype(str).str.strip()
mapping_target = mapping_target[['_GENE_ID_NORMALIZED', '_SYMBOL']].copy()
duplicate_check = mapping_target.groupby('_GENE_ID_NORMALIZED')['_SYMBOL'].nunique()
require(int((duplicate_check > 1).sum()) == 0, 'Conflicting symbols found for same GeneID.')
mapping_target = mapping_target.drop_duplicates('_GENE_ID_NORMALIZED', keep='first').reset_index(drop=True)
require(len(mapping_target) == N_GENES, f'Mapping coverage is {len(mapping_target)} instead of {N_GENES}.')
require(mapping_target['_SYMBOL'].nunique() == N_GENES, 'Expected 9554 unique symbols.')
ID_TO_SYMBOL = dict(zip(mapping_target['_GENE_ID_NORMALIZED'], mapping_target['_SYMBOL']))
UNIVERSE_SYMBOLS = set(ID_TO_SYMBOL.values())

def attach_symbols(df):
    x = df.copy()
    x['_GENE_ID_NORMALIZED'] = normalize_gene_id(x['canonical_NCBI_Gene_ID'])
    x['gene_symbol'] = x['_GENE_ID_NORMALIZED'].map(ID_TO_SYMBOL)
    require(x['gene_symbol'].notna().all(), 'Gene symbol mapping failure.')
    return x
state_meta = attach_symbols(state_meta)
state_disease = attach_symbols(state_disease)

def read_gmt_lines(path, gene_start_index):
    result = {}
    with Path(path).open('r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            parts = line.rstrip('\n\r').split('\t')
            if len(parts) <= gene_start_index:
                continue
            term = parts[0].strip()
            genes = {x.strip() for x in parts[gene_start_index:] if x.strip()}
            result[term] = genes
    return result

def effective_sets(gene_sets):
    result = {}
    for term, genes in gene_sets.items():
        genes_in_universe = set(genes) & UNIVERSE_SYMBOLS
        if MIN_GENESET_SIZE <= len(genes_in_universe) <= MAX_GENESET_SIZE:
            result[term] = genes_in_universe
    return result

def load_frozen_library(path, expected_raw, expected_effective):
    candidates = []
    for start_index in [2, 1]:
        raw = read_gmt_lines(path, start_index)
        eff = effective_sets(raw)
        candidates.append((start_index, raw, eff))
    matching = [x for x in candidates if len(x[1]) == expected_raw and len(x[2]) == expected_effective]
    require(len(matching) >= 1, f'Frozen GMT {path.name} did not reproduce historical counts. Observed candidates={[(len(x[1]), len(x[2])) for x in candidates]}')
    matching = sorted(matching, key=lambda x: 0 if x[0] == 2 else 1)
    return matching[0]
REACTOME_GENE_START, REACTOME_RAW, REACTOME_EFFECTIVE = load_frozen_library(REACTOME_GMT, expected_raw=1818, expected_effective=1024)
GO_GENE_START, GO_RAW, GO_EFFECTIVE = load_frozen_library(GO_BP_GMT, expected_raw=5343, expected_effective=2006)
print()
print('Reactome raw/effective:', len(REACTOME_RAW), len(REACTOME_EFFECTIVE))
print('GO BP raw/effective:', len(GO_RAW), len(GO_EFFECTIVE))
protocol = {'protocol_id': 'AML_GRAPHRX_R5_4A_BIOLOGICAL_CONSOLIDATION_V1', 'parent_R5_3_manifest_sha256': sha256_file(F513), 'R5_3_transportability_supported': False, 'all_six_states_analyzed': EXPECTED_STATES, 'state_selection_based_on_R5_3_performance': False, 'GNN_training': False, 'classifier_training': False, 'reclustering': False, 'threshold_tuning': False, 'top_N_gene_selection': False, 'pathway_download_during_R5_4A': False, 'mapping_source': str(MAPPING_FILE), 'mapping_sha256': sha256_file(MAPPING_FILE), 'Reactome_snapshot': str(REACTOME_GMT), 'Reactome_snapshot_sha256': REACTOME_GMT_SHA, 'GO_BP_snapshot': str(GO_BP_GMT), 'GO_BP_snapshot_sha256': GO_BP_GMT_SHA, 'GSEApy_version': str(gp.__version__), 'GSEA': {'ranking': 'all 9554 genes ranked by frozen R5-2G state meta_Z', 'permutations': GSEA_PERMUTATIONS, 'min_size': MIN_GENESET_SIZE, 'max_size': MAX_GENESET_SIZE, 'FDR_threshold': FDR_THRESHOLD}, 'ORA': {'input': 'CONCORDANT_DISEASE_LINKED_STATE_GENE=True', 'direction_split': 'UP and DOWN analyzed separately', 'universe': 'all 9554 mapped genes', 'test': 'one-sided hypergeometric', 'multiple_testing': 'BH separately by state x direction x library', 'FDR_threshold': FDR_THRESHOLD}, 'gene_prioritization': 'all concordant disease-linked genes retained; pathway-support counts added descriptively; no top-N inferential cutoff', 'high_confidence_claim': 'reserved until actual DepMap validation in R5-4B'}
write_once_or_verify(F514, json_bytes(protocol))
mapping_library_audit = pd.DataFrame([{'item': 'mapped_GeneIDs', 'observed': len(mapping_target), 'expected': 9554, 'status': 'PASS'}, {'item': 'unique_gene_symbols', 'observed': len(UNIVERSE_SYMBOLS), 'expected': 9554, 'status': 'PASS'}, {'item': 'Reactome_raw_terms', 'observed': len(REACTOME_RAW), 'expected': 1818, 'status': 'PASS'}, {'item': 'Reactome_effective_terms', 'observed': len(REACTOME_EFFECTIVE), 'expected': 1024, 'status': 'PASS'}, {'item': 'GO_BP_raw_terms', 'observed': len(GO_RAW), 'expected': 5343, 'status': 'PASS'}, {'item': 'GO_BP_effective_terms', 'observed': len(GO_EFFECTIVE), 'expected': 2006, 'status': 'PASS'}])
write_once_or_verify(F515, csv_bytes(mapping_library_audit, ['item']))

def find_result_column(dataframe, names):
    lookup = {str(c).strip().lower(): c for c in dataframe.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None

def normalize_gsea_results(result, state_id, library_name):
    x = result.copy()
    if x.index.name is not None or not isinstance(x.index, pd.RangeIndex):
        x = x.reset_index()
    term_col = find_result_column(x, ['Term', 'term'])
    nes_col = find_result_column(x, ['NES'])
    es_col = find_result_column(x, ['ES'])
    nom_col = find_result_column(x, ['NOM p-val', 'NOM p_value', 'NOM_p_value'])
    fdr_col = find_result_column(x, ['FDR q-val', 'FDR q_value', 'FDR_q_value'])
    lead_col = find_result_column(x, ['Lead_genes', 'Lead genes', 'Leading_edge_genes'])
    require(term_col is not None, 'GSEA result lacks Term.')
    require(nes_col is not None, 'GSEA result lacks NES.')
    require(fdr_col is not None, 'GSEA result lacks FDR.')
    output = pd.DataFrame({'state_id': state_id, 'library': library_name, 'Term': x[term_col].astype(str), 'ES': pd.to_numeric(x[es_col], errors='coerce') if es_col is not None else np.nan, 'NES': pd.to_numeric(x[nes_col], errors='coerce'), 'NOM_p_value': pd.to_numeric(x[nom_col], errors='coerce') if nom_col is not None else np.nan, 'FDR_q_value': pd.to_numeric(x[fdr_col], errors='coerce'), 'Leading_edge_genes': x[lead_col].fillna('').astype(str) if lead_col is not None else ''})
    output['direction'] = np.where(output['NES'] > 0, 'UP_IN_STATE', 'DOWN_IN_STATE')
    output['SIGNIFICANT_FDR_LT_0_05'] = output['FDR_q_value'] < FDR_THRESHOLD
    return output
reactome_gsea_frames = []
go_gsea_frames = []
for state_number, state_id in enumerate(EXPECTED_STATES, start=1):
    print()
    print('=' * 90)
    print('GSEA:', state_id)
    print('=' * 90)
    x = state_meta.loc[state_meta['state_id'].eq(state_id)].copy()
    require(len(x) == N_GENES, f'{state_id}: expected 9554 ranked genes; observed {len(x)}')
    require(x['gene_symbol'].nunique() == N_GENES, f'{state_id}: duplicate symbols.')
    ranked = x[['gene_symbol', 'meta_Z']].rename(columns={'gene_symbol': 'gene', 'meta_Z': 'score'})
    ranked['score'] = pd.to_numeric(ranked['score'], errors='raise')
    ranked = ranked.sort_values(['score', 'gene'], ascending=[False, True], kind='mergesort').reset_index(drop=True)
    pre_reactome = gp.prerank(rnk=ranked, gene_sets=REACTOME_EFFECTIVE, min_size=MIN_GENESET_SIZE, max_size=MAX_GENESET_SIZE, permutation_num=GSEA_PERMUTATIONS, seed=SEED + state_number, outdir=None, threads=4, verbose=False)
    reactome_result = normalize_gsea_results(pre_reactome.res2d, state_id, 'Reactome_2022')
    reactome_gsea_frames.append(reactome_result)
    pre_go = gp.prerank(rnk=ranked, gene_sets=GO_EFFECTIVE, min_size=MIN_GENESET_SIZE, max_size=MAX_GENESET_SIZE, permutation_num=GSEA_PERMUTATIONS, seed=SEED + 100 + state_number, outdir=None, threads=4, verbose=False)
    go_result = normalize_gsea_results(pre_go.res2d, state_id, 'GO_Biological_Process_2025')
    go_gsea_frames.append(go_result)
    print(state_id, 'Reactome FDR<0.05 =', int(reactome_result['SIGNIFICANT_FDR_LT_0_05'].sum()), '| GO BP FDR<0.05 =', int(go_result['SIGNIFICANT_FDR_LT_0_05'].sum()))
reactome_gsea = pd.concat(reactome_gsea_frames, ignore_index=True)
go_gsea = pd.concat(go_gsea_frames, ignore_index=True)
write_once_or_verify(F516, csv_bytes(reactome_gsea, ['state_id', 'FDR_q_value', 'Term']))
write_once_or_verify(F517, csv_bytes(go_gsea, ['state_id', 'FDR_q_value', 'Term']))

def run_ora(selected_symbols, gene_sets, state_id, direction, library_name):
    selected_symbols = set(selected_symbols) & UNIVERSE_SYMBOLS
    N = len(selected_symbols)
    M = len(UNIVERSE_SYMBOLS)
    rows = []
    for term, pathway_genes in gene_sets.items():
        pathway_genes = pathway_genes & UNIVERSE_SYMBOLS
        K = len(pathway_genes)
        overlap = sorted(selected_symbols & pathway_genes)
        k = len(overlap)
        p_value = float(hypergeom.sf(k - 1, M, K, N))
        if N > 0 and K > 0:
            fold_enrichment = k / N / (K / M)
        else:
            fold_enrichment = np.nan
        rows.append({'state_id': state_id, 'direction': direction, 'library': library_name, 'Term': term, 'Input_genes_n': N, 'GeneSet_n': K, 'Overlap_n': k, 'Fold_enrichment': float(fold_enrichment), 'P_value': p_value, 'Overlap_genes': ';'.join(overlap)})
    result = pd.DataFrame(rows)
    result['FDR_BH'] = bh_fdr(result['P_value'].to_numpy(dtype=float))
    result['SIGNIFICANT_FDR_LT_0_05'] = result['FDR_BH'] < FDR_THRESHOLD
    return result
reactome_ora_frames = []
go_ora_frames = []
for state_id in EXPECTED_STATES:
    x = state_disease.loc[state_disease['state_id'].eq(state_id) & state_disease['CONCORDANT_DISEASE_LINKED_STATE_GENE']].copy()
    state_z = pd.to_numeric(x['meta_Z'], errors='raise')
    x['_DIRECTION'] = np.where(state_z > 0, 'UP', 'DOWN')
    for direction in ['UP', 'DOWN']:
        genes = x.loc[x['_DIRECTION'].eq(direction), 'gene_symbol'].tolist()
        reactome_ora_frames.append(run_ora(genes, REACTOME_EFFECTIVE, state_id, direction, 'Reactome_2022'))
        go_ora_frames.append(run_ora(genes, GO_EFFECTIVE, state_id, direction, 'GO_Biological_Process_2025'))
reactome_ora = pd.concat(reactome_ora_frames, ignore_index=True)
go_ora = pd.concat(go_ora_frames, ignore_index=True)
write_once_or_verify(F518, csv_bytes(reactome_ora, ['state_id', 'direction', 'FDR_BH', 'Term']))
write_once_or_verify(F519, csv_bytes(go_ora, ['state_id', 'direction', 'FDR_BH', 'Term']))
pathway_summary_rows = []
for state_id in EXPECTED_STATES:
    row = {'state_id': state_id}
    for name, table in [('Reactome', reactome_gsea), ('GO_BP', go_gsea)]:
        temp = table.loc[table['state_id'].eq(state_id)]
        row[f'{name}_GSEA_sig_n'] = int(temp['SIGNIFICANT_FDR_LT_0_05'].sum())
        row[f'{name}_GSEA_UP_sig_n'] = int((temp['SIGNIFICANT_FDR_LT_0_05'] & temp['direction'].eq('UP_IN_STATE')).sum())
        row[f'{name}_GSEA_DOWN_sig_n'] = int((temp['SIGNIFICANT_FDR_LT_0_05'] & temp['direction'].eq('DOWN_IN_STATE')).sum())
    for name, table in [('Reactome', reactome_ora), ('GO_BP', go_ora)]:
        temp = table.loc[table['state_id'].eq(state_id)]
        for direction in ['UP', 'DOWN']:
            temp_direction = temp.loc[temp['direction'].eq(direction)]
            row[f'{name}_ORA_{direction}_sig_n'] = int(temp_direction['SIGNIFICANT_FDR_LT_0_05'].sum())
    pathway_summary_rows.append(row)
pathway_summary = pd.DataFrame(pathway_summary_rows)
write_once_or_verify(F520, csv_bytes(pathway_summary, ['state_id']))
ora_support = defaultdict(lambda: {'Reactome_ORA_sig_terms': 0, 'GO_BP_ORA_sig_terms': 0})
for library_name, table, key_name in [('Reactome', reactome_ora, 'Reactome_ORA_sig_terms'), ('GO_BP', go_ora, 'GO_BP_ORA_sig_terms')]:
    significant = table.loc[table['SIGNIFICANT_FDR_LT_0_05']]
    for row in significant.itertuples(index=False):
        genes = [x for x in str(row.Overlap_genes).split(';') if x]
        for gene in genes:
            ora_support[row.state_id, gene][key_name] += 1
gsea_support = defaultdict(lambda: {'Reactome_GSEA_leading_edge_sig_terms': 0, 'GO_BP_GSEA_leading_edge_sig_terms': 0})
for table, key_name in [(reactome_gsea, 'Reactome_GSEA_leading_edge_sig_terms'), (go_gsea, 'GO_BP_GSEA_leading_edge_sig_terms')]:
    significant = table.loc[table['SIGNIFICANT_FDR_LT_0_05']]
    for row in significant.itertuples(index=False):
        genes = [x.strip() for x in str(row.Leading_edge_genes).replace(',', ';').split(';') if x.strip()]
        for gene in genes:
            gsea_support[row.state_id, gene][key_name] += 1
priority = state_disease.loc[state_disease['CONCORDANT_DISEASE_LINKED_STATE_GENE']].copy()
priority['state_meta_Z'] = pd.to_numeric(priority['meta_Z'], errors='raise')
priority['AML_control_meta_Z'] = pd.to_numeric(priority['AML_control_meta_Z'], errors='raise')
priority['joint_min_abs_Z'] = np.minimum(np.abs(priority['state_meta_Z']), np.abs(priority['AML_control_meta_Z']))
reactome_ora_count = []
go_ora_count = []
reactome_gsea_count = []
go_gsea_count = []
for row in priority.itertuples(index=False):
    key = (row.state_id, row.gene_symbol)
    reactome_ora_count.append(ora_support[key]['Reactome_ORA_sig_terms'])
    go_ora_count.append(ora_support[key]['GO_BP_ORA_sig_terms'])
    reactome_gsea_count.append(gsea_support[key]['Reactome_GSEA_leading_edge_sig_terms'])
    go_gsea_count.append(gsea_support[key]['GO_BP_GSEA_leading_edge_sig_terms'])
priority['Reactome_ORA_sig_term_count'] = reactome_ora_count
priority['GO_BP_ORA_sig_term_count'] = go_ora_count
priority['Reactome_GSEA_leading_edge_sig_term_count'] = reactome_gsea_count
priority['GO_BP_GSEA_leading_edge_sig_term_count'] = go_gsea_count
priority['total_significant_pathway_support_count'] = priority['Reactome_ORA_sig_term_count'] + priority['GO_BP_ORA_sig_term_count'] + priority['Reactome_GSEA_leading_edge_sig_term_count'] + priority['GO_BP_GSEA_leading_edge_sig_term_count']
priority['BIOLOGICALLY_CONSOLIDATED_PRE_DEPMAP'] = priority['total_significant_pathway_support_count'] > 0
priority['priority_stage'] = 'PRE_DEPMAP'
priority = priority.sort_values(['state_id', 'BIOLOGICALLY_CONSOLIDATED_PRE_DEPMAP', 'total_significant_pathway_support_count', 'joint_min_abs_Z', 'gene_symbol'], ascending=[True, False, False, False, True], kind='mergesort').reset_index(drop=True)
write_once_or_verify(F521, csv_bytes(priority))
gene_priority_summary_rows = []
for state_id in EXPECTED_STATES:
    x = priority.loc[priority['state_id'].eq(state_id)]
    gene_priority_summary_rows.append({'state_id': state_id, 'concordant_disease_linked_genes_n': len(x), 'biologically_consolidated_pre_DepMap_n': int(x['BIOLOGICALLY_CONSOLIDATED_PRE_DEPMAP'].sum()), 'genes_with_Reactome_ORA_support_n': int((x['Reactome_ORA_sig_term_count'] > 0).sum()), 'genes_with_GO_BP_ORA_support_n': int((x['GO_BP_ORA_sig_term_count'] > 0).sum()), 'genes_in_significant_Reactome_GSEA_leading_edges_n': int((x['Reactome_GSEA_leading_edge_sig_term_count'] > 0).sum()), 'genes_in_significant_GO_BP_GSEA_leading_edges_n': int((x['GO_BP_GSEA_leading_edge_sig_term_count'] > 0).sum())})
gene_priority_summary = pd.DataFrame(gene_priority_summary_rows)
write_once_or_verify(F522, csv_bytes(gene_priority_summary, ['state_id']))
gate_rows = []

def add_gate(gate, passed, detail):
    gate_rows.append({'gate': gate, 'status': 'PASS' if bool(passed) else 'FAIL', 'detail': str(detail)})
add_gate('R5_3_negative_transportability_preserved', validation_value == 'false', validation_value)
add_gate('exact_six_states_analyzed', observed_states == EXPECTED_STATES, observed_states)
add_gate('all_9554_GeneIDs_mapped', len(mapping_target) == 9554, len(mapping_target))
add_gate('all_9554_symbols_unique', len(UNIVERSE_SYMBOLS) == 9554, len(UNIVERSE_SYMBOLS))
add_gate('Reactome_historical_raw_term_count', len(REACTOME_RAW) == 1818, len(REACTOME_RAW))
add_gate('Reactome_historical_effective_term_count', len(REACTOME_EFFECTIVE) == 1024, len(REACTOME_EFFECTIVE))
add_gate('GO_BP_historical_raw_term_count', len(GO_RAW) == 5343, len(GO_RAW))
add_gate('GO_BP_historical_effective_term_count', len(GO_EFFECTIVE) == 2006, len(GO_EFFECTIVE))
add_gate('no_state_selected_by_R5_3_AUROC', True, False)
add_gate('no_GNN_training', True, False)
add_gate('no_classifier_training', True, False)
add_gate('no_reclustering', True, False)
add_gate('no_threshold_tuning', True, False)
add_gate('no_topN_gene_selection', True, False)
add_gate('no_new_pathway_download', True, False)
hard_gates = pd.DataFrame(gate_rows)
require(hard_gates['status'].eq('PASS').all(), 'R5-4A HARD GATE FAILURE:\n' + hard_gates.loc[hard_gates['status'].ne('PASS')].to_string(index=False))
write_once_or_verify(F523, csv_bytes(hard_gates, ['gate']))
manifest_rows = []

def add_manifest(role, path):
    path = Path(path)
    require(path.is_file(), f'Manifest artifact missing: {path}')
    manifest_rows.append({'role': role, 'path': str(path), 'size_bytes': int(path.stat().st_size), 'sha256': sha256_file(path), 'protocol_id': protocol['protocol_id']})
for role, path in [('parent_state_gene_meta', F493), ('parent_state_disease_linked_genes', F508), ('parent_state_disease_summary', F509), ('parent_R5_3_summary', F511), ('parent_R5_3_hard_gates', F512), ('parent_R5_3_manifest', F513), ('R4A_mapping', MAPPING_FILE), ('R4A_library_audit', R4A_LIBRARY_AUDIT), ('Reactome_snapshot', REACTOME_GMT), ('GO_BP_snapshot', GO_BP_GMT), ('protocol', F514), ('mapping_library_audit', F515), ('Reactome_GSEA', F516), ('GO_BP_GSEA', F517), ('Reactome_ORA', F518), ('GO_BP_ORA', F519), ('pathway_summary', F520), ('gene_prioritization', F521), ('gene_priority_summary', F522), ('hard_gates', F523)]:
    add_manifest(role, path)
manifest = pd.DataFrame(manifest_rows)
write_once_or_verify(F524, csv_bytes(manifest, ['role']))
print()
print('=' * 105)
print('R5-4A — PATHWAY SUMMARY')
print('=' * 105)
print(pathway_summary.to_string(index=False))
print()
print('=' * 105)
print('R5-4A — GENE PRIORITY SUMMARY')
print('=' * 105)
print(gene_priority_summary.to_string(index=False))
print()
print('=' * 105)
print('R5-4A — TOP 15 PRE-DEPMAP GENES PER STATE')
print('=' * 105)
display_columns = ['state_id', 'gene_symbol', 'state_meta_Z', 'AML_control_meta_Z', 'joint_min_abs_Z', 'total_significant_pathway_support_count', 'BIOLOGICALLY_CONSOLIDATED_PRE_DEPMAP']
for state_id in EXPECTED_STATES:
    print()
    print(state_id)
    print(priority.loc[priority['state_id'].eq(state_id), display_columns].head(15).to_string(index=False))
print()
print('=' * 105)
print('R5-4A — HARD GATES')
print('=' * 105)
print(hard_gates.to_string(index=False))
print()
print('R5-4A COMPLETE ✅')
print('NO MODEL / CLUSTER / STATE RETUNING PERFORMED.')
print('NEXT = R5-4B OFFICIAL DEPMAP FUNCTIONAL VALIDATION')
