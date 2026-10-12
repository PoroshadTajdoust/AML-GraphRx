# Manuscript-linked analysis: S03/S06 expression and shared-pathway contrast only
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
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, hypergeom
import torch
SEED = 20261006
N_PERM = 1000
N_DEGREE_BINS = 10
STATE3 = 'AML_STATE_03'
STATE6 = 'AML_STATE_06'
EXPECTED_N_GENES = 9554
EXPECTED_ROBUST_AML_GENES = 3901
EXPECTED_STATE3_CONCORDANT = 348
EXPECTED_STATE6_CONCORDANT = 476
EXPECTED_TOPOLOGY_SHA = '97e214d5eea1b2e04125d4fcd0f5a0703f1f67c6fde02316eb51d2b3f0f7f9e7'
EXPECTED_SOURCE_BIOLOGICAL_EDGES = 95728
EXPECTED_SOURCE_SELF_LOOPS = 4
EXPECTED_ANALYSIS_NONSELF_EDGES = 95724
EXPECTED_MESSAGE_ARCS_ALLOWED = {95728, 116866}
EXPECTED_RELATIONS = {'signaling', 'PPI', 'TF_regulation', 'kinase_substrate'}
PROJECT = Path(_release_os.environ.get('AML_GRAPHRX_ROOT', '/content/drive/MyDrive/AML_GraphRx'))
AUDIT = PROJECT / 'results' / 'audit'
STEP5 = PROJECT / 'data' / 'patient_graphs' / '20260823T123040Z'
F428 = AUDIT / '428_R4A_entrez_to_symbol_mapping.csv'
F493 = AUDIT / '493_R5G_state_gene_meta_statistics.csv'
F507 = AUDIT / '507_R5V_AML_control_gene_meta_statistics.csv'
F508 = AUDIT / '508_R5V_state_disease_linked_genes.csv'
F513 = AUDIT / '513_R5V_manifest.csv'
F516 = AUDIT / '516_R5BIO_state_Reactome_GSEA.csv'
F517 = AUDIT / '517_R5BIO_state_GO_BP_GSEA.csv'
F521 = AUDIT / '521_R5BIO_gene_prioritization.csv'
F523 = AUDIT / '523_R5BIO_hard_gates.csv'
F524 = AUDIT / '524_R5BIO_manifest.csv'
F525 = AUDIT / '525_R5BR_predeclared_protocol.json'
F525A = AUDIT / '525A_R5BR_self_loop_technical_amendment.json'
F526 = AUDIT / '526_R5BR_topology_audit.csv'
F527 = AUDIT / '527_R5BR_state03_state06_gene_contrast.csv'
F528 = AUDIT / '528_R5BR_shared_pathway_analysis.csv'
F529 = AUDIT / '529_R5BR_cross_state_graph_edges.csv'
F530 = AUDIT / '530_R5BR_bridge_candidates.csv'
F531 = AUDIT / '531_R5BR_degree_matched_permutation.csv'
F532 = AUDIT / '532_R5BR_final_summary.csv'
F533 = AUDIT / '533_R5BR_hard_gates.csv'
F534 = AUDIT / '534_R5BR_manifest.csv'

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    path = Path(path)
    require(path.is_file(), f'Missing file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
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

def bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])

def normalize_gene_id(series):
    return series.astype(str).str.strip().str.replace('\\.0$', '', regex=True)

def parse_gene_list(x):
    if pd.isna(x):
        return set()
    value = str(x).strip()
    if not value:
        return set()
    return {gene.strip() for gene in value.replace(',', ';').split(';') if gene.strip()}

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
for path in [F428, F493, F507, F508, F513, F516, F517, F521, F523, F524]:
    require(path.is_file(), f'Missing required artifact: {path}')
manifest_r5v = pd.read_csv(F513)
manifest_r5bio = pd.read_csv(F524)

def verify_in_manifest(path, manifest):
    rows = manifest.loc[manifest['path'].astype(str).map(lambda x: Path(x).name).eq(path.name)]
    require(len(rows) == 1, f'{path.name} missing from parent manifest.')
    expected = str(rows['sha256'].iloc[0])
    observed = sha256_file(path)
    require(observed == expected, f'SHA mismatch: {path.name}')
for path in [F507, F508]:
    verify_in_manifest(path, manifest_r5v)
for path in [F493, F516, F517, F521, F523]:
    verify_in_manifest(path, manifest_r5bio)
print('PASS: frozen R5 input chains verified.')
protocol = {'protocol_id': 'AML_GRAPHRX_R5_4A2_STATE03_STATE06_BRIDGE_V1', 'analysis_type': 'targeted exploratory mechanistic follow-up', 'comparison': [STATE3, STATE6], 'reason_for_comparison': 'State03 was recurrent under the frozen R5-3 validation assignment procedure whereas State06 was not recovered.', 'independent_validation_claim': False, 'retraining': False, 'reclustering': False, 'threshold_tuning': False, 'DepMap_used': False, 'primary_network_test': 'cross-state biological-edge count between exclusive State03 and State06 concordant disease-linked genes', 'network_null': '1000 degree-decile-matched permutations restricted to the 3901 robust AML genes', 'candidate_definition': 'descriptive graph connectors only; no candidate is declared causal or functionally essential at this stage', 'directed_upstream_relations': ['signaling', 'TF_regulation', 'kinase_substrate'], 'PPI_treated_as': 'undirected', 'no_topN_inference': True, 'next_and_final_validation': 'DepMap CRISPR dependency'}
write_once_or_verify(F525, json_bytes(protocol))
technical_amendment = {'amendment_id': 'AML_GRAPHRX_R5_4A2_SELF_LOOP_TECHNICAL_AMENDMENT_V1', 'parent_protocol': protocol['protocol_id'], 'reason': 'Initial implementation removed self-loops before reconstructing the historical biological source graph, yielding 95724 rather than the locked 95728 source edges.', 'observed_before_any_bridge_result': True, 'scientific_threshold_changed': False, 'candidate_rule_changed': False, 'null_model_changed': False, 'state_definition_changed': False, 'correction': 'Preserve self-loops while auditing the locked source biological graph; exclude self-loops only for cross-state bridge statistics.', 'expected_locked_source_edges': EXPECTED_SOURCE_BIOLOGICAL_EDGES, 'expected_source_self_loops': EXPECTED_SOURCE_SELF_LOOPS, 'expected_bridge_analysis_nonself_edges': EXPECTED_ANALYSIS_NONSELF_EDGES, 'claim_boundary': 'This is an implementation/audit correction and not an outcome-driven analytical modification.'}
write_once_or_verify(F525A, json_bytes(technical_amendment))
state_meta = pd.read_csv(F493, dtype={'canonical_NCBI_Gene_ID': str})
aml_meta = pd.read_csv(F507, dtype={'canonical_NCBI_Gene_ID': str})
state_disease = pd.read_csv(F508, dtype={'canonical_NCBI_Gene_ID': str})
priority = pd.read_csv(F521, dtype={'canonical_NCBI_Gene_ID': str})
reactome = pd.read_csv(F516)
gobp = pd.read_csv(F517)
for df, column in [(state_meta, 'STATE_SIGNATURE_GENE'), (aml_meta, 'ROBUST_AML_GENE'), (state_disease, 'CONCORDANT_DISEASE_LINKED_STATE_GENE'), (priority, 'BIOLOGICALLY_CONSOLIDATED_PRE_DEPMAP')]:
    if column in df.columns:
        df[column] = bool_series(df[column])
mapping_raw = pd.read_csv(F428, dtype=str, keep_default_na=False)
state3_all = state_meta.loc[state_meta['state_id'].eq(STATE3)].copy()
require(len(state3_all) == EXPECTED_N_GENES, 'State03 does not contain 9554 genes.')
universe_ids = set(normalize_gene_id(state3_all['canonical_NCBI_Gene_ID']))
candidate_id_columns = []
for column in mapping_raw.columns:
    values = set(normalize_gene_id(mapping_raw[column]))
    coverage = len(universe_ids & values)
    candidate_id_columns.append((coverage, column))
candidate_id_columns.sort(reverse=True)
coverage, ID_COL = candidate_id_columns[0]
require(coverage == EXPECTED_N_GENES, 'Could not identify full-coverage GeneID mapping column.')
mapping_raw['_gid'] = normalize_gene_id(mapping_raw[ID_COL])
mapping_subset = mapping_raw.loc[mapping_raw['_gid'].isin(universe_ids)].copy()
symbol_candidates = [c for c in mapping_subset.columns if 'symbol' in c.lower() and c != ID_COL]
require(len(symbol_candidates) >= 1, 'Could not identify symbol column.')
SYMBOL_COL = symbol_candidates[0]
mapping_subset['_symbol'] = mapping_subset[SYMBOL_COL].astype(str).str.strip()
symbol_conflict = mapping_subset.groupby('_gid')['_symbol'].nunique()
require(int((symbol_conflict > 1).sum()) == 0, 'Conflicting symbols for same GeneID.')
mapping_subset = mapping_subset[['_gid', '_symbol']].drop_duplicates('_gid', keep='first').reset_index(drop=True)
require(len(mapping_subset) == EXPECTED_N_GENES, f'Gene mapping coverage changed: {len(mapping_subset)}')
require(mapping_subset['_symbol'].nunique() == EXPECTED_N_GENES, 'Expected 9554 unique symbols.')
id_to_symbol = dict(zip(mapping_subset['_gid'], mapping_subset['_symbol']))

def add_symbol(df):
    x = df.copy()
    x['_gid'] = normalize_gene_id(x['canonical_NCBI_Gene_ID'])
    x['gene_symbol'] = x['_gid'].map(id_to_symbol)
    require(x['gene_symbol'].notna().all(), 'Gene-symbol mapping failure.')
    return x
state_meta = add_symbol(state_meta)
aml_meta = add_symbol(aml_meta)
s3 = state_meta.loc[state_meta['state_id'].eq(STATE3)].copy()
s6 = state_meta.loc[state_meta['state_id'].eq(STATE6)].copy()
require(len(s3) == EXPECTED_N_GENES, 'State03 gene universe changed.')
require(len(s6) == EXPECTED_N_GENES, 'State06 gene universe changed.')
keep_cols = ['gene_index_0based', 'canonical_NCBI_Gene_ID', 'gene_symbol', 'meta_Z', 'STATE_SIGNATURE_GENE']
s3 = s3[keep_cols].rename(columns={'meta_Z': 'state03_meta_Z', 'STATE_SIGNATURE_GENE': 'state03_signature'})
s6 = s6[keep_cols].rename(columns={'meta_Z': 'state06_meta_Z', 'STATE_SIGNATURE_GENE': 'state06_signature'})
gene_contrast = s3.merge(s6[['gene_index_0based', 'state06_meta_Z', 'state06_signature']], on='gene_index_0based', how='inner', validate='one_to_one')
aml_small = aml_meta[['gene_index_0based', 'AML_control_meta_Z', 'ROBUST_AML_GENE']].copy()
gene_contrast = gene_contrast.merge(aml_small, on='gene_index_0based', how='left', validate='one_to_one')
priority['gene_index_0based'] = pd.to_numeric(priority['gene_index_0based'], errors='raise').astype(int)
p3 = priority.loc[priority['state_id'].eq(STATE3)].copy()
p6 = priority.loc[priority['state_id'].eq(STATE6)].copy()
require(len(p3) == EXPECTED_STATE3_CONCORDANT, f'Expected 348 State03 concordant genes; observed={len(p3)}')
require(len(p6) == EXPECTED_STATE6_CONCORDANT, f'Expected 476 State06 concordant genes; observed={len(p6)}')
S3_SET = set(p3['gene_index_0based'])
S6_SET = set(p6['gene_index_0based'])
S3_ONLY = S3_SET - S6_SET
S6_ONLY = S6_SET - S3_SET
SHARED = S3_SET & S6_SET
gene_contrast['state03_concordant_disease_linked'] = gene_contrast['gene_index_0based'].isin(S3_SET)
gene_contrast['state06_concordant_disease_linked'] = gene_contrast['gene_index_0based'].isin(S6_SET)

def classify_gene(row):
    in3 = bool(row['state03_concordant_disease_linked'])
    in6 = bool(row['state06_concordant_disease_linked'])
    if in3 and in6:
        sign3 = np.sign(row['state03_meta_Z'])
        sign6 = np.sign(row['state06_meta_Z'])
        if sign3 == sign6:
            return 'SHARED_SAME_DIRECTION'
        else:
            return 'SHARED_OPPOSITE_DIRECTION'
    if in3:
        return 'STATE03_ONLY'
    if in6:
        return 'STATE06_ONLY'
    return 'OTHER'
gene_contrast['S03_S06_class'] = gene_contrast.apply(classify_gene, axis=1)
gene_contrast['delta_meta_Z_S03_minus_S06'] = gene_contrast['state03_meta_Z'] - gene_contrast['state06_meta_Z']
gene_contrast['state03_absZ_rank'] = np.abs(gene_contrast['state03_meta_Z']).rank(method='min', ascending=False)
gene_contrast['state06_absZ_rank'] = np.abs(gene_contrast['state06_meta_Z']).rank(method='min', ascending=False)
write_once_or_verify(F527, csv_bytes(gene_contrast, ['S03_S06_class', 'gene_index_0based']))
spearman_all = spearmanr(gene_contrast['state03_meta_Z'], gene_contrast['state06_meta_Z'])
union_mask = gene_contrast['gene_index_0based'].isin(S3_SET | S6_SET)
spearman_union = spearmanr(gene_contrast.loc[union_mask, 'state03_meta_Z'], gene_contrast.loc[union_mask, 'state06_meta_Z'])
robust_mask = aml_meta['ROBUST_AML_GENE']
robust = aml_meta.loc[robust_mask].copy()
require(len(robust) == EXPECTED_ROBUST_AML_GENES, f'Expected 3901 robust AML genes; observed={len(robust)}')
OVERLAP_K = len(SHARED)
OVERLAP_EXPECTED = len(S3_SET) * len(S6_SET) / EXPECTED_ROBUST_AML_GENES
OVERLAP_FOLD = OVERLAP_K / OVERLAP_EXPECTED
OVERLAP_HYPERGEOM_P = float(hypergeom.sf(OVERLAP_K - 1, EXPECTED_ROBUST_AML_GENES, len(S3_SET), len(S6_SET)))

def pathway_overlap(df, library_name):
    x = df.copy()
    if 'SIGNIFICANT_FDR_LT_0_05' in x.columns:
        x['SIGNIFICANT_FDR_LT_0_05'] = bool_series(x['SIGNIFICANT_FDR_LT_0_05'])
        x = x.loc[x['SIGNIFICANT_FDR_LT_0_05']]
    else:
        x = x.loc[x['FDR_q_value'] < 0.05]
    x3 = x.loc[x['state_id'].eq(STATE3)].copy()
    x6 = x.loc[x['state_id'].eq(STATE6)].copy()
    shared = x3.merge(x6, on='Term', how='inner', suffixes=('_S03', '_S06'))
    rows = []
    for row in shared.itertuples(index=False):
        nes3 = float(getattr(row, 'NES_S03'))
        nes6 = float(getattr(row, 'NES_S06'))
        if np.sign(nes3) == np.sign(nes6):
            direction_relation = 'SAME_DIRECTION'
        else:
            direction_relation = 'OPPOSITE_DIRECTION'
        lead3 = parse_gene_list(getattr(row, 'Leading_edge_genes_S03', ''))
        lead6 = parse_gene_list(getattr(row, 'Leading_edge_genes_S06', ''))
        leading_shared = sorted(lead3 & lead6)
        rows.append({'library': library_name, 'Term': getattr(row, 'Term'), 'NES_STATE03': nes3, 'NES_STATE06': nes6, 'FDR_STATE03': float(getattr(row, 'FDR_q_value_S03')), 'FDR_STATE06': float(getattr(row, 'FDR_q_value_S06')), 'direction_relationship': direction_relation, 'leading_edge_S03_n': len(lead3), 'leading_edge_S06_n': len(lead6), 'shared_leading_edge_n': len(leading_shared), 'shared_leading_edge_genes': ';'.join(leading_shared)})
    result = pd.DataFrame(rows)
    set3 = set(x3['Term'])
    set6 = set(x6['Term'])
    union = set3 | set6
    intersection = set3 & set6
    jaccard = len(intersection) / len(union) if len(union) > 0 else np.nan
    summary = {'library': library_name, 'state03_significant_terms_n': len(set3), 'state06_significant_terms_n': len(set6), 'shared_significant_terms_n': len(intersection), 'shared_term_Jaccard': jaccard, 'shared_same_direction_n': int(result['direction_relationship'].eq('SAME_DIRECTION').sum()) if len(result) > 0 else 0, 'shared_opposite_direction_n': int(result['direction_relationship'].eq('OPPOSITE_DIRECTION').sum()) if len(result) > 0 else 0}
    return (result, summary)
reactome_overlap, reactome_summary = pathway_overlap(reactome, 'Reactome_2022')
go_overlap, go_summary = pathway_overlap(gobp, 'GO_Biological_Process_2025')
pathway_overlap_table = pd.concat([reactome_overlap, go_overlap], ignore_index=True)
write_once_or_verify(F528, csv_bytes(pathway_overlap_table, ['library', 'direction_relationship', 'Term']))
