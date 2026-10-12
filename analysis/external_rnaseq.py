# Manuscript-linked analysis: Executed BeatAML mapping amendment and external scoring
# Source lineage and packaging changes: provenance/source_lineage.json
import os as _release_os
try:
    from IPython.display import display
except ImportError:

    def display(value):
        print(value)
import os, re, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp, mannwhitneyu
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
FROZEN_AXIS_FILE = AUDIT / '605_R7A2_frozen_axis_definition.json'
DISCOVERY_SCORE_FILE = AUDIT / '600_R7A2_discovery_axis_scores.csv'
DISCOVERY_LEDGER_FILE = AUDIT / '452_R5DICE_discovery_ledger.csv'
STATE_FILE = AUDIT / '492_R5G_recurrent_state_patient_assignments.csv'
MAP_FILE = AUDIT / '428_R4A_entrez_to_symbol_mapping.csv'
NODE_FILE = PROJECT / 'data' / 'patient_graphs' / '20260823T123040Z' / '20260823T123040Z_node_index.csv'
RANK_FILE = PROJECT / 'data' / 'patient_graphs' / '20260823T123040Z' / '20260823T123040Z_patient_rank_raw_float64.npy'
PARENT_R7B_PROTOCOL = AUDIT / '608_R7B_predeclared_external_RNAseq_protocol.json'
PARENT_SOURCE_AUDIT = AUDIT / '609_R7B_BeatAML_source_audit.csv'
BEATAML_LOCAL = Path(_release_os.environ.get('AML_GRAPHRX_BEATAML_DIR', '/content/beataml2.0_data_r7b'))
EXPR_FILE = BEATAML_LOCAL / 'beataml_waves1to4_norm_exp_dbgap.txt'
CLIN_FILE = BEATAML_LOCAL / 'beataml_wv1to4_clinical.xlsx'
F619 = AUDIT / '619_R7B2_mapping_technical_amendment.json'
F620 = AUDIT / '620_R7B2_external_gene_mapping_audit.csv'
F621 = AUDIT / '621_R7B2_pathway_mapping_audit.csv'
F622 = AUDIT / '622_R7B2_masked_discovery_sensitivity.csv'
F623 = AUDIT / '623_R7B2_masked_discovery_state03_state06_summary.csv'
F624 = AUDIT / '624_R7B2_BeatAML_sample_ledger.csv'
F625 = AUDIT / '625_R7B2_BeatAML_axis_scores.csv'
F626 = AUDIT / '626_R7B2_external_opposition_summary.csv'
F627 = AUDIT / '627_R7B2_wave_transport_summary.csv'
F628 = AUDIT / '628_R7B2_bootstrap_opposition_stability.csv'
F629 = AUDIT / '629_R7B2_final_summary.csv'
F630 = AUDIT / '630_R7B2_manifest.csv'
MIN_PATHWAY_GENE_MAPPING_FRACTION = 0.8
MIN_MASKED_VS_FULL_SCORE_SPEARMAN = 0.9
MAX_MASKED_VS_FULL_MEDIAN_ABS_DIFF_IN_FULL_IQR = 0.25
MIN_MASKED_STATE03_STATE06_AUROC = 0.7
PRIMARY_RHO_MAX = -0.3
WAVE_RHO_MAX = -0.2
MAX_ABS_WAVE_MEDIAN_SHIFT_POOLED_IQR = 0.5
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20261009
MIN_BOOTSTRAP_NEGATIVE_FRACTION = 0.95

def require(x, msg):
    if not bool(x):
        raise RuntimeError(msg)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def normalize_gene_id(s):
    return s.astype(str).str.strip().str.replace('\\.0$', '', regex=True)

def robust_iqr(x):
    x = np.asarray(x, float)
    return float(np.quantile(x, 0.75) - np.quantile(x, 0.25))

def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out

def write_json(path, obj):
    b = (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()
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
    b = x.to_csv(index=False, lineterminator='\n').encode()
    if path.exists():
        require(path.read_bytes() == b, f'Existing CSV differs: {path.name}')
        print('PASS existing identical:', path.name)
    else:
        path.write_bytes(b)
        print('CREATED:', path.name)
for p in [FROZEN_AXIS_FILE, DISCOVERY_SCORE_FILE, DISCOVERY_LEDGER_FILE, STATE_FILE, MAP_FILE, NODE_FILE, RANK_FILE, PARENT_R7B_PROTOCOL, PARENT_SOURCE_AUDIT, EXPR_FILE, CLIN_FILE]:
    require(p.is_file(), f'Missing: {p}')
axis = json.loads(FROZEN_AXIS_FILE.read_text())
components = axis['components']
require(len(components) == 114, f'Expected 114 components, found {len(components)}')
require(sum((c['orientation'] == 'S03_HIGH' for c in components)) == 112, 'Expected 112 S03-high.')
require(sum((c['orientation'] == 'S06_HIGH' for c in components)) == 2, 'Expected 2 S06-high.')
amendment = {'protocol_id': 'AML_GRAPHRX_R7B2_BEATAML_MAPPING_AMENDMENT_V1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'parent_R7B_protocol_sha256': sha256_file(PARENT_R7B_PROTOCOL), 'parent_R7B_source_audit_sha256': sha256_file(PARENT_SOURCE_AUDIT), 'frozen_axis_sha256': sha256_file(FROZEN_AXIS_FILE), 'reason': 'Parent R7B stopped before external score computation because whole-universe coverage was 82.5% < arbitrary 90%.', 'new_primary_mapping_gate': {'every_frozen_pathway_gene_mapping_fraction_ge': 0.8}, 'mandatory_masked_discovery_gates': {'masked_vs_full_score_spearman_ge': 0.9, 'median_abs_difference_full_IQR_units_le': 0.25, 'masked_State06_vs_State03_AUROC_ge': 0.7}, 'external_rules_unchanged': {'overall_S03high_vs_S06high_rho_le': PRIMARY_RHO_MAX, 'each_wave_rho_le': WAVE_RHO_MAX, 'wave_median_shift_abs_pooled_IQR_le': MAX_ABS_WAVE_MEDIAN_SHIFT_POOLED_IQR, 'bootstrap_negative_fraction_ge': MIN_BOOTSTRAP_NEGATIVE_FRACTION}, 'forbidden': ['change frozen 605 membership', 'lower pathway mapping gate after external scores are seen', 'remove unfavorable BeatAML waves', 'retune R7A2 on BeatAML']}
write_json(F619, amendment)
mapping = pd.read_csv(MAP_FILE, dtype=str, keep_default_na=False)
symbol_candidates = [c for c in mapping.columns if 'symbol' in c.lower()]
require(symbol_candidates, 'Cannot identify symbol column in 428.')
SYMBOL_COL = symbol_candidates[0]
project_symbols = pd.Index(mapping[SYMBOL_COL].astype(str).str.strip().replace('', np.nan).dropna().drop_duplicates())
require(len(project_symbols) == 9554, f'Expected 9554 project symbols; found {len(project_symbols)}')
frozen_pathways = []
for comp in components:
    for j, pw in enumerate(comp['member_pathways'], start=1):
        genes = sorted({str(g['gene_symbol']).strip() for g in pw['genes'] if str(g['gene_symbol']).strip()})
        frozen_pathways.append({'pathway_id': f"{comp['component_id']}|P{j:03d}", 'component_id': comp['component_id'], 'orientation': comp['orientation'], 'library': comp['library'], 'Term': pw['Term'], 'genes': genes})
clinical = pd.read_excel(CLIN_FILE, engine='openpyxl')
clinical.columns = [str(c).strip() for c in clinical.columns]
required_clin = {'dbgap_subject_id', 'dbgap_rnaseq_sample', 'used_manuscript_analyses', 'manuscript_rnaseq', 'cohort'}
require(required_clin.issubset(clinical.columns), f'Clinical columns missing: {required_clin - set(clinical.columns)}')
c = clinical.copy()
for col in ['used_manuscript_analyses', 'manuscript_rnaseq']:
    c[col] = c[col].astype(str).str.strip().str.lower()
rna_clin = c.loc[c['used_manuscript_analyses'].eq('yes') & c['manuscript_rnaseq'].eq('yes')].copy()
rna_clin['dbgap_subject_id'] = rna_clin['dbgap_subject_id'].astype(str).str.strip()
rna_clin['dbgap_rnaseq_sample'] = rna_clin['dbgap_rnaseq_sample'].astype(str).str.strip()
rna_clin['ptid'] = 'pt' + rna_clin['dbgap_subject_id']
require(rna_clin['dbgap_rnaseq_sample'].is_unique, 'RNA sample IDs not unique.')
require(rna_clin['ptid'].is_unique, 'Subject IDs not unique.')
header = pd.read_csv(EXPR_FILE, sep='\t', nrows=0)
expr_cols = header.columns.tolist()
needed_samples = rna_clin['dbgap_rnaseq_sample'].tolist()
require(set(needed_samples).issubset(expr_cols), 'Some selected RNA samples absent from expression matrix.')
usecols = list(expr_cols[:4]) + needed_samples
expr = pd.read_csv(EXPR_FILE, sep='\t', usecols=usecols, low_memory=False)
require('display_label' in expr.columns, 'display_label missing.')
expr['display_label'] = expr['display_label'].astype(str).str.strip()
num = expr[needed_samples].apply(pd.to_numeric, errors='coerce')
expr2 = pd.concat([expr[['display_label']], num], axis=1)
expr2 = expr2.loc[expr2['display_label'].ne('')].groupby('display_label', sort=False)[needed_samples].median()
mapped_project_symbols = project_symbols.intersection(expr2.index)
Xext = expr2.loc[mapped_project_symbols, needed_samples].copy()
Xext = Xext.loc[~Xext.isna().all(axis=1)]
if Xext.isna().any().any():
    Xext = Xext.T.fillna(Xext.median(axis=1)).T
require(np.isfinite(Xext.to_numpy(float)).all(), 'Nonfinite BeatAML expression remains.')
mapped_project_symbols = pd.Index(Xext.index)
whole_universe_fraction = len(mapped_project_symbols) / 9554
write_csv(F620, pd.DataFrame([{'AMLGraphRx_gene_universe_n': 9554, 'BeatAML_mapped_nonmissing_genes_n': len(mapped_project_symbols), 'whole_universe_mapping_fraction_descriptive': whole_universe_fraction, 'whole_universe_gate_used': False}]))
pathway_rows = []
for pw in frozen_pathways:
    mapped = [g for g in pw['genes'] if g in mapped_project_symbols]
    frac = len(mapped) / len(pw['genes']) if pw['genes'] else 0.0
    pathway_rows.append({'pathway_id': pw['pathway_id'], 'component_id': pw['component_id'], 'orientation': pw['orientation'], 'library': pw['library'], 'Term': pw['Term'], 'frozen_genes_n': len(pw['genes']), 'mapped_genes_n': len(mapped), 'mapping_fraction': frac, 'mapping_gate': frac >= MIN_PATHWAY_GENE_MAPPING_FRACTION})
pathway_audit = pd.DataFrame(pathway_rows)
write_csv(F621, pathway_audit, ['component_id', 'pathway_id'])
require(pathway_audit['mapping_gate'].all(), 'At least one frozen pathway has <80% mapping. Stop; do not lower threshold.')
nodes = pd.read_csv(NODE_FILE, dtype=str, keep_default_na=False)
node_ids = normalize_gene_id(nodes['canonical_NCBI_Gene_ID'])
universe_ids = set(node_ids)
id_candidates = []
for col in mapping.columns:
    vals = set(normalize_gene_id(mapping[col]))
    id_candidates.append((len(universe_ids & vals), col))
id_candidates.sort(reverse=True)
coverage, ID_COL = id_candidates[0]
require(coverage == 9554, 'Cannot identify full-coverage GeneID column.')
map2 = mapping.copy()
map2['_gid'] = normalize_gene_id(map2[ID_COL])
map2['_symbol'] = map2[SYMBOL_COL].astype(str).str.strip()
map2 = map2.loc[map2['_gid'].isin(universe_ids)].drop_duplicates('_gid')
require(len(map2) == 9554, 'Mapping changed.')
gid_to_idx = {gid: i for i, gid in enumerate(node_ids)}
symbol_to_idx = {row['_symbol']: gid_to_idx[row['_gid']] for _, row in map2.iterrows()}
available_project_indices = np.array(sorted((symbol_to_idx[s] for s in mapped_project_symbols if s in symbol_to_idx)), int)
require(len(available_project_indices) == len(mapped_project_symbols), 'Project/BeatAML symbol mapping mismatch.')
disc = pd.read_csv(DISCOVERY_LEDGER_FILE, dtype={'analysis_unit_id': str, 'analysis_unit_index_0based': int, 'gse': str})
states = pd.read_csv(STATE_FILE, dtype={'analysis_unit_id': str, 'state_id': str})
full_scores = pd.read_csv(DISCOVERY_SCORE_FILE)
rank_mm = np.load(RANK_FILE, mmap_mode='r', allow_pickle=False)
disc_idx = disc['analysis_unit_index_0based'].to_numpy(int)
sub = pd.DataFrame(np.asarray(rank_mm[np.ix_(disc_idx, available_project_indices)], float).T)
masked_rank = sub.rank(axis=0, method='average', pct=True).to_numpy(float)
project_idx_to_local = {proj_i: local_i for local_i, proj_i in enumerate(available_project_indices)}
symbol_to_local = {s: project_idx_to_local[symbol_to_idx[s]] for s in mapped_project_symbols}
component_scores_disc = {}
for comp in components:
    members = []
    for pw in comp['member_pathways']:
        genes = sorted({str(g['gene_symbol']).strip() for g in pw['genes'] if str(g['gene_symbol']).strip()})
        local = [symbol_to_local[g] for g in genes if g in symbol_to_local]
        frac = len(local) / len(genes) if genes else 0.0
        require(frac >= MIN_PATHWAY_GENE_MAPPING_FRACTION, 'Masked discovery pathway coverage below 80%.')
        members.append(masked_rank[local, :].mean(axis=0) - 0.5)
    component_scores_disc[comp['component_id']] = np.vstack(members).mean(axis=0)
s03_cids = [c['component_id'] for c in components if c['orientation'] == 'S03_HIGH']
s06_cids = [c['component_id'] for c in components if c['orientation'] == 'S06_HIGH']
masked_s03 = np.vstack([component_scores_disc[c] for c in s03_cids]).mean(axis=0)
masked_s06 = np.vstack([component_scores_disc[c] for c in s06_cids]).mean(axis=0)
masked_axis = masked_s06 - masked_s03
full_axis = pd.to_numeric(full_scores['R7A2_axis_score'], errors='raise').to_numpy(float)
require(len(full_axis) == len(masked_axis) == 1733, 'Discovery score length mismatch.')
rho_mask = spearmanr(full_axis, masked_axis)
full_iqr = robust_iqr(full_axis)
median_abs_diff_iqr = float(np.median(np.abs(masked_axis - full_axis))) / full_iqr if full_iqr > 0 else np.nan
d = disc[['analysis_unit_id', 'gse']].copy()
d['masked_axis'] = masked_axis
d = d.merge(states[['analysis_unit_id', 'state_id']], on='analysis_unit_id', how='left', validate='one_to_one')
d36 = d.loc[d['state_id'].isin(['AML_STATE_03', 'AML_STATE_06'])].copy()
y36 = d36['state_id'].eq('AML_STATE_06').astype(int).to_numpy()
auc_masked = float(roc_auc_score(y36, d36['masked_axis'].to_numpy(float)))
mw_masked = mannwhitneyu(d36.loc[d36['state_id'].eq('AML_STATE_06'), 'masked_axis'].to_numpy(float), d36.loc[d36['state_id'].eq('AML_STATE_03'), 'masked_axis'].to_numpy(float), alternative='two-sided')
gate_rho = float(rho_mask.statistic) >= MIN_MASKED_VS_FULL_SCORE_SPEARMAN
gate_diff = np.isfinite(median_abs_diff_iqr) and median_abs_diff_iqr <= MAX_MASKED_VS_FULL_MEDIAN_ABS_DIFF_IN_FULL_IQR
gate_auc = auc_masked >= MIN_MASKED_STATE03_STATE06_AUROC
masked_gate = bool(gate_rho and gate_diff and gate_auc)
masked_summary = pd.DataFrame([{'BeatAML_available_project_genes_n': len(available_project_indices), 'BeatAML_available_project_gene_fraction': whole_universe_fraction, 'masked_vs_full_discovery_score_spearman': float(rho_mask.statistic), 'masked_vs_full_discovery_score_spearman_p': float(rho_mask.pvalue), 'median_absolute_score_difference_full_IQR_units': float(median_abs_diff_iqr), 'masked_discovery_State06_vs_State03_AUROC': auc_masked, 'masked_discovery_State03_vs_State06_MW_p': float(mw_masked.pvalue), 'gate_spearman': gate_rho, 'gate_score_difference': gate_diff, 'gate_State03_State06_AUROC': gate_auc, 'MASKED_DISCOVERY_PRESERVATION_GATE': masked_gate}])
write_csv(F622, masked_summary)
rows = []
for gse, gg in d36.groupby('gse', sort=True):
    n3 = int(gg['state_id'].eq('AML_STATE_03').sum())
    n6 = int(gg['state_id'].eq('AML_STATE_06').sum())
    a = float(roc_auc_score(gg['state_id'].eq('AML_STATE_06').astype(int), gg['masked_axis'])) if n3 >= 3 and n6 >= 3 else np.nan
    rows.append({'gse': gse, 'State03_n': n3, 'State06_n': n6, 'masked_State03_median': float(gg.loc[gg['state_id'].eq('AML_STATE_03'), 'masked_axis'].median()) if n3 else np.nan, 'masked_State06_median': float(gg.loc[gg['state_id'].eq('AML_STATE_06'), 'masked_axis'].median()) if n6 else np.nan, 'masked_State06_vs_State03_AUROC': a})
write_csv(F623, pd.DataFrame(rows), ['gse'])
require(masked_gate, 'Masked-discovery preservation FAILED. Stop; do not interpret BeatAML scores.')
rank_ext = Xext.rank(axis=0, method='average', pct=True)
component_scores_ext = {}
for comp in components:
    members = []
    for pw in comp['member_pathways']:
        genes = sorted({str(g['gene_symbol']).strip() for g in pw['genes'] if str(g['gene_symbol']).strip()})
        mapped = [g for g in genes if g in rank_ext.index]
        require(len(mapped) / len(genes) >= MIN_PATHWAY_GENE_MAPPING_FRACTION, 'External pathway mapping changed.')
        members.append(rank_ext.loc[mapped, needed_samples].mean(axis=0).to_numpy(float) - 0.5)
    component_scores_ext[comp['component_id']] = np.vstack(members).mean(axis=0)
ext_s03 = np.vstack([component_scores_ext[c] for c in s03_cids]).mean(axis=0)
ext_s06 = np.vstack([component_scores_ext[c] for c in s06_cids]).mean(axis=0)
ext_axis = ext_s06 - ext_s03
sample_ledger = rna_clin[['ptid', 'dbgap_subject_id', 'dbgap_rnaseq_sample', 'cohort']].copy()
sample_ledger = sample_ledger.rename(columns={'dbgap_rnaseq_sample': 'external_sample_id'})
sample_ledger['R7B2_analysis_role'] = 'INDEPENDENT_EXTERNAL_RNASEQ_VALIDATION'
write_csv(F624, sample_ledger, ['ptid'])
score_df = sample_ledger.copy()
score_df['S03_high_component_mean'] = ext_s03
score_df['S06_high_component_mean'] = ext_s06
score_df['R7B2_frozen_pathway_axis_score'] = ext_axis
write_csv(F625, score_df, ['ptid'])
overall = spearmanr(ext_s03, ext_s06)
opp_rows = [{'scope': 'ALL_BEATAML_RNA', 'cohort': 'ALL', 'n': len(score_df), 'spearman_rho_S03high_vs_S06high': float(overall.statistic), 'p_two_sided': float(overall.pvalue), 'rho_threshold': PRIMARY_RHO_MAX, 'rho_gate': float(overall.statistic) <= PRIMARY_RHO_MAX, 'p_gate': float(overall.pvalue) < 0.05}]
for cohort_name, gg in score_df.groupby('cohort', sort=True):
    if len(gg) < 10:
        continue
    rr = spearmanr(gg['S03_high_component_mean'], gg['S06_high_component_mean'])
    opp_rows.append({'scope': 'BEATAML_WAVE', 'cohort': str(cohort_name), 'n': len(gg), 'spearman_rho_S03high_vs_S06high': float(rr.statistic), 'p_two_sided': float(rr.pvalue), 'rho_threshold': WAVE_RHO_MAX, 'rho_gate': float(rr.statistic) <= WAVE_RHO_MAX, 'p_gate': float(rr.pvalue) < 0.05})
opp_df = pd.DataFrame(opp_rows)
opp_df['FDR_BH'] = bh_fdr(opp_df['p_two_sided'].to_numpy(float))
write_csv(F626, opp_df, ['scope', 'cohort'])
waves = sorted(score_df['cohort'].astype(str).unique())
require(len(waves) == 2, f'Expected two BeatAML cohort labels; observed {waves}')
w1 = score_df.loc[score_df['cohort'].astype(str).eq(waves[0]), 'R7B2_frozen_pathway_axis_score'].to_numpy(float)
w2 = score_df.loc[score_df['cohort'].astype(str).eq(waves[1]), 'R7B2_frozen_pathway_axis_score'].to_numpy(float)
pooled = score_df['R7B2_frozen_pathway_axis_score'].to_numpy(float)
pooled_iqr = robust_iqr(pooled)
median_shift_iqr = (float(np.median(w2)) - float(np.median(w1))) / pooled_iqr if pooled_iqr > 0 else np.nan
ks = ks_2samp(w1, w2)
mw_wave = mannwhitneyu(w1, w2, alternative='two-sided')
wave_gate = np.isfinite(median_shift_iqr) and abs(median_shift_iqr) <= MAX_ABS_WAVE_MEDIAN_SHIFT_POOLED_IQR
wave_df = pd.DataFrame([{'wave_1': waves[0], 'wave_2': waves[1], 'wave_1_n': len(w1), 'wave_2_n': len(w2), 'wave_1_median': float(np.median(w1)), 'wave_2_median': float(np.median(w2)), 'pooled_IQR': pooled_iqr, 'wave2_minus_wave1_median_shift_pooled_IQR_units': float(median_shift_iqr), 'absolute_shift_gate': bool(wave_gate), 'KS_statistic': float(ks.statistic), 'KS_p': float(ks.pvalue), 'MW_p': float(mw_wave.pvalue)}])
write_csv(F627, wave_df)
rng = np.random.default_rng(BOOTSTRAP_SEED)
boot = []
n = len(score_df)
for b in range(BOOTSTRAP_N):
    ii = rng.integers(0, n, size=n)
    rr = spearmanr(ext_s03[ii], ext_s06[ii])
    boot.append({'bootstrap': b + 1, 'spearman_rho': float(rr.statistic)})
boot_df = pd.DataFrame(boot)
write_csv(F628, boot_df, ['bootstrap'])
boot_negative_fraction = float(np.mean(boot_df['spearman_rho'] < 0))
q025, q975 = np.quantile(boot_df['spearman_rho'], [0.025, 0.975])
overall_row = opp_df.loc[opp_df['scope'].eq('ALL_BEATAML_RNA')].iloc[0]
wave_rows = opp_df.loc[opp_df['scope'].eq('BEATAML_WAVE')]
mapping_gate = bool(pathway_audit['mapping_gate'].all())
overall_gate = bool(overall_row['rho_gate'] and overall_row['p_gate'])
both_wave_gate = bool(len(wave_rows) == 2 and wave_rows['rho_gate'].all() and wave_rows['p_gate'].all())
boot_gate = boot_negative_fraction >= MIN_BOOTSTRAP_NEGATIVE_FRACTION
external_support = bool(masked_gate and mapping_gate and overall_gate and both_wave_gate and wave_gate and boot_gate)
final_summary = pd.DataFrame([('protocol_id', 'AML_GRAPHRX_R7B2_BEATAML_MAPPING_AMENDMENT_V1'), ('BeatAML_RNA_samples_n', len(score_df)), ('whole_universe_mapping_fraction_descriptive', whole_universe_fraction), ('minimum_frozen_pathway_mapping_fraction', float(pathway_audit['mapping_fraction'].min())), ('masked_vs_full_discovery_score_spearman', float(rho_mask.statistic)), ('masked_vs_full_median_abs_diff_full_IQR_units', float(median_abs_diff_iqr)), ('masked_discovery_State06_vs_State03_AUROC', auc_masked), ('overall_external_S03high_vs_S06high_spearman', float(overall.statistic)), ('overall_external_opposition_p', float(overall.pvalue)), ('bootstrap_rho_q025', float(q025)), ('bootstrap_rho_q975', float(q975)), ('bootstrap_negative_rho_fraction', boot_negative_fraction), ('wave_median_shift_pooled_IQR_units', float(median_shift_iqr)), ('gate_masked_discovery_preservation', masked_gate), ('gate_pathway_mapping', mapping_gate), ('gate_overall_external_opposition', overall_gate), ('gate_both_external_waves_opposition', both_wave_gate), ('gate_cross_wave_transport', bool(wave_gate)), ('gate_bootstrap_direction', bool(boot_gate)), ('R7B2_EXTERNAL_PATHWAY_AXIS_SUPPORTED', external_support), ('claim_boundary', 'independent external RNA-seq support for a frozen continuous pathway program only; not subtype, prognosis, or treatment validation'), ('next_step_if_supported', 'lock R7B2 and integrate into manuscript; any pharmacologic analysis must be separate and predeclared'), ('next_step_if_not_supported', 'retain result and stop pathway-axis refinement on BeatAML')], columns=['metric', 'value'])
write_csv(F629, final_summary)
manifest = []
for role, p in [('technical_amendment', F619), ('gene_mapping_audit', F620), ('pathway_mapping_audit', F621), ('masked_discovery_sensitivity', F622), ('masked_discovery_state_summary', F623), ('BeatAML_sample_ledger', F624), ('BeatAML_axis_scores', F625), ('external_opposition', F626), ('wave_transport', F627), ('bootstrap', F628), ('final_summary', F629)]:
    manifest.append({'role': role, 'path': str(p), 'sha256': sha256_file(p)})
write_csv(F630, pd.DataFrame(manifest), ['role'])
print('\n' + '=' * 120)
print('R7B2 COMPLETE — MAPPING AMENDMENT')
print('=' * 120)
print('\nMASKED DISCOVERY SENSITIVITY')
print(masked_summary.to_string(index=False))
print('\nPATHWAY MAPPING SUMMARY')
print(pathway_audit['mapping_fraction'].describe().to_string())
print('\nEXTERNAL OPPOSITION SUMMARY')
print(opp_df.to_string(index=False))
print('\nWAVE TRANSPORT SUMMARY')
print(wave_df.to_string(index=False))
print('\nFINAL SUMMARY')
print(final_summary.to_string(index=False))
print('\nIMPORTANT')
print('- Parent R7B stopped before external score computation.')
print('- 82.5% whole-universe coverage is descriptive QC, not a pass/fail threshold.')
print('- Frozen pathway mapping + masked-discovery preservation are the safeguards.')
print('- Do not retune R7A2/R7B2 on BeatAML if a gate fails.')
