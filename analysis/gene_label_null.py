# Manuscript-linked analysis: Rank-matched and secondary global gene-label nulls
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
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
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
F609 = AUDIT / '609_R7B_BeatAML_source_audit.csv'
F619 = AUDIT / '619_R7B2_mapping_technical_amendment.json'
F620 = AUDIT / '620_R7B2_external_gene_mapping_audit.csv'
F621 = AUDIT / '621_R7B2_pathway_mapping_audit.csv'
F624 = AUDIT / '624_R7B2_BeatAML_sample_ledger.csv'
F625 = AUDIT / '625_R7B2_BeatAML_axis_scores.csv'
F629 = AUDIT / '629_R7B2_final_summary.csv'
F630 = AUDIT / '630_R7B2_manifest.csv'
MAP_FILE = AUDIT / '428_R4A_entrez_to_symbol_mapping.csv'
BEATAML_REPO = 'https://github.com/biodev/beataml2.0_data.git'
BEATAML_TAG = 'v3.0'
BEATAML_LOCAL = Path(_release_os.environ.get('AML_GRAPHRX_BEATAML_DIR', '/content/beataml2.0_data_r7b'))
EXPR_FILE = BEATAML_LOCAL / 'beataml_waves1to4_norm_exp_dbgap.txt'
F631 = AUDIT / '631_R7C_predeclared_rank_composition_null_protocol.json'
F632 = AUDIT / '632_R7C_parent_integrity_audit.csv'
F633 = AUDIT / '633_R7C_frozen_weight_vector_audit.csv'
F634 = AUDIT / '634_R7C_observed_opposition_targets.csv'
F635 = AUDIT / '635_R7C_matched_gene_label_null_distribution.csv'
F636 = AUDIT / '636_R7C_global_gene_label_null_distribution.csv'
F637 = AUDIT / '637_R7C_null_test_summary.csv'
F638 = AUDIT / '638_R7C_balanced_component_count_sensitivity.csv'
F639 = AUDIT / '639_R7C_final_summary.csv'
F640 = AUDIT / '640_R7C_manifest.csv'
N_PERM_PRIMARY = 1000
N_PERM_SECONDARY = 1000
SEED_PRIMARY = 20261010
SEED_SECONDARY = 20261011
N_MEAN_RANK_STRATA = 10
PERM_BATCH = 40
PRIMARY_EMPIRICAL_P_MAX = 0.01
WAVE_FDR_MAX = 0.05
S06_COMPONENT_FDR_MAX = 0.05
RECON_TOL = 1e-10

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

def canonical_json_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')

def csv_bytes(df, sort_cols=None):
    x = df.copy()
    if sort_cols:
        valid = [c for c in sort_cols if c in x.columns]
        if valid:
            x = x.sort_values(valid, kind='mergesort').reset_index(drop=True)
    return x.to_csv(index=False, lineterminator='\n').encode('utf-8')

def write_once_or_verify(path, data):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == data, f'Existing artifact differs: {path.name}')
        print('PASS existing identical:', path.name)
    else:
        path.write_bytes(data)
        print('CREATED:', path.name)

def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out

def empirical_p_lower(null_values, observed):
    x = np.asarray(null_values, float)
    return float((1 + np.sum(x <= observed)) / (len(x) + 1))

def spearman_cols(A, B):
    """
    Column-wise Spearman correlation.
    A and B: n_samples x n_columns
    """
    RA = rankdata(A, axis=0, method='average')
    RB = rankdata(B, axis=0, method='average')
    RA = RA - RA.mean(axis=0, keepdims=True)
    RB = RB - RB.mean(axis=0, keepdims=True)
    den = np.sqrt((RA * RA).sum(axis=0) * (RB * RB).sum(axis=0))
    out = np.divide((RA * RB).sum(axis=0), den, out=np.full(A.shape[1], np.nan, dtype=float), where=den > 0)
    return out

def run(cmd, env=None):
    print('$', ' '.join(map(str, cmd)))
    subprocess.run(cmd, check=True, env=env)
for p in [F605, F607, F609, F619, F620, F621, F624, F625, F629, F630, MAP_FILE]:
    require(p.is_file(), f'Missing R7C parent input: {p}')
m607 = pd.read_csv(F607)
r605 = m607.loc[m607['path'].astype(str).map(lambda z: Path(z).name).eq(F605.name)]
require(len(r605) == 1, '605 not uniquely present in 607.')
require(str(r605['sha256'].iloc[0]) == sha256_file(F605), '605 hash mismatch vs 607.')
m630 = pd.read_csv(F630)
for p in [F619, F620, F621, F624, F625, F629]:
    r = m630.loc[m630['path'].astype(str).map(lambda z: Path(z).name).eq(p.name)]
    require(len(r) == 1, f'{p.name} not uniquely present in 630.')
    require(str(r['sha256'].iloc[0]) == sha256_file(p), f'{p.name} hash mismatch vs 630.')
s629 = dict(zip(pd.read_csv(F629)['metric'].astype(str), pd.read_csv(F629)['value'].astype(str)))
require(s629.get('R7B2_EXTERNAL_PATHWAY_AXIS_SUPPORTED', '').lower() == 'true', 'R7B2 support flag is not TRUE; R7C should not proceed.')
axis = json.loads(F605.read_text(encoding='utf-8'))
require(axis.get('axis_id') == 'AML_GRAPHRX_R7A2_COMPONENT_BALANCED_PATHWAY_AXIS_V1', 'Unexpected 605 axis ID.')
components = axis['components']
require(len(components) == 114, 'Frozen component count changed.')
require(sum((c['orientation'] == 'S03_HIGH' for c in components)) == 112, 'Expected 112 S03-high components.')
require(sum((c['orientation'] == 'S06_HIGH' for c in components)) == 2, 'Expected 2 S06-high components.')
protocol = {'protocol_id': 'AML_GRAPHRX_R7C_RANK_COMPOSITION_NULL_V1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'parent_axis_sha256': sha256_file(F605), 'parent_R7B2_final_summary_sha256': sha256_file(F629), 'scientific_question': 'Does the externally observed S03-high versus S06-high antagonism exceed the negative association expected from percentile-rank composition, frozen pathway/component overlap, and program-size asymmetry?', 'primary_null': {'name': 'mean-rank-stratified paired gene-label permutation', 'permutations': N_PERM_PRIMARY, 'seed': SEED_PRIMARY, 'strata': N_MEAN_RANK_STRATA, 'stratification_variable': 'mean BeatAML within-sample percentile rank per mapped project gene', 'same_bijection_for_all_frozen_weight_vectors': True, 'preserves': ['sample rank composition', 'weight distributions', 'S03/S06 overlap structure', 'component/pathway hierarchy after linear collapse', 'gene mean-rank stratum']}, 'secondary_null': {'name': 'unrestricted paired global gene-label permutation', 'permutations': N_PERM_SECONDARY, 'seed': SEED_SECONDARY, 'same_bijection_for_all_frozen_weight_vectors': True}, 'primary_statistic': 'Spearman rho between aggregate frozen S03-high and S06-high component means', 'tests': {'pooled_BeatAML_primary': {'alternative': 'more negative than primary null', 'empirical_p_le': PRIMARY_EMPIRICAL_P_MAX}, 'both_BeatAML_waves': {'alternative': 'more negative than primary null', 'BH_FDR_le': WAVE_FDR_MAX, 'observed_rho_must_be_negative': True}, 'both_individual_S06_components': {'alternative': 'more negative than primary null versus aggregate S03-high program', 'BH_FDR_le': S06_COMPONENT_FDR_MAX, 'observed_rho_must_be_negative': True}}, 'component_count_imbalance_sensitivity': {'method': 'enumerate all 2-of-112 S03-high component pairs and correlate their mean with the full two-component S06-high aggregate', 'role': 'descriptive robustness only; not a pass/fail gate'}, 'forbidden': ['change frozen 605 membership', 'change null after seeing permutation results', 'drop a BeatAML wave', 'select favorable S03 or S06 components', 'retune R7A2 or R7B2']}
write_once_or_verify(F631, canonical_json_bytes(protocol))
parent_rows = []
for p in [F605, F619, F620, F621, F624, F625, F629, F630]:
    parent_rows.append({'file': p.name, 'sha256': sha256_file(p), 'status': 'PASS'})
parent_audit = pd.DataFrame(parent_rows)
write_once_or_verify(F632, csv_bytes(parent_audit, ['file']))
source609 = pd.read_csv(F609)
expr_source_row = source609.loc[source609['file'].eq('beataml_waves1to4_norm_exp_dbgap.txt')]
require(len(expr_source_row) == 1, 'Expression file missing from 609 source audit.')
expected_expr_sha = str(expr_source_row['sha256'].iloc[0])
if not EXPR_FILE.is_file():
    if shutil.which('git-lfs') is None:
        run(['apt-get', '-qq', 'update'])
        run(['apt-get', '-qq', 'install', '-y', 'git-lfs'])
    run(['git', 'lfs', 'install'])
    if not BEATAML_LOCAL.exists():
        env = os.environ.copy()
        env['GIT_LFS_SKIP_SMUDGE'] = '1'
        run(['git', 'clone', '--depth', '1', '--branch', BEATAML_TAG, BEATAML_REPO, str(BEATAML_LOCAL)], env=env)
    run(['git', '-C', str(BEATAML_LOCAL), 'lfs', 'pull', '--include=beataml_waves1to4_norm_exp_dbgap.txt', '--exclude='])
require(EXPR_FILE.is_file(), 'BeatAML expression file unavailable.')
require(sha256_file(EXPR_FILE) == expected_expr_sha, 'BeatAML expression SHA mismatch vs 609.')
print('PASS: BeatAML expression SHA verified against 609.')
ledger = pd.read_csv(F624, dtype=str, keep_default_na=False)
scores625 = pd.read_csv(F625)
required_ledger = {'ptid', 'external_sample_id', 'cohort'}
require(required_ledger.issubset(ledger.columns), f'624 schema missing: {required_ledger - set(ledger.columns)}')
require(len(ledger) == 560, f'Expected 560 R7B2 external samples; found {len(ledger)}')
require(ledger['external_sample_id'].is_unique, 'External sample IDs not unique.')
waves = sorted(ledger['cohort'].unique().tolist())
require(len(waves) == 2, f'Expected two BeatAML waves; observed {waves}')
sample_ids = ledger['external_sample_id'].tolist()
header = pd.read_csv(EXPR_FILE, sep='\t', nrows=0)
expr_cols = header.columns.tolist()
require('display_label' in expr_cols[:4], 'display_label not found among first four BeatAML columns.')
require(set(sample_ids).issubset(expr_cols), 'Some 624 sample IDs absent from BeatAML expression file.')
usecols = list(expr_cols[:4]) + sample_ids
expr = pd.read_csv(EXPR_FILE, sep='\t', usecols=usecols, low_memory=False)
expr['display_label'] = expr['display_label'].astype(str).str.strip()
num = expr[sample_ids].apply(pd.to_numeric, errors='coerce')
expr2 = pd.concat([expr[['display_label']], num], axis=1)
expr2 = expr2.loc[expr2['display_label'].ne('')].groupby('display_label', sort=False)[sample_ids].median()
mapping = pd.read_csv(MAP_FILE, dtype=str, keep_default_na=False)
symbol_candidates = [c for c in mapping.columns if 'symbol' in c.lower()]
require(symbol_candidates, 'Could not identify symbol column in 428.')
symbol_col = symbol_candidates[0]
project_symbols = pd.Index(mapping[symbol_col].astype(str).str.strip().replace('', np.nan).dropna().drop_duplicates())
require(len(project_symbols) == 9554, f'Expected 9554 project symbols; found {len(project_symbols)}.')
mapped_symbols = project_symbols.intersection(expr2.index)
Xext = expr2.loc[mapped_symbols, sample_ids].copy()
Xext = Xext.loc[~Xext.isna().all(axis=1)]
if Xext.isna().any().any():
    Xext = Xext.T.fillna(Xext.median(axis=1)).T
require(np.isfinite(Xext.to_numpy(float)).all(), 'Nonfinite BeatAML expression remains.')
R = Xext.rank(axis=0, method='average', pct=True).to_numpy(dtype=np.float64)
gene_symbols = Xext.index.astype(str).tolist()
G, N = R.shape
require(N == 560, 'Unexpected BeatAML sample count after ranking.')
audit620 = pd.read_csv(F620)
expected_mapped_n = int(audit620['BeatAML_mapped_nonmissing_genes_n'].iloc[0])
require(G == expected_mapped_n, f'Mapped gene count changed: {G} vs R7B2 {expected_mapped_n}')
symbol_to_row = {g: i for i, g in enumerate(gene_symbols)}
component_weights = {}
component_meta = {}
for comp in components:
    member_vectors = []
    for pw in comp['member_pathways']:
        genes = sorted({str(g['gene_symbol']).strip() for g in pw['genes'] if str(g['gene_symbol']).strip()})
        mapped = [g for g in genes if g in symbol_to_row]
        require(len(genes) > 0, 'Frozen pathway has zero genes.')
        frac = len(mapped) / len(genes)
        require(frac >= 0.8, f"Frozen pathway mapping fell below 80%: {pw['Term']}")
        w = np.zeros(G, dtype=np.float64)
        rows = np.asarray([symbol_to_row[g] for g in mapped], dtype=int)
        w[rows] = 1.0 / len(rows)
        member_vectors.append(w)
    cw = np.vstack(member_vectors).mean(axis=0)
    require(abs(cw.sum() - 1.0) < 1e-12, 'Component weight vector does not sum to 1.')
    component_weights[comp['component_id']] = cw
    component_meta[comp['component_id']] = {'orientation': comp['orientation'], 'library': comp['library'], 'display_representative_term': comp['display_representative_term'], 'member_pathways_n': int(comp['member_pathways_n'])}
s03_cids = [c['component_id'] for c in components if c['orientation'] == 'S03_HIGH']
s06_cids = [c['component_id'] for c in components if c['orientation'] == 'S06_HIGH']
require(len(s03_cids) == 112 and len(s06_cids) == 2, 'Frozen orientation counts changed.')
w03 = np.vstack([component_weights[c] for c in s03_cids]).mean(axis=0)
w06 = np.vstack([component_weights[c] for c in s06_cids]).mean(axis=0)
w06a = component_weights[s06_cids[0]]
w06b = component_weights[s06_cids[1]]
for name, w in [('S03_AGG', w03), ('S06_AGG', w06), (s06_cids[0], w06a), (s06_cids[1], w06b)]:
    require(abs(w.sum() - 1.0) < 1e-12, f'{name} weights do not sum to 1.')
weight_audit = pd.DataFrame([{'weight_vector': name, 'orientation': 'S03_HIGH' if name == 'S03_AGG' else 'S06_HIGH', 'genes_nonzero_n': int(np.sum(w != 0)), 'weight_sum': float(w.sum()), 'max_abs_weight': float(np.max(np.abs(w)))} for name, w in [('S03_AGG', w03), ('S06_AGG', w06), (s06_cids[0], w06a), (s06_cids[1], w06b)]])
write_once_or_verify(F633, csv_bytes(weight_audit, ['weight_vector']))
s03_obs = w03 @ R - 0.5
s06_obs = w06 @ R - 0.5
s06a_obs = w06a @ R - 0.5
s06b_obs = w06b @ R - 0.5
axis_obs = s06_obs - s03_obs
s625 = scores625.copy()
require(len(s625) == 560, '625 sample count changed.')
if 'ptid' in s625.columns:
    require(s625['ptid'].astype(str).tolist() == ledger['ptid'].astype(str).tolist(), '625/624 ptid order mismatch.')
for col in ['S03_high_component_mean', 'S06_high_component_mean', 'R7B2_frozen_pathway_axis_score']:
    require(col in s625.columns, f'625 missing {col}')
maxdiff03 = float(np.max(np.abs(s03_obs - s625['S03_high_component_mean'].to_numpy(float))))
maxdiff06 = float(np.max(np.abs(s06_obs - s625['S06_high_component_mean'].to_numpy(float))))
maxdiffAxis = float(np.max(np.abs(axis_obs - s625['R7B2_frozen_pathway_axis_score'].to_numpy(float))))
require(max(maxdiff03, maxdiff06, maxdiffAxis) <= RECON_TOL, f'R7B2 score reconstruction mismatch: {max(maxdiff03, maxdiff06, maxdiffAxis)}')
print('PASS: exact R7B2 score reconstruction within', RECON_TOL)
wave_masks = {waves[0]: ledger['cohort'].eq(waves[0]).to_numpy(), waves[1]: ledger['cohort'].eq(waves[1]).to_numpy()}
observed_rows = []
rho_all = float(spearmanr(s03_obs, s06_obs).statistic)
observed_rows.append({'test_family': 'PROGRAM_OPPOSITION', 'scope': 'ALL', 'target': 'S03_AGG_vs_S06_AGG', 'n': N, 'observed_spearman_rho': rho_all})
for wv, mask in wave_masks.items():
    observed_rows.append({'test_family': 'PROGRAM_OPPOSITION', 'scope': wv, 'target': 'S03_AGG_vs_S06_AGG', 'n': int(mask.sum()), 'observed_spearman_rho': float(spearmanr(s03_obs[mask], s06_obs[mask]).statistic)})
for cid, sc in [(s06_cids[0], s06a_obs), (s06_cids[1], s06b_obs)]:
    observed_rows.append({'test_family': 'S06_COMPONENT_OPPOSITION', 'scope': 'ALL', 'target': f'S03_AGG_vs_{cid}', 'n': N, 'observed_spearman_rho': float(spearmanr(s03_obs, sc).statistic)})
observed_rows.append({'test_family': 'S06_COMPONENT_RELATION', 'scope': 'ALL', 'target': f'{s06_cids[0]}_vs_{s06_cids[1]}', 'n': N, 'observed_spearman_rho': float(spearmanr(s06a_obs, s06b_obs).statistic)})
observed_df = pd.DataFrame(observed_rows)
write_once_or_verify(F634, csv_bytes(observed_df, ['test_family', 'scope', 'target']))
gene_mean_rank = R.mean(axis=1)
order = np.argsort(gene_mean_rank, kind='mergesort')
stratum = np.empty(G, dtype=int)
for rankpos, gi in enumerate(order):
    stratum[gi] = min(N_MEAN_RANK_STRATA - 1, rankpos * N_MEAN_RANK_STRATA // G)
stratum_indices = [np.where(stratum == k)[0] for k in range(N_MEAN_RANK_STRATA)]
require(all((len(x) > 0 for x in stratum_indices)), 'Empty mean-rank stratum.')
WBASE = np.vstack([w03, w06, w06a, w06b])
K = WBASE.shape[0]

def make_permutation(rng, matched):
    if not matched:
        return rng.permutation(G)
    p = np.arange(G)
    for idx in stratum_indices:
        p[idx] = rng.permutation(idx)
    return p

def run_null(n_perm, seed, matched):
    rng = np.random.default_rng(seed)
    rows = []
    for start in range(0, n_perm, PERM_BATCH):
        b = min(PERM_BATCH, n_perm - start)
        W = np.empty((K * b, G), dtype=np.float64)
        for j in range(b):
            p = make_permutation(rng, matched=matched)
            W[j * K:(j + 1) * K, :] = WBASE[:, p]
        S = R.T @ W.T
        S = S.reshape(N, b, K)
        A_all = S[:, :, 0]
        B_all = S[:, :, 1]
        C1_all = S[:, :, 2]
        C2_all = S[:, :, 3]
        rho_all_batch = spearman_cols(A_all, B_all)
        rho_c1_batch = spearman_cols(A_all, C1_all)
        rho_c2_batch = spearman_cols(A_all, C2_all)
        m1 = wave_masks[waves[0]]
        m2 = wave_masks[waves[1]]
        rho_w1_batch = spearman_cols(A_all[m1, :], B_all[m1, :])
        rho_w2_batch = spearman_cols(A_all[m2, :], B_all[m2, :])
        for j in range(b):
            rows.append({'permutation': start + j + 1, 'rho_all': float(rho_all_batch[j]), f'rho_{waves[0]}': float(rho_w1_batch[j]), f'rho_{waves[1]}': float(rho_w2_batch[j]), f'rho_{s06_cids[0]}': float(rho_c1_batch[j]), f'rho_{s06_cids[1]}': float(rho_c2_batch[j])})
        if (start + b) % 200 == 0 or start + b == n_perm:
            print(f"Null {('MATCHED' if matched else 'GLOBAL')}: {start + b}/{n_perm}")
    return pd.DataFrame(rows)
matched_null = run_null(N_PERM_PRIMARY, SEED_PRIMARY, matched=True)
write_once_or_verify(F635, csv_bytes(matched_null, ['permutation']))
global_null = run_null(N_PERM_SECONDARY, SEED_SECONDARY, matched=False)
write_once_or_verify(F636, csv_bytes(global_null, ['permutation']))
null_tests = []
obs_all = float(observed_df.loc[observed_df['test_family'].eq('PROGRAM_OPPOSITION') & observed_df['scope'].eq('ALL'), 'observed_spearman_rho'].iloc[0])
null_tests.append({'family': 'PROGRAM_POOLED', 'test': 'ALL', 'observed_rho': obs_all, 'matched_null_mean': float(matched_null['rho_all'].mean()), 'matched_null_q005': float(matched_null['rho_all'].quantile(0.005)), 'matched_null_q025': float(matched_null['rho_all'].quantile(0.025)), 'matched_null_q975': float(matched_null['rho_all'].quantile(0.975)), 'matched_empirical_p_lower': empirical_p_lower(matched_null['rho_all'], obs_all), 'global_null_mean': float(global_null['rho_all'].mean()), 'global_empirical_p_lower': empirical_p_lower(global_null['rho_all'], obs_all)})
wave_ps = []
wave_tmp = []
for wv in waves:
    obs = float(observed_df.loc[observed_df['test_family'].eq('PROGRAM_OPPOSITION') & observed_df['scope'].eq(wv), 'observed_spearman_rho'].iloc[0])
    col = f'rho_{wv}'
    pmat = empirical_p_lower(matched_null[col], obs)
    wave_ps.append(pmat)
    wave_tmp.append((wv, obs, col, pmat))
wave_q = bh_fdr(wave_ps)
for (wv, obs, col, pmat), q in zip(wave_tmp, wave_q):
    null_tests.append({'family': 'PROGRAM_WAVE', 'test': wv, 'observed_rho': obs, 'matched_null_mean': float(matched_null[col].mean()), 'matched_null_q005': float(matched_null[col].quantile(0.005)), 'matched_null_q025': float(matched_null[col].quantile(0.025)), 'matched_null_q975': float(matched_null[col].quantile(0.975)), 'matched_empirical_p_lower': pmat, 'matched_FDR_BH_within_family': float(q), 'global_null_mean': float(global_null[col].mean()), 'global_empirical_p_lower': empirical_p_lower(global_null[col], obs)})
comp_ps = []
comp_tmp = []
for cid in s06_cids:
    obs = float(observed_df.loc[observed_df['test_family'].eq('S06_COMPONENT_OPPOSITION') & observed_df['target'].eq(f'S03_AGG_vs_{cid}'), 'observed_spearman_rho'].iloc[0])
    col = f'rho_{cid}'
    pmat = empirical_p_lower(matched_null[col], obs)
    comp_ps.append(pmat)
    comp_tmp.append((cid, obs, col, pmat))
comp_q = bh_fdr(comp_ps)
for (cid, obs, col, pmat), q in zip(comp_tmp, comp_q):
    null_tests.append({'family': 'S06_COMPONENT', 'test': cid, 'observed_rho': obs, 'matched_null_mean': float(matched_null[col].mean()), 'matched_null_q005': float(matched_null[col].quantile(0.005)), 'matched_null_q025': float(matched_null[col].quantile(0.025)), 'matched_null_q975': float(matched_null[col].quantile(0.975)), 'matched_empirical_p_lower': pmat, 'matched_FDR_BH_within_family': float(q), 'global_null_mean': float(global_null[col].mean()), 'global_empirical_p_lower': empirical_p_lower(global_null[col], obs)})
null_summary = pd.DataFrame(null_tests)
write_once_or_verify(F637, csv_bytes(null_summary, ['family', 'test']))
s03_component_scores = {}
for cid in s03_cids:
    s03_component_scores[cid] = component_weights[cid] @ R - 0.5
pair_rows = []
for i, (c1, c2) in enumerate(combinations(s03_cids, 2), start=1):
    balanced_s03 = 0.5 * (s03_component_scores[c1] + s03_component_scores[c2])
    rho = float(spearmanr(balanced_s03, s06_obs).statistic)
    pair_rows.append({'pair_index': i, 'S03_component_1': c1, 'S03_component_2': c2, 'rho_two_S03_components_vs_two_S06_component_aggregate': rho})
balanced_df = pd.DataFrame(pair_rows)
require(len(balanced_df) == 6216, f'Expected 6216 S03 pairs; found {len(balanced_df)}')
write_once_or_verify(F638, csv_bytes(balanced_df, ['pair_index']))
balanced_rhos = balanced_df['rho_two_S03_components_vs_two_S06_component_aggregate'].to_numpy(float)
pooled_row = null_summary.loc[null_summary['family'].eq('PROGRAM_POOLED')].iloc[0]
wave_rows = null_summary.loc[null_summary['family'].eq('PROGRAM_WAVE')].copy()
comp_rows = null_summary.loc[null_summary['family'].eq('S06_COMPONENT')].copy()
gate_primary_pooled = bool(obs_all < 0 and float(pooled_row['matched_empirical_p_lower']) <= PRIMARY_EMPIRICAL_P_MAX)
gate_both_waves = bool(len(wave_rows) == 2 and (wave_rows['observed_rho'] < 0).all() and (wave_rows['matched_FDR_BH_within_family'] <= WAVE_FDR_MAX).all())
gate_both_s06_components = bool(len(comp_rows) == 2 and (comp_rows['observed_rho'] < 0).all() and (comp_rows['matched_FDR_BH_within_family'] <= S06_COMPONENT_FDR_MAX).all())
r7c_supported = bool(gate_primary_pooled and gate_both_waves and gate_both_s06_components)
s06_component_relation = float(observed_df.loc[observed_df['test_family'].eq('S06_COMPONENT_RELATION'), 'observed_spearman_rho'].iloc[0])
final_summary = pd.DataFrame([('protocol_id', 'AML_GRAPHRX_R7C_RANK_COMPOSITION_NULL_V1'), ('BeatAML_samples_n', N), ('mapped_project_genes_n', G), ('primary_matched_permutations_n', N_PERM_PRIMARY), ('secondary_global_permutations_n', N_PERM_SECONDARY), ('observed_pooled_S03high_vs_S06high_rho', obs_all), ('primary_matched_null_mean_rho', float(matched_null['rho_all'].mean())), ('primary_matched_null_q005', float(matched_null['rho_all'].quantile(0.005))), ('primary_matched_null_q025', float(matched_null['rho_all'].quantile(0.025))), ('primary_matched_null_q975', float(matched_null['rho_all'].quantile(0.975))), ('primary_matched_empirical_p_lower', float(pooled_row['matched_empirical_p_lower'])), ('secondary_global_null_mean_rho', float(global_null['rho_all'].mean())), ('secondary_global_empirical_p_lower', float(pooled_row['global_empirical_p_lower'])), ('S06_component_1_id', s06_cids[0]), ('S06_component_1_rho_vs_S03_aggregate', float(comp_rows.loc[comp_rows['test'].eq(s06_cids[0]), 'observed_rho'].iloc[0])), ('S06_component_1_matched_FDR', float(comp_rows.loc[comp_rows['test'].eq(s06_cids[0]), 'matched_FDR_BH_within_family'].iloc[0])), ('S06_component_2_id', s06_cids[1]), ('S06_component_2_rho_vs_S03_aggregate', float(comp_rows.loc[comp_rows['test'].eq(s06_cids[1]), 'observed_rho'].iloc[0])), ('S06_component_2_matched_FDR', float(comp_rows.loc[comp_rows['test'].eq(s06_cids[1]), 'matched_FDR_BH_within_family'].iloc[0])), ('S06_component_1_vs_component_2_rho_descriptive', s06_component_relation), ('balanced_2of112_S03_pairs_n', int(len(balanced_df))), ('balanced_S03_pair_rho_median', float(np.median(balanced_rhos))), ('balanced_S03_pair_rho_q025', float(np.quantile(balanced_rhos, 0.025))), ('balanced_S03_pair_rho_q975', float(np.quantile(balanced_rhos, 0.975))), ('balanced_S03_pair_fraction_negative', float(np.mean(balanced_rhos < 0))), ('gate_primary_pooled_matched_null', gate_primary_pooled), ('gate_both_BeatAML_waves_matched_null', gate_both_waves), ('gate_both_S06_components_matched_null', gate_both_s06_components), ('R7C_RANK_COMPOSITION_ROBUSTNESS_SUPPORTED', r7c_supported), ('claim_boundary', 'supports gene-identity-specific external antagonistic pathway structure beyond rank-composition null; does not establish discrete subtype, causality, prognosis, or treatment response'), ('next_step_if_supported', 'freeze R7C and proceed to a separately predeclared pharmacogenomic analysis without altering R7A2/R7B2/R7C'), ('next_step_if_not_supported', 'retain negative robustness result and do not pursue pathway-axis pharmacogenomic claims as if externally established')], columns=['metric', 'value'])
write_once_or_verify(F639, csv_bytes(final_summary))
manifest_rows = []
for role, p in [('protocol', F631), ('parent_integrity_audit', F632), ('weight_vector_audit', F633), ('observed_targets', F634), ('matched_null_distribution', F635), ('global_null_distribution', F636), ('null_test_summary', F637), ('balanced_component_count_sensitivity', F638), ('final_summary', F639)]:
    manifest_rows.append({'role': role, 'path': str(p), 'sha256': sha256_file(p)})
manifest = pd.DataFrame(manifest_rows)
write_once_or_verify(F640, csv_bytes(manifest, ['role']))
try:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.hist(matched_null['rho_all'].to_numpy(float), bins=40)
    ax.axvline(obs_all, linewidth=2)
    ax.set_xlabel('Spearman rho under matched gene-label null')
    ax.set_ylabel('Permutations')
    ax.set_title('R7C primary rank-composition null calibration')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'R7C_matched_null_primary_300dpi.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR / 'R7C_matched_null_primary.pdf', bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.hist(balanced_rhos, bins=50)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel('rho: mean of 2 S03 components vs 2-component S06 aggregate')
    ax.set_ylabel('All 2-of-112 S03 pairs')
    ax.set_title('R7C component-count imbalance sensitivity')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'R7C_balanced_component_count_300dpi.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR / 'R7C_balanced_component_count.pdf', bbox_inches='tight')
    plt.close(fig)
except Exception as e:
    print('WARNING: figure generation skipped:', repr(e))
print('\n' + '=' * 120)
print('R7C COMPLETE — RANK-COMPOSITION NULL CALIBRATION')
print('=' * 120)
print('\nOBSERVED TARGETS')
print(observed_df.to_string(index=False))
print('\nNULL TEST SUMMARY')
print(null_summary.to_string(index=False))
print('\nBALANCED COMPONENT-COUNT SENSITIVITY')
print(pd.DataFrame([{'pairs_n': len(balanced_df), 'rho_median': float(np.median(balanced_rhos)), 'rho_q025': float(np.quantile(balanced_rhos, 0.025)), 'rho_q975': float(np.quantile(balanced_rhos, 0.975)), 'fraction_negative': float(np.mean(balanced_rhos < 0))}]).to_string(index=False))
print('\nFINAL SUMMARY')
print(final_summary.to_string(index=False))
print('\nIMPORTANT')
print('- Frozen R7A2/R7B2 definitions were not changed.')
print('- The primary null preserves rank composition, frozen weight overlap, and mean-rank strata.')
print('- The two S06 components were tested separately; neither was selected post hoc.')
print('- The 2-of-112 S03 analysis is descriptive only and does not alter the axis.')
print('- If R7C fails, do not tune the null or pathway axis on BeatAML.')
