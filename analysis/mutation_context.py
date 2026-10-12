# Manuscript-linked analysis: Mutation associations and training-only ridge model
# Source lineage and packaging changes: provenance/source_lineage.json
import os as _release_os
try:
    from IPython.display import display
except ImportError:

    def display(value):
        print(value)
import os, json, hashlib, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error
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
F625 = AUDIT / '625_R7B2_BeatAML_axis_scores.csv'
F629 = AUDIT / '629_R7B2_final_summary.csv'
F630 = AUDIT / '630_R7B2_manifest.csv'
F639 = AUDIT / '639_R7C_final_summary.csv'
F640 = AUDIT / '640_R7C_manifest.csv'
BEATAML_REPO = 'https://github.com/biodev/beataml2.0_data.git'
BEATAML_TAG = 'v3.0'
BEATAML_LOCAL = Path(_release_os.environ.get('AML_GRAPHRX_BEATAML_DIR', '/content/beataml2.0_data_r7b'))
CLIN_FILE = BEATAML_LOCAL / 'beataml_wv1to4_clinical.xlsx'
MUT_FILE = BEATAML_LOCAL / 'beataml_wes_wv1to4_mutations_dbgap.txt'
F641 = AUDIT / '641_R8G_predeclared_genomic_anchoring_protocol.json'
F642 = AUDIT / '642_R8G_BeatAML_WES_source_audit.csv'
F643 = AUDIT / '643_R8G_RNA_DNA_overlap_audit.csv'
F644 = AUDIT / '644_R8G_common_mutation_matrix.csv'
F645 = AUDIT / '645_R8G_common_mutation_feature_audit.csv'
F646 = AUDIT / '646_R8G_cross_wave_mutation_axis_associations.csv'
F647 = AUDIT / '647_R8G_replicated_mutation_axis_associations.csv'
F648 = AUDIT / '648_R8G_genomic_ridge_predictions.csv'
F649 = AUDIT / '649_R8G_genomic_reducibility_summary.csv'
F650 = AUDIT / '650_R8G_final_summary.csv'
F651 = AUDIT / '651_R8G_manifest.csv'
MIN_MUTATED_WAVE12 = 10
MIN_MUTATED_WAVE34 = 3
MIN_WILDTYPE_PER_WAVE = 10
DISCOVERY_FDR = 0.05
VALIDATION_FDR = 0.05
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
RIDGE_CV_FOLDS = 5
RIDGE_SEED = 20261010

def require(x, msg):
    if not bool(x):
        raise RuntimeError(msg)

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

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
        v = [c for c in sort_cols if c in x.columns]
        if v:
            x = x.sort_values(v, kind='mergesort').reset_index(drop=True)
    b = x.to_csv(index=False, lineterminator='\n').encode()
    if path.exists():
        require(path.read_bytes() == b, f'Existing CSV differs: {path.name}')
        print('PASS existing identical:', path.name)
    else:
        path.write_bytes(b)
        print('CREATED:', path.name)

def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    n = len(p)
    if n == 0:
        return np.array([], float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out

def rank_biserial_from_u(U, n1, n0):
    return float(2.0 * U / (n1 * n0) - 1.0) if n1 and n0 else np.nan

def run(cmd, env=None):
    print('$', ' '.join(map(str, cmd)))
    subprocess.run(cmd, check=True, env=env)
for p in [F625, F629, F630, F639, F640]:
    require(p.is_file(), f'Missing parent artifact: {p}')
m630 = pd.read_csv(F630)
for p in [F625, F629]:
    r = m630.loc[m630['path'].astype(str).map(lambda z: Path(z).name).eq(p.name)]
    require(len(r) == 1 and str(r['sha256'].iloc[0]) == sha256_file(p), f'{p.name} integrity mismatch.')
m640 = pd.read_csv(F640)
r639 = m640.loc[m640['path'].astype(str).map(lambda z: Path(z).name).eq(F639.name)]
require(len(r639) == 1 and str(r639['sha256'].iloc[0]) == sha256_file(F639), '639 integrity mismatch.')
s639_df = pd.read_csv(F639)
s639 = dict(zip(s639_df['metric'].astype(str), s639_df['value'].astype(str)))
require(s639.get('R7C_RANK_COMPOSITION_ROBUSTNESS_SUPPORTED', '').lower() == 'true', 'R7C robustness support is not TRUE. R8G must not proceed.')
protocol = {'protocol_id': 'AML_GRAPHRX_R8G_BEATAML_GENOMIC_ANCHORING_V1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'parent_R7B2_axis_scores_sha256': sha256_file(F625), 'parent_R7C_final_summary_sha256': sha256_file(F639), 'external_WES_source': {'repository': BEATAML_REPO, 'tag': BEATAML_TAG, 'file': MUT_FILE.name}, 'sample_rule': 'intersection of the exact 560 R7B2 RNA-score subjects with BeatAML subjects having manuscript_dnaseq == yes', 'mutation_matrix_rule': 'follow BeatAML manuscript preprocessing: merge by dbgap_dnaseq_sample / dbgap_sample_id; binary gene mutation presence; clinical overrides for NPM1, FLT3-ITD, RUNX1, ASXL1, TP53', 'common_feature_rule': {'custom_capture_context': 'Nimblegen Custom Capture symbols plus FLT3_ITD, following official preprocessing logic', 'minimum_mutated_Waves1plus2': MIN_MUTATED_WAVE12, 'minimum_mutated_Waves3plus4': MIN_MUTATED_WAVE34, 'minimum_wildtype_each_wave': MIN_WILDTYPE_PER_WAVE}, 'primary_association': {'test': 'two-sided Mann-Whitney U: frozen R7B2 axis in mutated vs wild-type subjects', 'effect': 'median difference and rank-biserial correlation', 'discovery': 'Waves1+2 BH FDR <= 0.05', 'validation': 'Waves3+4 BH FDR <= 0.05 across the same prequalified features', 'replication': 'FDR significant in both waves AND same effect direction'}, 'genomic_reducibility': {'model': 'Ridge regression on the same common binary mutation features', 'training': 'Waves1+2 only', 'alpha_selection': '5-fold shuffled CV within Waves1+2 only over fixed alpha grid', 'evaluation': 'Waves3+4 only', 'metrics': ['R2', 'MAE', 'Spearman observed-vs-predicted']}, 'forbidden': ['change the frozen pathway score', 'select genes by Waves3+4 association performance', 'drop a wave or sample because results are unfavorable', 'claim the pathway axis is genetically independent merely from a nonsignificant mutation test', 'claim causal genomic control']}
write_json(F641, protocol)
if not BEATAML_LOCAL.exists():
    if shutil.which('git-lfs') is None:
        run(['apt-get', '-qq', 'update'])
        run(['apt-get', '-qq', 'install', '-y', 'git-lfs'])
    run(['git', 'lfs', 'install'])
    env = os.environ.copy()
    env['GIT_LFS_SKIP_SMUDGE'] = '1'
    run(['git', 'clone', '--depth', '1', '--branch', BEATAML_TAG, BEATAML_REPO, str(BEATAML_LOCAL)], env=env)
if shutil.which('git-lfs') is None:
    run(['apt-get', '-qq', 'update'])
    run(['apt-get', '-qq', 'install', '-y', 'git-lfs'])
run(['git', 'lfs', 'install'])
tag = subprocess.check_output(['git', '-C', str(BEATAML_LOCAL), 'describe', '--tags', '--exact-match'], text=True).strip()
commit = subprocess.check_output(['git', '-C', str(BEATAML_LOCAL), 'rev-parse', 'HEAD'], text=True).strip()
require(tag == BEATAML_TAG, f'BeatAML repo is not pinned to {BEATAML_TAG}.')
if not MUT_FILE.is_file() or MUT_FILE.stat().st_size < 1024:
    run(['git', '-C', str(BEATAML_LOCAL), 'lfs', 'pull', f'--include={MUT_FILE.name},{CLIN_FILE.name}', '--exclude='])
require(MUT_FILE.is_file() and MUT_FILE.stat().st_size > 1024, 'BeatAML WES file unavailable.')
require(CLIN_FILE.is_file() and CLIN_FILE.stat().st_size > 1024, 'BeatAML clinical file unavailable.')
source_audit = pd.DataFrame([{'file': MUT_FILE.name, 'bytes': MUT_FILE.stat().st_size, 'sha256': sha256_file(MUT_FILE), 'tag': tag, 'commit': commit}, {'file': CLIN_FILE.name, 'bytes': CLIN_FILE.stat().st_size, 'sha256': sha256_file(CLIN_FILE), 'tag': tag, 'commit': commit}])
write_csv(F642, source_audit, ['file'])
scores = pd.read_csv(F625, dtype={'ptid': str, 'external_sample_id': str, 'cohort': str})
require(len(scores) == 560 and scores['ptid'].is_unique, '625 must contain 560 unique BeatAML subjects.')
require('R7B2_frozen_pathway_axis_score' in scores.columns, '625 axis score column missing.')
clinical = pd.read_excel(CLIN_FILE, engine='openpyxl')
clinical.columns = [str(c).strip() for c in clinical.columns]
required_clin = {'dbgap_subject_id', 'dbgap_dnaseq_sample', 'manuscript_dnaseq', 'cohort', 'NPM1', 'FLT3-ITD', 'RUNX1', 'ASXL1', 'TP53'}
require(required_clin.issubset(clinical.columns), f'Clinical schema missing: {required_clin - set(clinical.columns)}')
clinical['ptid'] = 'pt' + clinical['dbgap_subject_id'].astype(str).str.strip()
clinical['manuscript_dnaseq'] = clinical['manuscript_dnaseq'].astype(str).str.strip().str.lower()
dna_clin = clinical.loc[clinical['manuscript_dnaseq'].eq('yes') & clinical['ptid'].isin(scores['ptid'])].copy()
require(dna_clin['ptid'].is_unique, 'More than one DNA record per selected BeatAML subject.')
dna_clin['dbgap_dnaseq_sample'] = dna_clin['dbgap_dnaseq_sample'].astype(str).str.strip()
require(dna_clin['dbgap_dnaseq_sample'].ne('').all(), 'Missing DNA sample ID.')
overlap = scores[['ptid', 'cohort', 'R7B2_frozen_pathway_axis_score']].merge(dna_clin[['ptid', 'dbgap_dnaseq_sample', 'cohort']].rename(columns={'cohort': 'clinical_cohort'}), on='ptid', how='inner', validate='one_to_one')
require((overlap['cohort'] == overlap['clinical_cohort']).all(), 'RNA score / DNA cohort mismatch.')
overlap_audit = pd.DataFrame([{'metric': 'R7B2_RNA_subjects_n', 'value': 560}, {'metric': 'RNA_DNA_overlap_n', 'value': len(overlap)}, {'metric': 'Waves1+2_overlap_n', 'value': int(overlap['cohort'].eq('Waves1+2').sum())}, {'metric': 'Waves3+4_overlap_n', 'value': int(overlap['cohort'].eq('Waves3+4').sum())}])
write_csv(F643, overlap_audit)
muts = pd.read_csv(MUT_FILE, sep='\t', low_memory=False)
required_mut = {'dbgap_sample_id', 'symbol', 'capture_type'}
require(required_mut.issubset(muts.columns), f'WES schema missing: {required_mut - set(muts.columns)}')
muts['dbgap_sample_id'] = muts['dbgap_sample_id'].astype(str).str.strip()
muts['symbol'] = muts['symbol'].astype(str).str.strip()
muts_m = dna_clin[['ptid', 'dbgap_dnaseq_sample']].merge(muts, left_on='dbgap_dnaseq_sample', right_on='dbgap_sample_id', how='inner')
raw = muts_m.loc[muts_m['symbol'].ne(''), ['ptid', 'symbol']].drop_duplicates().assign(value=1).pivot_table(index='ptid', columns='symbol', values='value', aggfunc='max', fill_value=0)
mutmat = pd.DataFrame(index=dna_clin['ptid'].tolist())
mutmat = mutmat.join(raw, how='left').fillna(0).astype(np.int8)
clin_idx = dna_clin.set_index('ptid')

def positive_string(s):
    return s.astype(str).str.strip().str.lower().eq('positive').astype(np.int8)
mutmat['NPM1'] = positive_string(clin_idx.loc[mutmat.index, 'NPM1']).to_numpy()
mutmat['FLT3_ITD'] = positive_string(clin_idx.loc[mutmat.index, 'FLT3-ITD']).to_numpy()
for g in ['RUNX1', 'ASXL1', 'TP53']:
    mutmat[g] = clin_idx.loc[mutmat.index, g].notna().astype(np.int8).to_numpy()
custom_symbols = set(muts_m.loc[muts_m['capture_type'].astype(str).eq('Nimblegen Custom Capture'), 'symbol'].dropna().astype(str).str.strip())
custom_symbols.discard('')
custom_symbols.add('FLT3_ITD')
wave_by_pt = dna_clin.set_index('ptid')['cohort'].astype(str)
train_pts = [p for p in mutmat.index if wave_by_pt.loc[p] == 'Waves1+2']
test_pts = [p for p in mutmat.index if wave_by_pt.loc[p] == 'Waves3+4']
feature_rows = []
for g in sorted(set(mutmat.columns) & custom_symbols):
    nmut12 = int(mutmat.loc[train_pts, g].sum())
    nwt12 = len(train_pts) - nmut12
    nmut34 = int(mutmat.loc[test_pts, g].sum())
    nwt34 = len(test_pts) - nmut34
    eligible = nmut12 >= MIN_MUTATED_WAVE12 and nmut34 >= MIN_MUTATED_WAVE34 and (nwt12 >= MIN_WILDTYPE_PER_WAVE) and (nwt34 >= MIN_WILDTYPE_PER_WAVE)
    feature_rows.append({'gene': g, 'mutated_Waves1+2_n': nmut12, 'wildtype_Waves1+2_n': nwt12, 'mutated_Waves3+4_n': nmut34, 'wildtype_Waves3+4_n': nwt34, 'prequalified': eligible})
feature_audit = pd.DataFrame(feature_rows)
features = feature_audit.loc[feature_audit['prequalified'], 'gene'].tolist()
require(len(features) >= 1, 'No mutation feature passed predeclared cross-wave count rules.')
write_csv(F645, feature_audit, ['gene'])
mut_out = mutmat.loc[:, features].copy()
mut_out.insert(0, 'ptid', mut_out.index)
mut_out.insert(1, 'cohort', [wave_by_pt.loc[p] for p in mut_out.index])
write_csv(F644, mut_out.reset_index(drop=True), ['cohort', 'ptid'])
axis_map = scores.set_index('ptid')['R7B2_frozen_pathway_axis_score'].astype(float)
assoc_rows = []
for wave in ['Waves1+2', 'Waves3+4']:
    pts = train_pts if wave == 'Waves1+2' else test_pts
    for g in features:
        m = mutmat.loc[pts, g].to_numpy(int)
        y = axis_map.loc[pts].to_numpy(float)
        a = y[m == 1]
        b = y[m == 0]
        U = mannwhitneyu(a, b, alternative='two-sided')
        assoc_rows.append({'wave': wave, 'gene': g, 'mutated_n': len(a), 'wildtype_n': len(b), 'mutated_median_axis': float(np.median(a)), 'wildtype_median_axis': float(np.median(b)), 'delta_median_mutated_minus_wildtype': float(np.median(a) - np.median(b)), 'rank_biserial': rank_biserial_from_u(float(U.statistic), len(a), len(b)), 'MW_p_two_sided': float(U.pvalue)})
assoc = pd.DataFrame(assoc_rows)
assoc['BH_FDR_within_wave'] = np.nan
for wave, idx in assoc.groupby('wave').groups.items():
    assoc.loc[idx, 'BH_FDR_within_wave'] = bh_fdr(assoc.loc[idx, 'MW_p_two_sided'].to_numpy(float))
write_csv(F646, assoc, ['wave', 'gene'])
a12 = assoc.loc[assoc['wave'].eq('Waves1+2')].set_index('gene')
a34 = assoc.loc[assoc['wave'].eq('Waves3+4')].set_index('gene')
rep_rows = []
for g in features:
    e12 = float(a12.loc[g, 'delta_median_mutated_minus_wildtype'])
    e34 = float(a34.loc[g, 'delta_median_mutated_minus_wildtype'])
    q12 = float(a12.loc[g, 'BH_FDR_within_wave'])
    q34 = float(a34.loc[g, 'BH_FDR_within_wave'])
    rep_rows.append({'gene': g, 'Waves1+2_delta_median': e12, 'Waves1+2_FDR': q12, 'Waves3+4_delta_median': e34, 'Waves3+4_FDR': q34, 'same_direction': bool(np.sign(e12) == np.sign(e34) and e12 != 0 and (e34 != 0)), 'discovery_significant': bool(q12 <= DISCOVERY_FDR), 'validation_significant': bool(q34 <= VALIDATION_FDR), 'cross_wave_replicated': bool(q12 <= DISCOVERY_FDR and q34 <= VALIDATION_FDR and (np.sign(e12) == np.sign(e34)) and (e12 != 0) and (e34 != 0))})
rep = pd.DataFrame(rep_rows)
write_csv(F647, rep, ['gene'])
X12 = mutmat.loc[train_pts, features].to_numpy(float)
X34 = mutmat.loc[test_pts, features].to_numpy(float)
y12 = axis_map.loc[train_pts].to_numpy(float)
y34 = axis_map.loc[test_pts].to_numpy(float)
pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge())])
cv = KFold(n_splits=RIDGE_CV_FOLDS, shuffle=True, random_state=RIDGE_SEED)
grid = GridSearchCV(pipe, param_grid={'ridge__alpha': RIDGE_ALPHAS}, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, refit=True)
grid.fit(X12, y12)
pred12 = grid.predict(X12)
pred34 = grid.predict(X34)
rho34 = spearmanr(y34, pred34)
pred_rows = []
for wave, pts, y, pred in [('Waves1+2', train_pts, y12, pred12), ('Waves3+4', test_pts, y34, pred34)]:
    for p, obs, pr in zip(pts, y, pred):
        pred_rows.append({'ptid': p, 'wave': wave, 'observed_axis': float(obs), 'predicted_axis_from_mutations': float(pr)})
pred_df = pd.DataFrame(pred_rows)
write_csv(F648, pred_df, ['wave', 'ptid'])
reducibility = pd.DataFrame([('mutation_features_n', len(features)), ('selected_ridge_alpha_training_only', float(grid.best_params_['ridge__alpha'])), ('Waves1+2_training_R2_descriptive', float(r2_score(y12, pred12))), ('Waves3+4_test_R2', float(r2_score(y34, pred34))), ('Waves3+4_test_MAE', float(mean_absolute_error(y34, pred34))), ('Waves3+4_test_Spearman', float(rho34.statistic)), ('Waves3+4_test_Spearman_p', float(rho34.pvalue))], columns=['metric', 'value'])
write_csv(F649, reducibility)
n_disc_sig = int(rep['discovery_significant'].sum())
n_val_sig = int(rep['validation_significant'].sum())
n_repl = int(rep['cross_wave_replicated'].sum())
n_same = int(rep['same_direction'].sum())
final_summary = pd.DataFrame([('protocol_id', 'AML_GRAPHRX_R8G_BEATAML_GENOMIC_ANCHORING_V1'), ('RNA_DNA_overlap_n', len(overlap)), ('Waves1+2_RNA_DNA_n', len(train_pts)), ('Waves3+4_RNA_DNA_n', len(test_pts)), ('prequalified_common_mutation_features_n', len(features)), ('same_direction_mutation_associations_n', n_same), ('Waves1+2_FDR_significant_mutations_n', n_disc_sig), ('Waves3+4_FDR_significant_mutations_n', n_val_sig), ('cross_wave_FDR_replicated_mutations_n', n_repl), ('mutation_only_Waves3+4_test_R2', float(r2_score(y34, pred34))), ('mutation_only_Waves3+4_test_Spearman', float(rho34.statistic)), ('mutation_only_Waves3+4_test_Spearman_p', float(rho34.pvalue)), ('R8G_GENOMIC_ANCHORING_COMPLETED', True), ('claim_boundary', 'genomic context for the already-frozen external pathway axis; does not redefine the axis and does not establish causality or genetic independence'), ('next_step', 'integrate replicated genomic associations, if any, as external context; preserve null findings and avoid retuning the pathway program')], columns=['metric', 'value'])
write_csv(F650, final_summary)
manifest = []
for role, p in [('protocol', F641), ('source_audit', F642), ('overlap_audit', F643), ('mutation_matrix', F644), ('feature_audit', F645), ('wave_associations', F646), ('replication_summary', F647), ('ridge_predictions', F648), ('genomic_reducibility', F649), ('final_summary', F650)]:
    manifest.append({'role': role, 'path': str(p), 'sha256': sha256_file(p)})
write_csv(F651, pd.DataFrame(manifest), ['role'])
try:
    import matplotlib.pyplot as plt
    plot = rep.copy()
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.scatter(plot['Waves1+2_delta_median'], plot['Waves3+4_delta_median'], s=24)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel('Waves1+2 mutation effect on frozen axis')
    ax.set_ylabel('Waves3+4 mutation effect on frozen axis')
    ax.set_title('R8G cross-wave genomic anchoring')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'R8G_cross_wave_mutation_effects_300dpi.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR / 'R8G_cross_wave_mutation_effects.pdf', bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(y34, pred34, s=22, alpha=0.7)
    ax.set_xlabel('Observed frozen pathway-axis score')
    ax.set_ylabel('Mutation-only predicted score')
    ax.set_title('R8G Waves3+4 genomic reducibility')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'R8G_mutation_only_external_prediction_300dpi.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR / 'R8G_mutation_only_external_prediction.pdf', bbox_inches='tight')
    plt.close(fig)
except Exception as e:
    print('WARNING: figure generation skipped:', repr(e))
print('\n' + '=' * 120)
print('R8G COMPLETE — BEATAML GENOMIC ANCHORING')
print('=' * 120)
print('\nRNA/DNA OVERLAP')
print(overlap_audit.to_string(index=False))
print('\nCOMMON MUTATION FEATURE SUMMARY')
print(feature_audit.loc[feature_audit['prequalified']].to_string(index=False))
print('\nREPLICATED MUTATION–AXIS ASSOCIATIONS')
rep_hit = rep.loc[rep['cross_wave_replicated']]
print(rep_hit.to_string(index=False) if len(rep_hit) else 'No cross-wave FDR-replicated mutations.')
print('\nGENOMIC REDUCIBILITY SUMMARY')
print(reducibility.to_string(index=False))
print('\nFINAL SUMMARY')
print(final_summary.to_string(index=False))
print('\nIMPORTANT')
print('- The frozen pathway axis was not changed.')
print('- Mutation features were prequalified by cross-wave counts, not by axis association.')
print('- Waves3+4 were not used for feature selection or ridge tuning.')
print('- Zero replicated mutations is a valid result and must be retained if observed.')
