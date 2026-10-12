# Manuscript-linked analysis: Verified analysis-unit merge and corrected odds-ratio labels
# Source lineage and packaging changes: provenance/source_lineage.json
import os as _release_os
try:
    from IPython.display import display
except ImportError:

    def display(value):
        print(value)
pass
pass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency
try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    raise ImportError('Install statsmodels before this analysis')
    from statsmodels.stats.multitest import multipletests
PROJECT = Path(_release_os.environ.get('AML_GRAPHRX_ROOT', '/content/drive/MyDrive/AML_GraphRx'))
AUDIT_DIR = PROJECT / 'results' / 'audit'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def find_exact_file(root, name):
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(name)
    return sorted(hits, key=lambda x: (len(str(x)), str(x)))[0]
F492 = find_exact_file(PROJECT, '492_R5G_recurrent_state_patient_assignments.csv')
FCLIN = PROJECT / 'config' / 'verified_clinical_analysis_unit_map_v1.csv'
if not FCLIN.exists():
    raise FileNotFoundError(f'\nExpected authoritative analysis-unit clinical map not found:\n{FCLIN}\n\nDo NOT substitute the sample-level map.')
print('\nAUTHORITATIVE FILES')
print('-' * 100)
print('STATE ASSIGNMENTS:', F492)
print('CLINICAL MASTER:', FCLIN)
print('F492 SHA256:', sha256_file(F492))
print('FCLIN SHA256:', sha256_file(FCLIN))
states = pd.read_csv(F492, dtype=str, keep_default_na=False, low_memory=False)
clin = pd.read_csv(FCLIN, dtype=str, keep_default_na=False, low_memory=False)
required_clinical_columns = ['analysis_unit_id', 'age_years', 'sex_standard', 'FLT3_ITD', 'FLT3_TKD', 'NPM1_mut', 'CEBPA_mut', 'IDH1_mut', 'IDH2_mut', 'RUNX1_mut', 'RUNX1_RUNX1T1_fusion', 'cytogenetic_risk_standard']
missing_cols = [c for c in required_clinical_columns if c not in clin.columns]
if missing_cols:
    raise RuntimeError('Missing required clinical columns:\n' + '\n'.join(missing_cols))
if not states['analysis_unit_id'].is_unique:
    raise RuntimeError('HARD STOP: State assignment analysis_unit_id is not unique.')
if not clin['analysis_unit_id'].is_unique:
    dup = clin.loc[clin['analysis_unit_id'].duplicated(keep=False)].sort_values('analysis_unit_id')
    print('\nUnexpected duplicated analysis_unit_id in ANALYSIS-UNIT clinical master:')
    display(dup.head(50))
    raise RuntimeError('HARD STOP: verified_clinical_analysis_unit_map_v1.csv is unexpectedly non-unique. Do NOT deduplicate automatically.')
print('PASS: analysis_unit_id unique in both authoritative files ✅')
clin_small = clin[required_clinical_columns].copy()
df = states.merge(clin_small, on='analysis_unit_id', how='left', validate='one_to_one', indicator=True)
print('\nMERGE AUDIT')
print('-' * 100)
print(df['_merge'].value_counts(dropna=False).to_string())
missing_clinical_units = df.loc[df['_merge'] != 'both', ['analysis_unit_id', 'gse', 'state_id']]
if len(missing_clinical_units) > 0:
    print('\nUnits not found in clinical master:')
    display(missing_clinical_units.head(50))
    raise RuntimeError('HARD STOP: one or more discovery units were not found in clinical master.')
df = df.drop(columns='_merge')
df = df.loc[df['state_id'].isin(['AML_STATE_03', 'AML_STATE_06'])].copy()
n3 = int((df['state_id'] == 'AML_STATE_03').sum())
n6 = int((df['state_id'] == 'AML_STATE_06').sum())
print('\nSTATE COUNTS')
print('-' * 100)
print('State03:', n3)
print('State06:', n6)
assert n3 == 285, f'Expected State03 n=285; found {n3}'
assert n6 == 160, f'Expected State06 n=160; found {n6}'
print('\nPASS: State03/State06 clinical merge verified ✅')
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency
from statsmodels.stats.multitest import multipletests
assert len(df) == 445
assert (df['state_id'] == 'AML_STATE_03').sum() == 285
assert (df['state_id'] == 'AML_STATE_06').sum() == 160
print('PASS: input State03/State06 cohort = 445 patients')
MISSING_TOKENS = {'', 'na', 'n/a', 'nan', 'none', 'unknown', 'not available', 'not_available', 'missing'}

def clean_text(x):
    x = str(x).strip()
    if x.lower() in MISSING_TOKENS:
        return np.nan
    return x

def clean_binary(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    positive = {'1', '1.0', 'positive', 'pos', 'mut', 'mutated', 'mutation', 'yes', 'true', '+'}
    negative = {'0', '0.0', 'negative', 'neg', 'wt', 'wildtype', 'wild-type', 'no', 'false', '-'}
    if s in positive:
        return 1
    if s in negative:
        return 0
    return np.nan

def rank_biserial_from_u(U, n1, n2):
    if n1 == 0 or n2 == 0:
        return np.nan
    return 2 * U / (n1 * n2) - 1

def fmt_binary(pos, known):
    if known == 0:
        return 'NA'
    return f'{pos}/{known} ({100 * pos / known:.1f}%)'

def fmt_missing(missing, total):
    return f'{missing}/{total} ({100 * missing / total:.1f}%)'
rows = []
N3 = 285
N6 = 160
tmp = df[['state_id', 'age_years']].copy()
tmp['value'] = pd.to_numeric(tmp['age_years'], errors='coerce')
x3 = tmp.loc[tmp['state_id'] == 'AML_STATE_03', 'value'].dropna()
x6 = tmp.loc[tmp['state_id'] == 'AML_STATE_06', 'value'].dropna()
if len(x3) > 0 and len(x6) > 0:
    test = mannwhitneyu(x3, x6, alternative='two-sided')
    p = float(test.pvalue)
    effect = rank_biserial_from_u(float(test.statistic), len(x3), len(x6))
else:
    p = np.nan
    effect = np.nan
rows.append({'variable': 'Age, years', 'variable_type': 'continuous', 'State03': f'{x3.median():.1f} [IQR {x3.quantile(0.25):.1f}–{x3.quantile(0.75):.1f}], known n={len(x3)}', 'State06': f'{x6.median():.1f} [IQR {x6.quantile(0.25):.1f}–{x6.quantile(0.75):.1f}], known n={len(x6)}', 'State03_missing': fmt_missing(N3 - len(x3), N3), 'State06_missing': fmt_missing(N6 - len(x6), N6), 'test': 'Mann–Whitney U', 'effect_measure': 'rank-biserial correlation', 'effect_size': effect, 'p_value': p})
binary_vars = [('FLT3_ITD', 'FLT3-ITD'), ('FLT3_TKD', 'FLT3-TKD'), ('NPM1_mut', 'NPM1 mutation'), ('CEBPA_mut', 'CEBPA mutation'), ('IDH1_mut', 'IDH1 mutation'), ('IDH2_mut', 'IDH2 mutation'), ('RUNX1_mut', 'RUNX1 mutation'), ('RUNX1_RUNX1T1_fusion', 'RUNX1::RUNX1T1 fusion')]
for col, label in binary_vars:
    tmp = df[['state_id', col]].copy()
    tmp['value'] = tmp[col].map(clean_binary)
    known = tmp.dropna(subset=['value']).copy()
    known['value'] = known['value'].astype(int)
    tab = pd.crosstab(known['state_id'], known['value'])
    for s in ['AML_STATE_03', 'AML_STATE_06']:
        if s not in tab.index:
            tab.loc[s] = 0
    for v in [0, 1]:
        if v not in tab.columns:
            tab[v] = 0
    tab = tab.loc[['AML_STATE_03', 'AML_STATE_06'], [0, 1]]
    n3 = int(tab.loc['AML_STATE_03'].sum())
    n6 = int(tab.loc['AML_STATE_06'].sum())
    pos3 = int(tab.loc['AML_STATE_03', 1])
    pos6 = int(tab.loc['AML_STATE_06', 1])
    if tab.values.sum() > 0 and n3 > 0 and (n6 > 0):
        OR, p = fisher_exact(tab.values, alternative='two-sided')
        OR = float(OR)
        p = float(p)
    else:
        OR = np.nan
        p = np.nan
    rows.append({'variable': label, 'variable_type': 'binary', 'State03': fmt_binary(pos3, n3), 'State06': fmt_binary(pos6, n6), 'State03_missing': fmt_missing(N3 - n3, N3), 'State06_missing': fmt_missing(N6 - n6, N6), 'test': 'Fisher exact', 'effect_measure': 'odds ratio State03/State06', 'effect_size': OR, 'p_value': p})
tmp = df[['state_id', 'sex_standard']].copy()
tmp['value'] = tmp['sex_standard'].map(clean_text)
known = tmp.dropna(subset=['value']).copy()
sex_categories = sorted(known['value'].unique())
tab = pd.crosstab(known['state_id'], known['value'])
for s in ['AML_STATE_03', 'AML_STATE_06']:
    if s not in tab.index:
        tab.loc[s] = 0
tab = tab.loc[['AML_STATE_03', 'AML_STATE_06']]
n3 = int(tab.loc['AML_STATE_03'].sum())
n6 = int(tab.loc['AML_STATE_06'].sum())
if tab.shape == (2, 2):
    OR, p = fisher_exact(tab.values)
    effect_measure = 'odds ratio'
    effect = float(OR)
else:
    chi2, p, dof, expected = chi2_contingency(tab.values)
    effect_measure = "Cramér's V"
    n = tab.values.sum()
    effect = np.sqrt(chi2 / (n * min(tab.shape[0] - 1, tab.shape[1] - 1)))
rows.append({'variable': 'Sex', 'variable_type': 'categorical', 'State03': '; '.join([f'{k}: {int(v)}' for k, v in tab.loc['AML_STATE_03'].items()]), 'State06': '; '.join([f'{k}: {int(v)}' for k, v in tab.loc['AML_STATE_06'].items()]), 'State03_missing': fmt_missing(N3 - n3, N3), 'State06_missing': fmt_missing(N6 - n6, N6), 'test': 'Fisher exact' if tab.shape == (2, 2) else 'Chi-square', 'effect_measure': effect_measure, 'effect_size': effect, 'p_value': float(p)})
tmp = df[['state_id', 'cytogenetic_risk_standard']].copy()
tmp['value'] = tmp['cytogenetic_risk_standard'].map(clean_text)
known = tmp.dropna(subset=['value']).copy()
tab = pd.crosstab(known['state_id'], known['value'])
for s in ['AML_STATE_03', 'AML_STATE_06']:
    if s not in tab.index:
        tab.loc[s] = 0
tab = tab.loc[['AML_STATE_03', 'AML_STATE_06']]
n3 = int(tab.loc['AML_STATE_03'].sum())
n6 = int(tab.loc['AML_STATE_06'].sum())
if tab.shape[1] >= 2 and tab.values.sum() > 0:
    chi2, p, dof, expected = chi2_contingency(tab.values)
    n = tab.values.sum()
    cramer_v = np.sqrt(chi2 / (n * min(tab.shape[0] - 1, tab.shape[1] - 1)))
    expected_min = float(expected.min())
else:
    p = np.nan
    cramer_v = np.nan
    expected_min = np.nan
rows.append({'variable': 'Cytogenetic risk', 'variable_type': 'categorical', 'State03': '; '.join([f'{k}: {int(v)}' for k, v in tab.loc['AML_STATE_03'].items()]), 'State06': '; '.join([f'{k}: {int(v)}' for k, v in tab.loc['AML_STATE_06'].items()]), 'State03_missing': fmt_missing(N3 - n3, N3), 'State06_missing': fmt_missing(N6 - n6, N6), 'test': 'Chi-square', 'effect_measure': "Cramér's V", 'effect_size': cramer_v, 'p_value': float(p) if np.isfinite(p) else np.nan, 'minimum_expected_cell_count': expected_min})
out = pd.DataFrame(rows)
if 'minimum_expected_cell_count' not in out.columns:
    out['minimum_expected_cell_count'] = np.nan
mask = out['p_value'].notna()
out['FDR_BH'] = np.nan
out.loc[mask, 'FDR_BH'] = multipletests(out.loc[mask, 'p_value'].astype(float), method='fdr_bh')[1]

def interpret(row):
    if pd.isna(row['p_value']):
        return 'Not evaluable from available annotation'
    if pd.notna(row['FDR_BH']) and row['FDR_BH'] < 0.05:
        return 'Statistically supported difference after multiple-testing correction; interpret with annotation completeness in mind'
    return 'No statistically supported difference after multiple-testing correction'
out['reviewer_safe_interpretation'] = out.apply(interpret, axis=1)
preferred_order = ['Age, years', 'Sex', 'FLT3-ITD', 'FLT3-TKD', 'NPM1 mutation', 'CEBPA mutation', 'IDH1 mutation', 'IDH2 mutation', 'RUNX1 mutation', 'RUNX1::RUNX1T1 fusion', 'Cytogenetic risk']
out['_order'] = out['variable'].map({x: i for i, x in enumerate(preferred_order)})
out = out.sort_values('_order').drop(columns='_order').reset_index(drop=True)
F554 = AUDIT_DIR / '554_R5MS_state03_state06_subtype_table.csv'
out.to_csv(F554, index=False)
significant_after_FDR = out.loc[out['FDR_BH'].fillna(1.0) < 0.05, 'variable'].tolist()
summary = {'analysis': 'State03_vs_State06_available_clinical_subtype_annotations', 'state03_n': 285, 'state06_n': 160, 'combined_n': 445, 'clinical_master': str(FCLIN), 'clinical_master_SHA256': sha256_file(FCLIN), 'state_assignment_file': str(F492), 'state_assignment_SHA256': sha256_file(F492), 'analysis_unit_merge': 'exact_one_to_one', 'missing_values_imputed': False, 'automatic_deduplication': False, 'state_definition_modified': False, 'samples_removed_based_on_performance': False, 'multiple_testing': 'Benjamini-Hochberg across available inferential annotation comparisons', 'FDR_lt_0.05_variables': significant_after_FDR, 'claim_boundary': 'Absence of statistically supported differences in available annotations does not establish biological independence from AML subtype because annotation completeness varies across cohorts.'}
F555 = AUDIT_DIR / '555_R5MS_state03_state06_subtype_summary.json'
F555.write_text(json.dumps(summary, indent=2, sort_keys=True))
print('\n' + '=' * 120)
print('STATE03 vs STATE06 — FINAL CLINICAL/SUBTYPE TABLE')
print('=' * 120)
display(out[['variable', 'State03', 'State06', 'State03_missing', 'State06_missing', 'effect_measure', 'effect_size', 'p_value', 'FDR_BH', 'reviewer_safe_interpretation']])
print('\nFDR < 0.05 variables:')
print(significant_after_FDR if significant_after_FDR else 'NONE')
print('\nCREATED:')
print(F554)
print(F555)
print('\nPASS: no imputation')
print('PASS: no state modification')
print('PASS: no performance-based exclusion')
print('PASS: missingness explicitly reported')
binary_labels = ['FLT3-ITD', 'FLT3-TKD', 'NPM1 mutation', 'CEBPA mutation', 'IDH1 mutation', 'IDH2 mutation', 'RUNX1 mutation', 'RUNX1::RUNX1T1 fusion']
mask_binary = out['variable'].isin(binary_labels)
out.loc[mask_binary, 'effect_measure'] = 'odds ratio State06/State03'
out.loc[out['variable'].eq('Sex'), 'effect_measure'] = 'odds ratio from 2×2 sex table'
F554_CORR = AUDIT_DIR / '554A_R5MS_state03_state06_subtype_table_OR_corrected.csv'
out.to_csv(F554_CORR, index=False)
print('CREATED:', F554_CORR)
display(out[['variable', 'State03', 'State06', 'State03_missing', 'State06_missing', 'effect_measure', 'effect_size', 'p_value', 'FDR_BH', 'reviewer_safe_interpretation']])
