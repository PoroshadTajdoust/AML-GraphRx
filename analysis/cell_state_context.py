# Manuscript-linked analysis: Corrected six-score malignant-cell-state analysis
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
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
F650 = AUDIT / '650_R8G_final_summary.csv'
F651 = AUDIT / '651_R8G_manifest.csv'
F652 = AUDIT / '652_R8H_predeclared_malignant_hierarchy_protocol.json'
F653 = AUDIT / '653_R8H_BeatAML_celltype_source_audit.csv'
BEATAML_REPO = 'https://github.com/biodev/beataml2.0_data.git'
BEATAML_TAG = 'v3.0'
BEATAML_LOCAL = Path(_release_os.environ.get('AML_GRAPHRX_BEATAML_DIR', '/content/beataml2.0_data_r7b'))
CELLTYPE_FILE = BEATAML_LOCAL / 'beataml2_manuscript_vg_cts.xlsx'
F663 = AUDIT / '663_R8H2_vanGalen_source_correction_amendment.json'
F664 = AUDIT / '664_R8H2_workbook_layout_audit.csv'
F665 = AUDIT / '665_R8H2_normalized_celltype_matrix.csv'
F666 = AUDIT / '666_R8H2_axis_celltype_overlap_ledger.csv'
F667 = AUDIT / '667_R8H2_cross_wave_celltype_axis_associations.csv'
F668 = AUDIT / '668_R8H2_replicated_celltype_axis_associations.csv'
F669 = AUDIT / '669_R8H2_hierarchy_contrast_summary.csv'
F670 = AUDIT / '670_R8H2_celltype_ridge_predictions.csv'
F671 = AUDIT / '671_R8H2_celltype_reducibility_summary.csv'
F672 = AUDIT / '672_R8H2_final_summary.csv'
F673 = AUDIT / '673_R8H2_manifest.csv'
CELL_TYPES = ['HSC-like', 'Prog-like', 'GMP-like', 'ProMono-like', 'Mono-like', 'cDC-like']
PRIMITIVE = ['HSC-like', 'Prog-like']
INTERMEDIATE = ['GMP-like']
DIFFERENTIATED = ['ProMono-like', 'Mono-like', 'cDC-like']
MIN_OVERLAP_TOTAL = 100
MIN_OVERLAP_PER_WAVE = 30
DISCOVERY_FDR = 0.05
VALIDATION_FDR = 0.05
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
RIDGE_CV_FOLDS = 5
RIDGE_SEED = 20261010

def require(x, msg):
    if not bool(x):
        raise RuntimeError(msg)

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

def norm_name(x):
    return re.sub('[^a-z0-9]+', '', str(x).lower())

def run(cmd, env=None):
    print('$', ' '.join(map(str, cmd)))
    subprocess.run(cmd, check=True, env=env)
ALIASES = {'HSC-like': {'hsclike', 'hsc_like'}, 'Prog-like': {'proglike', 'prog_like', 'progenitorlike'}, 'GMP-like': {'gmplike', 'gmp_like'}, 'ProMono-like': {'promonolike', 'promono_like', 'promonocytelike'}, 'Mono-like': {'monolike', 'mono_like', 'monocytelike'}, 'cDC-like': {'cdclike', 'cdc_like', 'conventionaldendriticcelllike'}}
ALIASES_NORM = {k: {norm_name(v) for v in vals} | {norm_name(k)} for k, vals in ALIASES.items()}

def canonical_celltype(x):
    nx = norm_name(x)
    hits = [k for k, vals in ALIASES_NORM.items() if nx in vals]
    return hits[0] if len(hits) == 1 else None
for p in [F625, F629, F630, F639, F640, F650, F651, F652, F653]:
    require(p.is_file(), f'Missing parent artifact: {p}')
m630 = pd.read_csv(F630)
for p in [F625, F629]:
    r = m630.loc[m630['path'].astype(str).map(lambda z: Path(z).name).eq(p.name)]
    require(len(r) == 1 and str(r['sha256'].iloc[0]) == sha256_file(p), f'{p.name} integrity mismatch.')
m640 = pd.read_csv(F640)
r = m640.loc[m640['path'].astype(str).map(lambda z: Path(z).name).eq(F639.name)]
require(len(r) == 1 and str(r['sha256'].iloc[0]) == sha256_file(F639), '639 integrity mismatch.')
m651 = pd.read_csv(F651)
r = m651.loc[m651['path'].astype(str).map(lambda z: Path(z).name).eq(F650.name)]
require(len(r) == 1 and str(r['sha256'].iloc[0]) == sha256_file(F650), '650 integrity mismatch.')
s639df = pd.read_csv(F639)
s639 = dict(zip(s639df['metric'].astype(str), s639df['value'].astype(str)))
require(s639.get('R7C_RANK_COMPOSITION_ROBUSTNESS_SUPPORTED', '').lower() == 'true', 'R7C support flag is not TRUE.')
amendment = {'protocol_id': 'AML_GRAPHRX_R8H2_VANGALEN_HIERARCHY_CORRECTION_V1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'failed_R8H_v1_protocol_sha256': sha256_file(F652), 'failed_R8H_v1_source_audit_sha256': sha256_file(F653), 'parent_axis_scores_sha256': sha256_file(F625), 'reason': 'R8H v1 stopped before any cell-type/pathway-axis association because it incorrectly expected seven Zeng/LSPC cell types. The BeatAML v3.0 file beataml2_manuscript_vg_cts.xlsx contains six van Galen-derived malignant cell-type expression scores described by Bottomly et al. 2022.', 'correct_source_cell_types': CELL_TYPES, 'scientific_role': 'external biological contextualization of the already-frozen pathway axis against the six published BeatAML van Galen malignant cell-type scores', 'claim_boundary': 'these are transcriptome-derived scores; therefore this is convergent transcriptomic contextualization, not orthogonal validation or causality', 'layout_detection': 'support row-wise wide, transposed-wide, or long workbook layouts; layout is selected solely by exact controlled cell-type labels plus maximum overlap with frozen R7B2 identifiers; ambiguous ties hard-stop', 'association_rule': {'test': 'Spearman correlation', 'Waves1+2': 'BH FDR <=0.05', 'Waves3+4': 'BH FDR <=0.05', 'replication': 'same direction and FDR significant in both waves'}, 'predeclared_hierarchy_contrast': {'primitive': PRIMITIVE, 'intermediate': INTERMEDIATE, 'differentiated': DIFFERENTIATED, 'contrast': 'mean(primitive HSC/Prog) - mean(differentiated ProMono/Mono/cDC)', 'GMP_like_role': 'intermediate; excluded from endpoint contrast'}, 'forbidden': ['change frozen pathway-axis scores', 'choose cell types based on association result', 'drop a wave', 'replace van Galen states with Zeng LSPC states in this file', 'call cell-type score associations orthogonal validation']}
write_json(F663, amendment)
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
if not CELLTYPE_FILE.is_file() or CELLTYPE_FILE.stat().st_size < 1024:
    run(['git', '-C', str(BEATAML_LOCAL), 'lfs', 'pull', f'--include={CELLTYPE_FILE.name}', '--exclude='])
require(CELLTYPE_FILE.is_file() and CELLTYPE_FILE.stat().st_size > 1024, 'BeatAML cell-type workbook unavailable.')
old_source = pd.read_csv(F653)
oldrow = old_source.loc[old_source['file'].eq(CELLTYPE_FILE.name)]
require(len(oldrow) == 1, '653 does not contain the expected cell-type file.')
require(str(oldrow['sha256'].iloc[0]) == sha256_file(CELLTYPE_FILE), 'Cell-type workbook differs from R8H-v1 source audit.')
scores = pd.read_csv(F625, dtype=str, keep_default_na=False)
require(len(scores) == 560, 'Expected 560 R7B2 subjects.')
required = {'ptid', 'external_sample_id', 'cohort', 'R7B2_frozen_pathway_axis_score'}
require(required.issubset(scores.columns), f'625 missing {required - set(scores.columns)}')
scores['R7B2_frozen_pathway_axis_score'] = pd.to_numeric(scores['R7B2_frozen_pathway_axis_score'], errors='raise')
require(scores['ptid'].is_unique, '625 ptid duplicated.')
require(scores['external_sample_id'].is_unique, '625 external_sample_id duplicated.')
id_spaces = {'ptid': set(scores['ptid'].astype(str)), 'external_sample_id': set(scores['external_sample_id'].astype(str)), 'ptid_numeric': set(scores['ptid'].astype(str).str.replace('^pt', '', regex=True))}
xls = pd.ExcelFile(CELLTYPE_FILE, engine='openpyxl')
layout_candidates = []
diagnostic_rows = []
for sheet in xls.sheet_names:
    df = pd.read_excel(CELLTYPE_FILE, sheet_name=sheet, engine='openpyxl')
    df.columns = [str(c).strip() for c in df.columns]
    diagnostic_rows.append({'sheet': sheet, 'layout_probe': 'SHEET', 'rows': len(df), 'cols': len(df.columns), 'detail': ';'.join(map(str, df.columns[:30]))})
    wide_ct = {}
    for col in df.columns:
        can = canonical_celltype(col)
        if can:
            wide_ct.setdefault(can, []).append(col)
    if all((k in wide_ct and len(wide_ct[k]) == 1 for k in CELL_TYPES)):
        for id_col in df.columns:
            vals = set(df[id_col].astype(str).str.strip())
            for ns, target in id_spaces.items():
                ov = len(vals & target)
                if ov > 0:
                    layout_candidates.append({'sheet': sheet, 'layout': 'SAMPLE_ROWS_WIDE', 'overlap_n': ov, 'namespace': ns, 'id_column': id_col, 'celltype_columns': {k: wide_ct[k][0] for k in CELL_TYPES}, 'label_column': None, 'value_column': None})
    for label_col in df.columns:
        labels = df[label_col].astype(str).map(canonical_celltype)
        present = set(labels.dropna())
        if set(CELL_TYPES).issubset(present):
            counts = labels.value_counts()
            if not all((int(counts.get(k, 0)) == 1 for k in CELL_TYPES)):
                continue
            for ns, target in id_spaces.items():
                sample_cols = [c for c in df.columns if str(c).strip() in target]
                ov = len(sample_cols)
                if ov > 0:
                    layout_candidates.append({'sheet': sheet, 'layout': 'CELLTYPE_ROWS_TRANSPOSED', 'overlap_n': ov, 'namespace': ns, 'id_column': None, 'celltype_columns': None, 'label_column': label_col, 'value_column': None})
    for label_col in df.columns:
        can_series = df[label_col].astype(str).map(canonical_celltype)
        if len(set(can_series.dropna())) < len(CELL_TYPES):
            continue
        for id_col in df.columns:
            if id_col == label_col:
                continue
            vals = set(df[id_col].astype(str).str.strip())
            for ns, target in id_spaces.items():
                ov = len(vals & target)
                if ov == 0:
                    continue
                numeric_cols = []
                for vc in df.columns:
                    if vc in {label_col, id_col}:
                        continue
                    z = pd.to_numeric(df[vc], errors='coerce')
                    if int(z.notna().sum()) >= 6:
                        numeric_cols.append(vc)
                if len(numeric_cols) == 1:
                    layout_candidates.append({'sheet': sheet, 'layout': 'LONG', 'overlap_n': ov, 'namespace': ns, 'id_column': id_col, 'celltype_columns': None, 'label_column': label_col, 'value_column': numeric_cols[0]})
require(len(layout_candidates) > 0, 'No supported layout detected. R8H2 did not infer or fabricate a schema.')
maxov = max((x['overlap_n'] for x in layout_candidates))
best = [x for x in layout_candidates if x['overlap_n'] == maxov]
uniq = {}
for x in best:
    key = (x['sheet'], x['layout'], x['namespace'], x['id_column'], x['label_column'], x['value_column'], json.dumps(x['celltype_columns'], sort_keys=True) if x['celltype_columns'] else '')
    uniq[key] = x
best = list(uniq.values())
require(len(best) == 1, 'Ambiguous best workbook mapping. Candidates: ' + str([(x['sheet'], x['layout'], x['namespace'], x['overlap_n'], x['id_column'], x['label_column'], x['value_column']) for x in best]))
best = best[0]
require(best['overlap_n'] >= MIN_OVERLAP_TOTAL, f"Best workbook overlap is too small: {best['overlap_n']}")
for x in layout_candidates:
    diagnostic_rows.append({'sheet': x['sheet'], 'layout_probe': 'CANDIDATE', 'rows': np.nan, 'cols': np.nan, 'detail': json.dumps({'layout': x['layout'], 'namespace': x['namespace'], 'overlap_n': x['overlap_n'], 'id_column': x['id_column'], 'label_column': x['label_column'], 'value_column': x['value_column'], 'celltype_columns': x['celltype_columns'], 'selected': x is best}, sort_keys=True)})
layout_audit = pd.DataFrame(diagnostic_rows)
write_csv(F664, layout_audit, ['sheet', 'layout_probe'])
print('SELECTED WORKBOOK LAYOUT:', best)
df = pd.read_excel(CELLTYPE_FILE, sheet_name=best['sheet'], engine='openpyxl')
df.columns = [str(c).strip() for c in df.columns]
ns = best['namespace']
if best['layout'] == 'SAMPLE_ROWS_WIDE':
    out = pd.DataFrame()
    rawid = df[best['id_column']].astype(str).str.strip()
    out['join_id'] = rawid
    for ct in CELL_TYPES:
        out[ct] = pd.to_numeric(df[best['celltype_columns'][ct]], errors='coerce')
elif best['layout'] == 'CELLTYPE_ROWS_TRANSPOSED':
    labels = df[best['label_column']].astype(str).map(canonical_celltype)
    rows = {}
    for ct in CELL_TYPES:
        ri = np.where(labels.eq(ct).to_numpy())[0]
        require(len(ri) == 1, f'Expected exactly one {ct} row.')
        rows[ct] = ri[0]
    target = id_spaces[ns]
    sample_cols = [c for c in df.columns if str(c).strip() in target]
    out = pd.DataFrame({'join_id': [str(c).strip() for c in sample_cols]})
    for ct in CELL_TYPES:
        out[ct] = [pd.to_numeric(pd.Series([df.iloc[rows[ct]][c]]), errors='coerce').iloc[0] for c in sample_cols]
elif best['layout'] == 'LONG':
    z = df[[best['id_column'], best['label_column'], best['value_column']]].copy()
    z['join_id'] = z[best['id_column']].astype(str).str.strip()
    z['canonical_cell_type'] = z[best['label_column']].astype(str).map(canonical_celltype)
    z['value'] = pd.to_numeric(z[best['value_column']], errors='coerce')
    z = z.loc[z['canonical_cell_type'].isin(CELL_TYPES)]
    require(not z.duplicated(['join_id', 'canonical_cell_type']).any(), 'Long layout has duplicate subject × cell-type values.')
    out = z.pivot(index='join_id', columns='canonical_cell_type', values='value').reset_index()
else:
    raise RuntimeError('Unexpected layout.')
require(out['join_id'].is_unique, 'Normalized cell-type subject identifiers are duplicated.')
require(set(CELL_TYPES).issubset(out.columns), 'Normalized matrix is missing cell types.')
write_csv(F665, out, ['join_id'])
score_join = scores.copy()
if ns == 'ptid':
    score_join['join_id'] = score_join['ptid'].astype(str)
elif ns == 'external_sample_id':
    score_join['join_id'] = score_join['external_sample_id'].astype(str)
elif ns == 'ptid_numeric':
    score_join['join_id'] = score_join['ptid'].astype(str).str.replace('^pt', '', regex=True)
else:
    raise RuntimeError('Unexpected identifier namespace.')
merged = score_join[['join_id', 'ptid', 'external_sample_id', 'cohort', 'R7B2_frozen_pathway_axis_score']].merge(out[['join_id'] + CELL_TYPES], on='join_id', how='inner', validate='one_to_one')
merged['complete_six_scores'] = merged[CELL_TYPES].notna().all(axis=1)
analysis = merged.loc[merged['complete_six_scores']].copy()
require(len(analysis) >= MIN_OVERLAP_TOTAL, f'Too few complete six-score subjects: {len(analysis)}')
for wave in ['Waves1+2', 'Waves3+4']:
    require(int(analysis['cohort'].eq(wave).sum()) >= MIN_OVERLAP_PER_WAVE, f'Too few subjects in {wave}.')
write_csv(F666, merged, ['cohort', 'ptid'])
assoc_rows = []
for wave in ['Waves1+2', 'Waves3+4']:
    gg = analysis.loc[analysis['cohort'].eq(wave)]
    y = gg['R7B2_frozen_pathway_axis_score'].to_numpy(float)
    for ct in CELL_TYPES:
        x = gg[ct].to_numpy(float)
        rr = spearmanr(x, y)
        assoc_rows.append({'wave': wave, 'cell_type': ct, 'n': len(gg), 'spearman_rho': float(rr.statistic), 'p_two_sided': float(rr.pvalue)})
assoc = pd.DataFrame(assoc_rows)
assoc['BH_FDR_within_wave'] = np.nan
for wave, idx in assoc.groupby('wave').groups.items():
    assoc.loc[idx, 'BH_FDR_within_wave'] = bh_fdr(assoc.loc[idx, 'p_two_sided'].to_numpy(float))
write_csv(F667, assoc, ['wave', 'cell_type'])
a12 = assoc.loc[assoc['wave'].eq('Waves1+2')].set_index('cell_type')
a34 = assoc.loc[assoc['wave'].eq('Waves3+4')].set_index('cell_type')
rep_rows = []
for ct in CELL_TYPES:
    r12 = float(a12.loc[ct, 'spearman_rho'])
    r34 = float(a34.loc[ct, 'spearman_rho'])
    q12 = float(a12.loc[ct, 'BH_FDR_within_wave'])
    q34 = float(a34.loc[ct, 'BH_FDR_within_wave'])
    rep_rows.append({'cell_type': ct, 'Waves1+2_rho': r12, 'Waves1+2_FDR': q12, 'Waves3+4_rho': r34, 'Waves3+4_FDR': q34, 'same_direction': bool(np.sign(r12) == np.sign(r34) and r12 != 0 and (r34 != 0)), 'cross_wave_replicated': bool(q12 <= DISCOVERY_FDR and q34 <= VALIDATION_FDR and (np.sign(r12) == np.sign(r34)) and (r12 != 0) and (r34 != 0))})
rep = pd.DataFrame(rep_rows)
write_csv(F668, rep, ['cell_type'])
analysis['primitive_score'] = analysis[PRIMITIVE].mean(axis=1)
analysis['intermediate_GMP_score'] = analysis[INTERMEDIATE].mean(axis=1)
analysis['differentiated_score'] = analysis[DIFFERENTIATED].mean(axis=1)
analysis['primitive_minus_differentiated'] = analysis['primitive_score'] - analysis['differentiated_score']
contrast_rows = []
for wave in ['Waves1+2', 'Waves3+4']:
    gg = analysis.loc[analysis['cohort'].eq(wave)]
    y = gg['R7B2_frozen_pathway_axis_score'].to_numpy(float)
    for name in ['primitive_score', 'intermediate_GMP_score', 'differentiated_score', 'primitive_minus_differentiated']:
        rr = spearmanr(gg[name].to_numpy(float), y)
        contrast_rows.append({'wave': wave, 'hierarchy_metric': name, 'n': len(gg), 'spearman_rho': float(rr.statistic), 'p_two_sided': float(rr.pvalue)})
contrast = pd.DataFrame(contrast_rows)
contrast['BH_FDR_within_wave'] = np.nan
for wave, idx in contrast.groupby('wave').groups.items():
    contrast.loc[idx, 'BH_FDR_within_wave'] = bh_fdr(contrast.loc[idx, 'p_two_sided'].to_numpy(float))
write_csv(F669, contrast, ['wave', 'hierarchy_metric'])
train = analysis.loc[analysis['cohort'].eq('Waves1+2')].copy()
test = analysis.loc[analysis['cohort'].eq('Waves3+4')].copy()
X12 = train[CELL_TYPES].to_numpy(float)
X34 = test[CELL_TYPES].to_numpy(float)
y12 = train['R7B2_frozen_pathway_axis_score'].to_numpy(float)
y34 = test['R7B2_frozen_pathway_axis_score'].to_numpy(float)
pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge())])
cv = KFold(n_splits=RIDGE_CV_FOLDS, shuffle=True, random_state=RIDGE_SEED)
grid = GridSearchCV(pipe, param_grid={'ridge__alpha': RIDGE_ALPHAS}, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, refit=True)
grid.fit(X12, y12)
pred12 = grid.predict(X12)
pred34 = grid.predict(X34)
rho34 = spearmanr(y34, pred34)
pred_rows = []
for wave, frame, obs, pred in [('Waves1+2', train, y12, pred12), ('Waves3+4', test, y34, pred34)]:
    for pt, o, p in zip(frame['ptid'], obs, pred):
        pred_rows.append({'ptid': pt, 'wave': wave, 'observed_axis': float(o), 'predicted_axis_from_6_vanGalen_scores': float(p)})
write_csv(F670, pd.DataFrame(pred_rows), ['wave', 'ptid'])
reducibility = pd.DataFrame([('celltype_features_n', 6), ('selected_ridge_alpha_training_only', float(grid.best_params_['ridge__alpha'])), ('Waves1+2_training_R2_descriptive', float(r2_score(y12, pred12))), ('Waves3+4_test_R2', float(r2_score(y34, pred34))), ('Waves3+4_test_MAE', float(mean_absolute_error(y34, pred34))), ('Waves3+4_test_Spearman', float(rho34.statistic)), ('Waves3+4_test_Spearman_p', float(rho34.pvalue))], columns=['metric', 'value'])
write_csv(F671, reducibility)
n_rep = int(rep['cross_wave_replicated'].sum())
n_same = int(rep['same_direction'].sum())
hc = contrast.loc[contrast['hierarchy_metric'].eq('primitive_minus_differentiated')].copy()
require(len(hc) == 2, 'Hierarchy contrast missing one wave.')
h1 = hc.loc[hc['wave'].eq('Waves1+2')].iloc[0]
h2 = hc.loc[hc['wave'].eq('Waves3+4')].iloc[0]
hierarchy_contrast_replicated = bool(float(h1['BH_FDR_within_wave']) <= 0.05 and float(h2['BH_FDR_within_wave']) <= 0.05 and (np.sign(float(h1['spearman_rho'])) == np.sign(float(h2['spearman_rho']))) and (float(h1['spearman_rho']) != 0) and (float(h2['spearman_rho']) != 0))
context_supported = bool(n_rep >= 1 or hierarchy_contrast_replicated)
final_summary = pd.DataFrame([('protocol_id', 'AML_GRAPHRX_R8H2_VANGALEN_HIERARCHY_CORRECTION_V1'), ('selected_workbook_layout', best['layout']), ('selected_workbook_sheet', best['sheet']), ('selected_identifier_namespace', best['namespace']), ('workbook_identifier_overlap_n', best['overlap_n']), ('complete_six_score_analysis_n', len(analysis)), ('Waves1+2_complete_n', len(train)), ('Waves3+4_complete_n', len(test)), ('vanGalen_malignant_celltypes_n', 6), ('same_direction_celltype_associations_n', n_same), ('cross_wave_FDR_replicated_celltypes_n', n_rep), ('hierarchy_contrast_Waves1+2_rho', float(h1['spearman_rho'])), ('hierarchy_contrast_Waves1+2_FDR', float(h1['BH_FDR_within_wave'])), ('hierarchy_contrast_Waves3+4_rho', float(h2['spearman_rho'])), ('hierarchy_contrast_Waves3+4_FDR', float(h2['BH_FDR_within_wave'])), ('hierarchy_contrast_cross_wave_replicated', hierarchy_contrast_replicated), ('celltype_only_Waves3+4_test_R2', float(r2_score(y34, pred34))), ('celltype_only_Waves3+4_test_Spearman', float(rho34.statistic)), ('celltype_only_Waves3+4_test_Spearman_p', float(rho34.pvalue)), ('R8H2_VANGALEN_HIERARCHY_CONTEXT_SUPPORTED', context_supported), ('claim_boundary', 'external biological contextualization with six transcriptome-derived van Galen malignant-cell scores; not orthogonal validation, causal differentiation, or discrete subtype proof'), ('R8G_context', 'common mutations did not show cross-wave FDR replication and mutation-only held-out reconstruction was poor; no claim of genetic independence'), ('next_step_if_supported', 'freeze R8H2 and stop transcriptomic score development; integrate R7B2/R7C/R8G/R8H2 into manuscript before any separately predeclared pharmacogenomic analysis'), ('next_step_if_not_supported', 'retain null hierarchy contextualization and stop adding transcriptomic reinterpretation layers')], columns=['metric', 'value'])
write_csv(F672, final_summary)
manifest = []
for role, p in [('source_correction_amendment', F663), ('workbook_layout_audit', F664), ('normalized_celltype_matrix', F665), ('axis_celltype_overlap', F666), ('cross_wave_associations', F667), ('replicated_associations', F668), ('hierarchy_contrast', F669), ('ridge_predictions', F670), ('celltype_reducibility', F671), ('final_summary', F672)]:
    manifest.append({'role': role, 'path': str(p), 'sha256': sha256_file(p)})
write_csv(F673, pd.DataFrame(manifest), ['role'])
try:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.scatter(rep['Waves1+2_rho'], rep['Waves3+4_rho'], s=38)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    for _, r in rep.iterrows():
        ax.text(r['Waves1+2_rho'], r['Waves3+4_rho'], r['cell_type'], fontsize=8)
    ax.set_xlabel('Waves1+2 rho with frozen pathway axis')
    ax.set_ylabel('Waves3+4 rho with frozen pathway axis')
    ax.set_title('R8H2 van Galen malignant-cell context')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'R8H2_vanGalen_cross_wave_300dpi.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR / 'R8H2_vanGalen_cross_wave.pdf', bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(y34, pred34, s=22, alpha=0.7)
    ax.set_xlabel('Observed frozen pathway-axis score')
    ax.set_ylabel('Predicted from six van Galen scores')
    ax.set_title('R8H2 Waves3+4 hierarchy reducibility')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'R8H2_vanGalen_external_prediction_300dpi.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR / 'R8H2_vanGalen_external_prediction.pdf', bbox_inches='tight')
    plt.close(fig)
except Exception as e:
    print('WARNING: figure generation skipped:', repr(e))
print('\n' + '=' * 120)
print('R8H2 COMPLETE — CORRECTED VAN GALEN MALIGNANT-CELL HIERARCHY')
print('=' * 120)
print('\nSELECTED WORKBOOK LAYOUT')
print(pd.DataFrame([{'sheet': best['sheet'], 'layout': best['layout'], 'namespace': best['namespace'], 'overlap_n': best['overlap_n']}]).to_string(index=False))
print('\nCROSS-WAVE CELL-TYPE ASSOCIATIONS')
print(assoc.to_string(index=False))
print('\nREPLICATED CELL-TYPE ASSOCIATIONS')
hits = rep.loc[rep['cross_wave_replicated']]
print(hits.to_string(index=False) if len(hits) else 'No cross-wave FDR-replicated van Galen cell-type associations.')
print('\nHIERARCHY CONTRAST')
print(contrast.to_string(index=False))
print('\nCELL-TYPE REDUCIBILITY')
print(reducibility.to_string(index=False))
print('\nFINAL SUMMARY')
print(final_summary.to_string(index=False))
print('\nIMPORTANT')
print('- R8H-v1 stopped before any association and remains preserved.')
print('- R8H2 uses the correct six van Galen malignant cell-type scores described for BeatAML.')
print('- The frozen R7A2/R7B2/R7C pathway axis was not changed.')
print('- All six cell-type scores are tested; none is selected post hoc.')
print('- These cell-type scores are transcriptome-derived, so this is contextualization, not orthogonal validation.')
