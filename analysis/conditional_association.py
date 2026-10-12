# Manuscript-linked analysis: Six-covariate rank adjustment and resampling
# Source lineage and packaging changes: provenance/source_lineage.json
import os as _release_os
try:
    from IPython.display import display
except ImportError:

    def display(value):
        print(value)
from pathlib import Path
from io import BytesIO
import hashlib
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import matplotlib.pyplot as plt
try:
    pass
    pass
except ImportError:
    pass
ROOT = Path(_release_os.environ.get('AML_GRAPHRX_ROOT', '/content/drive/MyDrive/AML_GraphRx'))
AUDIT = ROOT / 'results' / 'audit'
AUDIT.mkdir(parents=True, exist_ok=True)
F625 = AUDIT / '625_R7B2_BeatAML_axis_scores.csv'
F666 = AUDIT / '666_R8H2_axis_celltype_overlap_ledger.csv'
F630 = AUDIT / '630_R7B2_manifest.csv'
F673 = AUDIT / '673_R8H2_manifest.csv'
F629 = AUDIT / '629_R7B2_final_summary.csv'
F639 = AUDIT / '639_R7C_final_summary.csv'
EXPECTED_SHA = {F625.name: '3b9a414709e26efe2a359f9b5b3e29e937923accb9f67554a451bba162ee1c9c', F666.name: '2d1539a5a13d7ca2b677c22832744ff2a8d91b6f49c70c659b30a342adc9cf48', F629.name: '5ef24268ccd8247e7e1f01ee51093a0e34ad2d6f5c653da0fe32a52b382b402c', F639.name: 'ad61859f4ff2c0eb5c5cdf0afcc57ae8e9eef19306301030e79c67adcba0443d'}
WAVES = ('Waves1+2', 'Waves3+4')
EXPECTED_WAVE_N = {'Waves1+2': 372, 'Waves3+4': 188}
EXPECTED_WAVE_RHO = {'Waves1+2': -0.7096100513931298, 'Waves3+4': -0.6202052710187879}
EXPECTED_POOLED_RHO = -0.6853854763567486
CELL = ['HSC-like', 'Prog-like', 'GMP-like', 'ProMono-like', 'Mono-like', 'cDC-like']
XCOL = 'S03_high_component_mean'
YCOL = 'S06_high_component_mean'
AXIS = 'R7B2_frozen_pathway_axis_score'
SEED = 20261010
N_BOOT = 2000
N_PERM = 2000
VERSION = 'AML_GRAPHRX_R9Q_CELLSTATE_CONDITIONAL_BEATAML_V1'
F740 = AUDIT / '740_R9Q_preanalysis_protocol.json'
F741 = AUDIT / '741_R9Q_input_integrity.csv'
F742 = AUDIT / '742_R9Q_wave_conditional_effects.csv'
F743 = AUDIT / '743_R9Q_bootstrap_distribution.csv'
F744 = AUDIT / '744_R9Q_permutation_distribution.csv'
F745 = AUDIT / '745_R9Q_summary.csv'
F746 = AUDIT / '746_R9Q_manifest.csv'
F747PNG = AUDIT / '747_R9Q_raw_vs_cellstate_adjusted.png'
F747PDF = AUDIT / '747_R9Q_raw_vs_cellstate_adjusted.pdf'

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)

def sha(path):
    path = Path(path)
    require(path.is_file(), f'Missing required file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_once(path, payload):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == payload, f'Existing output is different. Original file preserved:\n{path.name}')
        print('PASS unchanged:', path.name)
        return
    temp = path.with_name(path.name + '.partial')
    temp.write_bytes(payload)
    os.replace(temp, path)
    print('CREATED:', path.name)

def save_json(path, obj):
    payload = (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    write_once(path, payload)

def save_csv(path, frame, keys=None):
    z = frame.copy()
    if keys is not None:
        z = z.sort_values(keys, kind='mergesort').reset_index(drop=True)
    payload = z.to_csv(index=False, lineterminator='\n', float_format='%.17g').encode('utf-8')
    write_once(path, payload)

def check_manifest(manifest_path, target_path):
    m = pd.read_csv(manifest_path, dtype=str)
    require({'path', 'sha256'}.issubset(m.columns), f'Manifest schema mismatch: {manifest_path.name}')
    match = m.loc[m['path'].map(lambda x: Path(str(x)).name).eq(target_path.name)]
    require(len(match) == 1, f'Not uniquely in manifest: {target_path.name}')
    require(str(match['sha256'].iloc[0]) == sha(target_path), f'Manifest SHA mismatch: {target_path.name}')

def rank_scaled(values):
    v = np.asarray(values, dtype=float)
    return rankdata(v, method='average') / (len(v) + 1.0)

def residualized_effect(frame):
    n = len(frame)
    rx = rank_scaled(frame[XCOL])
    ry = rank_scaled(frame[YCOL])
    rc = np.column_stack([rank_scaled(frame[c]) for c in CELL])
    z = np.column_stack([np.ones(n), rc])
    bx = np.linalg.lstsq(z, rx, rcond=None)[0]
    by = np.linalg.lstsq(z, ry, rcond=None)[0]
    ex = rx - z @ bx
    ey = ry - z @ by
    require(np.std(ex) > 1e-12, 'S03 residual variance is numerically zero.')
    require(np.std(ey) > 1e-12, 'S06 residual variance is numerically zero.')
    r = float(np.corrcoef(ex, ey)[0, 1])
    design_rank = int(np.linalg.matrix_rank(z))
    return (r, ex, ey, design_rank)

def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    require(np.isfinite(p).all(), 'Non-finite permutation P value.')
    require(((p >= 0) & (p <= 1)).all(), 'Permutation P value outside [0,1].')
    n = len(p)
    order = np.argsort(p, kind='stable')
    ranked_p = p[order]
    q_ranked = ranked_p * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.minimum(q_ranked, 1.0)
    return q
for file in (F625, F666, F629, F639):
    observed = sha(file)
    expected = EXPECTED_SHA[file.name]
    require(observed == expected, f'Frozen parent SHA mismatch:\n{file.name}\nObserved: {observed}\nExpected: {expected}')
for p in (F630, F673):
    require(p.is_file(), f'Missing frozen manifest: {p}')
check_manifest(F630, F625)
check_manifest(F673, F666)
for path, flag in ((F629, 'R7B2_EXTERNAL_PATHWAY_AXIS_SUPPORTED'), (F639, 'R7C_RANK_COMPOSITION_ROBUSTNESS_SUPPORTED')):
    s = pd.read_csv(path, dtype=str)
    require({'metric', 'value'}.issubset(s.columns), f'Unexpected summary schema: {path.name}')
    kv = dict(zip(s['metric'], s['value']))
    require(str(kv.get(flag, '')).strip().lower() == 'true', f'Required support flag is not TRUE: {flag}')
print('PASS: frozen R7B2/R7C/R8H2 integrity verified.')
protocol = {'protocol_id': VERSION, 'role': 'post-hoc fixed sensitivity analysis in previously assessed BeatAML subjects', 'scientific_question': 'Does external S03-high vs S06-high opposition remain after adjustment for all six malignant-cell-state scores?', 'inputs': {p.name: sha(p) for p in (F625, F666, F629, F639, F630, F673)}, 'frozen_components': [XCOL, YCOL], 'axis_identity_check': f'{AXIS} == {YCOL} - {XCOL}', 'covariates': CELL, 'stratification': list(WAVES), 'raw_effect': 'Spearman correlation within each wave', 'adjusted_effect': 'Pearson correlation between residuals obtained after separately OLS-residualizing rank-transformed S03 and S06 component scores on an intercept and all six rank-transformed cell-state scores.', 'bootstrap': {'replicates_per_wave': N_BOOT, 'sampling_unit': 'subject', 'rerank_and_refit_covariate_residualization': True, 'seed': SEED}, 'permutation': {'replicates_per_wave': N_PERM, 'method': 'shuffle one residual vector within wave', 'tail': 'negative', 'seed': SEED, 'P_floor': f'1/{N_PERM + 1}', 'limitation': 'Approximate residual-permutation sensitivity test; not exact causal inference.'}, 'multiplicity': 'Benjamini-Hochberg over the two wave-level permutation P values', 'result_label': 'Conditional transcriptomic association; not independent external or orthogonal validation', 'no_drug_response': True, 'no_training_or_modification_of_GNN': True, 'no_change_to_frozen_pathway_axis': True, 'claim_boundaries': ['All adjusted variables arise from transcriptomes.', 'Two wave groups are subsets of one BeatAML study.', 'Previous raw correlations were already observed.', 'Residual correlation does not establish causality.', 'No prognostic or treatment response claim is tested.']}
save_json(F740, protocol)
a = pd.read_csv(F625, dtype={'ptid': str, 'external_sample_id': str, 'cohort': str})
b = pd.read_csv(F666, dtype={'ptid': str, 'external_sample_id': str, 'cohort': str})
keys = ['ptid', 'external_sample_id', 'cohort']
require(set(keys + [XCOL, YCOL, AXIS]).issubset(a.columns), '625 schema mismatch.')
require(set(keys + CELL + [AXIS]).issubset(b.columns), '666 schema mismatch.')
for label, z in (('625', a), ('666', b)):
    require(len(z) == 560, f'{label}: expected 560 samples, found {len(z)}.')
    require(not z['ptid'].duplicated().any(), f'{label}: duplicated ptid.')
    require(not z['external_sample_id'].duplicated().any(), f'{label}: duplicated external_sample_id.')
    require(set(z['cohort']) == set(WAVES), f'{label}: unexpected wave labels.')
d = a[keys + [XCOL, YCOL, AXIS]].merge(b[keys + CELL + [AXIS]], on=keys, how='inner', validate='one_to_one', suffixes=('', '_from666'))
require(len(d) == 560, 'Frozen 625/666 merge did not preserve all 560 samples.')
numeric_cols = [XCOL, YCOL, AXIS, AXIS + '_from666'] + CELL
for c in numeric_cols:
    d[c] = pd.to_numeric(d[c], errors='raise')
require(np.isfinite(d[numeric_cols].to_numpy(float)).all(), 'Nonfinite input scores.')
require(np.max(np.abs(d[AXIS] - d[AXIS + '_from666'])) < 1e-12, 'Frozen axis differs between input files.')
require(np.max(np.abs(d[AXIS] - (d[YCOL] - d[XCOL]))) < 1e-12, 'Frozen axis does not equal S06 - S03.')
train_ids = set(d.loc[d['cohort'].eq(WAVES[0]), 'ptid'])
test_ids = set(d.loc[d['cohort'].eq(WAVES[1]), 'ptid'])
require(not train_ids & test_ids, 'Patient appears in both wave groups.')
raw_pooled = float(spearmanr(d[XCOL], d[YCOL]).statistic)
require(abs(raw_pooled - EXPECTED_POOLED_RHO) < 1e-09, f'Pooled frozen Spearman mismatch: {raw_pooled}')
for wave in WAVES:
    di = d.loc[d['cohort'].eq(wave)]
    require(len(di) == EXPECTED_WAVE_N[wave], f'{wave}: sample count changed.')
    observed = float(spearmanr(di[XCOL], di[YCOL]).statistic)
    require(abs(observed - EXPECTED_WAVE_RHO[wave]) < 1e-09, f'{wave}: raw Spearman mismatch: {observed}')
input_audit = pd.DataFrame([{'role': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size} for p in (F625, F666, F629, F639, F630, F673)])
save_csv(F741, input_audit, ['role'])
print('PASS: 560 samples aligned and known raw Spearman values reproduced.')
rng = np.random.default_rng(SEED)
result_rows = []
boot_rows = []
perm_rows = []
plotdata = {}
for wave in WAVES:
    dd = d.loc[d['cohort'].eq(wave)].sort_values('ptid').reset_index(drop=True)
    raw = float(spearmanr(dd[XCOL], dd[YCOL]).statistic)
    adjusted, ex, ey, cov_rank = residualized_effect(dd)
    plotdata[wave] = (dd[XCOL].to_numpy(float), dd[YCOL].to_numpy(float), ex.copy(), ey.copy())
    valid_boot = []
    for i in range(N_BOOT):
        ix = rng.integers(0, len(dd), size=len(dd))
        sampled = dd.iloc[ix]
        try:
            effect, _, _, _ = residualized_effect(sampled)
        except RuntimeError:
            effect = np.nan
        boot_rows.append({'wave': wave, 'resample': i + 1, 'partial_rank_r': effect})
        if np.isfinite(effect):
            valid_boot.append(effect)
    require(len(valid_boot) >= int(0.95 * N_BOOT), f'{wave}: too many invalid bootstrap samples.')
    lo, hi = np.quantile(valid_boot, [0.025, 0.975])
    negative_or_equal = 0
    for i in range(N_PERM):
        perm_y = rng.permutation(ey)
        rho_perm = float(np.corrcoef(ex, perm_y)[0, 1])
        negative_or_equal += int(rho_perm <= adjusted)
        perm_rows.append({'wave': wave, 'permutation': i + 1, 'null_partial_rank_r': rho_perm})
    p_perm = (1 + negative_or_equal) / (N_PERM + 1)
    result_rows.append({'wave': wave, 'n': len(dd), 'nuisance_cell_scores_n': len(CELL), 'covariate_design_rank': cov_rank, 'raw_spearman_r': raw, 'cellscore_adjusted_partial_rank_r': adjusted, 'absolute_r_attenuation': abs(raw) - abs(adjusted), 'bootstrap_CI95_lower': float(lo), 'bootstrap_CI95_upper': float(hi), 'negative_or_equal_null_permutations_n': negative_or_equal, 'residual_permutation_p_lower': p_perm, 'no_compound_or_clinical_endpoint_used': True})
    print(f'{wave}: n={len(dd)} | raw rho={raw:.4f} | adjusted r={adjusted:.4f} | 95% CI=[{lo:.4f}, {hi:.4f}] | null count={negative_or_equal}/{N_PERM}')
res = pd.DataFrame(result_rows)
res['BH_q_two_wave_permutation'] = bh_fdr(res['residual_permutation_p_lower'].to_numpy(float))
res['negative_conditional_association_with_CI'] = (res['cellscore_adjusted_partial_rank_r'] < 0) & (res['bootstrap_CI95_upper'] < 0)
res['negative_conditional_association_permutation_BH'] = (res['cellscore_adjusted_partial_rank_r'] < 0) & (res['BH_q_two_wave_permutation'] <= 0.05)
k = len(CELL)
weights = res['n'].to_numpy(float) - k - 3.0
values = np.clip(res['cellscore_adjusted_partial_rank_r'].to_numpy(float), -0.999999, 0.999999)
pooled_fisher_r = float(np.tanh(np.average(np.arctanh(values), weights=weights)))
save_csv(F742, res, ['wave'])
save_csv(F743, pd.DataFrame(boot_rows), ['wave', 'resample'])
save_csv(F744, pd.DataFrame(perm_rows), ['wave', 'permutation'])
summary = pd.DataFrame([('protocol_id', VERSION), ('analysis_type', 'post-hoc conditional sensitivity on previously used BeatAML'), ('BeatAML_subjects_n', 560), ('Waves1+2_n', 372), ('Waves3+4_n', 188), ('pooled_raw_rho_reproduced', raw_pooled), ('descriptive_fisher_weighted_conditional_r', pooled_fisher_r), ('both_wave_conditional_r_negative', bool((res['cellscore_adjusted_partial_rank_r'] < 0).all())), ('both_wave_bootstrap_upper_below_zero', bool((res['bootstrap_CI95_upper'] < 0).all())), ('both_wave_permutation_BH_le_005', bool((res['BH_q_two_wave_permutation'] <= 0.05).all())), ('claim_boundary', 'Within-BeatAML conditional transcriptomic association only; not evidence of GNN advantage, orthogonal mechanism, prognostic value or drug sensitivity.')], columns=['metric', 'value'])
save_csv(F745, summary)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), constrained_layout=True)
wave_colors = {'Waves1+2': '#1761a0', 'Waves3+4': '#bf6732'}
for i, wave in enumerate(WAVES):
    xx, yy, rx, ry = plotdata[wave]
    color = wave_colors[wave]
    pairs = ((xx, yy), (rx, ry))
    for j, (px, py) in enumerate(pairs):
        ax = axes[i, j]
        ax.scatter(px, py, s=16, alpha=0.55, color=color, rasterized=True, edgecolor='none')
        ax.grid(True, alpha=0.18)
        if j == 0:
            coefficient = float(res.loc[res['wave'].eq(wave), 'raw_spearman_r'].iloc[0])
            ax.set_xlabel('Frozen S03-high component mean')
            ax.set_ylabel('Frozen S06-high component mean')
            ax.set_title(f'{wave} (n={len(xx)}) | Raw Spearman = {coefficient:.3f}', fontsize=10)
        else:
            ax.axhline(0, color='#777777', lw=0.7, alpha=0.5)
            ax.axvline(0, color='#777777', lw=0.7, alpha=0.5)
            coefficient = float(res.loc[res['wave'].eq(wave), 'cellscore_adjusted_partial_rank_r'].iloc[0])
            ax.set_xlabel('S03 residual rank after six-score adjustment')
            ax.set_ylabel('S06 residual rank after six-score adjustment')
            ax.set_title(f'{wave} | Adjusted partial-rank r = {coefficient:.3f}', fontsize=10)
fig.suptitle('Frozen BeatAML program opposition before and after cell-state score adjustment', fontsize=13, fontweight='bold')
for output, fmt in ((F747PNG, 'png'), (F747PDF, 'pdf')):
    buf = BytesIO()
    if fmt == 'pdf':
        fixed_metadata = {'Creator': 'AML-GraphRx R9Q', 'CreationDate': None, 'ModDate': None}
    else:
        fixed_metadata = {'Software': 'AML-GraphRx R9Q'}
    fig.savefig(buf, format=fmt, dpi=320, bbox_inches='tight', facecolor='white', metadata=fixed_metadata)
    write_once(output, buf.getvalue())
plt.close(fig)
paths = [F740, F741, F742, F743, F744, F745, F747PNG, F747PDF]
manifest = pd.DataFrame([{'file': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths])
save_csv(F746, manifest, ['file'])
print()
print('=' * 110)
print('R9Q COMPLETE — RETROSPECTIVE CONDITIONAL SENSITIVITY')
print('=' * 110)
print()
print('WAVE RESULTS')
print(res.to_string(index=False))
print()
print('SUMMARY')
print(summary.to_string(index=False))
print()
print('OUTPUT DIRECTORY:')
print(AUDIT)
print()
print('IMPORTANT')
print('- Frozen GNN and pathway axis were not changed.')
print('- All six cell-state scores were included.')
print('- Waves were analyzed separately.')
print('- No drug-response outcome was used.')
print('- This is a retrospective within-BeatAML sensitivity analysis.')
print('- Do not interpret it as causal or independent validation.')
