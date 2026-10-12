# Manuscript-linked analysis: Global held-out response comparison
# Source lineage and packaging changes: provenance/source_lineage.json
import os as _release_os
try:
    from IPython.display import display
except ImportError:

    def display(value):
        print(value)
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score
try:
    pass
    pass
except Exception:
    pass
PROJECT = Path(_release_os.environ.get('AML_GRAPHRX_ROOT', '/content/drive/MyDrive/AML_GraphRx'))
AUDIT = PROJECT / 'results' / 'audit'
AUDIT.mkdir(parents=True, exist_ok=True)
F625 = AUDIT / '625_R7B2_BeatAML_axis_scores.csv'
F629 = AUDIT / '629_R7B2_final_summary.csv'
F639 = AUDIT / '639_R7C_final_summary.csv'
F666 = AUDIT / '666_R8H2_axis_celltype_overlap_ledger.csv'
EXPECTED_PARENT_SHA = {F625.name: '3b9a414709e26efe2a359f9b5b3e29e937923accb9f67554a451bba162ee1c9c', F666.name: '2d1539a5a13d7ca2b677c22832744ff2a8d91b6f49c70c659b30a342adc9cf48', F639.name: 'ad61859f4ff2c0eb5c5cdf0afcc57ae8e9eef19306301030e79c67adcba0443d', F629.name: '5ef24268ccd8247e7e1f01ee51093a0e34ad2d6f5c653da0fe32a52b382b402c'}
RESPONSE_FILENAME = 'beataml_probit_curve_fits_v4_dbgap.txt'
EXPECTED_RESPONSE_SIZE = 19047078
EXPECTED_RESPONSE_SHA = 'd4bc5f0d91f66314107411e0f2511adc987e29df5b83d9b03df56d3d12928314'
CELL_TYPES = ['HSC-like', 'Prog-like', 'GMP-like', 'ProMono-like', 'Mono-like', 'cDC-like']
AXIS = 'R7B2_frozen_pathway_axis_score'
TRAIN_WAVE = 'Waves1+2'
TEST_WAVE = 'Waves3+4'
MIN_TRAIN = 60
MIN_TEST = 30
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
CV_FOLDS = 5
BOOTSTRAP_N = 2000
SEED = 20261010
VERSION = 'AML_GRAPHRX_R9P_GLOBAL_BLINDED_V1'
F718 = AUDIT / '718_R9P_GLOBAL_predeclared_protocol.json'
F719 = AUDIT / '719_R9P_GLOBAL_source_integrity.csv'
F720 = AUDIT / '720_R9P_GLOBAL_endpoint_eligibility_anonymous.csv'
F721 = AUDIT / '721_R9P_GLOBAL_analysis_QC.csv'
F722 = AUDIT / '722_R9P_GLOBAL_bootstrap.csv'
F723 = AUDIT / '723_R9P_GLOBAL_final_summary.csv'
F724 = AUDIT / '724_R9P_GLOBAL_manifest.csv'

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    path = Path(path)
    require(path.is_file(), f'Missing file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def json_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')

def write_once_bytes(path, data):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == data, f'Existing output differs; original retained:\n{path}')
        print('PASS existing identical:', path.name)
        return
    tmp = path.with_name(path.name + '.partial')
    tmp.write_bytes(data)
    os.replace(tmp, path)
    print('CREATED:', path.name)

def write_json(path, obj):
    write_once_bytes(path, json_bytes(obj))

def write_csv(path, df, sort_cols=None):
    x = df.copy()
    if sort_cols:
        valid = [c for c in sort_cols if c in x.columns]
        if valid:
            x = x.sort_values(valid, kind='mergesort').reset_index(drop=True)
    data = x.to_csv(index=False, lineterminator='\n', float_format='%.17g').encode('utf-8')
    write_once_bytes(path, data)

def truth(series):
    s = series.astype(str).str.strip().str.lower()
    allowed = {'true', 'false', '1', '0'}
    unexpected = set(s.unique()) - allowed
    require(len(unexpected) == 0, f'Unexpected boolean encoding: {sorted(unexpected)}')
    return s.isin(['true', '1'])

def anon_endpoint_id(value):
    digest = hashlib.sha256(('AMLGraphRx_R9P_GLOBAL|' + str(value).strip()).encode('utf-8')).hexdigest()
    return 'RESP_' + digest[:12]

def find_real_response_file():
    search_roots = [Path(_release_os.environ.get('AML_GRAPHRX_BEATAML_DIR', '/content/beataml2.0_data_r7b')), PROJECT / 'external', PROJECT / 'data']
    candidates = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        direct = root / RESPONSE_FILENAME
        if direct.is_file():
            key = str(direct.resolve())
            if key not in seen:
                seen.add(key)
                candidates.append(direct)
        try:
            for p in root.rglob(RESPONSE_FILENAME):
                if not p.is_file():
                    continue
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    candidates.append(p)
        except Exception:
            pass
    checked = []
    for p in candidates:
        size = p.stat().st_size
        if size != EXPECTED_RESPONSE_SIZE:
            checked.append({'path': str(p), 'bytes': size, 'status': 'SIZE_MISMATCH'})
            continue
        observed_sha = sha256_file(p)
        if observed_sha == EXPECTED_RESPONSE_SHA:
            return p
        checked.append({'path': str(p), 'bytes': size, 'sha256': observed_sha, 'status': 'SHA_MISMATCH'})
    raise RuntimeError(f'\nREAL BEATAML RESPONSE PAYLOAD NOT FOUND.\n\nRequired filename: {RESPONSE_FILENAME}\nExpected bytes: {EXPECTED_RESPONSE_SIZE}\nExpected SHA256: {EXPECTED_RESPONSE_SHA}\n\nThe small Git-LFS pointer is not accepted.\nPlace the authorized real payload somewhere under AML_GraphRx/external/ and rerun this exact code.\n\nChecked candidates: {checked}')

def fit_ridge_train_only(X, y, seed):
    require(len(y) >= MIN_TRAIN, 'Too few training observations.')
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge())])
    search = GridSearchCV(estimator=pipe, param_grid={'ridge__alpha': ALPHAS}, scoring='neg_mean_squared_error', cv=cv, n_jobs=1, refit=True, error_score='raise')
    search.fit(X, y)
    return search
for p in [F625, F666, F639, F629]:
    require(p.is_file(), f'Missing frozen parent: {p}')
    observed = sha256_file(p)
    expected = EXPECTED_PARENT_SHA[p.name]
    require(observed == expected, f'Frozen parent SHA mismatch:\n{p.name}\nObserved: {observed}\nExpected: {expected}')
print('PASS: exact frozen R7B2/R7C/R8H2 parent files verified.')
for path, flag in [(F629, 'R7B2_EXTERNAL_PATHWAY_AXIS_SUPPORTED'), (F639, 'R7C_RANK_COMPOSITION_ROBUSTNESS_SUPPORTED')]:
    temp = pd.read_csv(path, dtype=str)
    require({'metric', 'value'}.issubset(temp.columns), f'Unexpected summary schema: {path.name}')
    values = dict(zip(temp['metric'].astype(str), temp['value'].astype(str)))
    require(values.get(flag, '').lower() == 'true', f'Required support flag is not TRUE: {flag}')
print('PASS: frozen R7B2/R7C support flags verified.')
RESPONSE_FILE = find_real_response_file()
print('PASS: real BeatAML response payload verified.')
print('Response payload path:', RESPONSE_FILE)
print('Response payload SHA256:', sha256_file(RESPONSE_FILE))
protocol = {'protocol_id': VERSION, 'parents': EXPECTED_PARENT_SHA, 'response_payload_filename': RESPONSE_FILENAME, 'response_payload_bytes': EXPECTED_RESPONSE_SIZE, 'response_payload_sha256': EXPECTED_RESPONSE_SHA, 'scientific_question': 'Does the frozen pathway axis improve held-out prediction of the global ex-vivo response landscape beyond the six frozen malignant-cell-state scores?', 'baseline_predictors': CELL_TYPES, 'augmented_predictors': CELL_TYPES + [AXIS], 'training_split': TRAIN_WAVE, 'heldout_split': TEST_WAVE, 'response_row_eligibility': ['type == single-agent', 'status == train/test', 'paper_inclusion == TRUE', 'converged == TRUE', 'RNA sample ID present', 'finite reported AUC'], 'endpoint_eligibility': {'minimum_train_n': MIN_TRAIN, 'minimum_test_n': MIN_TEST, 'selection_basis': 'availability only'}, 'response_scaling': 'Each anonymous endpoint is standardized using Waves1+2 mean and SD only. The same training parameters are applied to Waves3+4.', 'model': 'standardized ridge regression fitted separately for each anonymous endpoint', 'alpha_grid': ALPHAS, 'inner_cv': '5-fold shuffled CV in Waves1+2 only', 'primary_global_metric': 'macro-average heldout delta-MSE across all availability-eligible anonymous endpoints; delta-MSE = baseline MSE - augmented MSE', 'inference': '2000 heldout subject-cluster bootstrap replicates; models are not refit', 'support_rule': 'observed macro delta-MSE > 0 AND bootstrap 95% lower bound > 0', 'endpoint_specific_significance_testing': False, 'endpoint_names_written_to_outputs': False, 'endpoint_ranking': False, 'test_set_tuning': False, 'seed': SEED}
write_json(F718, protocol)
source_audit = pd.DataFrame([{'role': 'response_payload', 'filename': RESPONSE_FILE.name, 'bytes': RESPONSE_FILE.stat().st_size, 'sha256': sha256_file(RESPONSE_FILE), 'status': 'EXACT_VERIFIED_REAL_PAYLOAD'}, {'role': 'frozen_axis', 'filename': F625.name, 'bytes': F625.stat().st_size, 'sha256': sha256_file(F625), 'status': 'FROZEN_PARENT_VERIFIED'}, {'role': 'frozen_cellstate_ledger', 'filename': F666.name, 'bytes': F666.stat().st_size, 'sha256': sha256_file(F666), 'status': 'FROZEN_PARENT_VERIFIED'}])
write_csv(F719, source_audit, ['role'])
axis_df = pd.read_csv(F625, dtype={'ptid': str, 'external_sample_id': str, 'cohort': str})
cell_df = pd.read_csv(F666, dtype={'ptid': str, 'external_sample_id': str, 'cohort': str})
KEYS = ['ptid', 'external_sample_id', 'cohort']
for d, name in [(axis_df, '625'), (cell_df, '666')]:
    require(len(d) == 560, f'{name}: expected 560 rows, found {len(d)}.')
    require(d['ptid'].is_unique, f'{name}: ptid must be unique.')
    require(d['external_sample_id'].is_unique, f'{name}: external_sample_id must be unique.')
    require(set(d['cohort'].astype(str)) == {TRAIN_WAVE, TEST_WAVE}, f'{name}: unexpected cohort labels.')
required_axis_cols = KEYS + [AXIS]
required_cell_cols = KEYS + CELL_TYPES + [AXIS]
require(set(required_axis_cols).issubset(axis_df.columns), f'625 missing columns: {set(required_axis_cols) - set(axis_df.columns)}')
require(set(required_cell_cols).issubset(cell_df.columns), f'666 missing columns: {set(required_cell_cols) - set(cell_df.columns)}')
paired = axis_df[required_axis_cols].merge(cell_df[required_cell_cols], on=KEYS, how='inner', suffixes=('', '_check'), validate='one_to_one')
require(len(paired) == 560, 'Frozen axis/cell-state ledgers do not align 1:1.')
require(np.allclose(paired[AXIS].to_numpy(dtype=float), paired[AXIS + '_check'].to_numpy(dtype=float), rtol=0, atol=1e-12), 'Frozen axis values disagree between 625 and 666.')
paired = paired.drop(columns=[AXIS + '_check'])
require(np.isfinite(paired[CELL_TYPES + [AXIS]].to_numpy(dtype=float)).all(), 'Missing/nonfinite frozen predictors.')
print('PASS: 560 frozen axis/cell-state records aligned exactly.')
response = pd.read_csv(RESPONSE_FILE, sep='\t', dtype={'dbgap_subject_id': str, 'dbgap_rnaseq_sample': str}, low_memory=False)
required_response_cols = {'dbgap_subject_id', 'dbgap_rnaseq_sample', 'inhibitor', 'type', 'status', 'paper_inclusion', 'converged', 'auc'}
require(required_response_cols.issubset(response.columns), f'Unexpected response schema. Missing: {required_response_cols - set(response.columns)}')
mask = response['type'].astype(str).str.strip().eq('single-agent') & response['status'].astype(str).str.strip().eq('train/test') & truth(response['paper_inclusion']) & truth(response['converged']) & response['dbgap_rnaseq_sample'].notna()
response = response.loc[mask].copy()
response['auc'] = pd.to_numeric(response['auc'], errors='coerce')
response = response.loc[np.isfinite(response['auc'].to_numpy(dtype=float))].copy()
response['ptid_response'] = 'pt' + response['dbgap_subject_id'].astype(str).str.strip().str.replace('\\.0$', '', regex=True)
response = response.rename(columns={'dbgap_rnaseq_sample': 'external_sample_id'})
response['response_id'] = response['inhibitor'].astype(str).str.strip().map(anon_endpoint_id)
require(response['response_id'].nunique() == response['inhibitor'].astype(str).str.strip().nunique(), 'Endpoint anonymization collision detected.')
response = response.drop(columns=['inhibitor'])
require(not response.duplicated(['external_sample_id', 'response_id']).any(), 'Duplicate RNA-sample/anonymous-endpoint response rows found. No automatic averaging is allowed.')
merged = response.merge(paired, on='external_sample_id', how='inner', validate='many_to_one')
require(len(merged) > 0, 'No response rows matched the frozen RNA samples.')
require((merged['ptid'].astype(str) == merged['ptid_response'].astype(str)).all(), 'Subject-ID contradiction between response table and frozen RNA ledger.')
require(not merged.duplicated(['ptid', 'response_id']).any(), 'More than one response row exists for the same subject/anonymous endpoint.')
train_subjects = set(merged.loc[merged['cohort'].eq(TRAIN_WAVE), 'ptid'].astype(str))
test_subjects = set(merged.loc[merged['cohort'].eq(TEST_WAVE), 'ptid'].astype(str))
require(len(train_subjects & test_subjects) == 0, 'A subject appears in both Waves1+2 and Waves3+4.')
elig_rows = []
for response_id, d in merged.groupby('response_id', sort=True):
    train_n = int(d.loc[d['cohort'].eq(TRAIN_WAVE)].shape[0])
    test_n = int(d.loc[d['cohort'].eq(TEST_WAVE)].shape[0])
    eligible = bool(train_n >= MIN_TRAIN and test_n >= MIN_TEST)
    elig_rows.append({'response_id': response_id, 'train_n': train_n, 'test_n': test_n, 'eligible': eligible})
elig = pd.DataFrame(elig_rows)
write_csv(F720, elig, ['response_id'])
eligible_ids = elig.loc[elig['eligible'], 'response_id'].tolist()
require(len(eligible_ids) > 0, f'No anonymous endpoint met train >= {MIN_TRAIN} and test >= {MIN_TEST}.')
qc = pd.DataFrame([{'frozen_predictor_subjects_n': len(paired), 'matched_response_subjects_n': len(train_subjects | test_subjects), 'matched_train_subjects_n': len(train_subjects), 'matched_test_subjects_n': len(test_subjects), 'anonymous_endpoints_total_n': len(elig), 'anonymous_endpoints_eligible_n': len(eligible_ids), 'train_test_subject_overlap_n': len(train_subjects & test_subjects), 'endpoint_selection_uses_model_performance': False, 'TEST_used_for_tuning': False}])
write_csv(F721, qc)
print('\n' + '=' * 100)
print('R9P GLOBAL PRE-MODEL QC')
print('=' * 100)
print(qc.to_string(index=False))
endpoint_internal = {}
delta_mse_observed = []
delta_r2_observed = []
for number, response_id in enumerate(eligible_ids, start=1):
    d = merged.loc[merged['response_id'].eq(response_id)].copy()
    tr = d.loc[d['cohort'].eq(TRAIN_WAVE)].sort_values('ptid').copy()
    te = d.loc[d['cohort'].eq(TEST_WAVE)].sort_values('ptid').copy()
    require(len(tr) >= MIN_TRAIN, 'Internal training-count error.')
    require(len(te) >= MIN_TEST, 'Internal test-count error.')
    y_train_raw = tr['auc'].to_numpy(dtype=float)
    y_test_raw = te['auc'].to_numpy(dtype=float)
    train_mean = float(np.mean(y_train_raw))
    train_sd = float(np.std(y_train_raw, ddof=1))
    if not np.isfinite(train_sd) or train_sd <= 1e-12:
        print(f'Skipped {number}/{len(eligible_ids)} anonymous endpoint: constant TRAIN outcome.')
        continue
    y_train = (y_train_raw - train_mean) / train_sd
    y_test = (y_test_raw - train_mean) / train_sd
    endpoint_seed = int(hashlib.sha256((response_id + '|' + str(SEED)).encode('utf-8')).hexdigest()[:8], 16)
    baseline = fit_ridge_train_only(tr[CELL_TYPES].to_numpy(dtype=float), y_train, endpoint_seed)
    pred_base = baseline.predict(te[CELL_TYPES].to_numpy(dtype=float))
    augmented_cols = CELL_TYPES + [AXIS]
    augmented = fit_ridge_train_only(tr[augmented_cols].to_numpy(dtype=float), y_train, endpoint_seed)
    pred_plus = augmented.predict(te[augmented_cols].to_numpy(dtype=float))
    base_mse = float(np.mean((y_test - pred_base) ** 2))
    plus_mse = float(np.mean((y_test - pred_plus) ** 2))
    delta_mse = base_mse - plus_mse
    delta_mse_observed.append(delta_mse)
    if np.var(y_test) > 1e-12:
        base_r2 = float(r2_score(y_test, pred_base))
        plus_r2 = float(r2_score(y_test, pred_plus))
        delta_r2_observed.append(plus_r2 - base_r2)
    endpoint_internal[response_id] = {'subject_ids': te['ptid'].astype(str).to_numpy(), 'y': y_test, 'pred_base': pred_base, 'pred_plus': pred_plus}
    print(f'Completed {number}/{len(eligible_ids)} anonymous response endpoints.')
evaluable_ids = sorted(endpoint_internal.keys())
require(len(evaluable_ids) > 0, 'No eligible endpoint was technically evaluable after TRAIN-only variance QC.')
observed_macro_delta_mse = float(np.mean(delta_mse_observed))
observed_median_delta_mse = float(np.median(delta_mse_observed))
fraction_endpoints_improved = float(np.mean(np.asarray(delta_mse_observed, dtype=float) > 0))
if len(delta_r2_observed):
    observed_median_delta_r2 = float(np.median(delta_r2_observed))
else:
    observed_median_delta_r2 = np.nan
rng = np.random.default_rng(SEED)
test_subject_universe = np.array(sorted(test_subjects), dtype=object)
require(len(test_subject_universe) >= MIN_TEST, 'Too few heldout subjects for bootstrap.')
boot_rows = []
for b in range(BOOTSTRAP_N):
    sampled_subjects = rng.choice(test_subject_universe, size=len(test_subject_universe), replace=True)
    endpoint_deltas = []
    for response_id in evaluable_ids:
        store = endpoint_internal[response_id]
        lookup = {sid: i for i, sid in enumerate(store['subject_ids'])}
        idx = [lookup[sid] for sid in sampled_subjects if sid in lookup]
        if len(idx) < 5:
            continue
        idx = np.asarray(idx, dtype=int)
        yb = store['y'][idx]
        pb = store['pred_base'][idx]
        pp = store['pred_plus'][idx]
        base_mse_b = float(np.mean((yb - pb) ** 2))
        plus_mse_b = float(np.mean((yb - pp) ** 2))
        endpoint_deltas.append(base_mse_b - plus_mse_b)
    require(len(endpoint_deltas) >= max(1, int(0.8 * len(evaluable_ids))), 'Too few endpoints represented in a bootstrap replicate.')
    boot_rows.append({'bootstrap_index': b + 1, 'macro_delta_MSE': float(np.mean(endpoint_deltas)), 'endpoints_contributing_n': len(endpoint_deltas)})
boot = pd.DataFrame(boot_rows)
require(len(boot) == BOOTSTRAP_N, 'Bootstrap replicate count changed unexpectedly.')
write_csv(F722, boot, ['bootstrap_index'])
ci_low = float(np.quantile(boot['macro_delta_MSE'], 0.025))
ci_high = float(np.quantile(boot['macro_delta_MSE'], 0.975))
bootstrap_positive_fraction = float(np.mean(boot['macro_delta_MSE'].to_numpy(dtype=float) > 0))
GLOBAL_SUPPORTED = bool(observed_macro_delta_mse > 0 and ci_low > 0)
summary = pd.DataFrame([('protocol_id', VERSION), ('training_wave', TRAIN_WAVE), ('heldout_wave', TEST_WAVE), ('frozen_predictor_subjects_n', len(paired)), ('matched_response_subjects_n', len(train_subjects | test_subjects)), ('matched_train_subjects_n', len(train_subjects)), ('matched_test_subjects_n', len(test_subjects)), ('anonymous_endpoints_total_n', len(elig)), ('anonymous_endpoints_count_eligible_n', len(eligible_ids)), ('anonymous_endpoints_technically_evaluable_n', len(evaluable_ids)), ('observed_macro_delta_MSE', observed_macro_delta_mse), ('observed_median_endpoint_delta_MSE', observed_median_delta_mse), ('fraction_endpoints_with_lower_test_MSE_after_axis', fraction_endpoints_improved), ('observed_median_endpoint_delta_R2_descriptive', observed_median_delta_r2), ('bootstrap_macro_delta_MSE_q025', ci_low), ('bootstrap_macro_delta_MSE_q975', ci_high), ('bootstrap_positive_fraction', bootstrap_positive_fraction), ('R9P_GLOBAL_INCREMENT_SUPPORTED', GLOBAL_SUPPORTED), ('TEST_used_for_tuning', False), ('endpoint_specific_significance_testing', False), ('endpoint_names_written_to_outputs', False), ('endpoint_ranking_performed', False), ('frozen_axis_changed', False), ('cell_state_scores_changed', False), ('claim_boundary', 'global heldout ex-vivo response-prediction increment of the frozen pathway axis beyond six transcriptome-derived malignant-cell-state scores; not endpoint-specific sensitivity, clinical response, treatment benefit, causal effect, or treatment utility')], columns=['metric', 'value'])
write_csv(F723, summary)
manifest_rows = []
for role, path in [('protocol', F718), ('source_integrity', F719), ('endpoint_eligibility_anonymous', F720), ('analysis_QC', F721), ('bootstrap', F722), ('final_summary', F723)]:
    manifest_rows.append({'role': role, 'path': str(path), 'sha256': sha256_file(path)})
write_csv(F724, pd.DataFrame(manifest_rows), ['role'])
print('\n' + '=' * 110)
print('R9P GLOBAL COMPLETE — BLINDED HELDOUT INCREMENTAL RESPONSE TEST')
print('=' * 110)
print('\nSOURCE INTEGRITY')
print(source_audit.to_string(index=False))
print('\nANALYSIS QC')
print(qc.to_string(index=False))
print('\nGLOBAL HELDOUT RESULT')
print(summary.to_string(index=False))
print('\nIMPORTANT')
print('- TRAIN = Waves1+2 only.')
print('- TEST = Waves3+4 only.')
print('- TEST was never used for scaling or alpha selection.')
print('- Baseline = six frozen malignant-cell-state scores.')
print('- Augmented = same six scores + frozen pathway axis.')
print('- Endpoint inclusion was based on availability only.')
print('- Response identities were anonymized and are not written to outputs.')
print('- No endpoint-specific ranking or significance testing was performed.')
print('- No GNN/state/pathway/cell-state score was retrained.')
print('- If R9P_GLOBAL_INCREMENT_SUPPORTED is False, do not retune the axis.')
