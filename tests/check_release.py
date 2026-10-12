"""Validate release structure, seeds, hashes and reported run populations."""
from pathlib import Path
import ast, json, hashlib, csv
ROOT=Path(__file__).resolve().parents[1]
catalog=json.loads((ROOT/'config/analyses.json').read_text())
settings={x['analysis']:x['constants'] for x in json.loads((ROOT/'config/random_settings.json').read_text())}
for path in ROOT.rglob('*.py'):
    compile(path.read_text(),str(path),'exec')
for row in json.loads((ROOT/'provenance/source_lineage.json').read_text()):
    assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()==row['packaged_sha256'],row['file']
assert len(catalog)==18
assert settings['gene_discovery']['SEED']==20261004
assert settings['representation_ablation']['SEED']==20261004
assert settings['representation_ablation']['RAND_SEEDS']==list(range(20261011,20261016))
assert settings['representation_ablation']['SHUFFLE_SEEDS']==list(range(20261021,20261026))
expected=[20261031,20261101,20261102]
for stage in ['message_ablation','cross_seed_consistency']:assert settings[stage]['SEEDS']==expected
assert settings['message_ablation']['REWIRE_SEED']==20261029
assert settings['external_rnaseq']['BOOTSTRAP_SEED']==20261009
assert settings['conditional_association']['SEED']==20261010
assert settings['conditional_association']['N_BOOT']==2000
assert settings['conditional_association']['N_PERM']==2000
def rows(path):
    with (ROOT/path).open() as f:return list(csv.DictReader(f))
a=rows('source_tables/new/753_R10A_arm_state_discovery_metrics.csv')
b=rows('source_tables/new/764_R10B_arm_discovery_metrics.csv')
c=rows('source_tables/new/771_R10C_cross_seed_pairwise_consistency.csv')
assert len(a)==12 and len(b)==9 and len(c)==9
assert sorted({int(r['seed']) for r in b})==expected
for arm in ['REAL_GRAPH','DEGREE_PRESERVING_REWIRE','NO_MESSAGE_PASSING']:
    assert sorted(int(r['seed']) for r in b if r['arm']==arm)==expected
    assert len([r for r in c if r['arm']==arm])==3
assert all(x['matches_included_source'] for x in json.loads((ROOT/'provenance/drive_protocol_verification.json').read_text()))
print('PASS: 18 scripts compile; source hashes and frozen seeds match; all 12 representation runs, 9 message fits and 9 seed-pair comparisons are retained.')
