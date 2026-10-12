"""Replay the already-reported conditional resampling with authentic subjects.

This verification adds no scientific analysis. It extracts the unchanged
rank-adjustment functions from the released script and compares every draw.
"""
from pathlib import Path
import ast, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
tree=ast.parse((ROOT/'analysis/conditional_association.py').read_text())
scope={'np':np,'pd':pd,'rankdata':rankdata}
names={'WAVES','CELL','XCOL','YCOL','SEED','N_BOOT','N_PERM'}
keep=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id in names for x in node.targets):keep.append(node)
    if isinstance(node,ast.FunctionDef) and node.name in {'require','rank_scaled','residualized_effect'}:keep.append(node)
exec(compile(ast.Module(body=keep,type_ignores=[]),'<unchanged conditional functions>','exec'),scope)
scores=pd.read_csv(ROOT/'source_tables/context/625_R7B2_BeatAML_axis_scores.csv',dtype={'ptid':str})
cell=pd.read_csv(ROOT/'source_tables/context/666_R8H2_axis_celltype_overlap_ledger.csv',dtype={'ptid':str})
keys=['ptid','external_sample_id','cohort']
d=scores.merge(cell[keys+list(scope['CELL'])],on=keys,validate='one_to_one')
stored_b=pd.read_csv(ROOT/'source_tables/new/743_R9Q_bootstrap_distribution.csv')
stored_p=pd.read_csv(ROOT/'source_tables/new/744_R9Q_permutation_distribution.csv')
rng=np.random.default_rng(scope['SEED']);report=[]
for wave in scope['WAVES']:
    dd=d.loc[d.cohort.eq(wave)].sort_values('ptid').reset_index(drop=True)
    adjusted,ex,ey,_=scope['residualized_effect'](dd)
    boots=[]
    for _ in range(scope['N_BOOT']):
        ix=rng.integers(0,len(dd),size=len(dd))
        boots.append(scope['residualized_effect'](dd.iloc[ix])[0])
    perms=[float(np.corrcoef(ex,rng.permutation(ey))[0,1]) for _ in range(scope['N_PERM'])]
    eb=stored_b.loc[stored_b.wave.eq(wave),'partial_rank_r'].to_numpy()
    ep=stored_p.loc[stored_p.wave.eq(wave),'null_partial_rank_r'].to_numpy()
    np.testing.assert_allclose(boots,eb,rtol=0,atol=1e-10)
    np.testing.assert_allclose(perms,ep,rtol=0,atol=1e-10)
    report.append({'wave':wave,'n':len(dd),'adjusted':adjusted,'bootstrap_draws':len(boots),'permutation_draws':len(perms),'max_bootstrap_absolute_difference':float(np.max(np.abs(np.asarray(boots)-eb))),'max_permutation_absolute_difference':float(np.max(np.abs(np.asarray(perms)-ep)))})
    print('PASS',wave,': all 2,000 bootstrap and 2,000 permutation draws match')
out=ROOT/'verification';out.mkdir(exist_ok=True)
(out/'conditional_resampling_replay.json').write_text(json.dumps(report,indent=2))
