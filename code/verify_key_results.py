"""Verify derived statistics from the included real project tables; no model retraining."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'source_tables/context';N=ROOT/'source_tables/new'
def read(folder,prefix):return pd.read_csv(next(folder.glob(prefix+'*.csv')))
scores=read(C,'625');cell=read(C,'666');effects=read(N,'742');boot=read(N,'743')
keys=['ptid','external_sample_id','cohort']
features=['HSC-like','Prog-like','GMP-like','ProMono-like','Mono-like','cDC-like']
joined=scores.merge(cell[keys+features],on=keys,how='inner',validate='one_to_one')
assert len(joined)==560 and joined[features].notna().all().all()
report={'scope':'Recalculate conditional coefficients, stored bootstrap quantiles and matched Jaccards only. No GNN, partitioning or new bootstrap run.','R9Q':[],'R10C':[]}
for _,r in effects.iterrows():
 z=joined[joined.cohort.eq(r.wave)];n=len(z);assert n==r.n
 array=z[['S03_high_component_mean','S06_high_component_mean']+features].to_numpy(float)
 ranks=np.column_stack([rankdata(array[:,i],method='average')/(n+1) for i in range(array.shape[1])])
 design=np.column_stack([np.ones(n),ranks[:,2:]])
 residuals=ranks[:,:2]-design@np.linalg.lstsq(design,ranks[:,:2],rcond=None)[0]
 coefficient=float(np.corrcoef(residuals.T)[0,1])
 assert abs(coefficient-r.cellscore_adjusted_partial_rank_r)<1e-10
 draws=boot[boot.wave.eq(r.wave)].partial_rank_r;assert len(draws)==2000
 ci=np.quantile(draws,[.025,.975]);assert np.allclose(ci,[r.bootstrap_CI95_lower,r.bootstrap_CI95_upper],atol=1e-10,rtol=0)
 report['R9Q'].append({'wave':r.wave,'n':n,'recomputed_r':coefficient,'stored_bootstrap_95_interval':ci.tolist(),'verified':True})
pairs=read(N,'771');matches=read(N,'772')
for _,r in pairs.iterrows():
 z=matches[matches.arm.eq(r.arm)&matches.seed_a.eq(r.seed_a)&matches.seed_b.eq(r.seed_b)]
 weights=(z.size_a+z.size_b).to_numpy(float)
 coefficient=float(np.dot(weights,z.jaccard)/weights.sum())
 assert abs(coefficient-r.matched_recurrent_weighted_Jaccard)<1e-10
 assert weights.sum()==r.n_recurrent_a+r.n_recurrent_b
 report['R10C'].append({'arm':r.arm,'seed_a':int(r.seed_a),'seed_b':int(r.seed_b),'weighted_jaccard':coefficient,'verified':True})
assert len(report['R10C'])==9
out=ROOT/'verification/recomputed_key_results.json';out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2))
print('Verified both R9Q coefficients, two stored bootstrap intervals and all nine R10C weighted Jaccards.')
