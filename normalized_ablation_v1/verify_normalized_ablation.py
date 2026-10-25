"""Verify saved normalized-ablation results; does not train or re-cluster data."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'verification/recomputed_checks.json')
    args=ap.parse_args(); n=args.results; checks=[]
    def check(name,ok):
        if not bool(ok):raise ValueError(name)
        checks.append({'check':name,'passed':True})
    def close(a,b):return np.isclose(a,b,rtol=0,atol=1e-12,equal_nan=True)
    manifest=pd.read_csv(n/'sensitivity_outputs_manifest.csv')
    for r in manifest.itertuples():
        f=(n/r.path).resolve()
        check('Manifest path is inside results: '+r.path,f.is_relative_to(n.resolve()))
        check('SHA256: '+r.path,hashlib.sha256(f.read_bytes()).hexdigest()==r.sha256)
        check('Bytes: '+r.path,f.stat().st_size==r.bytes)
    protocol=json.loads((n/'sensitivity_protocol_and_inputs.json').read_text())
    m=pd.read_csv(n/'normalized_arm_metrics.csv')
    p=pd.read_csv(n/'normalized_cross_seed_pairwise.csv')
    s=pd.read_csv(n/'normalized_arm_summary.csv')
    seeds=[20261031,20261101,20261102]
    arms=['REAL_GRAPH','DEGREE_PRESERVING_REWIRE','NO_MESSAGE_PASSING']
    check('Exact recorded three seeds',protocol['seeds']==seeds)
    check('All nine arm and seed combinations retained',len(m)==9 and set(zip(m.arm,m.seed))=={(a,k) for a in arms for k in seeds})
    expected_pairs={(a,seeds[i],seeds[j]) for a in arms for i in range(3) for j in range(i+1,3)}
    check('All nine pair comparisons retained',len(p)==9 and set(zip(p.arm,p.seed_a,p.seed_b))==expected_pairs)
    cache={}
    for r in m.itertuples():
        with np.load(n/f'cache_normalized_{r.seed}_{r.arm}.npz',allow_pickle=False) as z:
            a={k:z[k] for k in z.files}
        cache[r.arm,r.seed]=a; labels=a['labels']
        check(f'{r.arm}/{r.seed}: 1733 assignments',len(labels)==len(a['raw'])==1733)
        check(f'{r.arm}/{r.seed}: coverage',close((labels>=0).mean(),r.recurrent_coverage))
        check(f'{r.arm}/{r.seed}: community count',len(np.unique(labels[labels>=0]))==r.recurrent_states_n)
        check(f'{r.arm}/{r.seed}: original embedding identity',str(a['source_sha256'])==r.embedding_sha256==protocol['embedding_sha256'][f'{r.arm}_{r.seed}'])
    recalculated=[]
    for r in p.itertuples():
        a,b=cache[r.arm,r.seed_a],cache[r.arm,r.seed_b]; la,lb=a['labels'],b['labels']
        ua,ub=np.unique(la[la>=0]),np.unique(lb[lb>=0])
        sa=np.array([(la==u).sum() for u in ua]); sb=np.array([(lb==u).sum() for u in ub])
        den=sa.sum()+sb.sum()
        if len(ua) and len(ub):
            j=np.array([[np.sum((la==u)&(lb==v))/np.sum((la==u)|(lb==v)) for v in ub] for u in ua])
            ia,ib=linear_sum_assignment(-j)
            score=float(np.sum((sa[ia]+sb[ib])*j[ia,ib])/den)
        else:
            score=0.0 if den else np.nan
        joint=float(((la>=0)&(lb>=0)).mean())
        ari=float(adjusted_rand_score(a['raw'],b['raw']))
        name=f'{r.arm}/{r.seed_a}/{r.seed_b}'
        check(name+': matched Jaccard',close(score,r.matched_recurrent_weighted_Jaccard))
        check(name+': joint coverage',close(joint,r.jointly_recurrent_fraction))
        check(name+': raw partition ARI',close(ari,r.raw_partition_ARI_all1733))
        recalculated.append({'arm':r.arm,'seed_a':int(r.seed_a),'seed_b':int(r.seed_b),'matched_jaccard':None if np.isnan(score) else score,'joint_coverage':joint,'raw_partition_ARI':ari})
    for r in s.itertuples():
        arm_m=m[m.arm.eq(r.arm)];arm_p=p[p.arm.eq(r.arm)]
        check(r.arm+': arm mean',close(arm_m.recurrent_coverage.mean(),r.mean_coverage))
        check(r.arm+': evaluable-pair median',close(arm_p.matched_recurrent_weighted_Jaccard.median(),r.median_matched_recurrent_Jaccard))
    out={'scope':'Verification of archived outputs and assignments only. No GNN training, normalization, projection, or clustering rerun.','checks':checks,'recomputed_pairs':recalculated,'seeds':seeds}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(f'PASS: {len(checks)} checks; all nine fits and nine pairs retained.')
    print(m.groupby('arm').recurrent_coverage.mean().mul(100).round(6).to_string())
if __name__=='__main__':main()
