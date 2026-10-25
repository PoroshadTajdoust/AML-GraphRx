# -*- coding: utf-8 -*-
"""AML-GraphRx: retrospective aligned-normalization sensitivity for 9 saved R10B embeddings.

Reuses exactly the helper settings/algorithms from the corrected R10C execution.
No GNN training, no threshold tuning, no selection of a favorable seed.
Results are exploratory and internal to 1733 microarray discovery profiles.
Never writes to results/audit or modifies any frozen input.
"""
from pathlib import Path
from itertools import combinations
import json, hashlib, os, sys, subprocess
from io import BytesIO
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
try:
    import igraph as ig
    import leidenalg
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'python-igraph', 'leidenalg'])
    import igraph as ig
    import leidenalg

PROJECT=Path('/content/drive/MyDrive/AML_GraphRx')
AUDIT=PROJECT/'results'/'audit'
STEP5=PROJECT/'data'/'patient_graphs'/'20260823T123040Z'
WORK=PROJECT/'results'/'r10b_graph_edge_ablation_v1'
OUT=PROJECT/'results'/'normalized_r10b_sensitivity_v1'
RANK=STEP5/'20260823T123040Z_patient_rank_raw_float64.npy'
DISC=AUDIT/'452_R5DICE_discovery_ledger.csv'
F481=AUDIT/'481_R5G_predeclared_protocol.json'
F483=AUDIT/'483_R5G_gene_embeddings.npy'
F486=AUDIT/'486_R5G_patient_projection.npy'
F488=AUDIT/'488_R5G_cross_cohort_MNN_edges.csv'
F489=AUDIT/'489_R5G_Leiden_raw_assignments.csv'
F492=AUDIT/'492_R5G_recurrent_state_patient_assignments.csv'
F497=AUDIT/'497_R5G_manifest.csv'
F760=AUDIT/'760_R10B_protocol.json'
F761=AUDIT/'761_R10B_parent_integrity.csv'
F762=AUDIT/'762_R10B_graph_rewiring_audit.csv'
F763=AUDIT/'763_R10B_training_history.csv'
F764=AUDIT/'764_R10B_arm_discovery_metrics.csv'
F765=AUDIT/'765_R10B_paired_descriptive_contrasts.csv'
F766=AUDIT/'766_R10B_final_summary.csv'
F767=AUDIT/'767_R10B_manifest.csv'
SEEDS=[20261031, 20261101, 20261102]
ARMS=['REAL_GRAPH','DEGREE_PRESERVING_REWIRE','NO_MESSAGE_PASSING']
N_GENES=9554; D=32; N_PATIENTS=1733; K=10; N_BOOT=30; EDGE_KEEP=.80
RES=1.0; MIN_N=40; MIN_COH=4; MAX_COH_SHARE=.4; MIN_JAC=.60
SPLIT_SEED=20261004
GSES=['GSE15434','GSE161532','GSE1729','GSE34860','GSE37021','GSE37642','GSE6891','GSE71014']
GSE_COUNTS=[251,61,43,78,97,562,537,104]

def require(cond,msg):
    if not bool(cond): raise RuntimeError(msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def write_once(path,payload):
    path=Path(path)
    if path.exists():
        require(path.read_bytes()==payload,'Existing result differs; preserved file: '+str(path))
        print('PASS existing',path.name,flush=True)
        return
    temp=path.with_name(path.name+'.partial')
    temp.write_bytes(payload)
    os.replace(temp,path)
    print('CREATED',path.name,flush=True)

def save_csv(path,df,keys=None):
    df=df.copy()
    if keys: df=df.sort_values(keys,kind='mergesort').reset_index(drop=True)
    write_once(path,df.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode())

def save_json(path,obj):
    write_once(path,(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())

def normalize_basis(e):
    # Exact normalize_basis from corrected R10C helper, applied ONCE to raw R10B outputs.
    e=np.asarray(e,dtype=np.float32).copy()
    require(e.shape==(N_GENES,D),'Bad basis shape')
    e-=e.mean(axis=0,keepdims=True)
    sd=e.std(axis=0,keepdims=True)
    e/=np.where(sd<1e-8,1.,sd)
    e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
    return e.astype(np.float32)

def patient_projection(X,gses,E):
    P=np.zeros((len(gses),D),dtype=np.float32)
    for g in GSES:
        ix=np.where(gses==g)[0]
        x=X[ix]
        med=np.median(x,axis=0,keepdims=True)
        p=((x-med)@E)/np.sqrt(N_GENES)
        m=np.median(p,axis=0,keepdims=True)
        q25=np.quantile(p,.25,axis=0,keepdims=True)
        q75=np.quantile(p,.75,axis=0,keepdims=True)
        scale=q75-q25
        scale[scale<1e-6]=1.
        p=np.clip((p-m)/scale,-8,8)
        p/=np.maximum(np.linalg.norm(p,axis=1,keepdims=True),1e-12)
        P[ix]=p.astype(np.float32)
    return P

def create_mnn(P,gses):
    edges=[]
    for i,g1 in enumerate(GSES):
        for g2 in GSES[i+1:]:
            a=np.where(gses==g1)[0]; b=np.where(gses==g2)[0]
            sim=P[a]@P[b].T
            ka=min(K,len(b)); kb=min(K,len(a))
            ab=np.argpartition(-sim,kth=ka-1,axis=1)[:,:ka]
            ba=np.argpartition(-sim.T,kth=kb-1,axis=1)[:,:kb]
            back=[set(z.tolist()) for z in ba]
            for ai in range(len(a)):
                for bj in ab[ai]:
                    bj=int(bj)
                    if ai in back[bj] and sim[ai,bj]>0:
                        edges.append((int(a[ai]),int(b[bj]),float(sim[ai,bj])))
    require(bool(edges),'No positive cross-cohort MNN edges')
    return edges

def leiden_graph(n,edges,seed):
    g=ig.Graph(n=n,edges=[(int(a),int(b)) for a,b,w in edges],directed=False)
    g.es['weight']=[float(w) for a,b,w in edges]
    return np.asarray(leidenalg.find_partition(g,leidenalg.RBConfigurationVertexPartition,
        weights='weight',resolution_parameter=RES,seed=int(seed)).membership,dtype=int)

def state_metrics(labels,edges,gses):
    ids=np.unique(labels)
    members={int(c):set(np.flatnonzero(labels==c).tolist()) for c in ids}
    endpoints=np.asarray([(a,b) for a,b,w in edges],dtype=int)
    weights=np.asarray([w for a,b,w in edges],dtype=float)
    n_keep=int(round(EDGE_KEEP*len(edges)))
    rng=np.random.default_rng(SPLIT_SEED+9000)
    js={int(c):[] for c in ids}
    for t in range(N_BOOT):
        idx=np.sort(rng.choice(len(edges),n_keep,replace=False))
        e=[(int(endpoints[j,0]),int(endpoints[j,1]),float(weights[j])) for j in idx]
        lab=leiden_graph(len(gses),e,SPLIT_SEED+t+1)
        cms=[set(np.flatnonzero(lab==c).tolist()) for c in np.unique(lab)]
        for c,s in members.items():
            js[c].append(max(len(s&ss)/len(s|ss) for ss in cms))
    median_j={c:float(np.median(v)) for c,v in js.items()}
    passed=[]
    for c,s in members.items():
        counts=pd.Series(gses[list(s)]).value_counts()
        if len(s)>=MIN_N and len(counts)>=MIN_COH and float(counts.max()/len(s))<=MAX_COH_SHARE and median_j[c]>=MIN_JAC:
            passed.append(c)
    valid=np.isin(labels,passed)
    nmi=float(normalized_mutual_info_score(gses[valid],labels[valid])) if valid.any() and len(set(labels[valid]))>=2 else float('nan')
    return dict(states=len(passed),coverage=float(valid.mean()),
        stability=float(np.median([median_j[c] for c in passed])) if passed else float('nan'),
        cohort_nmi=nmi,labels=np.where(valid,labels,-1),
        mnn_coverage=len({i for a,b,w in edges for i in (a,b)})/len(gses))

def manifest_check(path,file_list):
    m=pd.read_csv(path,dtype=str)
    require({'path','sha256'}.issubset(m.columns),'Manifest schema invalid: '+path.name)
    for p in file_list:
        require(p.is_file(),'MISSING input: '+str(p))
        ss=m.loc[m['path'].map(str).map(lambda x:Path(x).name).eq(p.name)]
        require(len(ss)==1,'Nonunique/missing manifest record: '+p.name)
        require(str(ss.sha256.iloc[0])==sha(p),'SHA mismatch: '+p.name)

def matched_recurrence(A,B,seed_a,seed_b,arm):
    ca=sorted(int(c) for c in np.unique(A['labels']) if c>=0)
    cb=sorted(int(c) for c in np.unique(B['labels']) if c>=0)
    a=[set(np.flatnonzero(A['labels']==c).tolist()) for c in ca]
    b=[set(np.flatnonzero(B['labels']==c).tolist()) for c in cb]
    m=max(len(a),len(b))
    mat=np.zeros((m,m),dtype=np.float64)
    for i,u in enumerate(a):
        for j,v in enumerate(b):
            mat[i,j]=len(u&v)/len(u|v)
    rr,cc=linear_sum_assignment(-mat)
    matched=[];details=[]
    for i,j in zip(rr,cc):
        if i<len(a) and j<len(b):
            si=len(a[i]);sj=len(b[j]);shared=len(a[i]&b[j]);u=len(a[i]|b[j])
            jj=float(mat[i,j]); matched.append((si,sj,jj))
            details.append(dict(arm=arm,seed_a=seed_a,seed_b=seed_b,state_a=ca[i],state_b=cb[j],
                size_a=si,size_b=sj,intersection_n=shared,union_n=u,jaccard=jj,matched=True))
        elif i<len(a):
            details.append(dict(arm=arm,seed_a=seed_a,seed_b=seed_b,state_a=ca[i],state_b=-1,
                size_a=len(a[i]),size_b=0,intersection_n=0,union_n=len(a[i]),jaccard=0.,matched=False))
        elif j<len(b):
            details.append(dict(arm=arm,seed_a=seed_a,seed_b=seed_b,state_a=-1,state_b=cb[j],
                size_a=0,size_b=len(b[j]),intersection_n=0,union_n=len(b[j]),jaccard=0.,matched=False))
    n_a=int(np.sum(A['labels']>=0));n_b=int(np.sum(B['labels']>=0))
    denom=(n_a+n_b)/2
    weight_num=sum(.5*(p+q)*j for p,q,j in matched)
    weighted_j=float(weight_num/denom) if denom else np.nan
    shared_mask=(A['labels']>=0)&(B['labels']>=0)
    na=int(np.unique(A['labels'][shared_mask]).size)
    nb=int(np.unique(B['labels'][shared_mask]).size)
    raw_ari=float(adjusted_rand_score(A['raw'],B['raw']))
    raw_nmi=float(normalized_mutual_info_score(A['raw'],B['raw']))
    joint_ari=float(adjusted_rand_score(A['labels'][shared_mask],B['labels'][shared_mask])) if (int(shared_mask.sum())>=2 and na>=2 and nb>=2) else np.nan
    pair=dict(arm=arm,seed_a=seed_a,seed_b=seed_b,recurrent_states_a=len(a),recurrent_states_b=len(b),
        n_recurrent_a=n_a,n_recurrent_b=n_b,n_jointly_recurrent=int(shared_mask.sum()),
        jointly_recurrent_fraction=float(shared_mask.mean()),raw_partition_ARI_all1733=raw_ari,
        raw_partition_NMI_all1733=raw_nmi,recurrent_partition_ARI_on_intersection=joint_ari,
        recurrent_intersection_states_a=na,recurrent_intersection_states_b=nb,
        matched_recurrent_weighted_Jaccard=weighted_j,matched_states_n=len(matched),
        unmatched_states_a=len(a)-len(matched),unmatched_states_b=len(b)-len(matched))
    return pair,details

def run():
    print('Step 1/4: validating frozen inputs...',flush=True)
    source=[RANK,DISC,F481,F483,F486,F488,F489,F492,F497,F760,F761,F762,F763,F764,F765,F766,F767]
    for p in source: require(p.is_file(),'Missing frozen file: '+str(p))
    manifest_check(F497,[DISC,F481,F483,F486,F488,F489,F492])
    manifest_check(F767,[F760,F761,F762,F763,F764,F765,F766])
    r10b=json.loads(F760.read_text())
    require(r10b['arms']==ARMS and r10b['seeds']==SEEDS,'R10B frozen arm/seed definitions differ')
    require(r10b['parent_sha256'][RANK.name]==sha(RANK),'Raw rank SHA differs from R10B')
    require(r10b['parent_sha256'][DISC.name]==sha(DISC),'Discovery ledger SHA differs from R10B')
    bmet=pd.read_csv(F764)
    require(len(bmet)==9 and not bmet[['seed','arm']].duplicated().any(),'R10B arm table invalid')
    manifest=pd.read_csv(F767,dtype=str)
    emb_files={}
    for arm in ARMS:
        for seed in SEEDS:
            row=bmet.loc[(bmet.arm==arm)&(bmet.seed==seed)]
            require(len(row)==1,'Missing/duplicated R10B arm-seed')
            row=row.iloc[0]
            p=WORK/f'{seed}_{arm}'/'gene_embedding32.npy'
            require(p.is_file() and str(p)==str(row.embeddings_path),'Embedding path mismatch: '+str(p))
            require(sha(p)==str(row.embedding_sha256),'R10B embedded SHA mismatch: '+str(p))
            ms=manifest.loc[manifest['path'].eq(str(p))]
            require(len(ms)==1 and str(ms.sha256.iloc[0])==sha(p),'R10B manifest SHA mismatch: '+str(p))
            emb_files[(arm,seed)]=p
    # The source data are read-only. Outputs are placed in a new, non-audit folder.
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'analysis_id':'AML_GRAPHRX_R10B_ALIGNED_NORMALIZATION_SENSITIVITY_V1',
      'analysis_type':'POST_HOC_EXPLORATORY_WITHIN_DISCOVERY',
      'original_1733_and_all_recurrence_thresholds_unchanged':True,
      'normalization':'Original R10C normalize_basis: float32 column centering/sd then gene-row L2 exactly once',
      'reference_R5G_embedding_re_normalized':False,
      'R10B_bases_raw_as_saved_then_normalized_once':True,
      'uses_original_R10C_helpers':True,
      'retained_graph_reconstruction_targets_unchanged':True,
      'arms':ARMS,'seeds':SEEDS,'N_BOOT':N_BOOT,'EDGE_KEEP':EDGE_KEEP,
      'main_metrics':['recurrent_coverage','raw_partition_ARI_all1733','matched_recurrent_weighted_Jaccard'],
      'nonclaims':['no external transport','no independent biological validation','not evidence of clinical utility'],
      'source_sha256':{p.name:sha(p) for p in source},
      'embedding_sha256':{f'{arm}_{seed}':sha(p) for (arm,seed),p in emb_files.items()}}
    save_json(OUT/'sensitivity_protocol_and_inputs.json',protocol)
    print('Step 2/4: checking original R5G pipeline fidelity...',flush=True)
    ledger=pd.read_csv(DISC)
    require({'analysis_unit_index_0based','gse'}.issubset(ledger.columns),'Bad ledger schema')
    require(len(ledger)==N_PATIENTS and ledger.analysis_unit_index_0based.is_unique,'Discovery rows changed')
    ix=ledger.analysis_unit_index_0based.to_numpy(int)
    gses=ledger.gse.astype(str).to_numpy()
    require(set(gses)==set(GSES) and [int((gses==g).sum()) for g in GSES]==GSE_COUNTS,'Discovery cohorts differ')
    r=np.load(RANK,mmap_mode='r',allow_pickle=False)
    require(r.ndim==2 and r.shape[1]==N_GENES and ix.max()<r.shape[0],'Rank data shape changed')
    X=np.array(r[ix,:],dtype=np.float32,copy=True)
    require(np.isfinite(X).all(),'Nonfinite discovery input')
    ref_e=np.load(F483,allow_pickle=False)
    require(ref_e.shape==(N_GENES,D),'R5G embedding shape mismatch')
    # Never normalize original already-normalized R5G basis a second time.
    baseP=patient_projection(X,gses,ref_e)
    require(float(np.max(np.abs(baseP-np.load(F486,allow_pickle=False))))<1e-4,'Original R5G projection mismatch')
    baseEdges=create_mnn(baseP,gses)
    oldEdges=pd.read_csv(F488)
    require({(a,b) for a,b,w in baseEdges}==set(zip(oldEdges.patient_node_u.astype(int),oldEdges.patient_node_v.astype(int))), 'Original MNN mismatch')
    refLabels=pd.read_csv(F489).set_index('analysis_unit_index_0based').loc[ix,'raw_community'].to_numpy(int)
    baseRaw=leiden_graph(N_PATIENTS,baseEdges,SPLIT_SEED)
    require(adjusted_rand_score(refLabels,baseRaw)>0.999999,'Original R5G Leiden mismatch')
    baseMet=state_metrics(baseRaw,baseEdges,gses)
    require(baseMet['states']==6 and abs(baseMet['coverage']-1720/1733)<1e-10,'Original R5G recurrent states mismatch')
    print('PASS R5G fidelity: 6 states, coverage=1720/1733, unchanged original embedding.',flush=True)
    print('Step 3/4: normalizing and re-evaluating ALL nine saved embeddings...',flush=True)
    arms={};metrics=[]
    for arm in ARMS:
        for seed in SEEDS:
            p=emb_files[(arm,seed)]
            e=np.load(p,allow_pickle=False)
            require(e.shape==(N_GENES,D) and np.isfinite(e).all(),'Bad R10B embedding')
            normalized=normalize_basis(e)  # exactly once, non-destructive copy
            require(np.isfinite(normalized).all(),'Nonfinite normalized embedding')
            # Restart-safe, immutable per-fit checkpoint. A cached fit is valid only
            # when its source-embedding SHA and normalized-basis digest both match.
            cache=OUT/f'cache_normalized_{seed}_{arm}.npz'
            basis_sha=hashlib.sha256(normalized.tobytes()).hexdigest()
            if cache.exists():
                with np.load(cache,allow_pickle=False) as z:
                    require(str(z['source_sha256'].item())==sha(p),'Cached source embedding SHA mismatch')
                    require(str(z['normalized_sha256'].item())==basis_sha,'Cached normalized basis SHA mismatch')
                    raw=z['raw'].astype(int)
                    labels=z['labels'].astype(int)
                    require(raw.shape==(N_PATIENTS,) and labels.shape==(N_PATIENTS,),'Cache label dimensions invalid')
                    met=dict(states=int(z['states']),coverage=float(z['coverage']),
                        stability=float(z['stability']),cohort_nmi=float(z['cohort_nmi']),
                        labels=labels)
                    edge_count=int(z['mnn_edges_n'])
                    require(np.isclose(met['coverage'],float(np.mean(labels>=0)),atol=1e-12),
                        'Cached coverage does not match labels')
                print('  RESUMED verified checkpoint:',cache.name,flush=True)
            else:
                P=patient_projection(X,gses,normalized)
                edges=create_mnn(P,gses)
                edge_count=len(edges)
                raw=leiden_graph(N_PATIENTS,edges,SPLIT_SEED)
                met=state_metrics(raw,edges,gses)
                buf=BytesIO()
                np.savez_compressed(buf,raw=raw.astype(np.int32),labels=met['labels'].astype(np.int32),
                    states=np.int32(met['states']),coverage=np.float64(met['coverage']),
                    stability=np.float64(met['stability']),cohort_nmi=np.float64(met['cohort_nmi']),
                    mnn_edges_n=np.int32(edge_count),source_sha256=np.str_(sha(p)),
                    normalized_sha256=np.str_(basis_sha))
                write_once(cache,buf.getvalue())
            row=bmet.loc[(bmet.arm==arm)&(bmet.seed==seed)].iloc[0]
            arms[(arm,seed)]={'raw':raw,'labels':met['labels'],'metric':met}
            metrics.append(dict(arm=arm,seed=seed,embedding_sha256=sha(p),
                normalized_once=True,mnn_edges_n=edge_count,recurrent_states_n=met['states'],
                recurrent_coverage=met['coverage'],median_recurrent_stability=met['stability'],
                state_cohort_NMI_descriptive=met['cohort_nmi'],
                ARI_vs_original_discovery_descriptive=float(adjusted_rand_score(baseMet['labels'],met['labels'])),
                previously_reported_raw_R10B_coverage=float(row.recurrent_coverage),
                normalized_minus_raw_R10B_coverage=float(met['coverage']-float(row.recurrent_coverage))))
            print(f'  {arm} / {seed}: recurrent states={met["states"]} coverage={met["coverage"]:.4f}',flush=True)
    metrics=pd.DataFrame(metrics)
    save_csv(OUT/'normalized_arm_metrics.csv',metrics,['arm','seed'])
    print('Step 4/4: cross-seed and paired-arm comparisons...',flush=True)
    pairs=[];matches=[]
    for arm in ARMS:
        for sa,sb in combinations(SEEDS,2):
            row,detail=matched_recurrence(arms[(arm,sa)],arms[(arm,sb)],sa,sb,arm)
            pairs.append(row);matches.extend(detail)
    pairs=pd.DataFrame(pairs)
    save_csv(OUT/'normalized_cross_seed_pairwise.csv',pairs,['arm','seed_a','seed_b'])
    save_csv(OUT/'normalized_hungarian_state_matches.csv',pd.DataFrame(matches),['arm','seed_a','seed_b','state_a','state_b'])
    summary=[]
    for arm in ARMS:
        mm=metrics.loc[metrics.arm==arm]
        pp=pairs.loc[pairs.arm==arm]
        summary.append(dict(arm=arm,three_seeds=3,
            mean_coverage=float(mm.recurrent_coverage.mean()),
            median_coverage=float(mm.recurrent_coverage.median()),
            median_raw_partition_ARI=float(pp.raw_partition_ARI_all1733.median()),
            median_matched_recurrent_Jaccard=float(pp.matched_recurrent_weighted_Jaccard.median()),
            median_jointly_recurrent_fraction=float(pp.jointly_recurrent_fraction.median()),
            median_within_run_stability=float(mm.median_recurrent_stability.median())))
    summary=pd.DataFrame(summary)
    save_csv(OUT/'normalized_arm_summary.csv',summary,['arm'])
    paired=[]
    for seed in SEEDS:
        real=float(metrics.loc[(metrics.arm=='REAL_GRAPH')&(metrics.seed==seed),'recurrent_coverage'].iloc[0])
        for control in ARMS[1:]:
            ref=float(metrics.loc[(metrics.arm==control)&(metrics.seed==seed),'recurrent_coverage'].iloc[0])
            paired.append(dict(seed=seed,comparator=control,real_graph_coverage=real,control_coverage=ref,
                real_minus_control_coverage=real-ref))
    save_csv(OUT/'normalized_paired_coverage_differences.csv',pd.DataFrame(paired),['comparator','seed'])
    generated=sorted(p for p in OUT.glob('*.csv') if p.name != 'sensitivity_outputs_manifest.csv')+sorted(OUT.glob('cache_normalized_*.npz'))+[OUT/'sensitivity_protocol_and_inputs.json']
    manifest_df=pd.DataFrame([dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in generated])
    save_csv(OUT/'sensitivity_outputs_manifest.csv',manifest_df,['path'])
    print('\nALL-NINE NORMALIZED METRICS\n',metrics[['arm','seed','recurrent_states_n','recurrent_coverage','normalized_minus_raw_R10B_coverage']].to_string(index=False))
    print('\nARM SUMMARY (DESCRIPTIVE ONLY)\n',summary.to_string(index=False))
    print('\nPAIRED COVERAGE DIFFERENCES\n',pd.DataFrame(paired).to_string(index=False))
    print('\nCOMPLETE. Frozen source files untouched. New results in:',OUT,flush=True)
    print('NOTE: Three paired seeds are NOT independent datasets or proof of external transport.',flush=True)

if __name__=='__main__':
    run()
