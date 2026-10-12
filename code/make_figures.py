from pathlib import Path
import json,re
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

W=Path(__file__).resolve().parents[1];R=W;S=W/'source_tables/context';D=W/'source_tables/discovery';N=W/'source_tables/new'
O=W;F=O/'figures';T=O/'source_data';F.mkdir(parents=True,exist_ok=True);T.mkdir(exist_ok=True)
PURPLE='#7551A0';TEAL='#168F91';NAVY='#345F84';GRAY='#7D8992';GOLD='#C19A58';INK='#223342';LIGHT='#DEE5EB'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.labelsize':10,'axes.titlesize':11,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A6B0B8','axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def save(fig,name):
    for ext in ['png','svg','pdf']:
        buf=BytesIO();fig.savefig(buf,format=ext,dpi=600,bbox_inches='tight');payload=buf.getvalue()
        if ext=='png':
            with Image.open(BytesIO(payload)) as im: im.verify()
        tmp=F/f'{name}.{ext}.partial';tmp.write_bytes(payload);tmp.replace(F/f'{name}.{ext}')
    plt.close(fig)
def source(df,name):df.to_csv(T/f'{name}.csv',index=False)
def title(ax,letter,s):ax.set_title(f'{letter}  {s}',loc='left',pad=14)
def grid(ax,axis='y'):ax.grid(axis=axis,color='#EDF1F4',lw=.7);ax.set_axisbelow(True)

st=pd.read_csv(D/'491_R5G_recurrent_state_summary.csv');st=st[st.RECURRENT_AML_STATE].sort_values('state_id');source(st,'Figure2_recurrent_states')
asgn=pd.read_csv(D/'492_R5G_recurrent_state_patient_assignments.csv');ct=pd.crosstab(asgn.gse,asgn.state_id);cols=[x for x in ct if x.startswith('AML_STATE')];ct=ct[cols];pr=ct.div(asgn.groupby('gse').size(),axis=0);source(pr.reset_index(),'Figure2_cohort_composition')
a=pd.read_csv(N/'753_R10A_arm_state_discovery_metrics.csv');source(a,'Figure2_representation_all_runs')
fig,axs=plt.subplots(2,2,figsize=(9.2,7.2),gridspec_kw={'wspace':.47,'hspace':.55})
ax=axs[0,0];im=ax.imshow(100*pr,cmap='Blues',vmin=0,vmax=45,aspect='auto');ax.set_xticks(range(6),[f'S{i:02}' for i in range(1,7)]);ax.set_yticks(range(8),pr.index,fontsize=8.5)
for i in range(8):
    for j in range(6):ax.text(j,i,f'{100*pr.iloc[i,j]:.0f}',ha='center',va='center',fontsize=8.5,color='white' if pr.iloc[i,j]>.27 else INK)
title(ax,'A','Community membership by cohort');ax.set_xlabel('Percent of all profiles in each cohort')
ax=axs[0,1];colors=[GRAY,GRAY,PURPLE,GRAY,GRAY,TEAL];ax.bar(range(6),st.patients_n,color=colors,width=.65)
for i,v in enumerate(st.patients_n):ax.text(i,v+9,str(v),ha='center',fontsize=9)
ax.set_xticks(range(6),[f'S{i:02}' for i in range(1,7)]);ax.set_ylim(0,465);ax.set_ylabel('AML profiles');title(ax,'B','Six recurrent communities');grid(ax)
ax=axs[1,0];families=['GNN32','PCA32','RANDOM32','GNN_SHUFFLE32'];labels=['GNN32\nn=1','PCA32\nn=1','Gaussian\nn=5','Row shuffle\nn=5'];cols2=[NAVY,GOLD,GRAY,PURPLE]
for i,(family,c) in enumerate(zip(families,cols2)):
    vals=a[a.family.eq(family)].recurrent_coverage.to_numpy()*100;offset=np.linspace(-.12,.12,len(vals)) if len(vals)>1 else np.array([0]);ax.scatter(i+offset,vals,s=42,c=c,zorder=3);med=np.median(vals);ax.plot([i-.23,i+.23],[med,med],c=c,lw=2)
ax.set_xticks(range(4),labels,fontsize=8.5);ax.set_ylim(0,106);ax.set_ylabel('Recurrent coverage (%)');title(ax,'C','Representation comparison');grid(ax)
ax=axs[1,1]
for i,(family,c) in enumerate(zip(families,cols2)):
    vals=a[a.family.eq(family)].median_recurrent_stability.to_numpy();off=np.linspace(-.12,.12,len(vals)) if len(vals)>1 else np.array([0]);ax.scatter(i+off,vals,s=42,c=c,zorder=3);ax.plot([i-.23,i+.23],[np.median(vals)]*2,c=c,lw=2)
ax.set_xticks(range(4),labels,fontsize=8.5);ax.set_ylim(.55,.96);ax.set_ylabel('Median edge-perturbation Jaccard');title(ax,'D','Within-run recurrence stability');grid(ax)
save(fig,'Figure2_discovery_representation')

b=pd.read_csv(N/'764_R10B_arm_discovery_metrics.csv');c=pd.read_csv(N/'771_R10C_cross_seed_pairwise_consistency.csv');source(b,'Figure3_message_ablation_all_fits');source(c,'Figure3_cross_seed_all_pairs')
arms=['REAL_GRAPH','DEGREE_PRESERVING_REWIRE','NO_MESSAGE_PASSING'];arm_names=['Biological\ngraph','Degree-preserving\nrewiring','No neighbor\nmessages'];acol=[TEAL,GOLD,GRAY];seeds=sorted(b.seed.unique());pairs=sorted(set(zip(c.seed_a,c.seed_b)));markers=['o','s','^']
fig,axs=plt.subplots(2,2,figsize=(9.2,7.0),gridspec_kw={'wspace':.38,'hspace':.57})
for k,(metric,ylab) in enumerate([('recurrent_states_n','Recurrent communities'),('recurrent_coverage','Recurrent coverage (%)')]):
    ax=axs[0,k]
    for j,seed in enumerate(seeds):
        vals=b[b.seed.eq(seed)].set_index('arm').loc[arms,metric].to_numpy()*(100 if k==1 else 1);ax.plot(range(3),vals,c='#B9C3CB',lw=1,zorder=1,linestyle=['-','--',':'][j]);
        for i,v in enumerate(vals):ax.scatter(i,v,s=52,c=acol[i],marker=markers[j],zorder=3)
    mean=b.groupby('arm')[metric].mean().loc[arms].to_numpy()*(100 if k==1 else 1)
    for i,v in enumerate(mean):ax.plot([i-.15,i+.15],[v,v],c=acol[i],lw=3)
    ax.set_xticks(range(3),arm_names,fontsize=8.5);ax.set_ylabel(ylab);ax.set_xlim(-.4,2.4);ax.set_ylim((0,106) if k==1 else (0,8));title(ax,'AB'[k],['Community counts across seeds','Paired graph-message comparison'][k]);grid(ax)
ax=axs[1,0]
for i,arm in enumerate(arms):
    vals=c[c.arm.eq(arm)].raw_partition_ARI_all1733.to_numpy();ax.scatter(i+np.linspace(-.09,.09,3),vals,c=acol[i],s=47);ax.plot([i-.18,i+.18],[np.median(vals)]*2,c=acol[i],lw=2)
ax.set_xticks(range(3),arm_names,fontsize=8.5);ax.set_ylabel('All-profile partition ARI');ax.set_ylim(0,.36);title(ax,'C','Agreement between fitted seeds');grid(ax)
ax=axs[1,1]
for i,arm in enumerate(arms):
    z=c[c.arm.eq(arm)];ax.scatter(z.jointly_recurrent_fraction*100,z.matched_recurrent_weighted_Jaccard,s=53,c=acol[i],label=['Biological graph','Rewiring','No messages'][i])
ax.set_xlabel('Joint recurrent coverage (%)');ax.set_ylabel('Matched weighted Jaccard');ax.set_xlim(-3,103);ax.set_ylim(0,.46);title(ax,'D','Agreement and shared membership');ax.legend(frameon=False,fontsize=8.5,loc='lower right');grid(ax)
save(fig,'Figure3_graph_message_ablation')

p=pd.read_csv(D/'528_R5BR_shared_pathway_analysis.csv');source(p,'Figure4_all_shared_pathways')
terms=['Mitochondrial Translation R-HSA-5368287','Mitochondrial Translation Initiation R-HSA-5368286','mRNA Splicing R-HSA-72172','Processing Of Capped Intron-Containing Pre-mRNA R-HSA-72203','Transport Of Mature Transcript To Cytoplasm R-HSA-72202','Chromatin Modifying Enzymes R-HSA-3247509','FLT3 Signaling R-HSA-9607240','RHO GTPase Cycle R-HSA-9012999']
sel=p.set_index('Term').loc[terms].reset_index();source(sel,'Figure4_displayed_pathways')
fig,axs=plt.subplots(1,2,figsize=(9.2,4.8),gridspec_kw={'width_ratios':[1,1.2],'wspace':1.20})
ax=axs[0]
for lib,color in [('Reactome_2022',PURPLE),('GO_Biological_Process_2025',TEAL)]:
    z=p[p.library.eq(lib)];ax.scatter(z.NES_STATE03,z.NES_STATE06,s=24,alpha=.72,c=color,label=f'{lib.split("_")[0]} (n={len(z)})',edgecolors='none')
ax.axhline(0,c=GRAY,lw=.7);ax.axvline(0,c=GRAY,lw=.7);ax.set_xlabel('S03 normalized enrichment score');ax.set_ylabel('S06 normalized enrichment score');title(ax,'A','Shared significant pathways');ax.legend(frameon=False,fontsize=8.2,loc='upper right')
ax=axs[1];labs=['Mitochondrial translation','Mitochondrial translation\ninitiation','mRNA splicing','Pre-mRNA processing','Mature transcript export','Chromatin modifying\nenzymes','FLT3 signaling','RHO GTPase cycle']
for i,r in sel.iterrows():ax.plot([r.NES_STATE03,r.NES_STATE06],[i,i],c=LIGHT,lw=2)
ax.scatter(sel.NES_STATE03,range(8),s=41,c=PURPLE,label='S03');ax.scatter(sel.NES_STATE06,range(8),s=41,c=TEAL,label='S06');ax.set_yticks(range(8),labs,fontsize=8.5);ax.invert_yaxis();ax.axvline(0,c=GRAY,lw=.7);ax.set_xlabel('Normalized enrichment score');title(ax,'B','Illustrative Reactome terms');ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=8.5)
save(fig,'Figure4_pathway_contrast')

ext=pd.read_csv(S/'625_R7B2_BeatAML_axis_scores.csv');nul=pd.read_csv(S/'635_R7C_matched_gene_label_null_distribution.csv');q=pd.read_csv(N/'742_R9Q_wave_conditional_effects.csv');source(ext,'Figure5_frozen_BeatAML_scores');source(nul,'Figure5_matched_gene_label_null');source(q,'Figure5_conditional_associations')
fig,axs=plt.subplots(2,2,figsize=(9.0,7.1),gridspec_kw={'hspace':.59,'wspace':.46})
for i,(wave,col) in enumerate(zip(['Waves1+2','Waves3+4'],[PURPLE,TEAL])):
    z=ext[ext.cohort.eq(wave)];ax=axs[0,i];ax.scatter(z.S03_high_component_mean,z.S06_high_component_mean,s=17,alpha=.65,c=col,edgecolors='none');rho=spearmanr(z.S03_high_component_mean,z.S06_high_component_mean).statistic;ax.set_xlabel('S03 aggregate expression score');ax.set_ylabel('S06 aggregate expression score');title(ax,'AB'[i],f'{wave}   n={len(z)}');ax.text(.98,.96,f'ρ = {rho:.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=11,color=col,fontweight='bold');ax.set_xlim(ext.S03_high_component_mean.min()-.01,ext.S03_high_component_mean.max()+.01);ax.set_ylim(ext.S06_high_component_mean.min()-.015,ext.S06_high_component_mean.max()+.025)
ax=axs[1,0];ax.hist(nul.rho_all,bins=35,color='#BACBD8',edgecolor='white');obs=spearmanr(ext.S03_high_component_mean,ext.S06_high_component_mean).statistic;ax.axvline(obs,color=TEAL,lw=2,label=f'Observed ρ = {obs:.3f}');ax.set_xlabel('Pooled correlation under matched null');ax.set_ylabel('Permutation count');title(ax,'C','Matched rank-composition null');ax.legend(frameon=False,fontsize=8.5,loc='upper left');ax.text(.04,.78,'0/1,000 ≤ observed\nP = 0.000999',transform=ax.transAxes,fontsize=9,va='top')
ax=axs[1,1]
for i,r in q.iterrows():
    col=[PURPLE,TEAL][i];ax.scatter(r.raw_spearman_r,i+.13,edgecolors=col,s=45,marker='o',facecolors='none',label='Raw Spearman' if i==0 else None)
    ax.errorbar(r.cellscore_adjusted_partial_rank_r,i-.13,xerr=[[r.cellscore_adjusted_partial_rank_r-r.bootstrap_CI95_lower],[r.bootstrap_CI95_upper-r.cellscore_adjusted_partial_rank_r]],fmt='s',c=col,capsize=3,ms=5,label='Adjusted partial rank' if i==0 else None)
ax.set_yticks([0,1],['Waves1+2\nn=372','Waves3+4\nn=188']);ax.invert_yaxis();ax.set_xlim(-.81,-.50);ax.set_ylim(1.5,-.55);ax.set_xlabel('Association coefficient');title(ax,'D','Adjustment for six cell-state scores');ax.legend(frameon=False,fontsize=8.1,loc='lower left');grid(ax,'x')
save(fig,'Figure5_external_validation')

cc=pd.read_csv(S/'667_R8H2_cross_wave_celltype_axis_associations.csv');pred=pd.read_csv(S/'670_R8H2_celltype_ridge_predictions.csv');source(cc,'Figure6_cell_state_associations');source(pred,'Figure6_cell_state_model_predictions');names=['HSC-like','Prog-like','GMP-like','ProMono-like','Mono-like','cDC-like']
fig,axs=plt.subplots(1,2,figsize=(9.0,4.0),gridspec_kw={'wspace':.42})
ax=axs[0]
for k,(wave,col) in enumerate(zip(['Waves1+2','Waves3+4'],[PURPLE,TEAL])):
    z=cc[cc.wave.eq(wave)].set_index('cell_type').loc[names]
    ax.scatter(z.spearman_rho,np.arange(6)+(k-.5)*.24,c=col,label=wave,s=45)
    for j,(_,r) in enumerate(z.iterrows()):
        if r.BH_FDR_within_wave<=.05:ax.text(r.spearman_rho+.017,j+(k-.5)*.24,'*',c=col,va='center',fontsize=11)
ax.set_yticks(range(6),names);ax.invert_yaxis();ax.axvline(0,c=GRAY,lw=.7);ax.set_xlim(-.53,.43);ax.set_xlabel('Spearman correlation with frozen axis');title(ax,'A','Malignant-cell-state expression scores');ax.legend(frameon=False,fontsize=8.5,loc='lower right');grid(ax,'x')
ax=axs[1];z=pred[pred.wave.eq('Waves3+4')];ax.scatter(z.observed_axis,z.predicted_axis_from_6_vanGalen_scores,c=TEAL,s=22,alpha=.7,edgecolors='none');lims=[min(z.observed_axis.min(),z.predicted_axis_from_6_vanGalen_scores.min())-.01,max(z.observed_axis.max(),z.predicted_axis_from_6_vanGalen_scores.max())+.01];ax.plot(lims,lims,c=LIGHT,lw=1,ls='--');ax.set_xlim(lims);ax.set_ylim(lims);ax.set_xlabel('Observed frozen pathway axis');ax.set_ylabel('Axis predicted by six-score model');title(ax,'B','Held-out Waves3+4   n=188');ax.text(.04,.96,'R² = 0.276\nSpearman ρ = 0.550',transform=ax.transAxes,va='top',fontsize=10,color=TEAL)
save(fig,'Figure6_cell_state_context')

ds=pd.read_csv(S/'600_R7A2_discovery_axis_scores.csv');loco=pd.read_csv(S/'557_R5X_LOCO_fold_summary.csv');source(ds,'FigureS1_discovery_axis');source(loco,'FigureS1_LOCO')
fig,axs=plt.subplots(1,2,figsize=(9,4.3),gridspec_kw={'wspace':.70})
ax=axs[0];co=sorted(ds.gse.unique());ax.boxplot([ds.loc[ds.gse.eq(x),'R7A2_axis_score'] for x in co],tick_labels=co,showfliers=False,patch_artist=True,boxprops={'facecolor':'#C8DEE0'});ax.tick_params(axis='x',rotation=55,labelsize=8);ax.set_ylabel('Frozen pathway axis');title(ax,'A','Discovery cohort distributions')
ax=axs[1];ax.barh(loco.heldout_gse,loco.heldout_confident_coverage*100,color=NAVY);ax.set_xlabel('Confident assignment (%)');ax.set_xlim(0,105);ax.tick_params(axis='y',labelsize=8.5);title(ax,'B','Corrective LOCO assessment');grid(ax,'x');save(fig,'FigureS1_discovery_transport')

cl=pd.read_csv(D/'554A_R5MS_state03_state06_subtype_table_OR_corrected.csv');source(cl,'FigureS2_annotation_table');vals=[[100-float(re.search(r'\(([\d.]+)%\)',r[x]).group(1)) for x in ['State03_missing','State06_missing']] for _,r in cl.iterrows()]
fig,ax=plt.subplots(figsize=(6.8,4.7));im=ax.imshow(vals,cmap='Blues',vmin=0,vmax=100,aspect='auto');ax.set_xticks([0,1],['S03   n=285','S06   n=160']);ax.set_yticks(range(len(cl)),cl.variable,fontsize=9)
for i,row in enumerate(vals):
    for j,v in enumerate(row):ax.text(j,i,f'{v:.1f}%',ha='center',va='center',color='white' if v>60 else INK)
cb=fig.colorbar(im,ax=ax,pad=.08);cb.set_label('Profiles with recorded annotation (%)');ax.set_title('Completeness of discovery annotations',pad=14);save(fig,'FigureS2_annotation_completeness')

history=pd.read_csv(N/'763_R10B_training_history.csv');source(history,'FigureS3_training_history');rw=pd.read_csv(N/'762_R10B_graph_rewiring_audit.csv');source(rw,'FigureS3_rewiring_audit')
fig,axs=plt.subplots(1,2,figsize=(9.0,4.1),gridspec_kw={'wspace':.43});ax=axs[0]
for i,arm in enumerate(arms):
    for j,seed in enumerate(seeds):
        z=history[(history.arm==arm)&(history.seed==seed)];ax.plot(z.epoch,z.mean_relation_BCE,color=acol[i],ls=['-','--',':'][j],alpha=.85,label=['Biological graph','Rewiring','No messages'][i] if j==0 else None)
ax.set_xlabel('Training epoch');ax.set_ylabel('Mean relation reconstruction loss');title(ax,'A','All nine completed encoder fits');ax.legend(frameon=False,fontsize=8.5);grid(ax)
ax=axs[1];x=['Signaling','PPI','TF\nregulation','Kinase–\nsubstrate'];vals=rw.set_index('relation').loc[['signaling','PPI','TF_regulation','kinase_substrate'],'directed_edge_jaccard'];ax.bar(range(4),vals,color=GOLD);ax.set_xticks(range(4),x,fontsize=8.5);ax.set_ylim(0,.09);ax.set_ylabel('Original–rewired source-edge Jaccard');title(ax,'B','Rewiring retained degrees');grid(ax);save(fig,'FigureS3_training_rewiring')

gb=W/'source_tables/global_response';boot=pd.read_csv(gb/'722_R9P_GLOBAL_bootstrap.csv');dep=pd.read_csv(S/'692_R8F2_full533_axis_dependency_associations.csv');source(boot,'FigureS4_global_response_bootstrap');source(dep,'FigureS4_dependency_results')
fig,axs=plt.subplots(1,2,figsize=(9.0,3.9),gridspec_kw={'wspace':.44});ax=axs[0];ax.hist(boot.macro_delta_MSE,bins=35,color='#B9C9D5',edgecolor='white');ax.axvline(0,c=GRAY,lw=1.5);ax.axvline(-.002820,c=NAVY,lw=2,label='Observed');ax.set_xlabel('Global test MSE improvement\nPositive favors adding the axis');ax.set_ylabel('Subject-bootstrap count');title(ax,'A','Exploratory drug-response comparison');ax.legend(frameon=False,fontsize=8.5)
ax=axs[1];rho_col=next(x for x in dep if 'spearman' in x.lower() and not x.lower().endswith('_p'));qcol=next(x for x in dep if 'FDR' in x and 'full' in x.lower());v=dep[[rho_col,qcol]].dropna();ax.scatter(v[rho_col],-np.log10(v[qcol].clip(lower=1e-300)),c=GRAY,s=18,alpha=.7,edgecolors='none');ax.axhline(-np.log10(.05),c=NAVY,ls='--',lw=1);ax.set_xlabel('Axis–dependency Spearman correlation');ax.set_ylabel('−log₁₀(full-family BH q)');title(ax,'B','AML cell-line dependencies');ax.text(.04,.88,'32 models; 532 evaluable genes',transform=ax.transAxes,fontsize=9);save(fig,'FigureS4_functional_context')

deck={'waves':[],'ablation':b[['seed','arm','recurrent_coverage']].to_dict('records')}
for wave in ['Waves1+2','Waves3+4']:
    z=ext[ext.cohort.eq(wave)];deck['waves'].append({'name':wave,'n':len(z),'x':z.S03_high_component_mean.tolist(),'y':z.S06_high_component_mean.tolist()})
(W/'deck_data.json').write_text(json.dumps(deck))
print('Built 9 scientific figures at 600 dpi and full-precision source tables')
