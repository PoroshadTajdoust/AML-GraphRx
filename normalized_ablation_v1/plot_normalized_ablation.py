"""Recreate manuscript Figure 3 from archived normalized-ablation tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent
N=W/'results'; F=W/'generated'; T=F
F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
def save(fig,name):
    for ext in ['png','svg','pdf']:
        fig.savefig(F/f'{name}.{ext}',dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)
m=pd.read_csv(N/'normalized_arm_metrics.csv');p=pd.read_csv(N/'normalized_cross_seed_pairwise.csv');arms=['REAL_GRAPH','DEGREE_PRESERVING_REWIRE','NO_MESSAGE_PASSING'];labs=['Biological\ngraph','Degree-preserving\nrewiring','No neighbor\nmessages'];cols=['#168F91','#D78B42','#66758A'];seeds=[20261031,20261101,20261102]
fig,axs=plt.subplots(2,2,figsize=(10.8,7.2),gridspec_kw={'wspace':.29,'hspace':.42})
for k,(metric,scale,title,ylab) in enumerate([('recurrent_coverage',100,'A  Coverage by paired initialization','Recurrent coverage (%)'),('recurrent_states_n',1,'B  Recurrent communities','Qualifying communities')]):
 ax=axs[0,k]
 for j,seed in enumerate(seeds):
  vals=m[m.seed.eq(seed)].set_index('arm').loc[arms,metric].to_numpy()*scale
  ax.plot(range(3),vals,c='#B8C2CA',ls=['-','--',':'][j],lw=1,zorder=1)
  for i,v in enumerate(vals):ax.scatter(i,v,c=cols[i],marker=['o','s','^'][j],s=52,zorder=3)
 for i,arm in enumerate(arms):
  v=m.loc[m.arm.eq(arm),metric].mean()*scale;ax.plot([i-.16,i+.16],[v,v],c=cols[i],lw=3)
 ax.set_xticks(range(3),labs);ax.set_xlim(-.45,2.45);ax.set_ylim(-3 if scale==100 else -.2,105 if scale==100 else 7);ax.set_ylabel(ylab);ax.set_title(title,loc='left',weight='bold',fontsize=12);ax.grid(axis='y',alpha=.18)
for k,(metric,scale,title,ylab) in enumerate([('matched_recurrent_weighted_Jaccard',1,'C  Matched recurrent-community agreement','Size-weighted matched Jaccard'),('jointly_recurrent_fraction',100,'D  Joint recurrent membership','Joint recurrent coverage (%)')]):
 ax=axs[1,k]
 for i,arm in enumerate(arms):
  vals=p.loc[p.arm.eq(arm),metric].dropna().to_numpy()*scale;ax.scatter(i+np.linspace(-.09,.09,len(vals)),vals,c=cols[i],s=48);v=np.median(vals);ax.plot([i-.18,i+.18],[v,v],c=cols[i],lw=3)
  ax.text(i,v+(.035 if scale==1 else 4),f'{v:.3f}' if scale==1 else f'{v:.2f}%',ha='center',fontsize=9)
 ax.set_xticks(range(3),labs);ax.set_xlim(-.45,2.45);ax.set_ylim((-.025,.58) if scale==1 else (-4,100));ax.set_title(title,loc='left',weight='bold',fontsize=12);ax.set_ylabel(ylab);ax.grid(axis='y',alpha=.18)
axs[1,0].text(2,.49,'2 defined pairs\n1 empty–empty pair: NA',ha='center',fontsize=8.5,color='#596774')
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0],[0],marker=x,c='#66758A',ls='none',label=f'Initialization {l}') for x,l in zip(['o','s','^'],'ABC')],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.015));save(fig,'Figure_3')
m.to_csv(T/'Figure3_normalized_runs.csv',index=False);p.to_csv(T/'Figure3_normalized_pairs.csv',index=False)

print('Figure 3 written to', F)
