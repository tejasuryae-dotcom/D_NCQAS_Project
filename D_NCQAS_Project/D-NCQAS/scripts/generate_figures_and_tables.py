#!/usr/bin/env python3
"""Recompute tables and figures entirely from a completed run. No embedded result claims."""
import argparse,sys,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from d_ncqas.evaluation import paired_comparison,holm_bonferroni
from d_ncqas.noise_models import HardwareTopology,OrnsteinUhlenbeckDriftProcess
ORDER=['Minimal-GHZ','Validation-Tree','Nominal-PPO','Domain-Random','No-Learning','No-Spatial','No-Temporal','No-CVaR','No-Complexity','D-NCQAS']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output
    if not (out/'RUN_COMPLETE').exists():raise SystemExit('Run incomplete; refusing final analyses')
    tables=out/'tables';figures=out/'figures';tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    df=pd.read_csv(out/'processed/ghz_seeds.csv');curves=pd.read_csv(out/'processed/learning_curves.csv')
    summary=df.groupby(['topology','n_qubits','regime','method'],sort=False).agg(n_seeds=('seed','nunique'),mean_fidelity=('fidelity_mean','mean'),sd_fidelity=('fidelity_mean','std'),mean_cvar20=('fidelity_cvar20','mean'),sd_cvar20=('fidelity_cvar20','std'),mean_p_ghz=('p_ghz_mean','mean'),mean_gates=('total_gates','mean'),mean_train_seconds=('train_time_sec','mean'),mean_training_calls=('sim_calls_train','mean')).reset_index()
    tests=[]
    for (kind,n),group in df[df.regime=='ou'].groupby(['topology','n_qubits']):
        x=group[group.method=='D-NCQAS'].set_index('seed')['fidelity_cvar20']
        for m in ORDER[:-1]:
            y=group[group.method==m].set_index('seed')['fidelity_cvar20'];paired=pd.concat([x,y],axis=1,keys=['x','y']).dropna().sort_index()
            if len(paired)<2:continue
            tests.append(dict(topology=kind,n_qubits=n,baseline=m,**paired_comparison(paired.x.values,paired.y.values)))
    stats=pd.DataFrame(tests);stats['p_holm_signflip']=holm_bonferroni(stats.p_signflip.tolist());stats['significant']=stats.p_holm_signflip<.05
    vr=pd.read_csv(out/'raw/vqe_observations.csv')
    vs=vr.groupby(['molecule','method','seed','regime'],sort=False).agg(energy_error=('energy_error','mean'),measured_energy=('measured_energy','mean'),optimizer_success=('optimizer_success','first')).reset_index()
    vt=vs.groupby(['molecule','method','regime'],sort=False).agg(n_seeds=('seed','nunique'),mean_error=('energy_error','mean'),sd_error=('energy_error','std'),convergence_rate=('optimizer_success','mean')).reset_index()
    for name,data in [('ghz_summary',summary),('statistical_tests',stats),('vqe_summary',vt)]:
        data.to_csv(tables/(name+'.csv'),index=False);(tables/(name+'.md')).write_text(data.to_markdown(index=False)+'\n')
    # Stable compatibility outputs for notebook consumers.
    summary.to_csv(out/'processed/ghz_summary_table.csv',index=False);vt.to_csv(out/'processed/vqe_summary_table.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':600})
    def save(fig,name):
        fig.savefig(figures/(name+'.png'),dpi=600,bbox_inches='tight');fig.savefig(figures/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    kinds=list(df.topology.unique());ns=sorted(df.n_qubits.unique())
    fig,axes=plt.subplots(len(kinds),len(ns),figsize=(5*len(ns),5*len(kinds)),squeeze=False,layout='constrained')
    for ki,kind in enumerate(kinds):
        for ni,n in enumerate(ns):
            ax=axes[ki,ni];sub=summary[(summary.topology==kind)&(summary.n_qubits==n)&(summary.regime=='ou')].set_index('method').reindex(ORDER)
            ax.errorbar(sub.mean_cvar20,np.arange(len(ORDER)),xerr=sub.sd_cvar20.fillna(0),fmt='o',capsize=3,color='#145f72');ax.set_yticks(range(len(ORDER)),ORDER);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_title(f'{kind}, {n} qubits');ax.set_xlabel('OU fidelity CVaR(0.20), mean ± seed SD');ax.grid(axis='x',alpha=.2)
    save(fig,'fig1_ghz_robustness')
    fig,axes=plt.subplots(1,len(kinds),figsize=(6*len(kinds),4),squeeze=False,layout='constrained')
    for ax,kind in zip(axes[0],kinds):
        for m in ['D-NCQAS','Nominal-PPO','Domain-Random','No-Learning']:
            sub=curves[(curves.topology==kind)&(curves.n_qubits==max(ns))&(curves.method==m)]
            gr=sub.groupby('generation').mean_fidelity;means=gr.mean();sd=gr.std().fillna(0)
            ax.plot(means.index,means,label=m);ax.fill_between(means.index,means-sd,means+sd,alpha=.12)
        ax.set(title=f'{kind}, {max(ns)} qubits',xlabel='Generation',ylabel='Mean rollout fidelity ± seed SD',ylim=(0,1));ax.legend(fontsize=8)
    save(fig,'fig2_learning')
    fig,axes=plt.subplots(1,len(kinds),figsize=(6*len(kinds),5),squeeze=False,layout='constrained')
    for ax,kind in zip(axes[0],kinds):
        sub=stats[(stats.topology==kind)&(stats.n_qubits==max(ns))].set_index('baseline').reindex(ORDER[:-1]);y=np.arange(len(sub))
        ax.errorbar(sub.delta,y,xerr=[sub.delta-sub.ci95_low,sub.ci95_high-sub.delta],fmt='o',capsize=3,color='#a43b32');ax.set_yticks(y,sub.index);ax.invert_yaxis();ax.axvline(0,color='gray',ls='--');ax.set(title=f'{kind}, {max(ns)} qubits',xlabel='D-NCQAS minus control: CVaR difference\npointwise 95% seed-bootstrap CI')
    save(fig,'fig3_paired_effects')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    sub=summary[(summary.regime=='ou')&(summary.n_qubits==max(ns))&(summary.topology==kinds[0])]
    for i,row in enumerate(sub.itertuples()):
        ax.scatter(row.mean_training_calls,row.mean_cvar20,s=60,label=row.method)
    ax.set(xlabel='Mean training + validation simulator calls',ylabel='Mean OU fidelity CVaR(0.20)',title='Search cost and robustness (equal architecture-rollout budgets)');ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8);ax.grid(alpha=.2)
    save(fig,'fig4_search_cost')
    mols=list(vt.molecule.unique());fig,axes=plt.subplots(1,len(mols),figsize=(5*len(mols),4),squeeze=False,layout='constrained')
    for ax,mol in zip(axes[0],mols):
        sub=vt[(vt.molecule==mol)&(vt.regime=='ou')];ax.errorbar(np.arange(len(sub)),sub.mean_error,yerr=sub.sd_error.fillna(0),fmt='o',capsize=4);ax.set_xticks(range(len(sub)),sub.method,rotation=25,ha='right');ax.set(title=mol,xlabel='Frozen-parameter fixed ansatz',ylabel='OU energy error / Ha (mean ± seed SD)');ax.axhline(0,color='gray',ls='--')
    save(fig,'fig5_vqe_validation')
    sens=pd.read_csv(out/'processed/sensitivity.csv')
    if len(sens):
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,key in zip(axes,['cvar_beta','b_max_final']):
            g=sens[sens.parameter==key].groupby('value').ou_fidelity_cvar20;ax.errorbar(g.mean().index,g.mean(),yerr=g.std(),fmt='o-',capsize=4);ax.set(xlabel=key,ylabel='OU fidelity CVaR(0.20)',title='Exploratory sensitivity, mean ± seed SD');ax.grid(alpha=.2)
        save(fig,'fig6_sensitivity')
    topo=HardwareTopology.synthetic_tree(5);traj=OrnsteinUhlenbeckDriftProcess(topo).sample_trajectory(np.random.default_rng(987),30)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for i,edge in enumerate(topo.edges):ax.plot([c.p2q_vec[i] for c in traj],label=str(edge))
    ax.set(xlabel='Discrete calibration epoch',ylabel='Two-qubit Pauli error probability',title='Illustrative synthetic OU + jump trajectory (not measured calibration)');ax.legend(title='Edge');save(fig,'fig7_synthetic_drift')
    report=dict(primary_endpoint='OU premeasurement fidelity CVaR(0.20) per training seed',primary_test='two-sided paired sign-flip; assumes symmetric null differences',holm_family=len(stats),significant_comparisons=int(stats.significant.sum()),positive_significant=int(((stats.delta>0)&stats.significant).sum()),negative_significant=int(((stats.delta<0)&stats.significant).sum()),ghz_seed_runs=len(df)//3,vqe_rows=len(vr),vqe_convergence_rate=float(vs.optimizer_success.mean()))
    (out/'metadata/analysis_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
