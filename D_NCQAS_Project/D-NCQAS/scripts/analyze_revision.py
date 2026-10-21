#!/usr/bin/env python3
"""Analyze the frozen revision protocol; fail on incomplete/mismatched paired data."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from d_ncqas.evaluation import paired_comparison,holm_bonferroni

ORDER=['Validation-Tree','No-Learning','Nominal-PPO','Domain-Random','CEM','Age-Evolution','D-NCQAS']
PHASES=['development_optimizer','development_baselines','development_scaling','final_study','vqe_convergence']


def compare(x,y):
    paired=pd.concat([x,y],axis=1,keys=['x','y']).sort_index()
    if paired.isna().any().any():raise ValueError('Unpaired seed data')
    result=paired_comparison(paired.x.values,paired.y.values)
    delta=paired.x.values-paired.y.values;rng=np.random.default_rng(81991)
    boot=delta[rng.integers(len(delta),size=(20000,len(delta)))].mean(axis=1)
    result.update(ci9875_low=float(np.quantile(boot,.00625)),ci9875_high=float(np.quantile(boot,.99375)),paired_sd=float(np.std(delta,ddof=1)))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--outputs',type=Path,default=ROOT/'outputs');args=ap.parse_args();base=args.outputs.resolve()
    for phase in PHASES:
        if not (base/phase/'RUN_COMPLETE').exists():raise SystemExit('Incomplete phase: '+phase)
    out=base/'analysis'
    for sub in ['tables','figures','raw','metadata']:(out/sub).mkdir(parents=True,exist_ok=True)
    rows=[];diagnostics=[];histories=[]
    rawfields=['id','phase','topology','n_qubits','method','seed','regime','scenario','scenario_id','fidelity','p_ghz']
    with (out/'raw/ghz_observations.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=rawfields);writer.writeheader()
        for phase in PHASES[:-1]:
            manifest=json.loads((base/phase/'manifest.json').read_text())
            for job in manifest['spec']['jobs']:
                d=json.loads((base/phase/'jobs'/f"{job['id']}.json").read_text());r=d['result'];cfg=job['training']
                ident=dict(id=job['id'],phase=job['phase'],study=phase,topology=cfg['topology_kind'],n_qubits=cfg['n_qubits'],method=job['method'],variant=job['variant'],seed=job['seed'],proposals=cfg['n_generations']*cfg['episodes_per_gen'])
                common=dict(**ident,validation_cvar=r['validation_cvar'],candidate_count=r['candidate_count'],sim_calls_train=r['sim_calls_train'],train_seconds=r['train_time_sec'],gates=r['circuit_metrics']['total_gates'])
                for row in d['summary']:rows.append({**common,**row})
                if 'policy_probes' in r:
                    probes=r['policy_probes'];h=r['history']
                    diagnostics.append(dict(**ident,initial_probe=np.mean(probes['initial']),final_probe=np.mean(probes['final']),probe_change=np.mean(probes['final'])-np.mean(probes['initial']),fraction_final_above_half=np.mean(np.array(probes['final'])>.50000001),parameter_delta=r['parameter_delta_l2'],mean_kl=np.mean([v['policy_kl'] for v in h]),max_probability_error=max(v['rollout_probability_error'] for v in h),mean_clip_fraction=np.mean([v['clip_fraction'] for v in h]),diagnostic_circuit_evaluations=64))
                for row in r['history']:histories.append({**ident,**row})
                if phase=='final_study':
                    for row in d['observations']:writer.writerow({**{k:ident[k] for k in rawfields if k in ident},**row})
    seeds=pd.DataFrame(rows);diag=pd.DataFrame(diagnostics);history=pd.DataFrame(histories)
    final=seeds[(seeds.phase=='final')&(seeds.regime=='ou')]
    aggregate=final.groupby(['seed','method']).fidelity_cvar20.mean().unstack('method')
    if len(aggregate)!=32 or aggregate.isna().any().any():raise ValueError('Incomplete primary seed matrix')
    spec=json.loads((base/'final_study/manifest.json').read_text())['spec']
    primary=pd.DataFrame([dict(comparator=method,**compare(aggregate['D-NCQAS'],aggregate[method])) for method in spec['primary_comparators']])
    primary['p_holm']=holm_bonferroni(primary.p_signflip.tolist());primary['significant']=primary.p_holm<.05
    secondary=[]
    for (kind,n),group in final.groupby(['topology','n_qubits']):
        wide=group.pivot(index='seed',columns='method',values='fidelity_cvar20')
        for method in ORDER[:-1]:secondary.append(dict(topology=kind,n_qubits=n,comparator=method,**compare(wide['D-NCQAS'],wide[method])))
    secondary=pd.DataFrame(secondary);secondary['p_holm_secondary']=holm_bonferroni(secondary.p_signflip.tolist())
    full=final[(final.topology=='tree')&(final.n_qubits==5)&(final.seed<1016)].set_index('seed').loc[lambda x:x.method=='D-NCQAS','fidelity_cvar20']
    ablations=[]
    for method,group in seeds[(seeds.phase=='ablation')&(seeds.regime=='ou')].groupby('method'):
        ablations.append(dict(comparator=method,**compare(full,group.set_index('seed').fidelity_cvar20)))
    ablations=pd.DataFrame(ablations);ablations['p_holm_secondary']=holm_bonferroni(ablations.p_signflip.tolist())
    aggregate.to_csv(out/'tables/primary_seed_matrix.csv')
    summary=final.groupby(['topology','n_qubits','method']).agg(n=('seed','nunique'),mean_cvar=('fidelity_cvar20','mean'),sd_cvar=('fidelity_cvar20','std'),mean_calls=('sim_calls_train','mean'),mean_gates=('gates','mean')).reset_index()
    summary['ci95_halfwidth']=stats.t.ppf(.975,summary.n-1)*summary.sd_cvar/np.sqrt(summary.n)
    vstarts=[];vseeds=[];vraw=[]
    for p in sorted((base/'vqe_convergence/jobs').glob('*.json')):
        d=json.loads(p.read_text());sp=d['spec'];ident=dict(molecule=Path(sp['molecule']).stem,seed=sp['seed'],layers=sp['layers'],training=sp['training'])
        for fit in d['fits']:
            for restart in fit['runs']:
                vstarts.append(dict(**ident,algorithm=fit['algorithm'],cap=fit['cap'],selected=restart['restart']==fit['best']['restart'],reference=d['reference_energy'],**{k:v for k,v in restart.items() if k!='angles'}))
        for row in d['observations']:vraw.append({**ident,**row})
        selected=d['fits'][-1]['best']
        for regime in ['clean','nominal','static','ou']:
            observations=[v for v in d['observations'] if v['regime']==regime]
            vseeds.append(dict(**ident,regime=regime,mean_energy_error=np.mean([v['energy_error'] for v in observations]),mean_measured_energy=np.mean([v['measured_energy'] for v in observations]),gradient_inf=selected['gradient_inf'],solver_success=selected['success'],stationary=selected['stationary']))
    vstarts=pd.DataFrame(vstarts);vseeds=pd.DataFrame(vseeds);vraw=pd.DataFrame(vraw)
    vsummary=vstarts[vstarts.selected].groupby(['algorithm','cap']).agg(fits=('success','size'),solver_success_rate=('success','mean'),stationarity_rate=('stationary','mean'),max_gradient=('gradient_inf','max'),mean_nfev=('nfev','mean')).reset_index()
    tables={'ghz_seed_results':seeds,'policy_diagnostics':diag,'learning_history':history,'primary_tests':primary,'secondary_task_tests':secondary,'ablation_tests':ablations,'ghz_summary':summary,'vqe_optimizer_runs':vstarts,'vqe_optimizer_summary':vsummary,'vqe_seed_results':vseeds}
    for name,df in tables.items():df.to_csv(out/'tables'/f'{name}.csv',index=False)
    for name in ['primary_tests','ablation_tests','ghz_summary','vqe_optimizer_summary']:(out/'tables'/f'{name}.md').write_text(tables[name].to_markdown(index=False)+'\n')
    vraw.to_csv(out/'raw/vqe_observations.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(out/'figures'/f'{name}.pdf',bbox_inches='tight');fig.savefig(out/'figures'/f'{name}.png',dpi=600,bbox_inches='tight');plt.close(fig)
    dev=diag[diag.study=='development_optimizer'];fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    variants=['sgd_0.003','adam_0.001','adam_0.003']
    for ax,n in zip(axes,[3,5]):
        g=dev[dev.n_qubits==n].groupby('variant').agg(initial=('initial_probe','mean'),final=('final_probe','mean'),sd=('final_probe','std')).reindex(variants)
        x=np.arange(3);ax.bar(x-.16,g.initial,.32,label='Initial policy');ax.bar(x+.16,g.final,.32,yerr=g.sd,capsize=3,label='Final policy');ax.set_xticks(x,variants,rotation=15);ax.set(ylim=(0,1),ylabel='Nominal probe fidelity: mean ± seed SD',title=f'Development: {n}-qubit tree');ax.legend()
    save(fig,'fig1_optimizer_diagnostics')
    scaling=seeds[(seeds.study=='development_scaling')&(seeds.regime=='ou')];fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,n in zip(axes,[3,5]):
        for method in ['D-NCQAS','No-Learning']:
            g=scaling[(scaling.n_qubits==n)&(scaling.method==method)].groupby('proposals').fidelity_cvar20
            ax.errorbar(g.mean().index,g.mean(),yerr=g.std(),fmt='o-',capsize=3,label=method)
        ax.set(title=f'Development: {n}-qubit tree',xlabel='Circuit proposals',ylabel='Selected-circuit OU CVaR: mean ± seed SD',ylim=(0,1));ax.legend()
    save(fig,'fig2_budget_scaling')
    tasks=[('tree',3),('tree',5),('chain',5)];fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for ax,(kind,n) in zip(axes,tasks):
        g=summary[(summary.topology==kind)&(summary.n_qubits==n)].set_index('method').reindex(ORDER);y=np.arange(len(g))
        ax.errorbar(g.mean_cvar,y,xerr=g.ci95_halfwidth,fmt='o',capsize=3,color='#176b78');ax.set_yticks(y,g.index);ax.invert_yaxis();ax.set(xlim=(0,1),title=f'{kind}, {n} qubits',xlabel='OU CVaR: mean with 95% t interval');ax.axvline(.5,ls=':',color='gray');ax.grid(axis='x',alpha=.2)
    save(fig,'fig3_final_robustness')
    fig,ax=plt.subplots(figsize=(8.3,4),layout='constrained');p=primary;y=np.arange(len(p))
    ax.errorbar(p.delta,y,xerr=[p.delta-p.ci9875_low,p.ci9875_high-p.delta],fmt='o',capsize=4,color='#ac4934');ax.set_yticks(y,[f'{m} (Holm p={pv:.3g})' for m,pv in zip(p.comparator,p.p_holm)]);ax.axvline(0,color='gray',ls='--');ax.axvline(.075,color='gray',ls=':');ax.set(xlabel='D-NCQAS minus comparator: task-averaged CVaR\n98.75% seed-bootstrap percentile interval',title='Four prespecified primary comparisons');save(fig,'fig4_primary_effects')
    fig,axes=plt.subplots(2,1,figsize=(6.4,6.8),layout='constrained')
    h=history[(history.phase=='final')&(history.topology=='tree')&(history.n_qubits==5)]
    for method in ['D-NCQAS','Domain-Random','Nominal-PPO','No-Learning']:
        sub=h[h.method==method];g=sub.groupby('generation').mean_fidelity;m=g.mean();sd=g.std();axes[0].plot(m.index,m,label=method);axes[0].fill_between(m.index,m-sd,m+sd,alpha=.1)
    axes[0].set(xlabel='Generation',ylabel='Rollout fidelity: mean ± seed SD',ylim=(0,1),title='Five-qubit tree training');axes[0].legend(fontsize=8)
    probes=diag[(diag.phase=='final')&(diag.n_qubits==5)&(diag.topology=='tree')];g=probes.groupby('method')[['initial_probe','final_probe']].mean();x=np.arange(len(g));axes[1].bar(x-.17,g.initial_probe,.34,label='Initial');axes[1].bar(x+.17,g.final_probe,.34,label='Final');axes[1].set_xticks(x,g.index,rotation=15);axes[1].set(ylabel='Mean nominal policy-probe fidelity',ylim=(0,1),title='Policy quality before selection');axes[1].legend();save(fig,'fig5_learning_and_probes')
    fig,axes=plt.subplots(2,1,figsize=(6.4,6.8),layout='constrained');labels=[f'{r.algorithm}\n{r.cap} '+('iterations' if r.algorithm=='BFGS' else 'evaluations') for r in vsummary.itertuples()];x=np.arange(len(vsummary));axes[0].bar(x-.17,vsummary.solver_success_rate,.34,label='Solver reports success');axes[0].bar(x+.17,vsummary.stationarity_rate,.34,label='Gradient ≤ 1e-6');axes[0].set_xticks(x,labels);axes[0].set(ylim=(0,1.05),ylabel='Fraction of selected fits',title='Uniform VQE convergence checks');axes[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,1.22),ncol=2)
    selected=vstarts[vstarts.selected];data=[selected[(selected.algorithm==r.algorithm)&(selected.cap==r.cap)].gradient_inf.clip(lower=1e-14) for r in vsummary.itertuples()];axes[1].boxplot(data,tick_labels=labels,showfliers=True);axes[1].set_yscale('log');axes[1].axhline(1e-6,ls='--',color='gray');axes[1].set(ylabel='Infinity norm of parameter-shift gradient',title='Independent stationarity diagnostic');save(fig,'fig6_vqe_convergence')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,mol in zip(axes,sorted(vseeds.molecule.unique())):
        vs=vseeds[(vseeds.molecule==mol)&(vseeds.regime=='ou')].copy();vs['method']=vs.training+'-L'+vs.layers.astype(str);g=vs.groupby('method').mean_energy_error
        ax.errorbar(np.arange(4),g.mean(),yerr=g.std(),fmt='o',capsize=3);ax.set_xticks(range(4),g.mean().index,rotation=25);ax.set(title=mol,ylabel='OU energy error / Ha: mean ± seed SD');ax.axhline(0,ls=':',color='gray')
    save(fig,'fig7_vqe_frozen_parameters')
    fig,ax=plt.subplots(figsize=(8.5,4),layout='constrained');a=ablations;y=np.arange(len(a));ax.errorbar(a.delta,y,xerr=[a.delta-a.ci95_low,a.ci95_high-a.delta],fmt='o',capsize=3);ax.set_yticks(y,[f'{m} (secondary Holm p={p:.3g})' for m,p in zip(a.comparator,a.p_holm_secondary)]);ax.axvline(0,color='gray',ls='--');ax.set(xlabel='Full minus ablation: five-qubit-tree CVaR\npointwise 95% bootstrap interval',title='Secondary ablations, 16 paired seeds');save(fig,'fig8_ablations')
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for row in summary[(summary.topology=='tree')&(summary.n_qubits==5)].itertuples():ax.scatter(row.mean_calls,row.mean_cvar,s=55,label=row.method)
    ax.set(xlabel='Mean search + candidate-validation circuit evaluations',ylabel='Mean OU fidelity CVaR',title='Five-qubit tree: cost and selected-circuit quality');ax.legend(bbox_to_anchor=(1.02,1),loc='upper left');ax.grid(alpha=.2);save(fig,'fig9_search_cost')
    report=dict(primary_unit='seed, equally averaged over three fixed tasks',primary_seeds=len(aggregate),primary_family=len(primary),positive_primary=int(((primary.delta>0)&primary.significant).sum()),negative_primary=int(((primary.delta<0)&primary.significant).sum()),main_runs=int((seeds.phase=='final').sum()/3),ablation_runs=int((seeds.phase=='ablation').sum()/3),development_runs=int((seeds.phase=='development').sum()/3),raw_ghz_observations=int(sum(len(json.loads((base/'final_study/jobs'/f"{j['id']}.json").read_text())['observations']) for j in spec['jobs'])),vqe_fits=int(vsummary.fits.iloc[0]),vqe_observations=len(vraw),vqe_bfgs_stationarity_rate=float(vsummary[vsummary.algorithm=='BFGS'].stationarity_rate.iloc[0]),vqe_bfgs_solver_success_rate=float(vsummary[vsummary.algorithm=='BFGS'].solver_success_rate.iloc[0]),max_rollout_probability_error=float(diag.max_probability_error.max()),planning_power=0.8045886665112788)
    (out/'metadata/analysis_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
