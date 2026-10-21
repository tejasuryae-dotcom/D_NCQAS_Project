#!/usr/bin/env python3
"""Validate every configured revision job, paired observation and numerical summary."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from provenance import verify_sources
from d_ncqas.agents import compute_cvar,build_action_catalog,compute_valid_action_mask
from d_ncqas.coevolution import TrainingConfig
from d_ncqas.simulator import GateOp,DensityMatrixSimulator,circuit_sha256
from d_ncqas.noise_models import HardwareTopology,NoiseConfig
from d_ncqas.evaluation import evaluate,make_scenarios,noise_dict,scenario_id,holm_bonferroni
from d_ncqas.vqe import MolecularHamiltonian,ansatz


def require(cond,message):
    if not cond:raise RuntimeError(message)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--outputs',type=Path,default=ROOT/'outputs');args=ap.parse_args();base=args.outputs.resolve();report={};all_jobs=[];metadata_migrations=[];expected_raw=[]
    for phase in ['development_optimizer','development_baselines','development_scaling','final_study']:
        folder=base/phase;require((folder/'RUN_COMPLETE').exists(),'Incomplete '+phase)
        require(json.loads((folder/'driver_manifest.json').read_text())['sha256']==hashlib.sha256((ROOT/'scripts/revision_experiments.py').read_bytes()).hexdigest(),'Experiment driver changed since provenance record')
        manifest=json.loads((folder/'manifest.json').read_text());planned=manifest['spec']['jobs']
        require(len({j['id'] for j in planned})==len(planned),'Duplicate planned IDs')
        require({p.stem for p in (folder/'jobs').glob('*.json')}=={j['id'] for j in planned},'Missing/extra jobs '+phase)
        metadata_migrations.extend(verify_sources(ROOT,manifest['scientific_sources']))
        for job in planned:
            d=json.loads((folder/'jobs'/f"{job['id']}.json").read_text());require(d['job']==job,'Job identity mismatch')
            r=d['result'];cfg=TrainingConfig(**job['training']);topo=HardwareTopology.synthetic_tree(cfg.n_qubits,kind=cfg.topology_kind);sim=DensityMatrixSimulator(cfg.n_qubits,topo)
            circuit=[GateOp(g['name'],tuple(g['qubits']),g['theta']) for g in r['gates']]
            require(cfg.n_qubits<=len(circuit)<=cfg.max_steps,'Circuit length violates common grammar')
            catalog=build_action_catalog(cfg.n_qubits,topo);prefix=[]
            for gate in circuit:
                sim.validate_gate(gate)
                require(gate in catalog,'Gate outside common action alphabet')
                require(compute_valid_action_mask(prefix,catalog,cfg.n_qubits,min_gates=cfg.n_qubits)[catalog.index(gate)],'Circuit violates common action mask')
                prefix.append(gate)
            require(r['circuit_metrics']['circuit_hash']==circuit_sha256(circuit),'Circuit digest mismatch')
            require(len(d['observations'])==3*job['test_draws'],'Wrong observation count')
            require(len({(row['regime'],row['scenario']) for row in d['observations']})==len(d['observations']),'Repeated observation')
            for row in d['observations']:require(-1e-12<=row['fidelity']<=1+1e-12 and -1e-12<=row['p_ghz']<=1+1e-12,'Unphysical observation')
            for row in d['summary']:
                obs=[o for o in d['observations'] if o['regime']==row['regime']];fs=[o['fidelity'] for o in obs]
                require(len(fs)==job['test_draws'],'Wrong regime count')
                require(abs(compute_cvar(fs)-row['fidelity_cvar20'])<1e-12,'CVaR mismatch')
                require(abs(np.mean(fs)-row['fidelity_mean'])<1e-12,'Mean mismatch')
                require(abs(np.mean([o['p_ghz'] for o in obs])-row['p_ghz_mean'])<1e-12,'Measured-probability mismatch')
            if 'policy_probes' in r:
                require(max(h['rollout_probability_error'] for h in r['history'])<=1e-12,'Rollout probability bookkeeping')
                if not cfg.train_policy:
                    require(r['parameter_delta_l2']==0,'Frozen policy changed')
                    require(r['policy_probes']['initial']==r['policy_probes']['final'],'Frozen probe mismatch')
            if phase=='final_study' and job['seed']==1000 and job['method'] in ['D-NCQAS','No-Learning','CEM','Age-Evolution']:
                replay,_=evaluate(circuit,topo,make_scenarios(topo,job['seed'],job['test_draws']))
                for fresh,recorded in zip(replay,d['observations']):
                    require(fresh['scenario_id']==recorded['scenario_id'],'Replay scenario mismatch')
                    require(abs(fresh['fidelity']-recorded['fidelity'])<1e-12 and abs(fresh['p_ghz']-recorded['p_ghz'])<1e-12,'Replay numerical mismatch')
            if phase=='final_study':expected_raw.extend(dict(id=job['id'],**row) for row in d['observations'])
            all_jobs.append(dict(study=phase,id=job['id'],phase=job['phase'],method=job['method'],seed=job['seed'],topology=cfg.topology_kind,n_qubits=cfg.n_qubits,observations=len(d['observations'])))
        report[phase+'_jobs']=len(planned)
    final_manifest=json.loads((base/'final_study/manifest.json').read_text())['spec']
    require(hashlib.sha256((ROOT/'docs/REVISION_PROTOCOL.md').read_bytes()).hexdigest()==final_manifest['protocol_sha256'],'Prospective protocol changed')
    raw=pd.read_csv(base/'analysis/raw/ghz_observations.csv')
    require(len(raw)==sum(j['observations'] for j in all_jobs if j['study']=='final_study'),'Aggregated GHZ row count')
    require(not raw.duplicated(['id','regime','scenario']).any(),'Duplicate aggregated GHZ rows')
    keys=['id','regime','scenario'];expected=pd.DataFrame(expected_raw).set_index(keys).sort_index();actual=raw.set_index(keys).sort_index()
    require(expected.index.equals(actual.index),'Aggregated raw job/draw identifiers mismatch')
    require((expected.scenario_id==actual.scenario_id).all(),'Aggregated raw scenario mismatch')
    require(np.allclose(expected[['fidelity','p_ghz']],actual[['fidelity','p_ghz']],atol=1e-12,rtol=0),'Aggregated raw numerical values mismatch')
    require(raw.groupby(['topology','n_qubits','seed','regime','scenario']).scenario_id.nunique().max()==1,'Unpaired GHZ scenarios')
    # Save complete shared scenario configurations once, without method duplication.
    with (base/'analysis/raw/scenarios.jsonl').open('w') as stream:
        for kind,n,seed in sorted({(j['topology'],j['n_qubits'],j['seed']) for j in all_jobs if j['study']=='final_study'}):
            topo=HardwareTopology.synthetic_tree(n,kind=kind)
            for regime,configs in make_scenarios(topo,seed,100).items():
                for i,cfg in enumerate(configs):stream.write(json.dumps(dict(topology=kind,n_qubits=n,seed=seed,regime=regime,scenario=i,scenario_id=scenario_id(cfg),noise=noise_dict(cfg)))+'\n')
    primary=pd.read_csv(base/'analysis/tables/primary_tests.csv');require(len(primary)==4 and set(primary.n_seeds)=={32},'Primary design mismatch')
    require(np.allclose(primary.p_holm,holm_bonferroni(primary.p_signflip.tolist())),'Holm correction mismatch')
    vfolder=base/'vqe_convergence';require((vfolder/'RUN_COMPLETE').exists(),'VQE incomplete')
    verify_sources(ROOT,json.loads((vfolder/'source_manifest.json').read_text())['scientific_sources'])
    require(json.loads((vfolder/'driver_manifest.json').read_text())['sha256']==hashlib.sha256((ROOT/'scripts/vqe_convergence.py').read_bytes()).hexdigest(),'VQE driver changed since provenance record')
    specs=json.loads((vfolder/'manifest.json').read_text())['jobs'];files=list((vfolder/'jobs').glob('*.json'));require(len(files)==len(specs)==120,'VQE count mismatch')
    seen=set();stationary=0;formal_success=0
    for path in files:
        d=json.loads(path.read_text());sp=d['spec'];require(sp in specs,'Unexpected VQE job');ident=(sp['molecule'],sp['seed'],sp['layers'],sp['training']);require(ident not in seen,'Duplicate VQE job');seen.add(ident)
        hp=ROOT/'data/hamiltonians'/sp['molecule'];require(hashlib.sha256(hp.read_bytes()).hexdigest()==d['hamiltonian_sha256'],'Hamiltonian changed')
        ham=MolecularHamiltonian.load(hp);topo=HardwareTopology.synthetic_tree(2);sim=DensityMatrixSimulator(2,topo);circuit=[GateOp(g['name'],tuple(g['qubits']),g['theta']) for g in d['gates']]
        for g in circuit:sim.validate_gate(g)
        require([(f['algorithm'],f['cap']) for f in d['fits']]==[('COBYLA',400),('COBYLA',2000),('BFGS',2000)],'VQE optimizer protocol mismatch')
        for fit in d['fits']:
            require(len(fit['runs'])==5,'VQE restart count')
            require(abs(min(r['energy'] for r in fit['runs'])-fit['best']['energy'])<1e-12,'Incorrect restart selection')
        best=d['fits'][-1]['best'];stationary+=best['stationary'];formal_success+=best['success']
        require(best['stationary']==(best['gradient_inf']<=1e-6),'Stationarity criterion mismatch')
        clean=NoiseConfig(2,topo.edges,np.zeros(2),np.zeros(1),np.zeros(2),idle_decoherence_rate=0)
        training_noise=clean if sp['training']=='Clean' else NoiseConfig(2,topo.edges,*topo.nominal_rates())
        angles=np.asarray(best['angles']);central=[]
        for i in range(len(angles)):
            offset=np.zeros_like(angles);offset[i]=1e-5
            plus=float(np.trace(ham.matrix@sim.execute(ansatz(angles+offset,sp['layers']),training_noise)).real)
            minus=float(np.trace(ham.matrix@sim.execute(ansatz(angles-offset,sp['layers']),training_noise)).real)
            central.append((plus-minus)/2e-5)
        require(max(abs(np.asarray(central)))<=1e-6,'Independent finite-difference stationarity failed')
        require(abs(max(abs(np.asarray(central)))-best['gradient_inf'])<1e-7,'Recorded gradient disagrees with finite differences')
        require(len(d['observations'])==301,'VQE test count')
        for row in d['observations']:require(row['energy_error']>=-1e-8 and abs(row['energy']-ham.e_fci-row['energy_error'])<1e-12,'VQE physical energy error')
        if sp['seed']==1000:
            clean=NoiseConfig(2,topo.edges,np.zeros(2),np.zeros(1),np.zeros(2),idle_decoherence_rate=0)
            noise=clean if sp['training']=='Clean' else NoiseConfig(2,topo.edges,*topo.nominal_rates())
            value=float(np.trace(ham.matrix@sim.execute(circuit,noise)).real)
            require(abs(value-best['energy'])<1e-10,'VQE energy replay mismatch')
    vr=pd.read_csv(base/'analysis/raw/vqe_observations.csv');require(len(vr)==36120,'VQE aggregated rows')
    require(not vr.duplicated(['molecule','seed','layers','training','regime','scenario']).any(),'Duplicate VQE observations')
    require(vr.groupby(['molecule','seed','regime','scenario']).scenario_id.nunique().max()==1,'Unpaired VQE scenarios')
    require(stationary==120,'A selected VQE fit has not met stationarity')
    report.update(status='PASS',raw_ghz_observations=len(raw),vqe_fits=120,vqe_observations=len(vr),vqe_stationary_fits=stationary,vqe_finite_difference_stationarity_checked=120,vqe_solver_success_fits=formal_success,primary_seed_pairs=32,primary_holm_family=4,selected_ghz_replay_jobs=12,vqe_energy_replay_jobs=12,replay_tolerance=1e-12,numerical_sources_match=True,version_only_metadata_migration=metadata_migrations[:1],prospective_protocol_hash_matches=True)
    (base/'analysis/metadata/integrity_report.json').write_text(json.dumps(report,indent=2)+'\n');pd.DataFrame(all_jobs).to_csv(base/'analysis/metadata/completed_jobs.csv',index=False);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
