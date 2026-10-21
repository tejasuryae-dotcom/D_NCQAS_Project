#!/usr/bin/env python3
"""Immutable run directory, complete raw observations, explicit JSON config."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,sys,json,time,hashlib,platform,logging,dataclasses
from pathlib import Path
import numpy as np,pandas as pd,scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from d_ncqas.noise_models import HardwareTopology,NoiseConfig
from d_ncqas.simulator import GateOp,DensityMatrixSimulator,compute_circuit_metrics,circuit_sha256
from d_ncqas.coevolution import method_configs,train_qas_method,TrainingConfig
from d_ncqas.agents import minimal_ghz,compute_cvar
from d_ncqas.evaluation import make_scenarios,evaluate,noise_dict,scenario_id
from d_ncqas.vqe import MolecularHamiltonian,optimize,measured_energy

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def gates(c):return [dataclasses.asdict(g) for g in c]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'configs/publication.json');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();cfg=json.loads(a.config.read_text());out=a.output.resolve()
    if out.exists() and not a.resume:raise SystemExit(f'Refusing to overwrite existing run: {out}; use --resume to recover checkpoints')
    if a.resume and out.exists():
        if json.loads((out/'metadata/config.json').read_text()) != cfg:raise SystemExit('Resume config mismatch')
        previous=json.loads((out/'metadata/source_hashes.json').read_text())
        for rel,digest in previous.items():
            if rel.startswith(('src/','data/')) and hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:raise SystemExit('Resume scientific source mismatch: '+rel)
    for sub in ('raw','processed','tables','figures','logs','checkpoints','metadata'):(out/sub).mkdir(parents=True,exist_ok=True)
    (out/'RUN_COMPLETE').unlink(missing_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'logs/execution.log'),logging.StreamHandler()])
    start=time.perf_counter();dump(out/'metadata/config.json',cfg)
    if a.resume and (out/'metadata/source_hashes.json').exists():
        old=json.loads((out/'metadata/source_hashes.json').read_text());dump(out/'metadata/pre_resume_source_hashes.json',old)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for pattern in ('src/**/*.py','scripts/*.py','data/**/*.json','configs/*.json') for p in ROOT.glob(pattern)}
    dump(out/'metadata/source_hashes.json',sources)
    summary=[];curves=[]
    with (out/'raw/ghz_observations.jsonl').open('w') as raw, (out/'raw/scenarios.jsonl').open('w') as scenario_file:
        for kind in cfg['topologies']:
            for n in cfg['qubits']:
                topo=HardwareTopology.synthetic_tree(n,kind=kind)
                methods=method_configs(n,kind,cfg['training'])
                for seed in cfg['seeds']:
                    bank=make_scenarios(topo,seed,cfg['test_draws'])
                    for regime,configs in bank.items():
                        for i,c in enumerate(configs):scenario_file.write(json.dumps(dict(topology=kind,n_qubits=n,seed=seed,regime=regime,scenario=i,scenario_id=scenario_id(c),noise=noise_dict(c)))+'\n')
                    for method in ['Minimal-GHZ','Validation-Tree']+methods:
                        t=time.perf_counter()
                        if isinstance(method,str):
                            name=method;circ=minimal_ghz(topo);train_calls=0;hist=[];trained={}
                            if name=='Validation-Tree':
                                val=make_scenarios(topo,seed,20,'validation')['ou'];sim=DensityMatrixSimulator(n,topo)
                                circuits=[minimal_ghz(topo,root) for root in range(n)]
                                scores=[compute_cvar([sim.ghz_metrics(c,cfgv)[1] for cfgv in val]) for c in circuits]
                                circ=circuits[int(np.argmax(scores))];train_calls=sim.sim_call_count
                        else:
                            name=method.method_name
                            checkpoint=out/'checkpoints'/f'{kind}_{n}_{name}_{seed}.json'
                            if a.resume and checkpoint.exists():
                                trained=json.loads(checkpoint.read_text());circ=[GateOp(g['name'],tuple(g['qubits']),g['theta']) for g in trained.pop('gates')]
                                for k in ('topology','n_qubits','method'):trained.pop(k,None)
                            else:
                                trained=train_qas_method(method,seed);circ=trained.pop('best_circuit')
                            train_calls=trained['sim_calls_train'];hist=trained['history']
                        training_time=trained.get('train_time_sec',time.perf_counter()-t);t=time.perf_counter()
                        observations,calls=evaluate(circ,topo,bank);eval_time=time.perf_counter()-t
                        st=compute_circuit_metrics(circ,n,topo);st.pop('edge_counts')
                        ident=dict(topology=kind,n_qubits=n,seed=seed,method=name)
                        dump(out/'checkpoints'/f'{kind}_{n}_{name}_{seed}.json',{**ident,"gates":gates(circ),**trained})
                        for row in observations:raw.write(json.dumps({**ident,**row})+'\n')
                        for regime in bank:
                            rows=[r for r in observations if r['regime']==regime];fs=[r['fidelity'] for r in rows]
                            summary.append(dict(**ident,regime=regime,fidelity_mean=float(np.mean(fs)),fidelity_cvar20=compute_cvar(fs),p_ghz_mean=float(np.mean([r['p_ghz'] for r in rows])),**st,train_time_sec=training_time,eval_time_sec=eval_time,sim_calls_train=train_calls,sim_calls_eval=calls))
                        curves.extend([{**ident,**h} for h in hist]);logging.info('%s n=%d seed=%d %s complete',kind,n,seed,name)
                    raw.flush();scenario_file.flush()
                    pd.DataFrame(summary).to_csv(out/'processed/ghz_seeds.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'processed/learning_curves.csv',index=False)
    vrows=[]
    # Molecular validation has its own accurately named fixed-ansatz baselines.
    for hp in sorted((ROOT/'data/hamiltonians').glob('*.json')):
        ham=MolecularHamiltonian.load(hp);topo=HardwareTopology.synthetic_tree(2)
        for seed in cfg['vqe']['seeds']:
            bank=make_scenarios(topo,seed,cfg['vqe']['test_draws'])
            clean=NoiseConfig(2,topo.edges,np.zeros(2),np.zeros(1),np.zeros(2),idle_decoherence_rate=0)
            nominal=NoiseConfig(2,topo.edges,*topo.nominal_rates())
            for layers in (1,2):
                for training_name,training_noise in [('Clean',clean),('Nominal',nominal)]:
                    name=f'{training_name}-HEA-L{layers}'
                    checkpoint=out/'checkpoints'/f'VQE_{hp.stem}_{name}_{seed}.json'
                    if a.resume and checkpoint.exists():
                        info=json.loads(checkpoint.read_text());circ=[GateOp(g['name'],tuple(g['qubits']),g['theta']) for g in info.pop('gates')]
                        for k in ('molecule','method','seed'):info.pop(k,None)
                    else:
                        circ,info=optimize(ham,topo,training_noise,layers,seed,cfg['vqe']['restarts'],cfg['vqe']['maxiter'])
                    dump(checkpoint,dict(molecule=ham.name,method=name,seed=seed,gates=gates(circ),**info))
                    sim=DensityMatrixSimulator(2,topo)
                    for regime,configs in {'clean':[clean],**bank}.items():
                        for i,c in enumerate(configs):
                            rho=sim.execute(circ,c);energy=float(np.trace(ham.matrix@rho).real)
                            vrows.append(dict(molecule=ham.name,method=name,seed=seed,regime=regime,scenario=i,scenario_id=scenario_id(c),energy=energy,energy_error=energy-ham.e_fci,measured_energy=measured_energy(rho,ham,sim,c.pm_vec),e_fci_active=ham.e_fci,optimizer_success=info['converged'],optimization_calls=info['sim_calls']))
            logging.info('VQE %s seed=%d complete',ham.name,seed)
    pd.DataFrame(vrows).to_csv(out/'raw/vqe_observations.csv',index=False)
    sens=[]
    if cfg.get('sensitivity'):
        topo=HardwareTopology.synthetic_tree(4)
        # Matched initialization and seed streams across settings; exploratory, not confirmatory.
        for key,values in [('cvar_beta',[.1,.2,.5,1.]),('b_max_final',[2.,4.,6.,8.])]:
            for value in values:
                for seed in cfg['seeds'][:5]:
                    tc=TrainingConfig(n_qubits=4,max_steps=10,**cfg['training'],**{key:value})
                    checkpoint=out/'checkpoints'/f'sensitivity_{key}_{value}_{seed}.json'
                    if a.resume and checkpoint.exists():
                        tr=json.loads(checkpoint.read_text());circ=[GateOp(g['name'],tuple(g['qubits']),g['theta']) for g in tr.pop('gates')]
                        for k in ('parameter','value'):tr.pop(k,None)
                    else:
                        tr=train_qas_method(tc,seed);circ=tr.pop('best_circuit')
                    bank=make_scenarios(topo,seed,cfg['test_draws']);rows,calls=evaluate(circ,topo,bank)
                    fs=[r['fidelity'] for r in rows if r['regime']=='ou']
                    ident=dict(parameter=key,value=value,seed=seed)
                    dump(out/'checkpoints'/f'sensitivity_{key}_{value}_{seed}.json',{**ident,"gates":gates(circ),**tr})
                    dump(out/'raw'/f'sensitivity_{key}_{value}_{seed}.json',rows)
                    sens.append(dict(**ident,ou_fidelity_cvar20=compute_cvar(fs),train_time_sec=tr['train_time_sec'],sim_calls_train=tr['sim_calls_train']))
                logging.info('Sensitivity %s=%s complete',key,value)
    pd.DataFrame(sens,columns=['parameter','value','seed','ou_fidelity_cvar20','train_time_sec','sim_calls_train']).to_csv(out/'processed/sensitivity.csv',index=False)
    dump(out/'metadata/environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,wall_seconds=time.perf_counter()-start))
    (out/'RUN_COMPLETE').write_text('All configured experiments finished.\n')
    logging.info('All experiments completed in %.1fs',time.perf_counter()-start)
if __name__=='__main__':main()
