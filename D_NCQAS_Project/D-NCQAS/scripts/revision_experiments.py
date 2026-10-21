#!/usr/bin/env python3
"""Isolated deterministic jobs; ordered aggregation; resumable checkpointed studies."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,dataclasses,hashlib,json,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from provenance import verify_sources
from d_ncqas.coevolution import TrainingConfig,train_qas_method
from d_ncqas.search_baselines import search
from d_ncqas.agents import minimal_ghz,compute_cvar
from d_ncqas.noise_models import HardwareTopology
from d_ncqas.evaluation import make_scenarios,evaluate
from d_ncqas.simulator import DensityMatrixSimulator,compute_circuit_metrics


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,allow_nan=False)+'\n');tmp.replace(path)


def run_job(job):
    cfg=TrainingConfig(**job['training']);seed=job['seed'];method=job['method']
    topo=HardwareTopology.synthetic_tree(cfg.n_qubits,kind=cfg.topology_kind)
    if method=='Validation-Tree':
        sim=DensityMatrixSimulator(cfg.n_qubits,topo);t=time.perf_counter()
        val=make_scenarios(topo,seed,20,'validation')['ou']
        options=[minimal_ghz(topo,r) for r in range(cfg.n_qubits)]
        scores=[compute_cvar([sim.ghz_metrics(c,cfgv)[1] for cfgv in val]) for c in options]
        result=dict(best_circuit=options[int(np.argmax(scores))],history=[],train_time_sec=time.perf_counter()-t,sim_calls_train=sim.sim_call_count,validation_cvar=max(scores),candidate_count=len(options))
    elif method in ('CEM','Age-Evolution','Uniform-Random'):
        result=search(cfg,seed,method,**job.get('baseline',{}))
    else:
        result=train_qas_method(cfg,seed)
    circuit=result.pop('best_circuit');result['gates']=[dataclasses.asdict(g) for g in circuit]
    bank=make_scenarios(topo,seed,job['test_draws'])
    observations,calls=evaluate(circuit,topo,bank)
    metrics=compute_circuit_metrics(circuit,cfg.n_qubits,topo);metrics.pop('edge_counts')
    result.update(circuit_metrics=metrics,sim_calls_eval=calls)
    summaries=[]
    for regime in bank:
        rows=[r for r in observations if r['regime']==regime]
        summaries.append(dict(regime=regime,fidelity_mean=float(np.mean([r['fidelity'] for r in rows])),fidelity_cvar20=compute_cvar([r['fidelity'] for r in rows]),p_ghz_mean=float(np.mean([r['p_ghz'] for r in rows]))))
    return dict(job=job,result=result,summary=summaries,observations=observations)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--job',type=Path);ap.add_argument('--result',type=Path);args=ap.parse_args()
    if args.job:
        atomic_json(args.result,run_job(json.loads(args.job.read_text())));return
    spec=json.loads(args.spec.read_text());out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))}
    manifest=dict(spec=spec,scientific_sources=sources)
    if (out/'manifest.json').exists():
        recorded=json.loads((out/'manifest.json').read_text())
        if recorded['spec']!=spec:raise SystemExit('Cannot resume changed study')
        verify_sources(ROOT,recorded['scientific_sources'])
        manifest=recorded  # Preserve the original snapshot, including historical version metadata.
    driver=out/'driver_manifest.json';driver_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if driver.exists() and json.loads(driver.read_text())['sha256']!=driver_hash:raise SystemExit('Cannot resume changed experiment driver')
    if not driver.exists():atomic_json(driver,dict(sha256=driver_hash,recorded_at='runner initialization'))
    atomic_json(out/'manifest.json',manifest);(out/'RUN_COMPLETE').unlink(missing_ok=True)
    def launch(job):
        name=job['id'];target=out/'jobs'/f'{name}.json'
        if target.exists():
            if json.loads(target.read_text())['job']!=job:raise RuntimeError('Job identity mismatch')
            return name,'cached'
        jp=out/'inputs'/f'{name}.json';atomic_json(jp,job)
        log=out/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
        with log.open('w') as f:
            res=subprocess.run([sys.executable,__file__,'--job',str(jp),'--result',str(target)],stdout=f,stderr=subprocess.STDOUT)
        if res.returncode:raise RuntimeError(f'Failed job {name}; see {log}')
        return name,'done'
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(launch,j) for j in spec['jobs']]
        for i,future in enumerate(as_completed(futures),1):
            name,status=future.result();print(f'{i}/{len(futures)} {name} {status}',flush=True)
    (out/'RUN_COMPLETE').write_text('All configured jobs completed successfully.\n')


if __name__=='__main__':main()
