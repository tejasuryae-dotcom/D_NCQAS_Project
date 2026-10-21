#!/usr/bin/env python3
"""Uniform optimizer-cap study and independent parameter-shift refinement."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,dataclasses,json,sys,time,hashlib,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from scipy.optimize import minimize
from provenance import source_hashes,verify_sources
from d_ncqas.vqe import MolecularHamiltonian,ansatz,measured_energy
from d_ncqas.noise_models import HardwareTopology,NoiseConfig
from d_ncqas.simulator import DensityMatrixSimulator
from d_ncqas.evaluation import make_scenarios,scenario_id


def objective(ham,topology,noise,layers):
    sim=DensityMatrixSimulator(2,topology)
    def energy(x):return float(np.trace(ham.matrix@sim.execute(ansatz(x,layers),noise)).real)
    def jac(x):
        grad=np.empty(len(x))
        for i in range(len(x)):
            plus=x.copy();minus=x.copy();plus[i]+=np.pi/2;minus[i]-=np.pi/2
            grad[i]=.5*(energy(plus)-energy(minus))
        return grad
    return energy,jac,sim


def run_fit(spec):
    ham=MolecularHamiltonian.load(ROOT/'data/hamiltonians'/spec['molecule'])
    topo=HardwareTopology.synthetic_tree(2)
    clean=NoiseConfig(2,topo.edges,np.zeros(2),np.zeros(1),np.zeros(2),idle_decoherence_rate=0)
    noise=clean if spec['training']=='Clean' else NoiseConfig(2,topo.edges,*topo.nominal_rates())
    rng=np.random.default_rng([spec['seed'],33,spec['layers']]);dim=2*(spec['layers']+1)
    starts=[np.zeros(dim)]+[rng.uniform(-np.pi,np.pi,dim) for _ in range(4)]
    fits=[];selected=None
    for algorithm,cap in [('COBYLA',400),('COBYLA',2000),('BFGS',2000)]:
        runs=[];energy,jac,sim=objective(ham,topo,noise,spec['layers']);start=time.perf_counter()
        for i,x0 in enumerate(starts):
            options={'maxiter':cap,'tol':1e-6,'rhobeg':.5} if algorithm=='COBYLA' else {'maxiter':cap,'gtol':1e-8}
            res=minimize(energy,x0,method=algorithm,jac=jac if algorithm=='BFGS' else None,options=options)
            g=jac(res.x);grad_inf=float(np.max(abs(g)))
            runs.append(dict(restart=i,energy=float(res.fun),angles=res.x.tolist(),success=bool(res.success),status=int(res.status),message=str(res.message),nfev=int(res.nfev),nit=int(getattr(res,'nit',-1)),gradient_inf=grad_inf,stationary=grad_inf<=1e-6))
        best=min(runs,key=lambda r:r['energy'])
        fits.append(dict(algorithm=algorithm,cap=cap,runs=runs,best=best,sim_calls=sim.sim_call_count,wall_seconds=time.perf_counter()-start))
        if algorithm=='BFGS':selected=best
    circuit=ansatz(selected['angles'],spec['layers']);sim=DensityMatrixSimulator(2,topo)
    observations=[]
    for regime,configs in {'clean':[clean],**make_scenarios(topo,spec['seed'],spec['test_draws'])}.items():
        for i,c in enumerate(configs):
            rho=sim.execute(circuit,c);physical=float(np.trace(ham.matrix@rho).real)
            observations.append(dict(regime=regime,scenario=i,scenario_id=scenario_id(c),energy=physical,energy_error=physical-ham.e_fci,measured_energy=measured_energy(rho,ham,sim,c.pm_vec)))
    return dict(spec=spec,hamiltonian_sha256=hashlib.sha256((ROOT/'data/hamiltonians'/spec['molecule']).read_bytes()).hexdigest(),reference_energy=ham.e_fci,fits=fits,selected_algorithm='BFGS-2000',gates=[dataclasses.asdict(g) for g in circuit],observations=observations)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--job',type=Path);ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
    if args.job:
        data=run_fit(json.loads(args.job.read_text()));tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(data,allow_nan=False)+'\n');tmp.replace(args.output);return
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    jobs=[dict(molecule=p.name,seed=seed,layers=layer,training=training,test_draws=100) for p in sorted((ROOT/'data/hamiltonians').glob('*.json')) for seed in range(1000,1010) for layer in [1,2] for training in ['Clean','Nominal']]
    manifest={'protocol':'same five initializations, COBYLA caps 400/2000, BFGS maxiter2000 with parameter-shift Jacobian; final algorithm fixed as BFGS; stationarity threshold 1e-6 Ha/radian; no global optimality claim','jobs':jobs}
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest:raise SystemExit('VQE config mismatch')
    source_path=out/'source_manifest.json'
    if source_path.exists():verify_sources(ROOT,json.loads(source_path.read_text())['scientific_sources'])
    else:source_path.write_text(json.dumps({'scientific_sources':source_hashes(ROOT),'recorded_at':'runner initialization'},indent=2)+'\n')
    driver=out/'driver_manifest.json';driver_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if driver.exists() and json.loads(driver.read_text())['sha256']!=driver_hash:raise SystemExit('Cannot resume changed VQE driver')
    if not driver.exists():driver.write_text(json.dumps(dict(sha256=driver_hash,recorded_at='runner initialization'),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'RUN_COMPLETE').unlink(missing_ok=True)
    for sub in ['jobs','inputs','logs']:(out/sub).mkdir(exist_ok=True)
    def launch(job):
        name=f"{Path(job['molecule']).stem}_{job['training']}_L{job['layers']}_{job['seed']}";target=out/'jobs'/f'{name}.json'
        if target.exists():
            saved=json.loads(target.read_text())
            if saved['spec']!=job:raise RuntimeError('Cached VQE specification mismatch: '+name)
            if saved['hamiltonian_sha256']!=hashlib.sha256((ROOT/'data/hamiltonians'/job['molecule']).read_bytes()).hexdigest():raise RuntimeError('Cached VQE Hamiltonian changed: '+name)
            return name
        source=out/'inputs'/f'{name}.json';source.write_text(json.dumps(job))
        with (out/'logs'/f'{name}.log').open('w') as log:
            result=subprocess.run([sys.executable,__file__,'--job',str(source),'--output',str(target)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('VQE job failed: '+name)
        return name
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(launch,j) for j in jobs]
        for i,future in enumerate(as_completed(futures),1):print(f'{i}/{len(jobs)} {future.result()}',flush=True)
    (out/'RUN_COMPLETE').write_text('All VQE convergence and evaluation jobs complete.\n')


if __name__=='__main__':main()
