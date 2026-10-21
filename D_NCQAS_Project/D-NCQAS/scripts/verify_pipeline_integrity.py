#!/usr/bin/env python3
"""Verify actual artifact content and raw-to-summary agreement, not asserted status labels."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from d_ncqas.agents import compute_cvar
from d_ncqas.simulator import GateOp,DensityMatrixSimulator,circuit_sha256
from d_ncqas.noise_models import HardwareTopology
from d_ncqas.evaluation import make_scenarios,evaluate

def require(cond,message):
    if not cond:raise RuntimeError(message)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output
    require((out/'RUN_COMPLETE').exists(),'Run incomplete')
    cfg=json.loads((out/'metadata/config.json').read_text());df=pd.read_csv(out/'processed/ghz_seeds.csv');raw=pd.read_json(out/'raw/ghz_observations.jsonl',lines=True)
    keys=['topology','n_qubits','seed','method','regime'];expected=len(cfg['topologies'])*len(cfg['qubits'])*len(cfg['seeds'])*10
    require(len(df)==expected*3,'Wrong GHZ summary count');require(not df.duplicated(keys).any(),'Duplicate GHZ summaries')
    require(len(raw)==expected*3*cfg['test_draws'],'Wrong raw row count');require(not raw.duplicated(keys+['scenario']).any(),'Duplicate raw observations')
    require(raw.groupby(['topology','n_qubits','seed','regime','scenario']).scenario_id.nunique().max()==1,'Scenario pairing failed')
    require(raw[['fidelity','p_ghz']].ge(-1e-10).all().all() and raw[['fidelity','p_ghz']].le(1+1e-10).all().all(),'Unphysical probability/fidelity')
    rebuilt=raw.groupby(keys).fidelity.agg(fidelity_mean='mean',fidelity_cvar20=lambda x:compute_cvar(x)).reset_index()
    joined=df.merge(rebuilt,on=keys,suffixes=('','_recomputed'),validate='one_to_one')
    for metric in ['fidelity_mean','fidelity_cvar20']:require(np.allclose(joined[metric],joined[metric+'_recomputed'],atol=1e-12),'Raw/summary mismatch '+metric)
    # Validate every circuit and replay selected entire scenario banks, including 5-qubit data.
    for ident,rows in df.groupby(['topology','n_qubits','method','seed']):
        kind,n,m,seed=ident;n=int(n);seed=int(seed);p=out/'checkpoints'/f'{kind}_{n}_{m}_{seed}.json';d=json.loads(p.read_text());circ=[GateOp(g['name'],tuple(g['qubits']),g['theta']) for g in d['gates']]
        sim=DensityMatrixSimulator(n,HardwareTopology.synthetic_tree(n,kind=kind))
        for gate in circ:sim.validate_gate(gate)
        require((rows.circuit_hash==circuit_sha256(circ)).all(),'Circuit hash mismatch')
        if m=='D-NCQAS' and seed==cfg['seeds'][0]:
            replay,_=evaluate(circ,sim.topology,make_scenarios(sim.topology,seed,cfg['test_draws']))
            for regime in ['nominal','static','ou']:
                fs=[r['fidelity'] for r in replay if r['regime']==regime]
                require(abs(compute_cvar(fs)-rows[rows.regime==regime].fidelity_cvar20.iloc[0])<1e-12,'Replay mismatch')
    vr=pd.read_csv(out/'raw/vqe_observations.csv');require(np.isfinite(vr.energy).all(),'Nonfinite VQE energy');require(vr.energy_error.min()>-1e-8,'Variational bound violated for physical energy')
    expected_vqe=3*4*len(cfg['vqe']['seeds'])*(1+3*cfg['vqe']['test_draws']);require(len(vr)==expected_vqe,'Wrong VQE observation count')
    require(not vr.duplicated(['molecule','method','seed','regime','scenario']).any(),'Duplicate VQE observations')
    require(vr.groupby(['molecule','seed','regime','scenario']).scenario_id.nunique().max()==1,'VQE scenario pairing failed')
    require(np.isfinite(vr.measured_energy).all(),'Nonfinite measured VQE energy')
    require(np.allclose(vr.energy-vr.e_fci_active,vr.energy_error,atol=1e-12),'VQE error mismatch')
    for p in (out/'checkpoints').glob('VQE_*.json'):
        d=json.loads(p.read_text());sim=DensityMatrixSimulator(2,HardwareTopology.synthetic_tree(2))
        for g in d['gates']:sim.validate_gate(GateOp(g['name'],tuple(g['qubits']),g['theta']))
    sensitivity_runs=0
    if cfg.get('sensitivity'):
        sd=pd.read_csv(out/'processed/sensitivity.csv');sensitivity_runs=8*min(5,len(cfg['seeds']))
        require(len(sd)==sensitivity_runs,'Wrong sensitivity run count')
        require(not sd.duplicated(['parameter','value','seed']).any(),'Duplicate sensitivity run')
        for r in sd.itertuples(index=False):
            prefix=f'sensitivity_{r.parameter}_{r.value}_{r.seed}'
            observations=json.loads((out/'raw'/f'{prefix}.json').read_text())
            require(len(observations)==3*cfg['test_draws'],'Wrong sensitivity observation count')
            fs=[o['fidelity'] for o in observations if o['regime']=='ou']
            require(abs(compute_cvar(fs)-r.ou_fidelity_cvar20)<1e-12,'Sensitivity summary mismatch')
            d=json.loads((out/'checkpoints'/f'{prefix}.json').read_text());sim=DensityMatrixSimulator(4,HardwareTopology.synthetic_tree(4))
            for g in d['gates']:sim.validate_gate(GateOp(g['name'],tuple(g['qubits']),g['theta']))
    for rel,digest in json.loads((out/'metadata/source_hashes.json').read_text()).items():
        if rel.startswith(('src/','data/')):require(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,'Scientific source changed: '+rel)
    report=dict(status='PASS',ghz_seed_runs=expected,raw_ghz_observations=len(raw),vqe_observations=len(vr),sensitivity_runs=sensitivity_runs,scenario_pairing='all methods matched in GHZ and VQE',raw_summary_agreement=True,checkpoint_gate_validation=True,replay_tolerance=1e-12)
    (out/'metadata/integrity_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
