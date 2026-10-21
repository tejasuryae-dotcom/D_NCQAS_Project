"""Independent categorical CEM and aging-evolution circuit-search controls.

These adapt generic optimization algorithms to the common legal gate grammar;
they are not reproductions of published quantum architecture-search packages.
"""
from collections import deque
import time
import numpy as np
from .agents import build_action_catalog, compute_valid_action_mask, compute_cvar
from .noise_models import HardwareTopology, NoiseConfig
from .simulator import DensityMatrixSimulator, circuit_sha256
from .evaluation import make_scenarios
from .coevolution import complexity


def search(cfg, seed, algorithm='CEM', elite_fraction=.25, smoothing=.7, tournament=4):
    start=time.perf_counter()
    topo=HardwareTopology.synthetic_tree(cfg.n_qubits,kind=cfg.topology_kind)
    catalog=build_action_catalog(cfg.n_qubits,topo);n_actions=len(catalog)
    rng=np.random.default_rng([seed,cfg.n_qubits,50911])
    sim=DensityMatrixSimulator(cfg.n_qubits,topo)
    nominal=NoiseConfig(cfg.n_qubits,topo.edges,*topo.nominal_rates())
    # Static training bank is disjoint from candidate-validation and final testing.
    train=make_scenarios(topo,seed+2000000,cfg.n_traj_samples,'validation')['ou']
    validation=make_scenarios(topo,seed,20,'validation')['ou']
    probabilities=np.full((cfg.max_steps,n_actions),1/n_actions)
    scored={};candidates={};history=[];population=deque()

    def decode(genome=None):
        circuit=[];tokens=[]
        for step in range(cfg.max_steps):
            mask=compute_valid_action_mask(circuit,catalog,cfg.n_qubits,min_gates=cfg.n_qubits)
            if genome is None or not mask[genome[step]]:
                p=probabilities[step]*mask;p/=p.sum();action=int(rng.choice(n_actions,p=p))
            else:action=int(genome[step])
            tokens.append(action)
            if catalog[action].name=='DONE':break
            circuit.append(catalog[action])
        # Suffix genes do not affect the phenotype; they remain available for mutation.
        suffix=rng.integers(n_actions,size=cfg.max_steps-len(tokens)).tolist()
        return circuit,np.asarray(tokens+suffix),len(tokens)

    def objective(circuit):
        key=circuit_sha256(circuit)
        if key not in scored:
            fs=[sim.ghz_metrics(circuit,c)[1] for c in train]
            scored[key]=.4*sim.ghz_metrics(circuit,nominal)[1]+.6*(.5*np.mean(fs)+.5*compute_cvar(fs))-complexity(circuit,topo,cfg)
            candidates[key]=circuit
        return scored[key]

    for generation in range(cfg.n_generations):
        batch=[]
        for _ in range(cfg.episodes_per_gen):
            genome=None
            if algorithm=='Age-Evolution' and len(population)>=cfg.episodes_per_gen:
                indices=rng.choice(len(population),min(tournament,len(population)),replace=False)
                parent=max((population[i] for i in indices),key=lambda item:item[0])
                genome=parent[1].copy();pos=int(rng.integers(cfg.max_steps))
                choices=np.delete(np.arange(n_actions),genome[pos]);genome[pos]=rng.choice(choices)
            circuit,genome,length=decode(genome);score=objective(circuit)
            batch.append((score,genome,length));population.append((score,genome))
            if len(population)>cfg.episodes_per_gen:population.popleft()
        if algorithm=='CEM':
            elite=sorted(batch,key=lambda item:item[0],reverse=True)[:max(1,int(np.ceil(elite_fraction*len(batch))))]
            counts=np.full_like(probabilities,.01)
            for _,tokens,length in elite:
                for step,action in enumerate(tokens[:length]):counts[step,action]+=1
            fitted=counts/counts.sum(axis=1,keepdims=True)
            probabilities=(1-smoothing)*probabilities+smoothing*fitted
        elif algorithm not in ('Age-Evolution','Uniform-Random'):
            raise ValueError('Unknown search algorithm')
        history.append(dict(generation=generation+1,mean_objective=float(np.mean([b[0] for b in batch])),best_objective=float(max(scored.values())),unique_candidates=len(candidates)))
    best=None;best_score=-np.inf
    for circuit in candidates.values():
        score=compute_cvar([sim.ghz_metrics(circuit,c)[1] for c in validation])
        if score>best_score:best_score=score;best=circuit
    return dict(best_circuit=best,history=history,candidate_count=len(candidates),
                proposals=cfg.n_generations*cfg.episodes_per_gen,validation_cvar=float(best_score),
                train_time_sec=time.perf_counter()-start,sim_calls_train=sim.sim_call_count,
                algorithm=algorithm,elite_fraction=elite_fraction,smoothing=smoothing,tournament=tournament,
                final_distribution=probabilities.tolist(),rng_state=rng.bit_generator.state)
