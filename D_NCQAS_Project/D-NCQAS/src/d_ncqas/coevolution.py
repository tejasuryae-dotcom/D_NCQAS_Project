"""On-policy PPO with matched architecture controls and constrained adversaries."""
from dataclasses import dataclass, asdict
import time
import numpy as np
from .noise_models import HardwareTopology, NoiseConfig, OrnsteinUhlenbeckDriftProcess, amplify_noise_config_spatial
from .simulator import DensityMatrixSimulator, GateOp, compute_circuit_metrics
from .agents import MLPPolicyValueNet, TrajectoryConditionedSaboteurAgent, build_action_catalog, compute_valid_action_mask, compute_cvar

@dataclass
class TrainingConfig:
    method_name: str = "D-NCQAS"
    n_qubits: int = 3
    topology_kind: str = "tree"
    max_steps: int = 10
    n_generations: int = 16
    episodes_per_gen: int = 16
    saboteur_episodes_per_gen: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.92
    n_traj_samples: int = 20
    cvar_beta: float = 0.20
    eta_tail: float = 0.5
    b_max_final: float = 6.0
    use_adversary: bool = True
    use_spatial_saboteur: bool = True
    use_temporal_ou: bool = True
    use_cvar_objective: bool = True
    use_hw_complexity: bool = True
    complexity_scale: float = 1.0
    train_policy: bool = True
    nominal_only: bool = False
    optimizer: str = "sgd"
    learning_rate: float = 0.003


def method_configs(n, kind, overrides=None):
    """Every ablation changes exactly its named flag; architecture policy/grammar are shared."""
    changes = {
        "D-NCQAS": {}, "No-Spatial": {"use_spatial_saboteur": False},
        "No-Temporal": {"use_temporal_ou": False}, "No-CVaR": {"use_cvar_objective": False},
        "No-Complexity": {"complexity_scale": 0.0},
        "Domain-Random": {"use_adversary": False},
        "Nominal-PPO": {"nominal_only": True, "use_adversary": False},
        "No-Learning": {"train_policy": False},
    }
    return [TrainingConfig(method_name=m, n_qubits=n, topology_kind=kind, max_steps=2*n+2,
                           **(overrides or {}), **flags) for m,flags in changes.items()]


def complexity(circuit, topology, cfg):
    if cfg.complexity_scale == 0:
        return 0.0
    st = compute_circuit_metrics(circuit, topology.n_qubits, topology)
    rates = topology.nominal_rates()[1]
    return cfg.complexity_scale * (0.003*st['n_1q'] + 0.012*np.dot(st['edge_counts'], rates/rates.mean()) + 0.004*st['depth'])


def state_dict(net):
    return {k: v.tolist() for k,v in vars(net).items() if isinstance(v,np.ndarray)}


def train_qas_method(cfg, seed):
    start = time.perf_counter()
    topo = HardwareTopology.synthetic_tree(cfg.n_qubits, kind=cfg.topology_kind)
    sim = DensityMatrixSimulator(cfg.n_qubits, topo)
    nom = NoiseConfig(cfg.n_qubits, topo.edges, *topo.nominal_rates())
    # Split purpose-specific streams. Changing a method label does not change initialization.
    streams = np.random.SeedSequence([seed, cfg.n_qubits, 311]).spawn(4)
    init_rng, act_rng, noise_rng, sab_rng = [np.random.default_rng(s) for s in streams]
    catalog = build_action_catalog(cfg.n_qubits, topo)
    dim = len(sim.pauli_observation(sim.initial_state_tensor(),0,cfg.max_steps))
    arch = MLPPolicyValueNet(dim,len(catalog),init_rng,lr=cfg.learning_rate,optimizer=cfg.optimizer)
    initial_arch = state_dict(arch)
    sab = TrajectoryConditionedSaboteurAgent(topo,sab_rng)
    sab.net.optimizer = cfg.optimizer
    sab.net.lr = cfg.learning_rate
    ou = OrnsteinUhlenbeckDriftProcess(topo)
    candidates, history = {}, []
    episode = 0
    for gen in range(cfg.n_generations):
        batch = [[] for _ in range(7)]
        sab_batch = [[] for _ in range(7)]
        episode_f, rewards, robust_values, penalties, shaping_values = [], [], [], [], []
        for ep in range(cfg.episodes_per_gen):
            episode += 1
            progress = episode / (cfg.n_generations*cfg.episodes_per_gen)
            budget = 2.0 + (cfg.b_max_final-2.0)*progress
            rho = sim.initial_state_tensor(); sim.sim_call_count += 1
            circuit=[]; obs_l=[]; acts=[]; masks=[]; vals=[]; old=[]; shaping=[]
            prev_f=sim.ghz_metrics_from_rho(rho,nom.pm_vec)[1]
            for step in range(cfg.max_steps):
                obs=sim.pauli_observation(rho,step,cfg.max_steps)
                mask=compute_valid_action_mask(circuit,catalog,cfg.n_qubits,min_gates=cfg.n_qubits)
                probs,value,_=arch.forward(obs,mask)
                action=int(act_rng.choice(len(catalog),p=probs))
                obs_l.append(obs);acts.append(action);masks.append(mask);vals.append(value);old.append(float(probs[action]))
                op=catalog[action]
                if op.name=='DONE':
                    shaping.append(0.0);break
                circuit.append(op);rho=sim.execute_step(rho,op,nom,step)
                fidelity=sim.ghz_metrics_from_rho(rho,nom.pm_vec)[1]
                # Dense optimization heuristic, explicitly not claimed policy invariant.
                shaping.append(0.4*(cfg.gamma*fidelity-prev_f));prev_f=fidelity
            f_nom=sim.ghz_metrics(circuit,nom)[1]
            stressed=[]
            # Always reserve identical architecture rollout counts; actual call costs are reported.
            for j in range(cfg.n_traj_samples):
                base=ou.sample_trajectory(noise_rng,n_epochs=6)[-1] if cfg.use_temporal_ou else nom
                features,_=sab.encode_features(circuit,base,f_nom,cfg.max_steps)
                if not cfg.use_spatial_saboteur:
                    features[5:]=0.0  # Conditioning-only ablation: action space and budget unchanged.
                sab_probs,sab_value,_=sab.net.forward(features)
                idx=int(sab_rng.choice(len(sab.templates),p=sab_probs)) if cfg.use_adversary else int(noise_rng.integers(len(sab.templates)))
                counts=compute_circuit_metrics(circuit,cfg.n_qubits,topo)['edge_counts']
                s1,s2,sm,used=sab.realize_action(idx,counts,budget)
                stress=amplify_noise_config_spatial(base,s1,s2,sm)
                f_st=sim.ghz_metrics(circuit,stress)[1]
                stressed.append(f_st)
                if cfg.use_adversary and ep < cfg.saboteur_episodes_per_gen and j == 0:
                    rs=(f_nom-f_st)-0.08*used/max(budget,1e-12)
                    row=[features,idx,np.ones(len(sab.templates),bool),rs-sab_value,rs,float(sab_probs[idx]),0]
                    for dest,value in zip(sab_batch,row):dest.append(value)
            robust=(1-cfg.eta_tail)*np.mean(stressed)+cfg.eta_tail*compute_cvar(stressed,cfg.cvar_beta) if cfg.use_cvar_objective else np.mean(stressed)
            reward=(f_nom if cfg.nominal_only else 0.4*f_nom+0.6*robust)-complexity(circuit,topo,cfg)
            robust_values.append(float(robust));penalties.append(float(complexity(circuit,topo,cfg)));shaping_values.append(float(sum(shaping)))
            shaping[-1]+=reward
            advantages=np.zeros(len(acts));gae=0
            for t in reversed(range(len(acts))):
                next_v=vals[t+1] if t+1<len(vals) else 0
                delta=shaping[t]+cfg.gamma*next_v-vals[t]
                gae=delta+cfg.gamma*cfg.gae_lambda*gae;advantages[t]=gae
            returns=advantages+np.array(vals)
            for dest,values in zip(batch,[obs_l,acts,masks,list(advantages),list(returns),old,[0]*len(acts)]):dest.extend(values)
            key=tuple(circuit);candidates[key]=None
            episode_f.append(f_nom);rewards.append(reward)
        before=np.array([arch.forward(o,m)[0] for o,m in zip(batch[0],batch[2])])
        recorded=np.array(batch[5]);actual=before[np.arange(len(before)),batch[1]]
        probability_error=float(np.max(abs(recorded-actual)))
        if probability_error>1e-12:raise RuntimeError("Rollout probability mismatch")
        weights_before=np.concatenate([getattr(arch,k).ravel() for k in initial_arch])
        grad=arch.update_ppo(*batch[:3],np.array(batch[3]),np.array(batch[4]),batch[5],epochs=3) if cfg.train_policy else 0.0
        if cfg.train_policy and cfg.use_adversary and sab_batch[0]:
            sab.net.update_ppo(*sab_batch[:3],np.array(sab_batch[3]),np.array(sab_batch[4]),sab_batch[5],epochs=3)
        after=np.array([arch.forward(o,m)[0] for o,m in zip(batch[0],batch[2])])
        logs=np.log(np.maximum(before,1e-12));newlogs=np.log(np.maximum(after,1e-12))
        ratio=after[np.arange(len(after)),batch[1]]/recorded
        history.append(dict(generation=gen+1,mean_fidelity=float(np.mean(episode_f)),mean_reward=float(np.mean(rewards)),gradient_norm=float(grad),budget=float(budget),
            rollout_probability_error=probability_error,policy_kl=float(np.mean(np.sum(before*(logs-newlogs),axis=1))),
            policy_entropy=float(np.mean(-np.sum(after*newlogs,axis=1))),clip_fraction=float(np.mean(abs(ratio-1)>.2)),
            parameter_step_l2=float(np.linalg.norm(np.concatenate([getattr(arch,k).ravel() for k in initial_arch])-weights_before)),
            mean_robust_fidelity=float(np.mean(robust_values)),mean_complexity_penalty=float(np.mean(penalties)),mean_shaping=float(np.mean(shaping_values))))
    # Shared validation set, separate from test set; identical selector for all PPO variants.
    # This prevents per-method selection objectives from confounding the final comparison.
    from .evaluation import make_scenarios
    val=make_scenarios(topo,seed,draws=20,split='validation')['ou']
    best=None;best_score=-np.inf
    for circ in candidates:
        fs=[sim.ghz_metrics(circ,c)[1] for c in val]
        score=compute_cvar(fs,0.2)
        if score>best_score:
            best_score=score;best=list(circ)
    # Fixed independent probe stream measures policy quality without best-of-run selection.
    def probe(net):
        rng=np.random.default_rng([seed,cfg.n_qubits,71111]);scores=[]
        for _ in range(32):
            rho=sim.initial_state_tensor();circuit=[]
            for step in range(cfg.max_steps):
                obs=sim.pauli_observation(rho,step,cfg.max_steps)
                mask=compute_valid_action_mask(circuit,catalog,cfg.n_qubits,min_gates=cfg.n_qubits)
                probabilities=net.forward(obs,mask)[0];op=catalog[int(rng.choice(len(catalog),p=probabilities))]
                if op.name=='DONE':break
                circuit.append(op);rho=sim.execute_step(rho,op,nom,step)
            scores.append(sim.ghz_metrics_from_rho(rho,nom.pm_vec)[1])
        return scores
    import copy
    initial_net=copy.deepcopy(arch)
    for key,value in initial_arch.items():setattr(initial_net,key,np.asarray(value))
    probes={'initial':probe(initial_net),'final':probe(arch)}
    parameter_delta=float(np.linalg.norm(np.concatenate([(getattr(arch,k)-np.array(v)).ravel() for k,v in initial_arch.items()])))
    metrics=compute_circuit_metrics(best,cfg.n_qubits,topo);metrics['edge_counts']=metrics['edge_counts'].tolist()
    return dict(best_circuit=best,policy_probes=probes,parameter_delta_l2=parameter_delta,config=asdict(cfg),seed=seed,circuit_stats=metrics,history=history,
                train_time_sec=time.perf_counter()-start,sim_calls_train=sim.sim_call_count,
                arch_ppo_updates=arch.update_count,architect=state_dict(arch),saboteur=state_dict(sab.net),
                validation_cvar=float(best_score),candidate_count=len(candidates),
                rng_states={'action':act_rng.bit_generator.state,'noise':noise_rng.bit_generator.state,'saboteur':sab_rng.bit_generator.state})
