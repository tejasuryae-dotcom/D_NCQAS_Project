"""Shared scenario bank, premeasurement fidelity endpoint and seed-level inference."""
import hashlib,json
import numpy as np
from scipy import stats
from .noise_models import HardwareTopology,NoiseConfig,OrnsteinUhlenbeckDriftProcess,amplify_noise_config_spatial
from .simulator import DensityMatrixSimulator
from .agents import compute_cvar

def noise_dict(c):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in vars(c).items()}

def scenario_id(c):
    return hashlib.sha256(json.dumps(noise_dict(c),sort_keys=True).encode()).hexdigest()

def make_scenarios(topology,seed,draws=100,split='test'):
    if split not in ('test','validation'):
        raise ValueError('Unknown split')
    rngs=[np.random.default_rng(s) for s in np.random.SeedSequence([900001 if split=='test' else 800001,seed,topology.n_qubits]).spawn(3)]
    p1,p2,pm=topology.nominal_rates();out={'nominal':[],'static':[],'ou':[]}
    ou=OrnsteinUhlenbeckDriftProcess(topology,kappa_ou=.30,sigma_ou=.42,drift_aging_shift=.35,tls_burst_prob=.12,dt=.30)
    for i in range(draws):
        jitter=rngs[0].uniform(.95,1.05,3)
        nom=NoiseConfig(topology.n_qubits,topology.edges,p1*jitter[0],p2*jitter[1],pm*jitter[2])
        out['nominal'].append(nom)
        out['static'].append(amplify_noise_config_spatial(NoiseConfig(topology.n_qubits,topology.edges,p1,p2,pm),rngs[1].uniform(1,2),rngs[1].uniform(1,4,len(p2)),rngs[1].uniform(1,2)))
        base=ou.sample_trajectory(rngs[2],n_epochs=6,permute_spatial_bottlenecks=i%4==0)[-1]
        out['ou'].append(amplify_noise_config_spatial(base,rngs[2].uniform(1,2),rngs[2].uniform(1,4,len(p2)),rngs[2].uniform(1,2)))
    return out

def evaluate(circuit,topology,bank):
    sim=DensityMatrixSimulator(topology.n_qubits,topology);rows=[]
    for regime,configs in bank.items():
        for i,c in enumerate(configs):
            p,f=sim.ghz_metrics(circuit,c)
            rows.append(dict(regime=regime,scenario=i,scenario_id=scenario_id(c),p_ghz=p,fidelity=f))
    return rows,sim.sim_call_count

def holm_bonferroni(p_values):
    p=np.asarray(p_values,dtype=float)
    if not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('Invalid p values')
    order=np.argsort(p);out=np.zeros(len(p));running=0
    for rank,i in enumerate(order):
        running=max(running,(len(p)-rank)*p[i]);out[i]=min(1,running)
    return out.tolist()

def paired_comparison(x,y,seed=123,n_boot=10000):
    """Seed is the unit. CI targets the same per-seed CVaR difference as the test."""
    x,y=np.asarray(x),np.asarray(y);d=x-y;n=len(d)
    if n<2:raise ValueError('Need at least two paired seeds')
    if np.all(abs(d)<1e-14):t,p=0.,1.
    elif np.std(d,ddof=1)<1e-14:t,p=float(np.sign(d.mean())*1e15),0.
    else:
        res=stats.ttest_rel(x,y);t,p=float(res.statistic),float(res.pvalue)
    rng=np.random.default_rng(seed);boot=d[rng.integers(n,size=(n_boot,n))].mean(axis=1)
    # Exact sign-flip test for <=20 paired seeds; prespecified primary test.
    if n<=20:
        flips=1-2*((np.arange(2**n)[:,None]>>np.arange(n))&1)
        p_sign=float(np.mean(abs(flips@d/n)>=abs(d.mean())-1e-14))
    else:
        flips=rng.choice([-1,1],size=(100000,n));p_sign=float((1+np.sum(abs(flips@d/n)>=abs(d.mean())-1e-14))/100001)
    return dict(n_seeds=n,delta=float(d.mean()),ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),p_signflip=p_sign,p_paired_t=p,t_paired=t)
