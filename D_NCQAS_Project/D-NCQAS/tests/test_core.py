import unittest,copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from d_ncqas.agents import MLPPolicyValueNet,TrajectoryConditionedSaboteurAgent,compute_cvar,minimal_ghz
from d_ncqas.simulator import DensityMatrixSimulator,GateOp,circuit_sha256,X_MAT,Y_MAT,Z_MAT,I2
from d_ncqas.noise_models import HardwareTopology,NoiseConfig,amplify_noise_config_spatial
from d_ncqas.evaluation import make_scenarios,evaluate,scenario_id,paired_comparison,holm_bonferroni
from d_ncqas.coevolution import TrainingConfig,train_qas_method,method_configs
from d_ncqas.vqe import MolecularHamiltonian,measured_energy,pauli_terms

class CoreTests(unittest.TestCase):
    def test_actor_positive_advantage(self):
        net=MLPPolicyValueNet(1,2,np.random.default_rng(0),hidden=2,lr=.01)
        for v in vars(net).values():
            if isinstance(v,np.ndarray):v[:]=0
        obs=[np.zeros(1)]*2;masks=[np.ones(2,bool)]*2
        net.update_ppo(obs,[0,1],masks,np.array([1.,-1.]),np.zeros(2),[.5,.5],ent_coef=0,epochs=1)
        self.assertGreater(net.forward(obs[0])[0][0],.5)

    def test_entropy_increases(self):
        net=MLPPolicyValueNet(1,2,np.random.default_rng(0),hidden=2,lr=.01)
        for v in vars(net).values():
            if isinstance(v,np.ndarray):v[:]=0
        net.b_pi[:]=[1.,-1.];obs=[np.zeros(1)]*2;masks=[np.ones(2,bool)]*2
        p=net.forward(obs[0])[0];h=lambda p:-np.dot(p,np.log(p))
        net.update_ppo(obs,[0,1],masks,np.zeros(2),np.zeros(2),p,epochs=1,ent_coef=.1)
        self.assertGreater(h(net.forward(obs[0])[0]),h(p))

    def test_full_backprop_finite_differences(self):
        net=MLPPolicyValueNet(2,3,np.random.default_rng(2),hidden=3,lr=1e-4)
        obs=[np.array([.2,.5]),np.array([-.2,.7])];mask=[np.array([True,True,False])]*2;acts=[0,1];adv=np.array([1.,-1.]);ret=np.array([.1,-.1]);old=[net.forward(o,m)[0][a] for o,m,a in zip(obs,mask,acts)]
        normalized=(adv-adv.mean())/(adv.std()+1e-8)
        def loss():
            total=0.
            for o,m,a,A,R,op in zip(obs,mask,acts,normalized,ret,old):
                p,v,_=net.forward(o,m);ratio=p[a]/op;entropy=-np.dot(p,np.log(np.maximum(p,1e-8)))
                total+=-min(ratio*A,np.clip(ratio,.8,1.2)*A)-.02*entropy+.25*(v-R)**2
            return total/2
        grads={}
        for name in ['W1','b1','W2','b2','W_pi','b_pi','W_v','b_v']:
            param=getattr(net,name);g=np.zeros_like(param)
            for idx in np.ndindex(param.shape):
                original=param[idx];eps=1e-6;param[idx]=original+eps;up=loss();param[idx]=original-eps;down=loss();param[idx]=original;g[idx]=(up-down)/(2*eps)
            grads[name]=g
        original=copy.deepcopy(net)
        net.update_ppo(obs,acts,mask,adv,ret,old,epochs=1)
        for name,g in grads.items():np.testing.assert_allclose((getattr(original,name)-getattr(net,name))/net.lr,g,atol=1e-7,rtol=1e-5)

    def test_budget_all_templates(self):
        for n in [3,4,5]:
            topo=HardwareTopology.synthetic_tree(n);sab=TrajectoryConditionedSaboteurAgent(topo,np.random.default_rng(0))
            for budget in [0,2,4,6,8]:
                for i in range(len(sab.templates)):
                    s1,s2,sm,used=sab.realize_action(i,np.ones(len(topo.edges)),budget)
                    actual=s1-1+2*np.mean(s2-1)+sm-1
                    self.assertLessEqual(actual,budget+1e-10);self.assertAlmostEqual(actual,used)

    def test_topology_rejects_illegal_gate(self):
        t=HardwareTopology.synthetic_tree(5);sim=DensityMatrixSimulator(5,t);c=NoiseConfig(5,t.edges,*t.nominal_rates())
        with self.assertRaises(ValueError):sim.execute([GateOp('CNOT',(2,3))],c)
        for g in [GateOp('BAD',(0,)),GateOp('CNOT',(1,1)),GateOp('H',(-1,))]:
            with self.assertRaises(ValueError):sim.execute([g],c)

    def test_ghz_exact_all_graphs(self):
        for kind in ['tree','chain']:
            for n in [2,3,4,5]:
                t=HardwareTopology.synthetic_tree(n,kind=kind);sim=DensityMatrixSimulator(n,t);z=NoiseConfig(n,t.edges,np.zeros(n),np.zeros(len(t.edges)),np.zeros(n),idle_decoherence_rate=0)
                self.assertTrue(np.allclose(sim.ghz_metrics(minimal_ghz(t),z),(1,1)))

    def test_channels_and_step_path(self):
        t=HardwareTopology.synthetic_tree(3);sim=DensityMatrixSimulator(3,t);circ=minimal_ghz(t)+[GateOp('RY',(2,),.31)]
        for family in ['depolarizing','amplitude_damping','phase_damping','asymmetric_pauli','readout','overrotation']:
            c=NoiseConfig(3,t.edges,*t.nominal_rates(),noise_family=family,overrotation_angle=.1)
            rho=sim.execute(circ,c);step=sim.initial_state_tensor()
            for i,op in enumerate(circ):step=sim.execute_step(step,op,c,i)
            np.testing.assert_allclose(rho,step,atol=1e-12);np.testing.assert_allclose(rho,rho.conj().T,atol=1e-12)
            self.assertAlmostEqual(np.trace(rho).real,1);self.assertGreaterEqual(np.linalg.eigvalsh(rho).min(),-1e-12)

    def test_two_qubit_pauli_reference(self):
        t=HardwareTopology.synthetic_tree(2);sim=DensityMatrixSimulator(2,t);v=np.array([1,2j,3,4j]);v=v/np.linalg.norm(v);rho=np.outer(v,v.conj());p=.21
        paulis=[I2,X_MAT,Y_MAT,Z_MAT];ref=(1-p)*rho
        for i,a in enumerate(paulis):
            for j,b in enumerate(paulis):
                if i==j==0:continue
                u=np.kron(a,b);ref+=p/15*u@rho@u.conj().T
        np.testing.assert_allclose(sim._apply_2q_depol(rho,0,1,p),ref,atol=1e-12)

    def test_cvar_fractional_mass(self):
        self.assertAlmostEqual(compute_cvar([1,2,3],.5),4/3)
        with self.assertRaises(ValueError):compute_cvar([],0.2)

    def test_scenarios_matched(self):
        t=HardwareTopology.synthetic_tree(3);bank=make_scenarios(t,1,5)
        a,_=evaluate(minimal_ghz(t),t,bank);b,_=evaluate(minimal_ghz(t)+[GateOp('X',(0,)),GateOp('X',(0,))],t,bank)
        self.assertEqual([r['scenario_id'] for r in a],[r['scenario_id'] for r in b])
        self.assertNotEqual(scenario_id(bank['ou'][0]),scenario_id(make_scenarios(t,1,5,'validation')['ou'][0]))

    def test_hash_full_precision(self):
        self.assertNotEqual(circuit_sha256([GateOp('RY',(0,),.100001)]),circuit_sha256([GateOp('RY',(0,),.100002)]))
        self.assertEqual(len(circuit_sha256([])),64)

    def test_paired_statistics(self):
        r=paired_comparison(np.ones(10),np.zeros(10));self.assertEqual(r['p_paired_t'],0);self.assertAlmostEqual(r['p_signflip'],2/1024)
        self.assertEqual(paired_comparison(np.ones(10),np.ones(10))['p_signflip'],1)
        np.testing.assert_allclose(holm_bonferroni([.01,.03,.04]),[.03,.06,.06])

    def test_hamiltonian_and_readout(self):
        t=HardwareTopology.synthetic_tree(2);sim=DensityMatrixSimulator(2,t)
        for p in (ROOT/'data/hamiltonians').glob('*.json'):
            ham=MolecularHamiltonian.load(p);w,v=np.linalg.eigh(ham.matrix);rho=np.outer(v[:,0],v[:,0].conj())
            self.assertAlmostEqual(measured_energy(rho,ham,sim,np.zeros(2)),ham.e_fci)
        # For Z readout on |00>, each Z expectation is 1-2*0.85*p_m.
        rho=sim.initial_state_tensor();ham=MolecularHamiltonian('test',np.kron(Z_MAT,I2),-1,{})
        self.assertAlmostEqual(measured_energy(rho,ham,sim,np.array([.1,.2])),.83)

    def test_independent_fermionic_hamiltonian_reconstruction(self):
        # Independent second-quantized construction from the saved chemist integrals.
        # Mode order alpha0, alpha1, beta0, beta1; q0 is the most significant bit.
        annihilators=[]
        for mode in range(4):
            a=np.zeros((16,16))
            for state in range(16):
                bits=[(state>>(3-q))&1 for q in range(4)]
                if bits[mode]:a[state^(1<<(3-mode)),state]=(-1)**sum(bits[:mode])
            annihilators.append(a)
        sector=[(1<<(3-a))+(1<<(1-b)) for a in range(2) for b in range(2)]
        for path in (ROOT/'data/hamiltonians').glob('*.json'):
            data=json.loads(path.read_text());h=np.asarray(data['h1']);eri=np.asarray(data['eri']);matrix=data['energy_core']*np.eye(16)
            for p in range(2):
                for q in range(2):
                    for spin in range(2):
                        ap=annihilators[p+2*spin];aq=annihilators[q+2*spin]
                        matrix+=h[p,q]*ap.T@aq
                    for r in range(2):
                        for ss in range(2):
                            for sigma in range(2):
                                for tau in range(2):
                                    ap=annihilators[p+2*sigma];aq=annihilators[q+2*sigma]
                                    ar=annihilators[r+2*tau];ass=annihilators[ss+2*tau]
                                    matrix+=.5*eri[p,q,r,ss]*ap.T@ar.T@ass@aq
            np.testing.assert_allclose(matrix[np.ix_(sector,sector)],data['matrix'],atol=1e-10)

    def test_training_reproducible_and_controls(self):
        cfg=TrainingConfig(n_generations=1,episodes_per_gen=3,n_traj_samples=3)
        a=train_qas_method(cfg,0);b=train_qas_method(cfg,0)
        self.assertEqual(a['best_circuit'],b['best_circuit']);self.assertGreater(a['arch_ppo_updates'],0)
        methods=method_configs(3,'tree');full=vars(methods[0])
        for m in methods[1:5]:self.assertEqual(sum(v!=full[k] for k,v in vars(m).items()),2)

if __name__=='__main__':unittest.main(verbosity=2)
