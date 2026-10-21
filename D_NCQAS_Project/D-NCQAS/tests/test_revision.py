import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from d_ncqas.agents import MLPPolicyValueNet
from d_ncqas.coevolution import TrainingConfig,train_qas_method
from d_ncqas.search_baselines import search
from d_ncqas.noise_models import HardwareTopology,NoiseConfig
from d_ncqas.simulator import DensityMatrixSimulator
from d_ncqas.vqe import MolecularHamiltonian


class RevisionTests(unittest.TestCase):
    def test_adam_first_step_bias_correction(self):
        net=MLPPolicyValueNet(1,2,np.random.default_rng(2),hidden=2,lr=.001,optimizer='adam')
        for value in vars(net).values():
            if isinstance(value,np.ndarray):value[:]=0
        net.update_ppo([np.zeros(1)]*2,[0,1],[np.ones(2,bool)]*2,np.array([1.,-1.]),np.zeros(2),[.5,.5],epochs=1,ent_coef=0)
        np.testing.assert_allclose(net.b_pi,[.001,-.001],atol=1e-10)

    def test_clipped_surrogate_has_zero_actor_gradient(self):
        net=MLPPolicyValueNet(1,2,np.random.default_rng(0),hidden=2)
        for value in vars(net).values():
            if isinstance(value,np.ndarray):value[:]=0
        before=net.b_pi.copy()
        net.update_ppo([np.zeros(1)]*2,[0,1],[np.ones(2,bool)]*2,np.array([1.,-1.]),np.zeros(2),[.25,.75],epochs=1,ent_coef=0)
        np.testing.assert_array_equal(before,net.b_pi)

    def test_diagnostics_and_frozen_policy(self):
        cfg=TrainingConfig(n_generations=2,episodes_per_gen=3,n_traj_samples=3,optimizer='adam',learning_rate=.003)
        learned=train_qas_method(cfg,19);cfg.train_policy=False;frozen=train_qas_method(cfg,19)
        self.assertGreater(learned['parameter_delta_l2'],0)
        self.assertEqual(frozen['parameter_delta_l2'],0)
        self.assertEqual(frozen['policy_probes']['initial'],frozen['policy_probes']['final'])
        for row in learned['history']:self.assertLessEqual(row['rollout_probability_error'],1e-12)

    def test_baseline_budget_and_legality(self):
        cfg=TrainingConfig(n_generations=2,episodes_per_gen=4,n_traj_samples=3,max_steps=8)
        sim=DensityMatrixSimulator(3,HardwareTopology.synthetic_tree(3))
        for algorithm in ['CEM','Age-Evolution']:
            a=search(cfg,9,algorithm);b=search(cfg,9,algorithm)
            self.assertEqual(a['proposals'],8);self.assertEqual(a['best_circuit'],b['best_circuit'])
            self.assertLessEqual(a['candidate_count'],8)
            for g in a['best_circuit']:sim.validate_gate(g)

    def test_parameter_shift_gradient(self):
        spec=importlib.util.spec_from_file_location('vqe_convergence',ROOT/'scripts/vqe_convergence.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        topo=HardwareTopology.synthetic_tree(2);noise=NoiseConfig(2,topo.edges,*topo.nominal_rates())
        for p in (ROOT/'data/hamiltonians').glob('*.json'):
            ham=MolecularHamiltonian.load(p);energy,jac,_=mod.objective(ham,topo,noise,2)
            x=np.linspace(-.6,.4,6);reference=[]
            for i in range(6):
                direction=np.zeros(6);direction[i]=1e-5
                reference.append((energy(x+direction)-energy(x-direction))/2e-5)
            np.testing.assert_allclose(jac(x),reference,atol=1e-9,rtol=1e-7)


if __name__=='__main__':unittest.main()
