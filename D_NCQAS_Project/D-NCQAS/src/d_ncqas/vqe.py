"""Traceable CAS(2,2) VQE validation, separate from GHZ architecture-search claims."""
from dataclasses import dataclass
import json,itertools
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from .simulator import GateOp,DensityMatrixSimulator,I2,X_MAT,Y_MAT,Z_MAT,H_MAT,S_MAT

@dataclass
class MolecularHamiltonian:
    name: str
    matrix: np.ndarray
    e_fci: float
    metadata: dict

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text());mat=np.asarray(d['matrix'],complex)
        if not np.allclose(mat,mat.conj().T) or abs(np.linalg.eigvalsh(mat)[0]-d['e_fci_active'])>1e-9:
            raise ValueError('Hamiltonian reference validation failed')
        return cls(d['name'],mat,d['e_fci_active'],d)


def ansatz(angles,layers):
    angles=iter(angles);ops=[]
    for layer in range(layers):
        ops += [GateOp('RY',(q,),float(next(angles))) for q in range(2)]
        ops += [GateOp('CNOT',(0,1))]
    ops += [GateOp('RY',(q,),float(next(angles))) for q in range(2)]
    return ops


def pauli_terms(matrix):
    out=[];paulis=[I2,X_MAT,Y_MAT,Z_MAT]
    for ids in itertools.product(range(4),repeat=2):
        op=np.kron(paulis[ids[0]],paulis[ids[1]])
        c=float(np.trace(op@matrix).real/4)
        if abs(c)>1e-12:out.append((c,ids))
    return out


def measured_energy(rho,ham,sim,pm):
    """Pauli-wise basis rotations followed by the SAME asymmetric assignment channel.

    Readout basis rotations assumed ideal; infinite-shot expectations. Not mitigation.
    """
    total=0.
    for c,ids in pauli_terms(ham.matrix):
        rots=[I2 if p in (0,3) else H_MAT if p==1 else H_MAT@S_MAT.conj().T for p in ids]
        u=np.kron(*rots);rot=u@rho@u.conj().T
        probs=sim.apply_readout_noise_probs(rot.diagonal().real,pm)
        eigen=np.ones(4)
        for q,p in enumerate(ids):
            if p:eigen*=sim.sign_z[q]
        total+=c*np.dot(probs,eigen)
    return float(total)


def optimize(ham,topology,noise,layers,seed,restarts=5,maxiter=400):
    sim=DensityMatrixSimulator(2,topology);rng=np.random.default_rng([seed,33,layers])
    def energy(x):return float(np.trace(ham.matrix@sim.execute(ansatz(x,layers),noise)).real)
    runs=[];best=None
    for restart in range(restarts):
        x0=np.zeros(2*(layers+1)) if restart==0 else rng.uniform(-np.pi,np.pi,2*(layers+1))
        res=minimize(energy,x0,method='COBYLA',options={'maxiter':maxiter,'tol':1e-6,'rhobeg':0.5})
        runs.append(dict(restart=restart,energy=float(res.fun),success=bool(res.success),status=int(res.status),message=str(res.message),nfev=int(res.nfev),angles=res.x.tolist()))
        if best is None or res.fun<best.fun:best=res
    return ansatz(best.x,layers),dict(angles=best.x.tolist(),runs=runs,energy=float(best.fun),sim_calls=sim.sim_call_count,converged=bool(best.success))
