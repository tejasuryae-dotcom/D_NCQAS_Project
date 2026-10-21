#!/usr/bin/env python3
"""Generate compact fixed-(N_alpha,N_beta) CAS(2,2) Hamiltonians with PySCF.

Basis |a,b> has alpha electron in active spatial orbital a and beta in b.
This is an explicit compact CI encoding, NOT a four-qubit Jordan-Wigner model.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys
from pathlib import Path
import numpy as np
import pyscf
from pyscf import gto,scf,mcscf,fci,ao2mo
# PySCF Linux memory probe assumes /proc; use resource in restricted containers.
if not Path(f"/proc/{os.getpid()}/statm").exists():
    import resource
    pyscf.lib.current_memory=lambda: (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 0)
ROOT=Path(__file__).resolve().parents[1]

def build(name,atom,charge):
    mol=gto.M(atom=atom,basis='sto-3g',unit='Angstrom',charge=charge,spin=0,verbose=0)
    mf=scf.RHF(mol);mf.conv_tol=1e-12;mf.kernel()
    if not mf.converged:raise RuntimeError('SCF did not converge')
    cas=mcscf.CASCI(mf,2,2)
    h1,ecore=cas.get_h1eff();eri=cas.get_h2eff()
    h2eff=fci.direct_spin1.absorb_h1e(h1,eri,2,(1,1),.5)
    columns=[]
    for i in range(4):
        ci=np.zeros((2,2));ci.flat[i]=1
        columns.append(fci.direct_spin1.contract_2e(h2eff,ci,2,(1,1)).reshape(-1))
    matrix=np.column_stack(columns)+ecore*np.eye(4)
    ref,ci=fci.direct_spin1.kernel(h1,eri,2,(1,1),ecore=ecore,tol=1e-12)
    assert np.allclose(matrix,matrix.T,atol=1e-12)
    assert abs(np.linalg.eigvalsh(matrix)[0]-ref)<1e-9
    return dict(name=name,geometry=atom,unit='Angstrom',charge=charge,spin=0,basis='sto-3g',
                active_electrons=[1,1],active_spatial_orbitals=2,ncore=cas.ncore,
                active_orbital_indices=list(range(cas.ncore,cas.ncore+2)),
                orbital_basis='canonical restricted Hartree-Fock; no orbital optimization',
                encoding='compact alpha-orbital/beta-orbital CI basis |a,b>, a,b in {0,1}',
                n_qubits=2,energy_unit='Hartree',energy_core=float(ecore),
                h1=h1.tolist(),eri=ao2mo.restore(1,eri,2).tolist(),mo_coeff=mf.mo_coeff.tolist(),
                matrix=matrix.tolist(),e_fci_active=float(ref),e_hf=float(mf.e_tot),
                reference_eigensolver_error=float(abs(np.linalg.eigvalsh(matrix)[0]-ref)),
                generator='PySCF CASCI integrals + direct_spin1 FCI',pyscf_version=pyscf.__version__)

def main():
    dest=ROOT/'data/hamiltonians';dest.mkdir(parents=True,exist_ok=True)
    for name,atom,charge in [('H2','H 0 0 0; H 0 0 0.735',0),('HeH+','He 0 0 0; H 0 0 0.775',1),('LiH','Li 0 0 0; H 0 0 1.6',0)]:
        d=build(name,atom,charge);(dest/(name.replace('+','plus')+'.json')).write_text(json.dumps(d,indent=2)+'\n')
        print(name,d['e_fci_active'],d['reference_eigensolver_error'])
if __name__=='__main__':main()
