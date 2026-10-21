"""
Fast Exact Density-Matrix Simulator with Precomputed Bitwise Channel Masks, Cached Unitaries,
Exact Simulation Call Telemetry, and Cryptographic Circuit Hashing.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from .noise_models import HardwareTopology, NoiseConfig


@dataclass(frozen=True)
class GateOp:
    name: str
    qubits: Tuple[int, ...]
    theta: float = 0.0

    def to_canonical_str(self) -> str:
        q_str = ",".join(str(q) for q in self.qubits)
        if self.name in ("RX", "RY", "RZ"):
            return f"{self.name}({q_str};theta={self.theta:.17g})"
        return f"{self.name}({q_str})"


def circuit_to_string(circuit: Sequence[GateOp]) -> str:
    if len(circuit) == 0:
        return "EMPTY"
    return " -> ".join(op.to_canonical_str() for op in circuit)


def circuit_sha256(circuit: Sequence[GateOp]) -> str:
    s = json.dumps([(op.name, op.qubits, float(op.theta).hex()) for op in circuit], separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


I2 = np.eye(2, dtype=np.complex128)
X_MAT = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y_MAT = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z_MAT = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H_MAT = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
S_MAT = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T_MAT = np.array([[1, 0], [0, np.exp(1j * np.pi / 4.0)]], dtype=np.complex128)


def rx_mat(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2.0), -1j * np.sin(theta / 2.0)
    return np.array([[c, s], [s, c]], dtype=np.complex128)


def ry_mat(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_mat(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=np.complex128)


def get_1q_unitary(op: GateOp, overrot: float = 0.0) -> np.ndarray:
    ang = op.theta + overrot
    if op.name == "H":
        u = H_MAT
    elif op.name == "X":
        u = X_MAT
    elif op.name == "Y":
        u = Y_MAT
    elif op.name == "Z":
        u = Z_MAT
    elif op.name == "S":
        u = S_MAT
    elif op.name == "T":
        u = T_MAT
    elif op.name == "RX":
        u = rx_mat(ang)
    elif op.name == "RY":
        u = ry_mat(ang)
    elif op.name == "RZ":
        u = rz_mat(ang)
    else:
        raise ValueError(f"Unknown 1Q gate: {op.name}")
    if overrot != 0.0 and op.name in ("H", "X", "Y", "Z", "S", "T"):
        u = rx_mat(overrot) @ u
    return u


def compute_circuit_metrics(circuit: Sequence[GateOp], n_qubits: int, topology: HardwareTopology) -> Dict[str, object]:
    n_1q = 0
    n_2q = 0
    qubit_layer = np.zeros(n_qubits, dtype=int)
    edge_counts = np.zeros(len(topology.edges), dtype=int)

    for op in circuit:
        if len(op.qubits) == 1:
            n_1q += 1
            q = op.qubits[0]
            qubit_layer[q] += 1
        elif len(op.qubits) == 2:
            n_2q += 1
            q0, q1 = op.qubits
            new_d = max(qubit_layer[q0], qubit_layer[q1]) + 1
            qubit_layer[q0] = new_d
            qubit_layer[q1] = new_d
            e_idx = topology.edge_index(q0, q1)
            edge_counts[e_idx] += 1

    depth = int(np.max(qubit_layer)) if len(circuit) > 0 else 0
    max_edge_share = float(np.max(edge_counts) / max(1, n_2q)) if len(edge_counts) > 0 else 0.0
    return {
        "total_gates": len(circuit),
        "n_1q": n_1q,
        "n_2q": n_2q,
        "depth": depth,
        "max_coupler_share": max_edge_share,
        "edge_counts": edge_counts,
        "circuit_hash": circuit_sha256(circuit),
        "circuit_str": circuit_to_string(circuit),
    }


class DensityMatrixSimulator:
    _CLASS_CACHE: Dict[int, Dict[str, object]] = {}

    def __init__(self, n_qubits: int, topology: HardwareTopology):
        self.n_qubits = n_qubits
        self.topology = topology
        self.dim = 2 ** n_qubits
        self.sim_call_count: int = 0
        if n_qubits not in self._CLASS_CACHE:
            self._CLASS_CACHE[n_qubits] = self._precompute_tables(n_qubits)
        tables = self._CLASS_CACHE[n_qubits]
        self.idx0_list: List[np.ndarray] = tables["idx0"]
        self.idx1_list: List[np.ndarray] = tables["idx1"]
        self.flip_1q: List[np.ndarray] = tables["flip_1q"]
        self.sign_z: List[np.ndarray] = tables["sign_z"]
        self.offdiag_mask_1q: List[np.ndarray] = tables["offdiag_mask_1q"]
        self.pair_flip_map: Dict[Tuple[int, int], List[np.ndarray]] = tables["pair_flip_map"]
        self.cnot_mats: Dict[Tuple[int, int], np.ndarray] = tables["cnot_mats"]
        self.unitary_cache: Dict[Tuple[str, Tuple[int, ...], int], np.ndarray] = {}

    @staticmethod
    def _precompute_tables(n: int) -> Dict[str, object]:
        dim = 2 ** n
        states = np.arange(dim)
        idx0_list, idx1_list, flip_1q, sign_z, offdiag_mask_1q = [], [], [], [], []
        for q in range(n):
            shift = n - 1 - q
            bit = (states >> shift) & 1
            i0 = states[bit == 0]
            i1 = i0 | (1 << shift)
            idx0_list.append(i0)
            idx1_list.append(i1)
            flip_1q.append(states ^ (1 << shift))
            sign_z.append(np.where(bit == 0, 1.0, -1.0))
            offdiag_mask_1q.append(bit[:, None] != bit[None, :])

        cnot_mats, pair_flip_map = {}, {}
        for q0 in range(n):
            for q1 in range(n):
                if q0 == q1:
                    continue
                s0 = n - 1 - q0
                s1 = n - 1 - q1
                perm = states ^ (((states >> s0) & 1) << s1)
                mat = np.zeros((dim, dim), dtype=np.complex128)
                mat[perm, states] = 1.0
                cnot_mats[(q0, q1)] = mat
                pair_flip_map[(q0, q1)] = [
                    states,
                    states ^ (1 << s1),
                    states ^ (1 << s0),
                    states ^ (1 << s0) ^ (1 << s1),
                ]
        return {
            "idx0": idx0_list,
            "idx1": idx1_list,
            "flip_1q": flip_1q,
            "sign_z": sign_z,
            "offdiag_mask_1q": offdiag_mask_1q,
            "pair_flip_map": pair_flip_map,
            "cnot_mats": cnot_mats,
        }

    def initial_state_tensor(self) -> np.ndarray:
        rho = np.zeros((self.dim, self.dim), dtype=np.complex128)
        rho[0, 0] = 1.0
        return rho

    def _get_full_1q_unitary(self, op: GateOp, overrot: float = 0.0) -> np.ndarray:
        ang_key = (float(op.theta).hex(), float(overrot).hex())
        key = (op.name, op.qubits, ang_key)
        if key not in self.unitary_cache:
            u2 = get_1q_unitary(op, overrot=overrot)
            q = op.qubits[0]
            mats = [I2 if i != q else u2 for i in range(self.n_qubits)]
            out = mats[0]
            for m in mats[1:]:
                out = np.kron(out, m)
            self.unitary_cache[key] = out
        return self.unitary_cache[key]

    def _apply_1q_pauli_channel(self, rho: np.ndarray, q: int, px: float, py: float, pz: float) -> np.ndarray:
        p_tot = px + py + pz
        if p_tot <= 0.0:
            return rho
        fl = self.flip_1q[q]
        sz = self.sign_z[q]
        z_mask = np.outer(sz, sz)
        rho_x = rho[np.ix_(fl, fl)]
        rho_z = rho * z_mask
        rho_y = rho_x * z_mask
        return (1.0 - p_tot) * rho + px * rho_x + py * rho_y + pz * rho_z

    def _apply_1q_depol(self, rho: np.ndarray, q: int, p: float) -> np.ndarray:
        if p <= 0.0:
            return rho
        p = min(p, 0.75)
        w = p / 3.0
        return self._apply_1q_pauli_channel(rho, q, w, w, w)

    def _apply_1q_phase_damp(self, rho: np.ndarray, q: int, lam: float) -> np.ndarray:
        if lam <= 0.0:
            return rho
        lam = min(lam, 0.99)
        out = rho.copy()
        out[self.offdiag_mask_1q[q]] *= np.sqrt(1.0 - lam)
        return out

    def _apply_1q_amp_damp(self, rho: np.ndarray, q: int, gamma: float) -> np.ndarray:
        if gamma <= 0.0:
            return rho
        gamma = min(gamma, 0.99)
        i0, i1 = self.idx0_list[q], self.idx1_list[q]
        sq = np.sqrt(1.0 - gamma)
        out = rho.copy()
        out[i1, :] *= sq
        out[:, i1] *= sq
        out[np.ix_(i0, i0)] += gamma * rho[np.ix_(i1, i1)]
        return out

    def _apply_2q_depol(self, rho: np.ndarray, q0: int, q1: int, p: float) -> np.ndarray:
        if p <= 0.0:
            return rho
        p_eff = min(p * (16.0 / 15.0), 1.0)
        m0 = ~self.offdiag_mask_1q[q0]
        m1 = ~self.offdiag_mask_1q[q1]
        rho_diag = rho * (m0 & m1)
        flips = self.pair_flip_map[(q0, q1)]
        tr_avg = 0.25 * (
            rho_diag[np.ix_(flips[0], flips[0])]
            + rho_diag[np.ix_(flips[1], flips[1])]
            + rho_diag[np.ix_(flips[2], flips[2])]
            + rho_diag[np.ix_(flips[3], flips[3])]
        )
        return (1.0 - p_eff) * rho + p_eff * tr_avg

    def execute_step(self, rho: np.ndarray, op: GateOp, noise_cfg: NoiseConfig, g_idx: int = 0) -> np.ndarray:
        """Applies a single gate operation and its corresponding hardware noise channel to rho."""
        self.validate_gate(op)
        if noise_cfg.n_qubits != self.n_qubits or noise_cfg.edges != self.topology.edges:
            raise ValueError("Noise configuration and simulator topology differ")
        fam = noise_cfg.noise_family
        local_p = None
        if noise_cfg.gate_local_depol is not None and g_idx < len(noise_cfg.gate_local_depol):
            local_p = float(noise_cfg.gate_local_depol[g_idx])

        if len(op.qubits) == 1:
            q = op.qubits[0]
            overrot = noise_cfg.overrotation_angle if fam == "overrotation" else 0.0
            u_full = self._get_full_1q_unitary(op, overrot=overrot)
            rho = u_full @ rho @ u_full.conj().T
            p1 = local_p if local_p is not None else float(noise_cfg.p1q_vec[q])

            if fam in ("depolarizing", "readout", "overrotation"):
                rho = self._apply_1q_depol(rho, q, p1)
            elif fam == "asymmetric_pauli":
                pa = min(p1 * 1.5, 0.75)
                rho = self._apply_1q_pauli_channel(rho, q, 0.2 * pa, 0.2 * pa, 0.6 * pa)
            elif fam == "amplitude_damping":
                gamma = noise_cfg.amp_damping_rate if noise_cfg.amp_damping_rate > 0 else p1 * 2.0
                rho = self._apply_1q_amp_damp(rho, q, gamma)
            elif fam == "phase_damping":
                lam = noise_cfg.phase_damping_rate if noise_cfg.phase_damping_rate > 0 else p1 * 2.5
                rho = self._apply_1q_phase_damp(rho, q, lam)

        elif len(op.qubits) == 2:
            q0, q1 = op.qubits
            e_idx = self.topology.edge_index(q0, q1)
            if fam == "overrotation" and noise_cfg.overrotation_angle != 0.0:
                u_rx = self._get_full_1q_unitary(GateOp("RX", (q0,), 0.0), overrot=noise_cfg.overrotation_angle)
                u_rz = self._get_full_1q_unitary(GateOp("RZ", (q1,), 0.0), overrot=noise_cfg.overrotation_angle)
                rho = u_rx @ rho @ u_rx.conj().T
                rho = u_rz @ rho @ u_rz.conj().T

            u_cnot = self.cnot_mats[(q0, q1)]
            rho = u_cnot @ rho @ u_cnot.T
            p2 = local_p if local_p is not None else float(noise_cfg.p2q_vec[e_idx])

            if fam in ("depolarizing", "readout", "overrotation"):
                rho = self._apply_2q_depol(rho, q0, q1, p2)
            elif fam == "asymmetric_pauli":
                rho = self._apply_2q_depol(rho, q0, q1, p2 * 0.6)
                pa = min(p2 * 0.5, 0.75)
                rho = self._apply_1q_pauli_channel(rho, q0, 0.2 * pa, 0.2 * pa, 0.6 * pa)
                rho = self._apply_1q_pauli_channel(rho, q1, 0.2 * pa, 0.2 * pa, 0.6 * pa)
            elif fam == "amplitude_damping":
                gamma = noise_cfg.amp_damping_rate if noise_cfg.amp_damping_rate > 0 else p2 * 1.5
                rho = self._apply_2q_depol(rho, q0, q1, p2 * 0.5)
                rho = self._apply_1q_amp_damp(rho, q0, gamma)
                rho = self._apply_1q_amp_damp(rho, q1, gamma)
            elif fam == "phase_damping":
                lam = noise_cfg.phase_damping_rate if noise_cfg.phase_damping_rate > 0 else p2 * 1.8
                rho = self._apply_2q_depol(rho, q0, q1, p2 * 0.5)
                rho = self._apply_1q_phase_damp(rho, q0, lam)
                rho = self._apply_1q_phase_damp(rho, q1, lam)

        # Serial-gate clock: every gate occupies one time slice on all qubits.
        # This same rule is applied in rollouts and complete circuit execution.
        for qi in range(self.n_qubits):
            rho = self._apply_1q_phase_damp(rho, qi, noise_cfg.idle_decoherence_rate)
        return rho

    def validate_gate(self, op: GateOp) -> None:
        if len(set(op.qubits)) != len(op.qubits) or any(q < 0 or q >= self.n_qubits for q in op.qubits):
            raise ValueError("Invalid qubit indices")
        if not np.isfinite(op.theta):
            raise ValueError("Nonfinite rotation angle")
        if op.name == "CNOT" and len(op.qubits) == 2:
            self.topology.edge_index(*op.qubits)
        elif op.name in {"H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ"} and len(op.qubits) == 1:
            pass
        else:
            raise ValueError(f"Invalid gate: {op}")

    def execute(self, circuit: Sequence[GateOp], noise_cfg: NoiseConfig) -> np.ndarray:
        self.sim_call_count += 1
        rho = self.initial_state_tensor()
        for g_idx, op in enumerate(circuit):
            rho = self.execute_step(rho, op, noise_cfg, g_idx)
        return rho

    def apply_readout_noise_probs(self, ideal_probs: np.ndarray, pm_vec: np.ndarray) -> np.ndarray:
        probs_tensor = ideal_probs.reshape((2,) * self.n_qubits)
        for q in range(self.n_qubits):
            pm = float(np.clip(pm_vec[q], 0.0, 0.48))
            p01, p10 = 0.85 * pm, 1.15 * pm
            mat = np.array([[1.0 - p01, p10], [p01, 1.0 - p10]], dtype=float)
            probs_tensor = np.moveaxis(np.tensordot(mat, probs_tensor, axes=([1], [q])), 0, q)
        return probs_tensor.reshape(-1)

    def ghz_metrics_from_rho(self, rho: np.ndarray, pm_vec: np.ndarray) -> Tuple[float, float]:
        """Return measured Z-basis success and premeasurement state fidelity separately."""
        noisy_probs = self.apply_readout_noise_probs(np.real(np.diag(rho)), pm_vec)
        p = float(noisy_probs[0] + noisy_probs[-1])
        f = 0.5 * float(np.real(rho[0, 0] + rho[-1, -1] + rho[0, -1] + rho[-1, 0]))
        return p, f

    def ghz_metrics(self, circuit: Sequence[GateOp], noise_cfg: NoiseConfig) -> Tuple[float, float]:
        return self.ghz_metrics_from_rho(self.execute(circuit, noise_cfg), noise_cfg.pm_vec)

    def entanglement_potential(self, rho: np.ndarray, pm_vec: np.ndarray) -> float:
        """
        Computes dense step-level shaping potential Phi(rho) in [0, 1] combining:
        - State-overlap fidelity F_GHZ(rho)
        - Operational success probability p_GHZ(rho)
        - Average single-qubit linear mixedness (not an entanglement witness for mixed states) 2*(1 - Tr(rho_q^2))
        """
        p_ghz, f_ghz = self.ghz_metrics_from_rho(rho, pm_vec)
        diag = np.real(np.diag(rho))
        ent_sum = 0.0
        for q in range(self.n_qubits):
            i0, i1 = self.idx0_list[q], self.idx1_list[q]
            r00 = float(np.sum(diag[i0]))
            r11 = float(np.sum(diag[i1]))
            r01 = complex(np.sum(rho[i0, i1]))
            purity = r00 * r00 + r11 * r11 + 2.0 * (r01.real ** 2 + r01.imag ** 2)
            ent_sum += max(0.0, min(1.0, 2.0 * (1.0 - purity)))
        avg_ent = ent_sum / self.n_qubits
        return 0.55 * f_ghz + 0.25 * p_ghz + 0.20 * avg_ent

    def pauli_observation(self, rho: np.ndarray, step: int, max_steps: int) -> np.ndarray:
        n = self.n_qubits
        obs: List[float] = []
        diag = np.real(np.diag(rho))
        x_vals, y_vals, z_vals = [], [], []
        for q in range(n):
            i0, i1 = self.idx0_list[q], self.idx1_list[q]
            r01 = np.sum(rho[i0, i1])
            x_vals.append(float(2.0 * np.real(r01)))
            y_vals.append(float(-2.0 * np.imag(r01)))
            z_vals.append(float(np.dot(diag, self.sign_z[q])))
        obs.extend(x_vals)
        obs.extend(y_vals)
        obs.extend(z_vals)
        for q in range(n - 1):
            obs.append(float(np.dot(diag, self.sign_z[q] * self.sign_z[q + 1])))
        obs.append(float(diag[0] + diag[-1]))
        obs.append(float(2.0 * np.real(rho[0, -1])))
        obs.append(float(step) / max(1, max_steps))
        return np.array(obs, dtype=np.float32)
