"""
Spatially Heterogeneous and Temporally Drifting Quantum Hardware Noise Models.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class HardwareTopology:
    """Represents an n-qubit quantum processor coupling topology and spatial baseline calibration."""
    n_qubits: int
    edges: List[Tuple[int, int]]
    base_p1q: float = 0.001
    base_p2q: float = 0.010
    base_pm: float = 0.020
    spatial_mult_1q: np.ndarray = field(default_factory=lambda: np.array([]))
    spatial_mult_2q: np.ndarray = field(default_factory=lambda: np.array([]))
    spatial_mult_m: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def synthetic_tree(cls, n_qubits: int, homogeneous: bool = False, kind: str = "tree") -> "HardwareTopology":
        """Synthetic connected graph and rates; not an IBM calibration snapshot."""
        if n_qubits < 2 or n_qubits > 8:
            raise ValueError("Supported qubit count is 2..8")
        if kind not in ("tree", "chain"):
            raise ValueError("kind must be tree or chain")
        edges = [(i, i + 1) for i in range(n_qubits - 1)]
        if kind == "tree" and n_qubits >= 4:
            edges = [(0, 1), (1, 2), (1, 3)] + [(i, i + 1) for i in range(3, n_qubits - 1)]
        m1 = np.linspace(0.8, 1.4, n_qubits)
        m2 = np.linspace(0.75, 1.75, len(edges))
        mm = np.linspace(0.8, 1.5, n_qubits)
        if homogeneous:
            m1, m2, mm = np.ones(n_qubits), np.ones(len(edges)), np.ones(n_qubits)
        return cls(n_qubits, edges, spatial_mult_1q=m1, spatial_mult_2q=m2, spatial_mult_m=mm)

    def nominal_rates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.base_p1q * self.spatial_mult_1q,
            self.base_p2q * self.spatial_mult_2q,
            self.base_pm * self.spatial_mult_m,
        )

    def edge_index(self, q_ctrl: int, q_targ: int) -> int:
        pair = (min(q_ctrl, q_targ), max(q_ctrl, q_targ))
        for idx, e in enumerate(self.edges):
            if (min(e[0], e[1]), max(e[0], e[1])) == pair:
                return idx
        raise ValueError(f"Unsupported coupling: {(q_ctrl, q_targ)}")


@dataclass
class NoiseConfig:
    n_qubits: int
    edges: List[Tuple[int, int]]
    p1q_vec: np.ndarray
    p2q_vec: np.ndarray
    pm_vec: np.ndarray
    noise_family: str = "depolarizing"
    gate_local_depol: Optional[np.ndarray] = None
    overrotation_angle: float = 0.0
    amp_damping_rate: float = 0.0
    phase_damping_rate: float = 0.0
    idle_decoherence_rate: float = 0.0015

    def __post_init__(self):
        for name, length in [("p1q_vec", self.n_qubits), ("p2q_vec", len(self.edges)), ("pm_vec", self.n_qubits)]:
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != (length,) or not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 1)):
                raise ValueError(f"Invalid {name}")
            setattr(self, name, a)
        if self.noise_family not in {"depolarizing", "readout", "overrotation", "amplitude_damping", "phase_damping", "asymmetric_pauli"}:
            raise ValueError("Unknown noise family")

    def copy_with_scale(self, scale: float) -> "NoiseConfig":
        return NoiseConfig(
            n_qubits=self.n_qubits,
            edges=list(self.edges),
            p1q_vec=np.clip(self.p1q_vec * scale, 0.0, 0.5),
            p2q_vec=np.clip(self.p2q_vec * scale, 0.0, 0.75),
            pm_vec=np.clip(self.pm_vec * scale, 0.0, 0.45),
            noise_family=self.noise_family,
            gate_local_depol=None if self.gate_local_depol is None else np.clip(self.gate_local_depol * scale, 0.0, 0.5),
            overrotation_angle=self.overrotation_angle * np.sqrt(scale),
            amp_damping_rate=np.clip(self.amp_damping_rate * scale, 0.0, 0.5),
            phase_damping_rate=np.clip(self.phase_damping_rate * scale, 0.0, 0.5),
            idle_decoherence_rate=np.clip(self.idle_decoherence_rate * scale, 0.0, 0.2),
        )


class OrnsteinUhlenbeckDriftProcess:
    """Spatially correlated Ornstein-Uhlenbeck (OU) calibration drift process with TLS bursts."""
    def __init__(
        self,
        topology: HardwareTopology,
        kappa_ou: float = 0.35,
        sigma_ou: float = 0.28,
        drift_aging_shift: float = 0.22,
        tls_burst_prob: float = 0.06,
        dt: float = 0.25,
    ):
        self.topology = topology
        self.kappa_ou = kappa_ou
        self.sigma_ou = sigma_ou
        self.drift_aging_shift = drift_aging_shift
        self.tls_burst_prob = tls_burst_prob
        self.dt = dt
        self.n_1q = topology.n_qubits
        self.n_2q = len(topology.edges)
        self.n_m = topology.n_qubits
        self.dim = self.n_1q + self.n_2q + self.n_m
        self.chol_corr = self._build_cholesky_correlation()

    def _build_cholesky_correlation(self) -> np.ndarray:
        corr = np.eye(self.dim)
        for i, e1 in enumerate(self.topology.edges):
            for j, e2 in enumerate(self.topology.edges):
                if i != j and (set(e1) & set(e2)):
                    corr[self.n_1q + i, self.n_1q + j] = 0.35
            for q in e1:
                corr[q, self.n_1q + i] = 0.25
                corr[self.n_1q + i, q] = 0.25
                corr[self.n_1q + self.n_2q + q, self.n_1q + i] = 0.20
                corr[self.n_1q + i, self.n_1q + self.n_2q + q] = 0.20
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-4)
        return np.linalg.cholesky(eigvecs @ np.diag(eigvals) @ eigvecs.T)

    def sample_trajectory(
        self,
        rng: np.random.Generator,
        n_epochs: int = 8,
        permute_spatial_bottlenecks: bool = False,
    ) -> List[NoiseConfig]:
        p1q_0, p2q_0, pm_0 = self.topology.nominal_rates()
        if permute_spatial_bottlenecks:
            p1q_0 = rng.permutation(p1q_0)
            p2q_0 = rng.permutation(p2q_0)
            pm_0 = rng.permutation(pm_0)

        y0 = np.log(np.concatenate([p1q_0, p2q_0, pm_0]))
        mu_drift = y0 + self.drift_aging_shift
        y = y0.copy()
        trajectory: List[NoiseConfig] = []

        for step in range(n_epochs):
            if step > 0:
                eps = rng.standard_normal(self.dim)
                d_w = np.sqrt(self.dt) * (self.chol_corr @ eps)
                tls_mask = rng.random(self.dim) < (self.tls_burst_prob * self.dt)
                tls_jump = tls_mask * rng.uniform(0.35, 0.90, size=self.dim)
                y = y + self.kappa_ou * (mu_drift - y) * self.dt + self.sigma_ou * d_w + tls_jump

            eta = np.exp(y)
            p1q_t = np.clip(eta[:self.n_1q], 1e-4, 0.15)
            p2q_t = np.clip(eta[self.n_1q:self.n_1q + self.n_2q], 1e-3, 0.35)
            pm_t = np.clip(eta[self.n_1q + self.n_2q:], 2e-3, 0.35)
            idle_rate = 0.0015 * float(np.mean(p1q_t / np.maximum(p1q_0, 1e-6)))
            trajectory.append(
                NoiseConfig(
                    n_qubits=self.topology.n_qubits,
                    edges=list(self.topology.edges),
                    p1q_vec=p1q_t,
                    p2q_vec=p2q_t,
                    pm_vec=pm_t,
                    noise_family="depolarizing",
                    idle_decoherence_rate=np.clip(idle_rate, 0.0005, 0.03),
                )
            )
        return trajectory


def amplify_noise_config_global(base_cfg: NoiseConfig, s_1q: float, s_2q: float, s_m: float) -> NoiseConfig:
    return NoiseConfig(
        n_qubits=base_cfg.n_qubits,
        edges=list(base_cfg.edges),
        p1q_vec=np.clip(base_cfg.p1q_vec * s_1q, 0.0, 0.35),
        p2q_vec=np.clip(base_cfg.p2q_vec * s_2q, 0.0, 0.50),
        pm_vec=np.clip(base_cfg.pm_vec * s_m, 0.0, 0.40),
        noise_family=base_cfg.noise_family,
        idle_decoherence_rate=np.clip(base_cfg.idle_decoherence_rate * (0.5 * (s_1q + s_2q)), 0.0, 0.04),
    )


def amplify_noise_config_spatial(base_cfg: NoiseConfig, s_1q: float, s_2q_edges: np.ndarray, s_m: float) -> NoiseConfig:
    return NoiseConfig(
        n_qubits=base_cfg.n_qubits,
        edges=list(base_cfg.edges),
        p1q_vec=np.clip(base_cfg.p1q_vec * s_1q, 0.0, 0.35),
        p2q_vec=np.clip(base_cfg.p2q_vec * s_2q_edges, 0.0, 0.55),
        pm_vec=np.clip(base_cfg.pm_vec * s_m, 0.0, 0.40),
        noise_family=base_cfg.noise_family,
        idle_decoherence_rate=np.clip(
            base_cfg.idle_decoherence_rate * (0.4 * s_1q + 0.6 * float(np.mean(s_2q_edges))),
            0.0,
            0.04,
        ),
    )
