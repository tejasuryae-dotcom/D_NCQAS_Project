"""
Reinforcement Learning Agents for Quantum Architecture Search and Hardware-Aware Sabotage.
Implements unrestricted sequential MDP gate selection over the full legal gate alphabet A_arch
with physics-informed no-op/cancellation masking.
"""
from typing import List, Optional, Sequence, Tuple
import numpy as np
from .noise_models import HardwareTopology, NoiseConfig
from .simulator import GateOp, compute_circuit_metrics


def compute_cvar(values: Sequence[float], beta: float = 0.20) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.ndim != 1 or len(arr) == 0 or not np.all(np.isfinite(arr)) or not 0 < beta <= 1:
        raise ValueError("CVaR requires finite nonempty samples and beta in (0,1]")
    mass = beta * len(arr)
    k = int(np.floor(mass))
    total = float(np.sum(arr[:k]))
    if k < len(arr):
        total += (mass - k) * arr[k]
    return total / mass


def build_action_catalog(n_qubits: int, topology: HardwareTopology, include_rotations: bool = True) -> List[GateOp]:
    """
    Constructs the complete discrete gate alphabet A_arch (Appendix A of Urbaneja et al.):
    - Single-qubit Clifford gates {H, X, Z, S} on all qubits q in {0..n-1}
    - Two-qubit entangling gates CNOT(q_i, q_j) in both directions on all hardware edges (q_i, q_j) in E
    - Parameterized rotations {RY(pi/2), RZ(pi/4), RX(pi/2)} on all qubits
    - Terminal DONE action
    """
    catalog: List[GateOp] = []
    for q in range(n_qubits):
        catalog.append(GateOp("H", (q,)))
        catalog.append(GateOp("X", (q,)))
        catalog.append(GateOp("Z", (q,)))
        catalog.append(GateOp("S", (q,)))
    for (q0, q1) in topology.edges:
        catalog.append(GateOp("CNOT", (q0, q1)))
        catalog.append(GateOp("CNOT", (q1, q0)))
    if include_rotations:
        for q in range(n_qubits):
            catalog.append(GateOp("RY", (q,), theta=np.pi / 2.0))
            catalog.append(GateOp("RZ", (q,), theta=np.pi / 4.0))
            catalog.append(GateOp("RX", (q,), theta=np.pi / 2.0))
    catalog.append(GateOp("DONE", ()))
    return catalog


def compute_valid_action_mask(
    circuit: Sequence[GateOp],
    catalog: Sequence[GateOp],
    n_qubits: int,
    obs: Optional[np.ndarray] = None,
    min_gates: int = 3,
) -> np.ndarray:
    """Common structural mask: minimum length and adjacent inverse Clifford pairs.

    No method-specific prior and no noisy-state threshold marketed as an exact no-op.
    """
    mask = np.ones(len(catalog), dtype=bool)
    mask[-1] = len(circuit) >= min_gates
    if circuit:
        last = circuit[-1]
        for i, op in enumerate(catalog[:-1]):
            if op.name == last.name and op.qubits == last.qubits and op.name in {"H", "X", "Z", "CNOT"}:
                mask[i] = False
    return mask


class MLPPolicyValueNet:
    """Two-hidden-layer (64, 64) tanh MLP for PPO actor-critic optimization."""
    def __init__(self, obs_dim: int, act_dim: int, rng: np.random.Generator, hidden: int = 64, lr: float = 4e-3, optimizer: str = "sgd"):
        if optimizer not in ("sgd", "adam"):
            raise ValueError("Unknown optimizer")
        self.optimizer = optimizer
        self._adam = {}
        self._adam_step = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.update_count: int = 0
        s1 = np.sqrt(2.0 / (obs_dim + hidden))
        s2 = np.sqrt(2.0 / (hidden + hidden))
        self.W1 = rng.normal(0.0, s1, size=(obs_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0.0, s2, size=(hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W_pi = rng.normal(0.0, 0.05, size=(hidden, act_dim))
        self.b_pi = np.zeros(act_dim)
        self.W_v = rng.normal(0.0, 0.05, size=(hidden, 1))
        self.b_v = np.zeros(1)

    def forward(self, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
        h1 = np.tanh(obs @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W_pi + self.b_pi
        if mask is not None and not np.any(mask):
            raise ValueError("All actions masked")
        if mask is not None:
            logits = np.where(mask, logits, -1e9)
        logits = logits - np.max(logits)
        exp_l = np.exp(logits)
        probs = exp_l / np.sum(exp_l)
        val = float((h2 @ self.W_v + self.b_v)[0])
        return probs, val, (h1, h2)

    def update_ppo(
        self,
        obs_list: List[np.ndarray],
        act_list: List[int],
        mask_list: List[np.ndarray],
        adv_list: np.ndarray,
        ret_list: np.ndarray,
        old_probs_list: List[float],
        clip_eps: float = 0.2,
        ent_coef: float = 0.02,
        epochs: int = 4,
    ) -> float:
        if len(obs_list) == 0:
            return 0.0
        if not (len(obs_list) == len(act_list) == len(mask_list) == len(adv_list) == len(ret_list) == len(old_probs_list)):
            raise ValueError("Unequal PPO batch lengths")
        self.update_count += 1
        adv_norm = (adv_list - np.mean(adv_list)) / (np.std(adv_list) + 1e-8)
        total_grad_norm = 0.0
        x = np.asarray(obs_list)
        masks = np.asarray(mask_list, dtype=bool)
        acts = np.asarray(act_list, dtype=int)
        old = np.asarray(old_probs_list, dtype=float)
        returns = np.asarray(ret_list, dtype=float)
        if np.any(old <= 0) or np.any(old > 1) or not np.all(masks[np.arange(len(x)), acts]):
            raise ValueError("Invalid recorded action probabilities or mask")
        if np.any(~masks.any(axis=1)):
            raise ValueError("All actions masked")
        ns = len(x)
        names = ['W1','b1','W2','b2','W_pi','b_pi','W_v','b_v']
        for _ in range(epochs):
            h1 = np.tanh(x @ self.W1 + self.b1)
            h2 = np.tanh(h1 @ self.W2 + self.b2)
            logits = np.where(masks, h2 @ self.W_pi + self.b_pi, -1e9)
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
            val = (h2 @ self.W_v + self.b_v)[:,0]
            ratio = probs[np.arange(ns), acts] / old
            active = ~(((adv_norm > 0) & (ratio > 1+clip_eps)) | ((adv_norm < 0) & (ratio < 1-clip_eps)))
            dlog = probs.copy(); dlog[np.arange(ns), acts] -= 1
            dlog *= (adv_norm * ratio * active)[:,None]
            logp = np.log(np.maximum(probs, 1e-8))
            entropy = -(probs * logp).sum(axis=1)
            dlog += ent_coef * probs * (logp + entropy[:,None])
            dv = 0.5 * np.clip(val - returns, -2, 2)[:,None]
            dz2 = (dlog @ self.W_pi.T + dv @ self.W_v.T) * (1-h2*h2)
            dz1 = (dz2 @ self.W2.T) * (1-h1*h1)
            grads = [x.T@dz1, dz1.sum(axis=0), h1.T@dz2, dz2.sum(axis=0),
                     h2.T@dlog, dlog.sum(axis=0), h2.T@dv, dv.sum(axis=0)]
            self._adam_step += 1
            for name, grad in zip(names, grads):
                g = np.clip(grad/ns, -1, 1)
                total_grad_norm += float(np.linalg.norm(g))
                if self.optimizer == 'adam':
                    m,v = self._adam.get(name, (np.zeros_like(g), np.zeros_like(g)))
                    m = .9*m + .1*g; v = .999*v + .001*g*g
                    self._adam[name] = (m,v)
                    step = (m/(1-.9**self._adam_step))/(np.sqrt(v/(1-.999**self._adam_step))+1e-8)
                else:
                    step = g
                getattr(self, name)[:] -= self.lr*step
        return total_grad_norm


class StaticSaboteurAgent:
    """Urbaneja et al. 3-scalar global Hardware-Aware Saboteur (Appendix B)."""
    def __init__(self, rng: np.random.Generator, w_1q: float = 1.0, w_2q: float = 2.0, w_m: float = 1.0, lambda_noise: float = 0.10):
        self.rng = rng
        self.w_1q, self.w_2q, self.w_m = w_1q, w_2q, w_m
        self.lambda_noise = lambda_noise
        self.actions = [(s1, s2, sm) for s1 in range(1, 6) for s2 in range(1, 6) for sm in range(1, 6)]
        self.budgets = np.array([w_1q * (a[0] - 1) + w_2q * (a[1] - 1) + w_m * (a[2] - 1) for a in self.actions], dtype=float)
        self.net = MLPPolicyValueNet(obs_dim=5, act_dim=len(self.actions), rng=rng, hidden=32, lr=5e-3)

    def encode_features(self, circuit: Sequence[GateOp], n_qubits: int, topology: HardwareTopology, m_nom: float, max_steps: int) -> np.ndarray:
        stats = compute_circuit_metrics(circuit, n_qubits, topology)
        return np.array([
            stats["total_gates"] / max(1, max_steps),
            stats["n_1q"] / max(1, max_steps),
            stats["n_2q"] / max(1, max_steps),
            stats["depth"] / max(1, max_steps),
            m_nom,
        ], dtype=np.float32)

    def select_action(self, feat: np.ndarray, b_max: float, deterministic: bool = False) -> Tuple[int, Tuple[float, float, float], float, float]:
        mask = self.budgets <= (b_max + 1e-6)
        if not np.any(mask):
            mask[0] = True
        probs, _, _ = self.net.forward(feat, mask)
        idx = int(np.argmax(probs)) if deterministic else int(self.rng.choice(len(self.actions), p=probs))
        s = self.actions[idx]
        return idx, (float(s[0]), float(s[1]), float(s[2])), float(self.budgets[idx]), float(probs[idx])


class TrajectoryConditionedSaboteurAgent:
    """Proposed D-NCQAS Topology- and Trajectory-Conditioned Hardware-Aware Saboteur (T-HAS)."""
    def __init__(self, topology: HardwareTopology, rng: np.random.Generator, w_1q: float = 1.0, w_2q: float = 2.0, w_m: float = 1.0, lambda_noise: float = 0.08):
        self.topology = topology
        self.rng = rng
        self.w_1q, self.w_2q, self.w_m = w_1q, w_2q, w_m
        self.lambda_noise = lambda_noise
        self.n_edges = len(topology.edges)
        self.templates: List[Tuple[float, int, float, float, float]] = []
        for s1 in (1.0, 2.0, 3.0):
            for sm in (1.0, 2.0, 3.5, 5.0):
                for s2_uni in (1.0, 2.0, 3.5, 5.0):
                    self.templates.append((s1, -1, s2_uni, s2_uni, sm))
                for e_idx in range(self.n_edges):
                    for s2_hot in (2.5, 4.0, 5.0):
                        self.templates.append((s1, e_idx, s2_hot, 1.2, sm))
        obs_dim = 5 + 2 * self.n_edges
        self.net = MLPPolicyValueNet(obs_dim=obs_dim, act_dim=len(self.templates), rng=rng, hidden=48, lr=5e-3)

    def encode_features(self, circuit: Sequence[GateOp], noise_cfg: NoiseConfig, m_nom: float, max_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        stats = compute_circuit_metrics(circuit, self.topology.n_qubits, self.topology)
        edge_counts = stats["edge_counts"].astype(float)
        edge_share = edge_counts / max(1.0, float(stats["n_2q"]))
        _, p2q_0, _ = self.topology.nominal_rates()
        drift_ratio = noise_cfg.p2q_vec / np.maximum(p2q_0, 1e-6)
        feat = np.concatenate([
            np.array([stats["total_gates"] / max(1, max_steps), stats["n_1q"] / max(1, max_steps), stats["n_2q"] / max(1, max_steps), stats["depth"] / max(1, max_steps), m_nom], dtype=np.float32),
            edge_share.astype(np.float32),
            np.clip(drift_ratio / 3.0, 0.0, 2.0).astype(np.float32),
        ])
        return feat, edge_counts

    def realize_action(self, template_idx: int, edge_counts: np.ndarray, b_max: float) -> Tuple[float, np.ndarray, float, float]:
        if b_max < 0:
            raise ValueError("Negative adversarial budget")
        s1, mode, s2_hot, s2_bg, sm = self.templates[template_idx]
        s2_vec = np.full(self.n_edges, s2_bg, dtype=float)
        if mode == -1:
            s2_vec[:] = s2_hot
        elif mode == -2:
            s2_vec[int(np.argmax(edge_counts))] = s2_hot
        else:
            s2_vec[mode] = s2_hot
        raw_b = self.w_1q * (s1 - 1.0) + self.w_2q * float(np.mean(s2_vec - 1.0)) + self.w_m * (sm - 1.0)
        if raw_b > b_max and raw_b > 1e-8:
            scale = b_max / raw_b
            s1 = 1.0 + (s1 - 1.0) * scale
            s2_vec = 1.0 + (s2_vec - 1.0) * scale
            sm = 1.0 + (sm - 1.0) * scale
            raw_b = b_max
        actual = self.w_1q*(s1-1) + self.w_2q*np.mean(s2_vec-1) + self.w_m*(sm-1)
        if actual > b_max + 1e-10:
            raise AssertionError("Budget projection failed")
        return float(s1), s2_vec, float(sm), float(actual)

    def select_action(self, circuit: Sequence[GateOp], noise_cfg: NoiseConfig, m_nom: float, b_max: float, max_steps: int, deterministic: bool = False) -> Tuple[int, np.ndarray, float, np.ndarray, float, float, float]:
        feat, edge_counts = self.encode_features(circuit, noise_cfg, m_nom, max_steps)
        probs, _, _ = self.net.forward(feat)
        idx = int(np.argmax(probs)) if deterministic else int(self.rng.choice(len(self.templates), p=probs))
        s1, s2_vec, sm, used_b = self.realize_action(idx, edge_counts, b_max)
        return idx, feat, s1, s2_vec, sm, used_b, float(probs[idx])


def minimal_ghz(topology: HardwareTopology, root: int = 0) -> List[GateOp]:
    """Breadth-first spanning tree, one H and n-1 legal CNOTs, no padding."""
    visited, queue = {root}, [root]
    ops = [GateOp("H", (root,))]
    while queue:
        q = queue.pop(0)
        for a, b in topology.edges:
            t = b if a == q else a if b == q else None
            if t is not None and t not in visited:
                visited.add(t); queue.append(t); ops.append(GateOp("CNOT", (q, t)))
    if len(visited) != topology.n_qubits:
        raise ValueError("Disconnected topology")
    return ops
