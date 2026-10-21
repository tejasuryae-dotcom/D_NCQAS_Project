# Review finding to implementation map

| Finding | Revision | Evidence |
|---|---|---|
| Reversed actor/entropy gradients | Correct minimization gradients | Finite-difference all-parameter test; actor and entropy direction tests |
| Sampling/recorded probability mismatch | Removed mixed action prior; sample the recorded categorical distribution | Direct on-policy rollout code |
| Stress exceeds budget | Deleted post-projection override; explicit budget assertion | Every template at five budgets over three sizes tested |
| Invalid hardware/topology claims | Synthetic connected tree/chain, no FakeQuito claim; illegal gates raise | All-checkpoint gate validation |
| Inconsistent step/full execution | Shared serial-gate phase-damping rule | Channel and step-path equivalence tests |
| Inexact unitary cache and hashes | Float-hex angle cache keys, full SHA-256 canonical gate serialization | Full-precision hash regression |
| Broken paired scenarios | Circuit-independent scenario bank with regime/split streams | All raw scenario IDs checked across methods |
| Confounded ablations | Common architecture policy/grammar/selector; one named flag per ablation | Configuration-difference tests; protocol |
| Artificially padded baseline | Minimal GHZ and root-selected tree controls | Exact ideal GHZ test on both graph families |
| Molecular matrices shifted to target | PySCF integrals and independently checked compact CI matrices | Three saved Hamiltonian generation records |
| Misnamed VQE methods | Separate fixed-ansatz, frozen-parameter validation | VQE checkpoint/restart metadata and raw outputs |
| Readout/fidelity conflation | Premeasurement state fidelity; assignment-noisy measurement separate | Pauli measurement reference test |
| Unimplemented mitigation | Removed mitigation claims and fabricated noise-reduction overlay | New figures contain no RC-ZNE results |
| Missing paths/dependencies | Stable generated summaries, exact runtime/notebook/chemistry locks | Clean setup and notebook execution logs |
| Stale documentation/results | Superseded version-2 claims with the frozen revision-3 protocol and data-derived article | Version-3 manuscript and analysis tables |
| Missing metadata | MIT, CFF, contributor guide, CI, source/artifact hashes | Supporting assets and manifests |

The corrected release deliberately narrows scientific scope. It does not preserve invalid four-qubit
molecular comparisons or claim a successful coevolutionary VQE implementation.

Revision 3 adds Adam/SGD diagnostics, vectorized PPO gradients, independent generic CEM and aging-evolution controls, development-only budget scaling, 32 fresh primary seed pairs, four prespecified primary tests, and matched VQE cap/parameter-shift convergence checks. See REVISION_PROTOCOL.md and manuscript/REVISION_RESPONSE.md.
