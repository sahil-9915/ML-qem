# Machine Learning for Quantum Error Mitigation

Reproduces and extends Section C of:
> Liao et al. *Machine Learning for Practical Quantum Error Mitigation*,
> Nature Machine Intelligence 6, 594–604 (2024). [arXiv:2309.17368](https://arxiv.org/abs/2309.17368)

## Key Results

|                        | H₂ (2 qubits) | LiH (6 qubits) |
|------------------------|----------------|-----------------|
| Pauli terms            | 5              | 62              |
| Features               | 13             | 33              |
| Unmitigated MAE        | 0.0349         | 0.0406          |
| ZNE MAE                | 0.0359 (worse) | 0.0419 (worse)  |
| RF MAE                 | 0.0084 (4.2×)  | 0.0135 (3.0×)   |
| MLP MAE                | 0.0092 (3.8×)  | 0.0147 (2.8×)   |
| Energy MAE (RF)        | 0.00461 Ha     | 0.00477 Ha      |

**Findings:**
- RF achieves chemical accuracy (1.4 mHa bias) on H₂ VQE with zero runtime overhead
- RF outperforms ZNE on both molecules while requiring no extra circuits at runtime
- RF outperforms MLP on both molecules: H₂ (4.2× vs 3.8×) and LiH (3.0× vs 2.8×), with the gap narrowing at larger system size
- RF is more data-efficient, reaching near-optimal performance with 50 training circuits on LiH

## Project Structure

```
ml-qem/
├── h2/                                 # H₂ molecule (2 qubits, FakeLimaV2)
│   ├── generate_data.ipynb             # Generate (noisy, ideal) pairs
│   ├── training_RF.ipynb               # Train Random Forest
│   ├── training_MLP.ipynb              # Train MLP, compare with RF
│   ├── vqe_optimization.ipynb          # VQE loop with RF correction
│   ├── zne_comparison.ipynb            # ZNE baseline comparison
│   ├── data_efficiency.ipynb           # Data efficiency & feature importance
│   ├── ideal_data.npy                  # Ideal expectation values (2000, 5)
│   ├── noisy_data.npy                  # Noisy expectation values (2000, 5)
│   ├── theta_samples.npy              # Parameter vectors (2000, 8)
│   └── dataset_meta.json              # Metadata
│
├── lih/                                # LiH molecule (6 qubits, FakeJakartaV2)
│   ├── lih_generate_data.ipynb         # Generate (noisy, ideal) pairs
│   ├── lih_train_mlqem.ipynb           # Train RF and MLP
│   ├── lih_zne_comparison.ipynb        # ZNE baseline comparison
│   ├── lih_data_efficiency.ipynb       # Data efficiency & feature importance
│   ├── lih_ideal_data.npy             # Ideal expectation values (2000, 62)
│   ├── lih_noisy_data.npy             # Noisy expectation values (2000, 62)
│   ├── lih_theta_samples.npy          # Parameter vectors (2000, 24)
│   └── lih_meta.json                  # Metadata
│
└── mlqem_report.pdf                    # Technical report
```

## Pipeline

Each molecule follows the same pipeline:

1. **generate_data** — Sample random θ, run circuits on noiseless and noisy simulators, collect paired expectation values
2. **training_RF / training_MLP** — Train models using symplectic Pauli encoding with per-qubit noise features
3. **zne_comparison** — Benchmark against digital Zero-Noise Extrapolation (gate folding, noise factors {1,3})
4. **data_efficiency** — Learning curves and RF feature importance analysis
5. **vqe_optimization** (H₂ only) — Full VQE loop with RF correction, demonstrating chemical accuracy

## Method

Following the paper's feature design (Fig. 7), each (circuit, observable) pair is encoded as:

| Feature | Description |
|---------|-------------|
| Noisy ⟨O⟩ | Implicitly encodes the quantum state |
| Symplectic encoding | x and z bits per qubit from the Pauli observable |
| Gate counts | Number of CX and SX gates (circuit complexity proxy) |
| Noise parameters | T1, T2, readout error per qubit, masked by observable support |

## Requirements

- Python 3.10+
- qiskit >= 1.4, qiskit-aer >= 0.16, qiskit-ibm-runtime >= 0.29
- scikit-learn, numpy, scipy, matplotlib
- PyTorch (for MLP training)

Note: `pyscf` and `qiskit-nature` are **not** required — the LiH Hamiltonian is pre-computed and hardcoded in the notebook.

## Reference

```
@article{liao2024mlqem,
  title={Machine Learning for Practical Quantum Error Mitigation},
  author={Liao, Haoran and Wang, Derek S. and Sitdikov, Iskandar and Salcedo, Ciro and Seif, Alireza and Minev, Zlatko K.},
  journal={Nature Machine Intelligence},
  volume={6},
  pages={594--604},
  year={2024}
}
```
