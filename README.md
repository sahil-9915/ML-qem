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
| ZNE-Linear MAE         | 0.0360 (worse) | 0.0418 (worse)  |
| ZNE-Quadratic MAE      | 0.0379 (worse) | 0.0439 (worse)  |
| RF MAE                 | 0.0084 (4.2×)  | 0.0135 (3.0×)   |
| MLP MAE                | 0.0076 (4.6×)  | 0.0122 (3.3×)   |
| Energy MAE (RF)        | 0.00461 Ha     | 0.00477 Ha      |
| Energy MAE (MLP)       | 0.00402 Ha     | 0.00440 Ha      |

**Key Findings:**
- MLP outperforms RF on both molecules when properly trained (1000–2000 epochs vs 300), with the difference confirmed by paired bootstrap test (p < 0.05)
- RF achieves chemical accuracy on H₂ VQE (0.05 mHa bias, 4/5 runs within 1.6 mHa) but fails on LiH (149.96 mHa, worse than unmitigated)
- MLP provides meaningful VQE improvement on LiH (66.73 mHa, 2.2× over unmitigated) through smoother corrections
- Both ZNE configurations (linear and quadratic Richardson extrapolation) perform worse than unmitigated on both molecules
- Feature ablation reveals symplectic encoding nearly halves MAE despite <0.05% impurity-based importance
- MLP shows essentially zero overfitting (train-test ratio ≈1.0) vs RF's moderate overfitting (≈2.0–2.5×)

## Project Structure

```
ml-qem/
├── h2/                                 # H₂ molecule (2 qubits, FakeLimaV2)
│   ├── generate_data.ipynb             # Generate (noisy, ideal) pairs
│   ├── training_RF.ipynb               # Train Random Forest
│   ├── training_MLP.ipynb              # Train MLP (1000 epochs), compare with RF
│   ├── h2_comprehensive_analysis.ipynb # Bootstrap CIs, ablation, residuals, significance test
│   ├── vqe_optimization.ipynb          # VQE with ideal/noisy/RF/MLP
│   ├── zne_comparison.ipynb            # ZNE baseline (linear + quadratic)
│   ├── data_efficiency.ipynb           # Data efficiency (5 seeds) & feature importance
│   ├── ideal_data.npy                  # Ideal expectation values (2000, 5)
│   ├── noisy_data.npy                  # Noisy expectation values (2000, 5)
│   ├── theta_samples.npy              # Parameter vectors (2000, 8)
│   └── dataset_meta.json              # Metadata
│
├── lih/                                # LiH molecule (6 qubits, FakeJakartaV2)
│   ├── lih_generate_data.ipynb         # Generate (noisy, ideal) pairs
│   ├── lih_training_RF.ipynb           # Train Random Forest
│   ├── lih_training_MLP.ipynb          # Train MLP (2000 epochs), compare with RF
│   ├── lih_comprehensive_analysis.ipynb# Bootstrap CIs, ablation, residuals, significance test
│   ├── lih_vqe_optimization.ipynb      # VQE with ideal/noisy/RF/MLP, 5-run multi-start
│   ├── lih_zne_comparison.ipynb        # ZNE baseline (linear + quadratic)
│   ├── lih_data_efficiency.ipynb       # Data efficiency (5 seeds) & feature importance
│   ├── lih_ideal_data.npy             # Ideal expectation values (2000, 62)
│   ├── lih_noisy_data.npy             # Noisy expectation values (2000, 62)
│   ├── lih_theta_samples.npy          # Parameter vectors (2000, 24)
│   └── lih_meta.json                  # Metadata
│
└── mlqem_report.pdf                    # Technical report
```

## Pipeline

Each molecule follows the same pipeline:

1. **generate_data** — Sample 2000 random θ ∈ [-π, π], run circuits on noiseless and noisy simulators, collect paired expectation values
2. **training_RF** — Train Random Forest (100 trees) using symplectic Pauli encoding with per-qubit noise features
3. **training_MLP** — Train MLP (64 hidden units, early stopping) for 1000–2000 epochs, compare with RF
4. **comprehensive_analysis** — Multi-seed MLP (5 seeds), bootstrap CIs, paired significance test, overfitting check, feature ablation, correction function visualisation, predicted vs ideal scatter, residual analysis
5. **zne_comparison** — Benchmark against digital ZNE with linear {1,3} and quadratic {1,3,5} Richardson extrapolation
6. **data_efficiency** — Learning curves with error bands (5 seeds, both models) and RF feature importance
7. **vqe_optimization** — Full VQE loop (COBYLA) with ideal/noisy/RF-mitigated/MLP-mitigated cost functions, 20 evaluations at θ*, 5-run multi-start analysis

## Method

Following the paper's feature design (Fig. 7), each (circuit, observable) pair is encoded as:

| Feature | Description |
|---------|-------------|
| Noisy ⟨O⟩ | Implicitly encodes the quantum state |
| Symplectic encoding | x and z bits per qubit from the Pauli observable |
| Gate counts | Number of CX and SX gates (circuit complexity proxy) |
| Noise parameters | T1, T2, readout error per qubit, masked by observable support |

## Extensions Beyond the Original Paper

| Extension | Finding |
|-----------|---------|
| MLP with proper training | MLP outperforms RF (p < 0.05) on both molecules |
| Quadratic ZNE | Higher-order extrapolation performs even worse due to stronger variance amplification |
| Feature ablation | Symplectic encoding is critical despite <0.05% impurity importance |
| VQE with MLP correction | RF wins on H₂ (low bias), MLP wins on LiH (lower bias and better corrections) |
| Multi-run VQE (5 starts) | RF achieves chemical accuracy on H₂ (4/5), fails on LiH (worse than noisy) |
| Bootstrap significance tests | Paired bootstrap CIs confirm MLP advantage on both molecules |
| Multi-seed MLP (5 seeds) | MLP extremely stable: 0.0076 ± 0.0001 (H₂), 0.0122 ± 0.0001 (LiH) |

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
