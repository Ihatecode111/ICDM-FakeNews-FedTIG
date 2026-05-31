# FedTIG: Model-Decoupled Federated Topological Alignment for Incomplete Multimodal Fake News Detection

This repository contains the core anonymous source code for the FedTIG framework, implementing the Alternating Optimization paradigm, Fact-Driven Imputation, and Evidential Graph Information Bottleneck.

## 1. Requirements
- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Geometric (PyG)
- POT (Python Optimal Transport) >= 0.9.0
- spaCy
- scikit-learn

## 2. Core Modules
- `fedtig_core.py`: Contains the complete implementation of the Client and Server architectures, including the Generator, Discriminator, Evidential GIB, and Wasserstein Barycenter synchronization.

## 3. How to Run
To reproduce the local alternating optimization training process (Step 1: Imputation, Step 2: Topology & Task), you can execute the core script with your customized data loaders:

```bash
python fedtig_core.py --num_clients 20 --communication_rounds 50 --local_epochs 5 --masking_rate 0.3
