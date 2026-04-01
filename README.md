# privacy-mcps
# Privacy-Preserving Microservices ML Framework for Medical Cyber-Physical Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.12+](https://img.shields.io/badge/tensorflow-2.12+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-24.0+-blue.svg)](https://www.docker.com/)
## Overview
This repository contains the implementation of a **microservice-based privacy-preserving deep learning framework** for intrusion detection in Medical Cyber-Physical Systems (MCPS). The framework integrates:

- **Paillier Partially Homomorphic Encryption (PHE)** for noise-free privacy preservation during distributed model training
- **Hybrid 1D-CNN-BiGRU-Attention** deep learning architecture for high-accuracy intrusion detection
- **Six-layer microservice architecture** deployed via Docker and Kubernetes for scalable edge-cloud analytics

> **Paper:** *Privacy-Preserving Microservices Machine Learning Framework in Medical Cyber-Physical Systems*  
> **Target Journal:** IEEE Internet of Things Journal

## Architecture

┌─────────────┐  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌─────────┐  ┌─────────────┐
│   Layer 1    │  │   Layer 2    │  │     Layer 3       │  │     Layer 4       │  │ Layer 5  │  │   Layer 6    │
│  Activity    │→│Cyber-Physical│→│    Privacy         │→│   Knowledge       │→│  Alert   │→│ Application  │
│  Receiver    │  │   System     │  │  Preserving       │  │  Aggregation      │  │         │  │   (App)      │
│   (ARMS)     │  │   (DPMS)     │  │  (ETMS×M)        │  │(DAMS,BMS,SELMS)  │  │  (AMS)  │  │(MLaMS,DMS)  │
└─────────────┘  └─────────────┘  └──────────────────┘  └──────────────────┘  └─────────┘  └─────────────┘
## Key Results

| Metric | CIC-IoMT2024 | Edge-IIoTset |
|--------|:------------:|:------------:|
| Binary Accuracy (PHE) | 98.89% | 98.65% |
| Multi-class Accuracy (PHE) | 98.21% | 97.68% |
| AUC (Binary) | 0.9978 | 0.9969 |
| Privacy-induced Accuracy Loss | **0.23%** | **0.22%** |
| FAR (Binary) | 1.11% | 1.35% |

## Repository Structure

```
privacy-mcps/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── configs/
│   ├── config.yaml                    # Main configuration
│   └── model_config.yaml              # Model hyperparameters
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_bigru_attention.py     # 1D-CNN-BiGRU-Attention model
│   │   └── attention.py               # Multi-Head Self-Attention layer
│   ├── privacy/
│   │   ├── __init__.py
│   │   ├── paillier_enc.py            # Paillier PHE encryption/decryption
│   │   ├── encoding.py                # Fixed-point encoding for real-valued params
│   │   └── secure_aggregation.py      # Homomorphic aggregation protocol
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── data_pipeline.py           # Data loading, preprocessing, SMOTE
│   ├── microservices/
│   │   ├── __init__.py
│   │   ├── proto/
│   │   │   └── inference.proto        # gRPC protocol buffer definition
│   │   ├── arms.py                    # Activity Receiver Microservice
│   │   ├── dpms.py                    # Data Preprocessing Microservice
│   │   ├── etms.py                    # Encrypted Training Microservice
│   │   ├── dams.py                    # Data Aggregation Microservice
│   │   ├── ams.py                     # Alert Microservice
│   │   └── mlams.py                   # ML Application Microservice (gRPC)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # All evaluation metrics
│   │   ├── statistical_tests.py       # T-tests, significance analysis
│   │   └── visualization.py           # Confusion matrices, ROC, convergence plots
│   └── utils/
│       ├── __init__.py
│       └── logger.py                  # Logging utilities
├── scripts/
│   ├── train_centralized.py           # Centralized training (baseline)
│   ├── train_distributed.py           # Distributed training with Paillier PHE
│   ├── evaluate.py                    # Full evaluation pipeline
│   ├── run_ablation.py                # Ablation experiments
│   ├── benchmark_sota.py              # SOTA benchmarking
│   └── run_all_experiments.sh         # Run all experiments end-to-end
├── docker/
│   ├── Dockerfile.etms                # ETMS container
│   ├── Dockerfile.dams                # DAMS container
│   ├── Dockerfile.mlams               # MLaMS inference container
│   └── docker-compose.yaml            # Multi-container orchestration
├── kubernetes/
│   ├── namespace.yaml
│   ├── etms-deployment.yaml
│   ├── dams-deployment.yaml
│   ├── mlams-deployment.yaml
│   └── ams-deployment.yaml
├── tests/
│   ├── test_model.py
│   ├── test_paillier.py
│   └── test_aggregation.py
├── data/                              # Placeholder (download instructions below)
│   └── .gitkeep
├── results/                           # Experiment outputs
│   └── .gitkeep
└── docs/
    ├── INSTALLATION.md
    └── EXPERIMENTS.md
```

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (recommended)
- Docker 24.0+ (for microservice deployment)
- Kubernetes 1.27+ (optional, for orchestration)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/privacy-mcps.git
cd privacy-mcps

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

1. **CIC-IoMT2024**: Download from [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/)
2. **Edge-IIoTset**: Download from [IEEE Dataport](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)

Place downloaded CSV files in the `data/` directory:
```
data/
├── CIC-IoMT2024.csv
└── Edge-IIoTset.csv
```

## Quick Start

### 1. Centralized Training (Baseline)
```bash
python scripts/train_centralized.py --dataset cic-iomt --task binary
python scripts/train_centralized.py --dataset edge-iiot --task multiclass
```

### 2. Distributed Training with Paillier PHE
```bash
python scripts/train_distributed.py --dataset cic-iomt --partitions 6 --key-length 2048
python scripts/train_distributed.py --dataset edge-iiot --partitions 6 --key-length 2048
```

### 3. Full Evaluation
```bash
python scripts/evaluate.py --dataset cic-iomt --model-path results/models/best_model.h5
```

### 4. Run All Experiments
```bash
bash scripts/run_all_experiments.sh
```

### 5. Docker Deployment
```bash
cd docker
docker-compose up --build
```

## Citation

If you use this code, please cite:

```bibtex
@article{author2026privacy,
  title={Privacy-Preserving Microservices Machine Learning Framework 
         in Medical Cyber-Physical Systems},
  author={Author Name},
  journal={IEEE Internet of Things Journal},
  year={2026},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Canadian Institute for Cybersecurity for the CIC-IoMT2024 dataset
- Ferrag et al. for the Edge-IIoTset dataset
- Paillier cryptosystem implementation based on the `phe` library
