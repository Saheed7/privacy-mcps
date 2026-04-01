# Experiment Guide

## Overview

This guide describes how to reproduce all experiments from the paper. The full experiment suite takes approximately 8-12 hours on a single NVIDIA RTX 3090 GPU.

## Quick Run (All Experiments)

```bash
bash scripts/run_all_experiments.sh
```

## Individual Experiments

### Experiment 1: Centralized Baseline (Tables VI-VIII)

Train the model without privacy or distribution:

```bash
# Binary classification
python scripts/train_centralized.py --dataset cic-iomt --task binary
python scripts/train_centralized.py --dataset edge-iiot --task binary

# Multi-class classification
python scripts/train_centralized.py --dataset cic-iomt --task multiclass
python scripts/train_centralized.py --dataset edge-iiot --task multiclass
```

### Experiment 2: Distributed Training with Paillier PHE (Tables IX-X)

```bash
# Default: 6 partitions, 2048-bit key
python scripts/train_distributed.py --dataset cic-iomt --task binary --partitions 6

# Test different partition counts
for M in 2 6 12 18 24; do
  python scripts/train_distributed.py --dataset cic-iomt --task binary --partitions $M
done

# Skip encryption for speed comparison
python scripts/train_distributed.py --dataset cic-iomt --task binary --skip-encryption
```

### Experiment 3: Full Evaluation with Figures (Figures 3-5)

```bash
python scripts/evaluate.py \
  --dataset cic-iomt \
  --task binary \
  --model-path results/models/distributed_cic-iomt_binary_M6.h5
```

### Experiment 4: Ablation Study (Table XVII)

```bash
python scripts/run_ablation.py --dataset cic-iomt --task binary
python scripts/run_ablation.py --dataset edge-iiot --task binary
```

### Experiment 5: SOTA Benchmarking (Table XVI)

```bash
python scripts/benchmark_sota.py --dataset cic-iomt --task binary
python scripts/benchmark_sota.py --dataset edge-iiot --task binary
```

### Experiment 6: Microservice Deployment

```bash
# Build and run with Docker Compose
cd docker
docker-compose up --build

# Test inference endpoint
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ..., 0.5]}'
```

## Output Structure

After running all experiments:

```
results/
├── models/                    # Trained model weights (.h5)
├── figures/                   # Generated plots (Figures 3-5)
├── tables/                    # JSON metrics and results
└── logs/                      # Training logs
```

## Configuration

Edit `configs/config.yaml` to modify:
- Dataset paths and parameters
- Number of partitions (M)
- Paillier key length
- Training hyperparameters
- Evaluation settings (number of runs, significance level)

Edit `configs/model_config.yaml` to modify:
- CNN filter counts and kernel sizes
- BiGRU hidden units
- Attention heads and dimensions
- Dropout rates
