#!/bin/bash
# Run All Experiments End-to-End
# Usage: bash scripts/run_all_experiments.sh

set -e
echo "============================================================"
echo "  Privacy-Preserving MCPS Framework — Full Experiment Suite"
echo "============================================================"

DATASETS=("cic-iomt" "edge-iiot")
TASKS=("binary" "multiclass")
PARTITIONS=(2 6 12 18 24)
OUTPUT="results"

mkdir -p $OUTPUT/{models,figures,tables,logs}

# 1. Centralized Training (Baseline)
echo -e "\n[1/5] CENTRALIZED TRAINING"
for ds in "${DATASETS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "  → $ds / $task"
    python scripts/train_centralized.py --dataset $ds --task $task --output-dir $OUTPUT \
      2>&1 | tee $OUTPUT/logs/centralized_${ds}_${task}.log
  done
done

# 2. Distributed Training with Paillier PHE
echo -e "\n[2/5] DISTRIBUTED TRAINING (Paillier PHE)"
for ds in "${DATASETS[@]}"; do
  for task in "${TASKS[@]}"; do
    for M in "${PARTITIONS[@]}"; do
      echo "  → $ds / $task / M=$M"
      python scripts/train_distributed.py --dataset $ds --task $task --partitions $M --output-dir $OUTPUT \
        2>&1 | tee $OUTPUT/logs/distributed_${ds}_${task}_M${M}.log
    done
  done
done

# 3. Distributed Training without encryption (plaintext baseline)
echo -e "\n[3/5] DISTRIBUTED TRAINING (Plaintext, for comparison)"
for ds in "${DATASETS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "  → $ds / $task / M=6 / plaintext"
    python scripts/train_distributed.py --dataset $ds --task $task --partitions 6 \
      --skip-encryption --output-dir $OUTPUT \
      2>&1 | tee $OUTPUT/logs/distributed_${ds}_${task}_plaintext.log
  done
done

# 4. Ablation Experiments
echo -e "\n[4/5] ABLATION EXPERIMENTS"
for ds in "${DATASETS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "  → $ds / $task"
    python scripts/run_ablation.py --dataset $ds --task $task --output-dir $OUTPUT \
      2>&1 | tee $OUTPUT/logs/ablation_${ds}_${task}.log
  done
done

# 5. SOTA Benchmarking
echo -e "\n[5/5] SOTA BENCHMARKING"
for ds in "${DATASETS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "  → $ds / $task"
    python scripts/benchmark_sota.py --dataset $ds --task $task --output-dir $OUTPUT \
      2>&1 | tee $OUTPUT/logs/benchmark_${ds}_${task}.log
  done
done

echo -e "\n============================================================"
echo "  ALL EXPERIMENTS COMPLETED"
echo "  Results saved to: $OUTPUT/"
echo "============================================================"
