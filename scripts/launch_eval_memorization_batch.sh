#!/bin/bash
# Batch launcher for eval_memorization.py: runs each checkpoint on a separate GPU

mkdir -p results/memorization_eval

# List of checkpoints (step-10000 ... step-80000, final)
CHECKPOINTS=(
  checkpoint-step-10000
  checkpoint-step-20000
  checkpoint-step-30000
  checkpoint-step-40000
  checkpoint-step-50000
  checkpoint-step-60000
  checkpoint-step-70000
  checkpoint-step-80000
  final
)

# List of GPUs to use (edit as needed)
GPUS=(0 1 2 3)

N_GPUS=${#GPUS[@]}

for i in "${!CHECKPOINTS[@]}"; do
  CKPT=${CHECKPOINTS[$i]}
  GPU=${GPUS[$((i%N_GPUS))]}
  echo "Launching $CKPT on GPU $GPU..."
  CUDA_VISIBLE_DEVICES=$GPU nohup python scripts/eval_memorization.py --only_checkpoint $CKPT > results/memorization_eval/eval_${CKPT}.log 2>&1 &
done

echo "All jobs launched. Monitor logs in results/memorization_eval/"
