#!/bin/bash
# Full training pipeline: SFT → Collect Activations → Train SAE → Evaluate

set -e  # Exit on error

echo "=========================================="
echo "STEP 1: SFT Base Model Training (80,000 steps)"
echo "=========================================="
echo "Configuration: batch_size=96, lr=1e-3→1e-4 (cosine), weight_decay=0.1"
echo "Following the paper's exact setup for bioS training"
echo ""
python scripts/02_sft_base_model.py \
    --model_name gpt2 \
    --max_steps 80000 \
    --batch_size 96 \
    --lr 1e-3 \
    --min_lr 1e-4 \
    --warmup_steps 1000 \
    --weight_decay 0.1 \
    --adam_epsilon 1e-6 \
    --output_dir models/base_sft

echo ""
echo "=========================================="
echo "STEP 2: Collect Activations"
echo "=========================================="
python scripts/03_collect_activations.py \
    --model_path models/base_sft/final \
    --qa_file data/generated/qa_train.jsonl \
    --output_file data/activations/train_activations.pkl \
    --layer -1

echo ""
echo "=========================================="
echo "STEP 3: Train SAE with Periodic Evaluation"
echo "=========================================="

# Train SAE for multiple cycles with evaluation
for cycle in {1..4}; do
    echo ""
    echo "--- Training Cycle $cycle/4 (500 epochs) ---"
    
    start_epoch=$(( (cycle - 1) * 500 ))
    
    python scripts/04_train_sae.py \
        --activation_file data/activations/train_activations.pkl \
        --output_dir models/sae_6slot \
        --epochs 500 \
        --batch_size 256 \
        --lr 1e-3 \
        --lambda_recon 1.0 \
        --lambda_sparse 0.1 \
        --lambda_align 2.0 \
        --lambda_indep 0.5 \
        --lambda_value 1.0 \
        --resume $([ $cycle -gt 1 ] && echo "models/sae_6slot/sae_final.pt" || echo "")
    
    echo ""
    echo "--- Evaluating after $((cycle * 500)) epochs ---"
    
    python scripts/05_evaluate_sae.py \
        --sae_checkpoint models/sae_6slot/sae_final.pt \
        --lm_model models/base_sft/final \
        --train_qa data/generated/qa_train.jsonl \
        --test_qa_ood data/generated/qa_test_ood.jsonl \
        --output_dir results/sae_eval_epoch$((cycle * 500))
    
    # Print quick summary
    echo ""
    echo "Results after $((cycle * 500)) epochs:"
    cat results/sae_eval_epoch$((cycle * 500))/binding_accuracy_results.json | grep -E "(slot_binding_acc|answer_acc)" | head -6
done

echo ""
echo "=========================================="
echo "TRAINING COMPLETE! (2000 total SAE epochs)"
echo "=========================================="
echo "Check results in:"
echo "  - results/sae_eval_epoch500/"
echo "  - results/sae_eval_epoch1000/"
echo "  - results/sae_eval_epoch1500/"
echo "  - results/sae_eval_epoch2000/"
echo ""
echo "Final model:"
echo "  - models/sae_6slot/sae_final.pt"
