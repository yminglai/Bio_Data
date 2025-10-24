"""
Sweep SFT checkpoints: For each checkpoint, collect activations (layer 1), train SAE, and evaluate on ID and OOD.
"""
import os
import subprocess
from pathlib import Path

# List of SFT checkpoints to evaluate
CHECKPOINTS = [
    f"models/base_sft/checkpoint-step-{step}" for step in range(10000, 90000, 10000)
] + ["models/base_sft/final"]

LAYER = 1  # Use layer 1 for SAE

# Paths
ID_QA = "data/generated/qa_train.jsonl"
OOD_QA = "data/generated/qa_test_ood.jsonl"

ID_ACTIV = "data/activations/train_activations_id_layer1.pkl"
OOD_ACTIV = "data/activations/test_activations_ood_layer1.pkl"

RESULTS_DIR = Path("results/sae_sweep_checkpoints")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

for ckpt in CHECKPOINTS:
    tag = Path(ckpt).name.replace("checkpoint-step-", "step") if "checkpoint" in ckpt else "final"
    print(f"\n=== Evaluating SFT checkpoint: {ckpt} (layer {LAYER}) ===")
    outdir = RESULTS_DIR / tag
    outdir.mkdir(exist_ok=True)

    # 1. Collect activations for ID
    subprocess.run([
        "python", "scripts/03_collect_activations.py",
        "--model_path", ckpt,
        "--train_qa", ID_QA,
        "--output", ID_ACTIV,
        "--layer", str(LAYER)
    ], check=True)

    # 2. Collect activations for OOD
    subprocess.run([
        "python", "scripts/03_collect_activations.py",
        "--model_path", ckpt,
        "--train_qa", OOD_QA,
        "--output", OOD_ACTIV,
        "--layer", str(LAYER)
    ], check=True)

    # 3. Train SAE on ID activations
    subprocess.run([
        "python", "scripts/04_train_sae.py",
        "--activation_file", ID_ACTIV,
        "--output_dir", str(outdir / "sae_6slot"),
        "--epochs", "1000",
        "--batch_size", "256",
        "--lr", "1e-3",
        "--lambda_recon", "1.0",
        "--lambda_sparse", "0.1",
        "--lambda_align", "2.0",
        "--lambda_indep", "0.5",
        "--lambda_value", "1.0"
    ], check=True)

    # 4. Evaluate SAE on both ID and OOD
    subprocess.run([
        "python", "scripts/05_evaluate_sae.py",
        "--sae_checkpoint", str(outdir / "sae_6slot/sae_final.pt"),
        "--lm_model", ckpt,
        "--train_qa", ID_QA,
        "--test_qa_ood", OOD_QA,
        "--output_dir", str(outdir / "eval")
    ], check=True)

    print(f"Results for {tag} saved in {outdir}/eval/")

print("\nAll checkpoints evaluated!")
