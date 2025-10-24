"""
Main Pipeline Runner
Executes all steps of the 1-to-1 SAE training pipeline.
"""
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        print(f"Command failed with return code {result.returncode}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run 1-to-1 SAE training pipeline")
    parser.add_argument('--skip_data', action='store_true', help='Skip data generation')
    parser.add_argument('--skip_sft', action='store_true', help='Skip base model SFT')
    parser.add_argument('--skip_activations', action='store_true', help='Skip activation collection')
    parser.add_argument('--skip_sae', action='store_true', help='Skip SAE training')
    parser.add_argument('--skip_eval', action='store_true', help='Skip evaluation')
    
    parser.add_argument('--model_name', type=str, default='gpt2', help='Base model name')
    parser.add_argument('--num_persons', type=int, default=1000, help='Number of persons to generate')
    parser.add_argument('--sft_epochs', type=int, default=3, help='SFT training epochs')
    parser.add_argument('--sae_epochs', type=int, default=100, help='SAE training epochs')
    
    args = parser.parse_args()
    
    scripts_dir = Path("scripts")
    
    # Step 1: Generate Dataset
    if not args.skip_data:
        if not run_command(
            ["python", str(scripts_dir / "01_generate_dataset.py")],
            "Step 1: Generate Dataset with Train/Test Split"
        ):
            return
    
    # Step 2: Fine-tune Base Model
    if not args.skip_sft:
        if not run_command(
            [
                "python", str(scripts_dir / "02_sft_base_model.py"),
                "--model_name", args.model_name,
                "--epochs", str(args.sft_epochs),
                "--train_qa", "data/generated/qa_train.jsonl",
                "--train_kg", "data/generated/train_kg.json",
                "--output_dir", "models/base_sft"
            ],
            "Step 2: Fine-tune Base LLM on Biography QA"
        ):
            return
    
    # Step 3: Collect Activations
    if not args.skip_activations:
        if not run_command(
            [
                "python", str(scripts_dir / "03_collect_activations.py"),
                "--model_path", "models/base_sft/final",
                "--train_qa", "data/generated/qa_train.jsonl",
                "--train_kg", "data/generated/train_kg.json",
                "--output", "data/activations/train_activations.pkl"
            ],
            "Step 3: Collect Activations from SFT Model"
        ):
            return
    
    # Step 4: Train SAE
    if not args.skip_sae:
        if not run_command(
            [
                "python", str(scripts_dir / "04_train_sae.py"),
                "--activations", "data/activations/train_activations.pkl",
                "--output_dir", "models/sae_6slot",
                "--epochs", str(args.sae_epochs),
                "--batch_size", "64",
                "--lr", "1e-3"
            ],
            "Step 4: Train Supervised 6-Slot SAE"
        ):
            return
    
    # Step 5: Evaluate
    if not args.skip_eval:
        if not run_command(
            [
                "python", str(scripts_dir / "05_evaluate_sae.py"),
                "--sae_checkpoint", "models/sae_6slot/sae_final.pt",
                "--lm_model", "models/base_sft/final",
                "--test_qa_id", "data/generated/qa_test_id.jsonl",
                "--test_qa_ood", "data/generated/qa_test_ood.jsonl",
                "--test_kg", "data/generated/test_kg.json",
                "--output_dir", "results/sae_eval"
            ],
            "Step 5: Evaluate SAE (Selectivity, Substitution, OOD)"
        ):
            return
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETE!")
    print("="*60)
    print("\nResults available in:")
    print("  - models/base_sft/final/          (Fine-tuned LLM)")
    print("  - models/sae_6slot/sae_final.pt   (Trained SAE)")
    print("  - results/sae_eval/               (Evaluation results)")
    print("\nCheck results/sae_eval/sae_evaluation.png for visualizations")
    print("="*60)

if __name__ == "__main__":
    main()
