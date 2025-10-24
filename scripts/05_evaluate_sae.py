"""
Step 5: Evaluate SAE - Verify 1-to-1 Mapping with Binding Accuracy
Tests: 
1. Question → Relation binding (does question activate correct slot?)
2. Relation → Answer binding (does slot predict correct answer?)
3. OOD generalization (in-distribution vs out-of-distribution templates)
"""
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

def normalize_date(date_str):
    """
    Normalize date strings to a common format for comparison.
    Handles formats like: "15,March,1985", "March 15, 1985", "1985-03-15", "15 March 1985"
    Returns: (day, month_name, year) tuple or None if parsing fails
    """
    date_str = date_str.strip().lower()
    
    # Try to extract day, month, year using various patterns
    # Pattern 1: "15,March,1985" or "15,march,1985"
    match = re.match(r'(\d{1,2}),\s*([a-z]+),\s*(\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return (int(day), month.lower(), int(year))
    
    # Pattern 2: "March 15, 1985" or "march 15 1985"
    match = re.match(r'([a-z]+)\s+(\d{1,2}),?\s+(\d{4})', date_str)
    if match:
        month, day, year = match.groups()
        return (int(day), month.lower(), int(year))
    
    # Pattern 3: "15 March 1985"
    match = re.match(r'(\d{1,2})\s+([a-z]+)\s+(\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return (int(day), month.lower(), int(year))
    
    # Pattern 4: "1985-03-15" or "1985/03/15"
    match = re.match(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', date_str)
    if match:
        year, month_num, day = match.groups()
        months = ['', 'january', 'february', 'march', 'april', 'may', 'june', 
                  'july', 'august', 'september', 'october', 'november', 'december']
        month = months[int(month_num)]
        return (int(day), month, int(year))
    
    return None

def compare_answers(gold_answer, gen_answer, rule_name):
    """
    Compare gold and generated answers with rule-specific handling.
    For dates: normalize and compare components.
    For others: flexible string matching.
    """
    gold_answer = gold_answer.strip().lower()
    gen_answer = gen_answer.strip().lower()
    
    # Special handling for dates
    if rule_name == 'birth_date':
        gold_date = normalize_date(gold_answer)
        gen_date = normalize_date(gen_answer)
        
        if gold_date and gen_date:
            # Compare day, month, year
            return gold_date == gen_date
        # Fallback to string matching if parsing fails
    
    # For non-dates or failed date parsing: flexible matching
    return (
        gold_answer in gen_answer or 
        gen_answer in gold_answer or
        gold_answer == gen_answer
    )

# Import SAE model
import sys
sys.path.append(str(Path(__file__).parent))
try:
    from scripts.train_sae_6slot import SupervisedSAE
except ImportError:
    # Try alternative import path
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_sae", Path(__file__).parent / "04_train_sae.py")
    train_sae_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_sae_module)
    SupervisedSAE = train_sae_module.SupervisedSAE

def load_sae(checkpoint_path, device):
    """Load trained SAE model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    d_model = checkpoint['d_model']
    n_slots = checkpoint['args']['n_slots']
    
    model = SupervisedSAE(d_model=d_model, n_slots=n_slots)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['args']

def evaluate_binding_accuracy(sae, lm_model, tokenizer, qa_file, kg_file, layer_idx=-1, split_name="test"):
    """
    Evaluate binding accuracy: Question → Relation → Answer
    Uses PURE QA format (no biography context) - knowledge retrieval mode.
    
    Returns:
        - slot_binding_acc: Does the question activate the correct slot?
        - answer_acc: Does the model generate the correct answer?
        - per_rule_metrics: Breakdown by each rule
    """
    device = next(lm_model.parameters()).device
    
    # Load data
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    results = []
    rule_names = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']
    
    # Metrics
    total = 0
    slot_correct = 0
    answer_correct = 0
    
    per_rule = defaultdict(lambda: {'total': 0, 'slot_correct': 0, 'answer_correct': 0})
    
    lm_model.eval()
    sae.eval()
    
    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc=f"Evaluating {split_name}"):
            # Pure QA format (no biography!)
            prompt = f"Q: {qa['question']}\nA:"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            # Get model output with activations
            outputs = lm_model(**inputs, output_hidden_states=True)
            
            # Extract activation from answer position
            hidden_states = outputs.hidden_states[layer_idx]
            last_position = inputs['input_ids'].shape[1] - 1
            h = hidden_states[0, last_position, :]  # [d_model]
            
            # Pass through SAE
            z, h_recon, logits = sae(h.unsqueeze(0), temperature=0.1)
            predicted_slot = logits.argmax(dim=-1).item()
            
            # Check slot binding: Does question activate correct slot?
            true_rule = qa['rule_idx']
            slot_is_correct = (predicted_slot == true_rule)
            
            # Generate answer
            generated = lm_model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(
                generated[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Clean up generated answer (remove everything after .<|endoftext|> or newline)
            if '.' in generated_text:
                generated_text = generated_text.split('.')[0].strip()
            if '\n' in generated_text:
                generated_text = generated_text.split('\n')[0].strip()
            
            # Check answer correctness (with rule-specific comparison)
            answer_is_correct = compare_answers(qa['answer'], generated_text, qa['rule_name'])
            
            # Update metrics
            total += 1
            if slot_is_correct:
                slot_correct += 1
            if answer_is_correct:
                answer_correct += 1
            
            # Per-rule metrics
            rule_name = qa['rule_name']
            per_rule[rule_name]['total'] += 1
            if slot_is_correct:
                per_rule[rule_name]['slot_correct'] += 1
            if answer_is_correct:
                per_rule[rule_name]['answer_correct'] += 1
            
            results.append({
                'question': qa['question'],
                'gold_answer': qa['answer'],
                'generated_answer': generated_text,
                'true_rule': true_rule,
                'predicted_slot': predicted_slot,
                'slot_correct': slot_is_correct,
                'answer_correct': answer_is_correct,
                'rule_name': rule_name,
            })
    
    # Calculate overall metrics
    slot_binding_acc = slot_correct / total if total > 0 else 0
    answer_acc = answer_correct / total if total > 0 else 0
    
    # Calculate per-rule metrics
    per_rule_metrics = {}
    for rule_name in rule_names:
        if per_rule[rule_name]['total'] > 0:
            per_rule_metrics[rule_name] = {
                'slot_binding_acc': per_rule[rule_name]['slot_correct'] / per_rule[rule_name]['total'],
                'answer_acc': per_rule[rule_name]['answer_correct'] / per_rule[rule_name]['total'],
                'count': per_rule[rule_name]['total']
            }
        else:
            per_rule_metrics[rule_name] = {
                'slot_binding_acc': 0.0,
                'answer_acc': 0.0,
                'count': 0
            }
    
    return {
        'slot_binding_acc': slot_binding_acc,
        'answer_acc': answer_acc,
        'total': total,
        'per_rule_metrics': per_rule_metrics,
        'detailed_results': results
    }

def test_slot_assignment(sae, lm_model, tokenizer, qa_file, kg_file, layer_idx=-1):
    """
    Test 2: Check if slots are assigned correctly to rules.
    Returns confusion matrix: [predicted_slot, true_rule]
    Uses pure QA format (no biography context).
    """
    device = next(lm_model.parameters()).device
    n_slots = sae.n_slots
    
    # Load data
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    confusion = np.zeros((n_slots, n_slots))
    
    lm_model.eval()
    sae.eval()
    
    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc="Testing slot assignment"):
            # Pure QA format (no biography!)
            prompt = f"Q: {qa['question']}\nA:"
            
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            outputs = lm_model(**inputs, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states[layer_idx]
            last_position = inputs['input_ids'].shape[1] - 1
            h = hidden_states[0, last_position, :].unsqueeze(0)
            
            _, _, logits = sae(h, temperature=0.1)
            predicted_slot = logits.argmax(dim=-1).item()
            
            confusion[predicted_slot, qa['rule_idx']] += 1
    
    # Normalize by true rule counts
    rule_counts = confusion.sum(axis=0, keepdims=True)
    confusion_norm = confusion / (rule_counts + 1e-9)
    
    return confusion, confusion_norm

def test_ood_generalization(lm_model, tokenizer, sae, qa_file, kg_file, layer_idx=-1):
    """
    Test 3: OOD generalization on unseen templates and persons.
    This just calls evaluate_binding_accuracy with OOD data.
    """
    return evaluate_binding_accuracy(sae, lm_model, tokenizer, qa_file, kg_file, layer_idx, split_name="OOD")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_checkpoint', type=str, default='models/sae_6slot/sae_final.pt')
    parser.add_argument('--lm_model', type=str, default='models/base_sft/final')
    parser.add_argument('--train_qa', type=str, default='data/generated/qa_train.jsonl',
                       help='Training QA for baseline')
    parser.add_argument('--train_kg', type=str, default='data/generated/train_kg.json')
    parser.add_argument('--test_qa_ood', type=str, default='data/generated/qa_test_ood.jsonl')
    parser.add_argument('--test_kg', type=str, default='data/generated/test_kg.json')
    parser.add_argument('--output_dir', type=str, default='results/sae_eval')
    parser.add_argument('--layer', type=int, default=-1)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load SAE
    print(f"Loading SAE from {args.sae_checkpoint}")
    sae, sae_args = load_sae(args.sae_checkpoint, device)
    
    # Load LM model
    print(f"Loading LM model from {args.lm_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model)
    lm_model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    rule_names = ['birth_date', 'birth_city', 'university', 'major', 'employer', 'work_city']
    
    # ===================================================================
    # BINDING ACCURACY EVALUATION
    # ===================================================================
    
    print("\n" + "="*60)
    print("BINDING ACCURACY EVALUATION")
    print("="*60)
    
    # Test on TRAIN set (sanity check - should be very high)
    print("\n=== Train Set (templates 0-1, train persons) ===")
    train_results = evaluate_binding_accuracy(
        sae, lm_model, tokenizer, args.train_qa, args.train_kg, args.layer, split_name="Train"
    )
    
    print(f"\nTrain Slot Binding Accuracy: {train_results['slot_binding_acc']:.3f}")
    print(f"Train Answer Accuracy: {train_results['answer_acc']:.3f}")
    print("\nPer-rule (Train):")
    for rule_name in rule_names:
        metrics = train_results['per_rule_metrics'][rule_name]
        print(f"  {rule_name:20s}: Slot={metrics['slot_binding_acc']:.3f}, Ans={metrics['answer_acc']:.3f} (n={metrics['count']})")
    
    # Test on OUT-OF-DISTRIBUTION test set (unseen templates 2-3, different persons)
    print("\n=== Out-of-Distribution Test Set (templates 2-3, test persons) ===")
    ood_results = evaluate_binding_accuracy(
        sae, lm_model, tokenizer, args.test_qa_ood, args.test_kg, args.layer, split_name="Test-OOD"
    )
    
    print(f"\nTest-OOD Slot Binding Accuracy: {ood_results['slot_binding_acc']:.3f}")
    print(f"Test-OOD Answer Accuracy: {ood_results['answer_acc']:.3f}")
    print("\nPer-rule (Test-OOD):")
    for rule_name in rule_names:
        metrics = ood_results['per_rule_metrics'][rule_name]
        print(f"  {rule_name:20s}: Slot={metrics['slot_binding_acc']:.3f}, Ans={metrics['answer_acc']:.3f} (n={metrics['count']})")
    
    # ===================================================================
    # CONFUSION MATRIX (Slot Assignment)
    # ===================================================================
    
    print("\n=== Slot Assignment Confusion Matrix (Test-ID) ===")
    confusion, confusion_norm = test_slot_assignment(
        sae, lm_model, tokenizer, args.test_qa_id, args.test_kg, args.layer
    )
    
    print("\nConfusion Matrix (rows=predicted slot, cols=true rule):")
    print(confusion_norm)
    
    diagonal_acc = np.trace(confusion_norm) / 6
    print(f"\nDiagonal accuracy: {diagonal_acc:.3f} (1.0 = perfect 1-to-1)")
    
    # ===================================================================
    # VISUALIZATIONS
    # ===================================================================
    
    print("\n=== Generating Visualizations ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Slot assignment confusion matrix
    sns.heatmap(
        confusion_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=rule_names,
        yticklabels=[f'Slot {i}' for i in range(6)],
        ax=axes[0, 0],
        cbar_kws={'label': 'Fraction'},
        vmin=0, vmax=1
    )
    axes[0, 0].set_title('Slot Assignment Confusion (Test-ID)\n(Should be diagonal for 1-to-1)')
    axes[0, 0].set_xlabel('True Rule')
    axes[0, 0].set_ylabel('Predicted Slot')
    
    # Plot 2: Binding accuracy comparison
    splits = ['Train', 'Test-OOD']
    slot_accs = [
        train_results['slot_binding_acc'],
        ood_results['slot_binding_acc']
    ]
    answer_accs = [
        train_results['answer_acc'],
        ood_results['answer_acc']
    ]
    
    x = np.arange(len(splits))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, slot_accs, width, label='Slot Binding Acc', alpha=0.8)
    axes[0, 1].bar(x + width/2, answer_accs, width, label='Answer Acc', alpha=0.8)
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Binding Accuracy: Question → Relation → Answer')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(splits)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Per-rule binding accuracy (Train)
    rule_slot_accs = [train_results['per_rule_metrics'][r]['slot_binding_acc'] for r in rule_names]
    rule_answer_accs = [train_results['per_rule_metrics'][r]['answer_acc'] for r in rule_names]
    
    x_rules = np.arange(len(rule_names))
    axes[1, 0].bar(x_rules - width/2, rule_slot_accs, width, label='Slot Binding', alpha=0.8)
    axes[1, 0].bar(x_rules + width/2, rule_answer_accs, width, label='Answer Acc', alpha=0.8)
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Per-Rule Accuracy (Train)')
    axes[1, 0].set_xticks(x_rules)
    axes[1, 0].set_xticklabels(rule_names, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1.0])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Per-rule binding accuracy (Test-OOD)
    ood_slot_accs = [ood_results['per_rule_metrics'][r]['slot_binding_acc'] for r in rule_names]
    ood_answer_accs = [ood_results['per_rule_metrics'][r]['answer_acc'] for r in rule_names]
    
    axes[1, 1].bar(x_rules - width/2, ood_slot_accs, width, label='Slot Binding', alpha=0.8)
    axes[1, 1].bar(x_rules + width/2, ood_answer_accs, width, label='Answer Acc', alpha=0.8)
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Per-Rule Accuracy (Test-OOD)')
    axes[1, 1].set_xticks(x_rules)
    axes[1, 1].set_xticklabels(rule_names, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1.0])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'binding_accuracy_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_dir / 'binding_accuracy_evaluation.png'}")
    plt.close()
    
    # ===================================================================
    # SAVE RESULTS
    # ===================================================================
    
    results = {
        'train': {
            'slot_binding_acc': float(train_results['slot_binding_acc']),
            'answer_acc': float(train_results['answer_acc']),
            'per_rule': {r: {k: float(v) for k, v in train_results['per_rule_metrics'][r].items()} 
                        for r in rule_names}
        },
        'test_ood': {
            'slot_binding_acc': float(ood_results['slot_binding_acc']),
            'answer_acc': float(ood_results['answer_acc']),
            'per_rule': {r: {k: float(v) for k, v in ood_results['per_rule_metrics'][r].items()} 
                        for r in rule_names}
        },
        'diagonal_accuracy': float(diagonal_acc),
        'confusion_matrix': confusion_norm.tolist(),
    }
    
    with open(output_dir / 'binding_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save sample predictions
    sample_predictions = {
        'train_samples': train_results['detailed_results'][:20],
        'test_ood_samples': ood_results['detailed_results'][:20]
    }
    
    with open(output_dir / 'sample_predictions.json', 'w') as f:
        json.dump(sample_predictions, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    
    # ===================================================================
    # SUMMARY
    # ===================================================================
    
    print("\n" + "="*60)
    print("SUMMARY: 1-to-1 SAE BINDING ACCURACY")
    print("="*60)
    print(f"\nQuestion → Relation Binding (Slot Activation):")
    print(f"  Train:    {train_results['slot_binding_acc']:.3f}")
    print(f"  Test-OOD: {ood_results['slot_binding_acc']:.3f}")
    
    print(f"\nRelation → Answer Binding (Answer Generation):")
    print(f"  Train:    {train_results['answer_acc']:.3f}")
    print(f"  Test-OOD: {ood_results['answer_acc']:.3f}")
    
    print(f"\nDiagonal Accuracy (1-to-1 mapping): {diagonal_acc:.3f}")
    
    # Success criteria
    success = (
        train_results['slot_binding_acc'] >= 0.85 and
        ood_results['slot_binding_acc'] >= 0.75 and
        diagonal_acc >= 0.85
    )
    
    print(f"\nOverall Assessment: {'✅ SUCCESSFUL' if success else '⚠️  NEEDS IMPROVEMENT'}")
    
    if not success:
        print("\nSuggestions:")
        if train_results['slot_binding_acc'] < 0.85:
            print("  - Increase lambda_align in SAE training")
            print("  - Train SAE for more epochs")
        if ood_results['slot_binding_acc'] < 0.75:
            print("  - Add more diverse question templates")
            print("  - Increase training data size")
        if diagonal_acc < 0.85:
            print("  - Check for slot collapse (multiple rules → same slot)")
            print("  - Increase lambda_indep to decorrelate slots")
    
    print("="*60)

if __name__ == "__main__":
    main()
