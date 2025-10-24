"""
Preview Training and Testing Data Samples
Generates .txt files showing what the model will see during training and testing.
"""
import json
from pathlib import Path

def main():
    data_dir = Path("data/generated")
    output_dir = Path("data/preview")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    with open(data_dir / "train_kg.json", 'r') as f:
        persons = json.load(f)
    
    with open(data_dir / "qa_train.jsonl", 'r') as f:
        train_qa = [json.loads(line) for line in f]
    
    with open(data_dir / "qa_test_ood.jsonl", 'r') as f:
        test_qa = [json.loads(line) for line in f]
    
    # ===================================================================
    # TRAINING DATA PREVIEW
    # ===================================================================
    
    with open(output_dir / "training_samples.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("TRAINING DATA SAMPLES (what model learns from)\n")
        f.write("="*70 + "\n\n")
        
        # Show first 5 persons' biographies (memorization phase)
        f.write("PHASE 1: BIOGRAPHY MEMORIZATION\n")
        f.write("-"*70 + "\n")
        f.write("Format: {biography}<|endoftext|>\n")
        f.write("Purpose: Model learns to store factual information in memory\n")
        f.write("-"*70 + "\n\n")
        
        for i, person in enumerate(persons[:5]):
            f.write(f"[Example {i+1}]\n")
            f.write(f"{person['biographies'][0]}<|endoftext|>\n\n")
        
        f.write("\n" + "="*70 + "\n\n")
        
        # Show QA training samples (retrieval phase)
        f.write("PHASE 2: QA TRAINING (Templates 0-1)\n")
        f.write("-"*70 + "\n")
        f.write("Format: Q: {question}\\nA: {answer}<|endoftext|>\n")
        f.write("(Note: \\n represents a real newline character in training)\n")
        f.write("Purpose: Model learns to retrieve stored facts via questions\n")
        f.write("-"*70 + "\n\n")
        
        # Group by rule to show template variety
        rules = ["birth_date", "birth_city", "university", "major", "employer", "company_city"]
        
        for rule in rules:
            rule_samples = [qa for qa in train_qa if qa['rule_name'] == rule][:3]
            f.write(f"\n--- Rule: {rule} ---\n\n")
            
            for i, qa in enumerate(rule_samples):
                f.write(f"[{rule} - Template {qa['template_idx']}]\n")
                f.write(f"Q: {qa['question']}\n")
                f.write(f"A: {qa['answer']}<|endoftext|>\n\n")
    
    print(f"✅ Training samples saved to: {output_dir / 'training_samples.txt'}")
    
    # ===================================================================
    # TESTING DATA PREVIEW
    # ===================================================================
    
    with open(output_dir / "testing_samples.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("TESTING DATA SAMPLES (OOD - what model is evaluated on)\n")
        f.write("="*70 + "\n\n")
        
        f.write("TESTING FORMAT (Templates 2-3, same persons)\n")
        f.write("-"*70 + "\n")
        f.write("Input:  Q: {question}\\nA:\n")
        f.write("Expected Output: {answer}\n")
        f.write("Purpose: Test if model can recall facts with UNSEEN question phrasings\n")
        f.write("-"*70 + "\n\n")
        
        # Group by rule to show OOD template variety
        for rule in rules:
            rule_samples = [qa for qa in test_qa if qa['rule_name'] == rule][:3]
            f.write(f"\n--- Rule: {rule} ---\n\n")
            
            for i, qa in enumerate(rule_samples):
                f.write(f"[{rule} - Template {qa['template_idx']} - OOD]\n")
                f.write(f"Person: {qa['full_name']}\n")
                f.write(f"Input:  Q: {qa['question']}\\nA:\n")
                f.write(f"Expected: {qa['answer']}\n\n")
    
    print(f"✅ Testing samples saved to: {output_dir / 'testing_samples.txt'}")
    
    # ===================================================================
    # STATISTICS SUMMARY
    # ===================================================================
    
    with open(output_dir / "data_statistics.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("DATA STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Persons: {len(persons)}\n")
        f.write(f"  - All persons used for both training and testing\n")
        f.write(f"  - Training: templates 0-1\n")
        f.write(f"  - Testing: templates 2-3 (OOD)\n\n")
        
        f.write(f"Training Data:\n")
        f.write(f"  - Biographies: {len(persons)} (1 per person)\n")
        f.write(f"  - QA pairs: {len(train_qa)}\n")
        f.write(f"  - Total training instances: {len(persons) + len(train_qa)}\n\n")
        
        f.write(f"Testing Data:\n")
        f.write(f"  - QA pairs (OOD): {len(test_qa)}\n\n")
        
        f.write("Per-Rule Breakdown:\n")
        for rule in rules:
            train_count = len([qa for qa in train_qa if qa['rule_name'] == rule])
            test_count = len([qa for qa in test_qa if qa['rule_name'] == rule])
            f.write(f"  {rule:20s}: Train={train_count:5d}, Test={test_count:5d}\n")
        
        f.write("\nTemplate Distribution:\n")
        f.write("  Training templates: 0, 1 (seen during training)\n")
        f.write("  Testing templates:  2, 3 (unseen - OOD generalization)\n\n")
        
        # Show template examples for one rule
        f.write("Example Template Variance (birth_date):\n")
        birth_date_samples = [qa for qa in train_qa + test_qa if qa['rule_name'] == 'birth_date']
        templates_by_idx = {}
        for qa in birth_date_samples:
            template_idx = qa['template_idx']
            if template_idx not in templates_by_idx:
                # Extract just the question pattern
                templates_by_idx[template_idx] = qa['question'].replace(qa['full_name'], '{NAME}')
        
        for idx in sorted(templates_by_idx.keys()):
            split = "TRAIN" if idx < 2 else "TEST-OOD"
            f.write(f"  Template {idx} [{split}]: {templates_by_idx[idx]}\n")
    
    print(f"✅ Statistics saved to: {output_dir / 'data_statistics.txt'}")
    
    print("\n" + "="*70)
    print("Preview files generated successfully!")
    print("="*70)
    print(f"\nReview these files in {output_dir}/:")
    print("  1. training_samples.txt   - What model learns from")
    print("  2. testing_samples.txt    - What model is evaluated on")
    print("  3. data_statistics.txt    - Overview of data distribution")

if __name__ == "__main__":
    main()
