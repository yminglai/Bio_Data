# 1-to-1 SAE Training Pipeline for 6-Rule Biography QA

This pipeline trains a **Supervised Sparse Autoencoder (SAE)** with exactly **6 latent features**, one per rule, on a biography question-answering task.

## Rules (Features)

1. **Birth Date** (f₁)
2. **Birth City** (f₂)
3. **University** (f₃)
4. **Major/Field** (f₄)
5. **Employer** (f₅)
6. **Work City** (f₆)

## Pipeline Overview

```
Step 1: Generate Dataset
├── Create synthetic persons with 6 attributes
├── Split train/test (disjoint persons)
├── Template split: 1 for training, 3 for testing per rule
└── Output: train_kg.json, test_kg.json, qa_*.jsonl

Step 2: Fine-tune Base LLM
├── Light SFT on biography + QA pairs
├── Ensures consistent answering before probing
└── Output: models/base_sft/final/

Step 3: Collect Activations
├── Extract hidden states at answer position
├── From final layer residual stream
└── Output: data/activations/train_activations.pkl

Step 4: Train 6-Slot SAE
├── Supervised slot assignment (1 slot per rule)
├── Losses: reconstruction + sparsity + alignment + independence + value
├── Gumbel-Softmax with temperature annealing
└── Output: models/sae_6slot/sae_final.pt

Step 5: Evaluate
├── Test 1: Slot assignment (confusion matrix)
├── Test 2: Ablation (selectivity)
├── Test 3: OOD generalization
└── Output: results/sae_eval/
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python run_pipeline.py --model_name gpt2 --num_persons 1000 --sft_epochs 3 --sae_epochs 100
```

### 3. Run Individual Steps

```bash
# Step 1: Generate data
python scripts/01_generate_dataset.py

# Step 2: Fine-tune base model
python scripts/02_sft_base_model.py \
  --model_name gpt2 \
  --epochs 3 \
  --train_qa data/generated/qa_train.jsonl \
  --train_kg data/generated/train_kg.json

# Step 3: Collect activations
python scripts/03_collect_activations.py \
  --model_path models/base_sft/final \
  --train_qa data/generated/qa_train.jsonl \
  --train_kg data/generated/train_kg.json

# Step 4: Train SAE
python scripts/04_train_sae.py \
  --activations data/activations/train_activations.pkl \
  --epochs 100 \
  --batch_size 64

# Step 5: Evaluate
python scripts/05_evaluate_sae.py \
  --sae_checkpoint models/sae_6slot/sae_final.pt \
  --lm_model models/base_sft/final
```

## Data Format

### Knowledge Graph (JSON)
```json
{
  "person_id": "person_0001",
  "full_name": "Alice Johnson",
  "birth_date": "12,March,1985",
  "birth_city": "Boston",
  "university": "MIT",
  "major": "Computer Science",
  "employer": "Google",
  "work_city": "Mountain View",
  "biographies": ["...", "..."]
}
```

### QA Pairs (JSONL)
```json
{
  "person_id": "person_0001",
  "rule_idx": 0,
  "rule_name": "birth_date",
  "question": "What is Alice Johnson's birth date?",
  "answer": "12,March,1985",
  "template_idx": 0,
  "split": "train"
}
```

## SAE Architecture

```python
class SupervisedSAE:
    encoder: Linear(d_model → 6)           # Maps activations to slot logits
    decoder: Linear(6 → d_model)           # Reconstructs from slots
    value_heads: ModuleList[6]             # Predict answer from active slot
    
    forward(h, temperature):
        logits = encoder(h)
        z = gumbel_softmax(logits, tau=temperature)  # Soft one-hot
        h_recon = decoder(z)
        return z, h_recon
```

## Loss Function

```
L = λ₁·L_recon + λ₂·L_sparse + λ₃·L_align + λ₄·L_indep + λ₅·L_value

where:
  L_recon  = ||h - h_recon||²              (reconstruction)
  L_sparse = H(z)                          (entropy, want low)
  L_align  = CE(z, rule_idx)               (supervised slot selection)
  L_indep  = ||Cov(z) - I||²              (decorrelate slots)
  L_value  = CE(value_head(z), answer)     (answer prediction)
```

## Evaluation Metrics

1. **Diagonal Accuracy**: Fraction of correct slot assignments (should be ~1.0)
2. **Ablation Impact**: Selectivity of each slot (should be diagonal)
3. **OOD Accuracy**: Performance on unseen templates and persons

## Expected Results

For a successful 1-to-1 mapping:

- **Diagonal Accuracy**: ≥ 0.95
- **OOD Slot Accuracy**: ≥ 0.90
- **Ablation Matrix**: Near-diagonal (>0.8 on diagonal)

## File Structure

```
bio_data/
├── data/
│   ├── entities/              # Input: names, cities, etc.
│   ├── templates/             # Biography templates
│   ├── qa_templates/          # Question templates
│   ├── generated/             # Generated KG and QA pairs
│   └── activations/           # Collected activations
├── models/
│   ├── base_sft/              # Fine-tuned LLM
│   └── sae_6slot/             # Trained SAE
├── results/
│   └── sae_eval/              # Evaluation results
├── scripts/
│   ├── 01_generate_dataset.py
│   ├── 02_sft_base_model.py
│   ├── 03_collect_activations.py
│   ├── 04_train_sae.py
│   └── 05_evaluate_sae.py
├── run_pipeline.py            # Main runner
└── requirements.txt
```

## Hyperparameters

### SFT (Step 2)
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 8
- Max length: 512 tokens

### SAE (Step 4)
- Slots: 6
- Epochs: 100
- Learning rate: 1e-3
- Batch size: 64
- Temperature: 1.0 → 0.1 (linear anneal)
- λ_recon: 1.0
- λ_sparse: 1e-3
- λ_align: 1.0
- λ_indep: 1e-2
- λ_value: 0.5

## Customization

### Change number of persons
Edit `scripts/01_generate_dataset.py`:
```python
num_persons = 2000  # Default: 1000
```

### Change base model
```bash
python run_pipeline.py --model_name gpt2-medium
```

### Adjust SAE hyperparameters
```bash
python scripts/04_train_sae.py \
  --lambda_align 2.0 \
  --lambda_value 1.0 \
  --temp_start 2.0 \
  --temp_end 0.05
```

## Troubleshooting

**Q: Diagonal accuracy is low (<0.8)**
- Increase `lambda_align` (e.g., 2.0)
- Increase training epochs
- Lower final temperature (e.g., 0.05)

**Q: OOD accuracy is low**
- Generate more training persons
- Add more biography variants
- Ensure template diversity

**Q: Slots don't reconstruct well**
- Increase `lambda_recon`
- Check activation collection layer
- Verify model convergence in Step 2

## Citation

Based on the supervised feature-aligned SAE approach with:
- Gumbel-Softmax for differentiable discrete selection
- Multi-objective loss for 1-to-1 constraint
- Contrastive template splits for OOD evaluation
