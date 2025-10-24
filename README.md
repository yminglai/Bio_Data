# 1-to-1 SAE: Supervised Sparse Autoencoder with Perfect Feature Binding

Physics of Language Models: Part 3.1, Knowledge Storage and Extraction

This repository implements a **supervised sparse autoencoder (SAE)** that achieves **perfect 1-to-1 mapping** between semantic concepts and latent features for biography question-answering.

## 🎯 Goal

Train an SAE with **exactly 6 latent features**, one per rule:
1. **Birth Date** (f₁)
2. **Birth City** (f₂)
3. **University** (f₃)
4. **Major/Field** (f₄)
5. **Employer** (f₅)
6. **Work City** (f₆)

Each feature must:
- ✅ Activate ONLY for its corresponding question type
- ✅ Contain sufficient information to answer the question
- ✅ Generalize to unseen question phrasings (OOD templates)

## 📋 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add 4th question template (if needed)
python scripts/add_fourth_template.py

# 3. Run full pipeline
python run_pipeline.py

# Or run steps individually:
python scripts/01_generate_dataset.py      # Generate data
python scripts/02_sft_base_model.py        # Fine-tune LLM
python scripts/03_collect_activations.py   # Collect activations
python scripts/04_train_sae.py             # Train SAE
python scripts/05_evaluate_sae.py          # Evaluate binding accuracy
```

## 🔬 Key Innovation: Binding Accuracy

Traditional SAE evaluation only checks reconstruction and sparsity. We introduce **binding accuracy**:

### Question → Relation Binding
Does a question activate the correct feature?
```
Question: "On what date was Alice born?"  (OOD template!)
Expected: Activate birth_date feature (slot 0)
```

### Relation → Answer Binding
Does the activated feature lead to the correct answer?
```
Slot 0 activated → Should generate correct birth date
```

### OOD Generalization
Does it work on UNSEEN question phrasings?
```
Train:    "What is {NAME}'s birth date?"
Test-OOD: "On what date was {NAME} born?"  ← NEW PHRASING
Expected: STILL activate slot 0
```

**This tests if the SAE learned semantic concepts, not template matching!**

## 📊 Evaluation Metrics

### Success Criteria
- **Train Slot Binding**: ≥ 0.95 (sanity check)
- **Test-ID Slot Binding**: ≥ 0.85 (generalization to new persons)
- **Test-OOD Slot Binding**: ≥ 0.75 (generalization to new phrasings) ⭐
- **Diagonal Accuracy**: ≥ 0.85 (1-to-1 mapping quality)

### Example Output
```
Question → Relation Binding (Slot Activation):
  Train:    0.98  ✅
  Test-ID:  0.92  ✅
  Test-OOD: 0.87  ✅  ← Generalizes to new phrasings!

Relation → Answer Binding (Answer Generation):
  Train:    0.95  ✅
  Test-ID:  0.89  ✅
  Test-OOD: 0.82  ✅

Diagonal Accuracy: 0.94  ✅
```

## 📁 Documentation

- **[PIPELINE_README.md](PIPELINE_README.md)**: Complete pipeline overview
- **[BINDING_ACCURACY.md](BINDING_ACCURACY.md)**: Binding accuracy evaluation details
- **[OOD_TEMPLATE_TEST.md](OOD_TEMPLATE_TEST.md)**: OOD generalization explained
- **[run_all.sh](run_all.sh)**: Automated full pipeline script

## 🎯 What Makes This 1-to-1 SAE Special?

### The OOD Template Test

The critical test: **Does each feature capture a semantic concept or just memorize templates?**

```
Rule: Birth Date

Training Template:
  "What is {NAME}'s birth date?"  → Slot 0 activates

OOD Test Templates (NEVER SEEN):
  "When was {NAME} born?"                        → Should activate Slot 0 ✅
  "Can you tell me the birth date of {NAME}?"   → Should activate Slot 0 ✅
  "On what date was {NAME} born?"               → Should activate Slot 0 ✅
```

If OOD accuracy ≥ 0.75, the SAE learned **semantic concepts**, not patterns!

## 🚀 Output Files

After running the pipeline:
```
models/
├── base_sft/final/              # Fine-tuned LLM
└── sae_6slot/sae_final.pt       # Trained SAE

results/sae_eval/
├── binding_accuracy_results.json      # All metrics
├── binding_accuracy_evaluation.png    # 4-panel visualization
└── sample_predictions.json            # Example predictions
```

## 📝 License

MIT License
