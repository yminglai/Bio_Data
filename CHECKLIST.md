# Pre-Flight Checklist

Before running the full pipeline, verify:

## ✅ Dependencies
- [ ] Python 3.8+ installed
- [ ] PyTorch installed (`pip install torch`)
- [ ] Transformers installed (`pip install transformers`)
- [ ] All requirements: `pip install -r requirements.txt`

## ✅ Data Files

### Question Templates (need 4 each)
- [ ] `data/qa_templates/birth_date_questions.txt` (4 templates)
- [ ] `data/qa_templates/birth_city_questions.txt` (4 templates)
- [ ] `data/qa_templates/university_questions.txt` (4 templates)
- [ ] `data/qa_templates/major_questions.txt` (4 templates)
- [ ] `data/qa_templates/employer_questions.txt` (4 templates)
- [ ] `data/qa_templates/company_city_questions.txt` (4 templates)

**Run if missing 4th template**: `python scripts/add_fourth_template.py`

### Entity Files
- [ ] `data/entities/first_names.txt`
- [ ] `data/entities/middle_names.txt`
- [ ] `data/entities/last_names.txt`
- [ ] `data/entities/birth_cities.txt`
- [ ] `data/entities/universities.txt`
- [ ] `data/entities/major_names.txt`
- [ ] `data/entities/companies.csv`
- [ ] `data/entities/work_cities.txt`

### Biography Templates
- [ ] `data/templates/birth_date_templates.txt`
- [ ] `data/templates/birth_city_templates.txt`
- [ ] `data/templates/university_templates.txt`
- [ ] `data/templates/major_templates.txt`
- [ ] `data/templates/employer_templates.txt`
- [ ] `data/templates/company_city_templates.txt`

## ✅ Scripts
- [ ] `scripts/01_generate_dataset.py`
- [ ] `scripts/02_sft_base_model.py`
- [ ] `scripts/03_collect_activations.py`
- [ ] `scripts/04_train_sae.py`
- [ ] `scripts/05_evaluate_sae.py`
- [ ] `scripts/add_fourth_template.py`

## ✅ Test Setup
```bash
python test_setup.py
```

Should see all ✓ checks pass.

## 🚀 Ready to Run!

### Option 1: Full Pipeline (Recommended)
```bash
python run_pipeline.py
```

### Option 2: Step by Step
```bash
# Step 1: Generate dataset (1000 persons)
python scripts/01_generate_dataset.py

# Step 2: Fine-tune base model (3 epochs)
python scripts/02_sft_base_model.py --model_name gpt2 --epochs 3

# Step 3: Collect activations
python scripts/03_collect_activations.py

# Step 4: Train SAE (100 epochs)
python scripts/04_train_sae.py --epochs 100

# Step 5: Evaluate binding accuracy ⭐
python scripts/05_evaluate_sae.py
```

### Option 3: Quick Test (Small Dataset)
```bash
bash quick_test.sh
```

## 📊 Expected Timeline

| Step | Time (CPU) | Time (GPU) | Disk Space |
|------|------------|------------|------------|
| 1. Generate Dataset | 2 min | 2 min | ~50 MB |
| 2. SFT Base Model | 4-6 hours | 30-60 min | ~2 GB |
| 3. Collect Activations | 30 min | 10 min | ~500 MB |
| 4. Train SAE | 2-3 hours | 15-30 min | ~100 MB |
| 5. Evaluate | 30 min | 10 min | ~50 MB |
| **Total** | **~8 hours** | **~2 hours** | **~3 GB** |

## 🎯 Success Indicators

After running, check `results/sae_eval/binding_accuracy_results.json`:

### Must Have
- [x] `test_ood.slot_binding_acc` ≥ 0.75
- [x] `diagonal_accuracy` ≥ 0.85
- [x] All per-rule OOD accuracies ≥ 0.70

### Great to Have
- [x] `test_ood.slot_binding_acc` ≥ 0.85
- [x] `test_ood.answer_acc` ≥ 0.80
- [x] `diagonal_accuracy` ≥ 0.90

## ⚠️ Troubleshooting

### Low OOD Accuracy (<0.70)
```bash
# Increase alignment supervision
python scripts/04_train_sae.py --lambda_align 2.0 --epochs 150
```

### Slot Collapse (same slot for multiple rules)
```bash
# Increase independence loss
python scripts/04_train_sae.py --lambda_indep 0.05 --epochs 150
```

### Poor Reconstruction
```bash
# Increase reconstruction weight
python scripts/04_train_sae.py --lambda_recon 2.0
```

### Out of Memory
```bash
# Reduce batch size
python scripts/02_sft_base_model.py --batch_size 4
python scripts/04_train_sae.py --batch_size 32
```

## 📖 Documentation

Read in this order:
1. `README.md` - Overview
2. `OOD_TEMPLATE_TEST.md` - Understand OOD test ⭐
3. `BINDING_ACCURACY.md` - Understand metrics
4. `PIPELINE_README.md` - Detailed pipeline guide
5. `SUMMARY.txt` - Quick reference

## 🎓 Key Concepts

Before running, understand:
- **1-to-1 mapping**: Each slot = exactly one semantic rule
- **Binding accuracy**: Question → Slot → Answer
- **OOD generalization**: Works on unseen phrasings
- **Template split**: 1 train, 3 test per rule

## 🏁 Final Check

```bash
# Quick verification
python -c "
import torch
import transformers
print('✅ PyTorch:', torch.__version__)
print('✅ Transformers:', transformers.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
"
```

You're ready! Run `python run_pipeline.py` 🚀
