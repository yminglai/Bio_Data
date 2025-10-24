# Binding Accuracy Evaluation for 1-to-1 SAE

## What is Binding Accuracy?

**Binding accuracy** measures whether the SAE achieves a true **1-to-1 mapping** between:
1. **Questions → Relations (Slots)**: When you see a question, does it activate the correct slot?
2. **Relations (Slots) → Answers**: Does the activated slot lead to the correct answer?

This is the key test for whether each SAE latent is truly specialized to exactly one rule.

---

## Two-Way Binding Test

### 1. Question → Relation Binding (Slot Activation)

**Question**: When the model sees a question, does it activate the correct slot?

**Example**:
```
Question: "What is Alice Johnson's birth date?"
True Rule: birth_date (rule_idx=0)
Expected: Slot 0 should activate
```

**Metric**: **Slot Binding Accuracy**
- Perfect 1-to-1: Each question type should consistently activate its corresponding slot
- Train: Should be very high (>0.95)
- Test-ID: Should be high (>0.85)
- Test-OOD: Should remain high (>0.75) even with unseen templates

### 2. Relation → Answer Binding (Answer Generation)

**Question**: Does the activated slot contain the right information to answer?

**Example**:
```
Slot 0 activated → Should lead to correct birth date answer
```

**Metric**: **Answer Accuracy**
- Measures if the model generates the correct answer
- Tests both slot selection AND value prediction

---

## Evaluation Splits

### Train Set
- **Purpose**: Sanity check
- **Data**: Same persons and templates used in training
- **Expected**: Very high accuracy (>0.95)
- **Interpretation**: If low, SAE training failed

### Test-ID (In-Distribution)
- **Purpose**: Generalization to new persons
- **Data**: 
  - ✅ New persons (never seen in training)
  - ✅ Same template style (template[1] from each rule)
- **Expected**: High accuracy (>0.85)
- **Interpretation**: Tests if SAE learned the rule, not just memorized persons

### Test-OOD (Out-of-Distribution)
- **Purpose**: Generalization to new question phrasings
- **Data**:
  - ✅ New persons (never seen in training)
  - ✅ Different templates (template[2], template[3] - unseen phrasings)
- **Expected**: Moderate-high accuracy (>0.75)
- **Interpretation**: Tests if SAE truly understands the rule concept, not just template matching

---

## Template Split Strategy

Each rule has **4 question templates**:

```
Template 0: TRAIN
  "What is {FULL_NAME}'s birth date?"

Template 1: TEST-ID (in-distribution)
  "When was {FULL_NAME} born?"

Template 2: TEST-OOD (out-of-distribution)
  "Can you tell me the birth date of {FULL_NAME}?"

Template 3: TEST-OOD (out-of-distribution)
  "On what date was {FULL_NAME} born?"
```

**Key**: Training uses ONLY template 0. Testing uses templates 1-3 on different persons.

---

## Example Results (Target)

### Successful 1-to-1 Mapping

```
Question → Relation Binding (Slot Activation):
  Train:    0.98  ✅
  Test-ID:  0.92  ✅
  Test-OOD: 0.87  ✅

Relation → Answer Binding (Answer Generation):
  Train:    0.95  ✅
  Test-ID:  0.89  ✅
  Test-OOD: 0.82  ✅

Diagonal Accuracy: 0.94  ✅
```

**Interpretation**: 
- ✅ Questions consistently activate the correct slot
- ✅ Slots contain the right information
- ✅ Generalization to unseen templates works
- ✅ TRUE 1-to-1 mapping achieved

### Failed Mapping (Needs Improvement)

```
Question → Relation Binding (Slot Activation):
  Train:    0.85  ⚠️
  Test-ID:  0.62  ❌
  Test-OOD: 0.45  ❌

Relation → Answer Binding (Answer Generation):
  Train:    0.78  ⚠️
  Test-ID:  0.55  ❌
  Test-OOD: 0.38  ❌

Diagonal Accuracy: 0.68  ❌
```

**Interpretation**:
- ❌ Slots are not consistently assigned to rules
- ❌ Multiple rules might map to the same slot (collapse)
- ❌ Poor generalization to new templates

**Fixes**:
1. Increase `lambda_align` (stronger supervision)
2. Increase `lambda_indep` (decorrelate slots)
3. Train longer (more epochs)
4. Add more diverse templates

---

## Per-Rule Analysis

The evaluation also breaks down accuracy **per rule**:

```
Per-Rule Accuracy (Test-OOD):
  birth_date    : Slot=0.95, Ans=0.88  ✅
  birth_city    : Slot=0.91, Ans=0.85  ✅
  university    : Slot=0.89, Ans=0.83  ✅
  major         : Slot=0.87, Ans=0.81  ✅
  employer      : Slot=0.86, Ans=0.79  ✅
  work_city     : Slot=0.84, Ans=0.78  ✅
```

**What to look for**:
- All rules should have similar accuracy (balanced)
- If one rule is much lower, it may be collapsed with another
- Slot accuracy should always be ≥ Answer accuracy

---

## Confusion Matrix

The **slot assignment confusion matrix** shows how often each slot is activated for each rule:

```
                 birth_date  birth_city  university  major  employer  work_city
Slot 0 (birth_date)   0.95       0.02       0.01     0.01     0.00      0.01
Slot 1 (birth_city)   0.02       0.93       0.00     0.02     0.01      0.02
Slot 2 (university)   0.01       0.01       0.94     0.02     0.01      0.01
Slot 3 (major)        0.01       0.02       0.02     0.92     0.02      0.01
Slot 4 (employer)     0.00       0.01       0.01     0.02     0.94      0.01
Slot 5 (work_city)    0.01       0.01       0.02     0.01     0.02      0.94
```

**Perfect 1-to-1**: Diagonal values close to 1.0, off-diagonal close to 0.0

**Diagonal Accuracy**: Average of diagonal = (0.95+0.93+0.94+0.92+0.94+0.94)/6 = 0.94

---

## Visualizations

The evaluation generates 4 plots:

### 1. Slot Assignment Confusion Matrix
- Heatmap showing slot → rule mapping
- Should be diagonal

### 2. Binding Accuracy Comparison
- Bar chart comparing Train, Test-ID, Test-OOD
- Both slot and answer accuracy

### 3. Per-Rule Accuracy (Test-ID)
- Which rules work well vs. struggle
- Identifies imbalanced slots

### 4. Per-Rule Accuracy (Test-OOD)
- OOD generalization per rule
- Critical for real 1-to-1 binding

---

## Running the Evaluation

```bash
python scripts/05_evaluate_sae.py \
  --sae_checkpoint models/sae_6slot/sae_final.pt \
  --lm_model models/base_sft/final \
  --train_qa data/generated/qa_train.jsonl \
  --train_kg data/generated/train_kg.json \
  --test_qa_id data/generated/qa_test_id.jsonl \
  --test_qa_ood data/generated/qa_test_ood.jsonl \
  --test_kg data/generated/test_kg.json \
  --output_dir results/sae_eval
```

**Outputs**:
- `binding_accuracy_results.json`: All metrics
- `binding_accuracy_evaluation.png`: 4-panel visualization
- `sample_predictions.json`: Example predictions for inspection

---

## Success Criteria

✅ **Successful 1-to-1 SAE**:
- Train slot binding ≥ 0.95
- Test-ID slot binding ≥ 0.85
- Test-OOD slot binding ≥ 0.75
- Diagonal accuracy ≥ 0.85
- All per-rule accuracies > 0.70

⚠️ **Needs Improvement**:
- Test-ID slot binding < 0.85
- Test-OOD slot binding < 0.75
- Diagonal accuracy < 0.85
- Large variance in per-rule accuracy

❌ **Failed**:
- Test-ID slot binding < 0.70
- Test-OOD slot binding < 0.50
- Diagonal accuracy < 0.70

---

## Why Binding Accuracy Matters

Traditional SAE evaluation only checks:
- Reconstruction error
- Sparsity

But for **interpretable, steerable** SAEs, we need:
- **Selectivity**: Each slot responds to exactly one concept
- **Sufficiency**: Each slot contains enough information to answer
- **Generalization**: Works on unseen data

**Binding accuracy** directly measures all three!
