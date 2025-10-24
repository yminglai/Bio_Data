# OOD Template Test: Does the SAE Learn the Semantic Concept?

## The Core Question

**When you see a NEW PHRASING of a "birth date" question, does it still activate the "birth date" feature?**

This tests whether the SAE learned the **semantic concept** of each rule, rather than just memorizing exact template strings.

---

## Example: Birth Date Rule

### Training (Template 0)
The model is trained ONLY on this phrasing:
```
"What is {FULL_NAME}'s birth date?"
```

### Test In-Distribution (Template 1)
Similar phrasing, slight variation:
```
"When was {FULL_NAME} born?"
```
✅ **Should activate birth_date feature** (rule_idx=0, Slot 0)

### Test OOD (Template 2)
Different phrasing:
```
"Can you tell me the birth date of {FULL_NAME}?"
```
✅ **Should STILL activate birth_date feature** if SAE learned the concept

### Test OOD (Template 3)
Yet another phrasing:
```
"On what date was {FULL_NAME} born?"
```
✅ **Should STILL activate birth_date feature** if SAE learned the concept

---

## What Makes This "Out-of-Distribution"?

### In-Distribution (ID)
- Template style similar to training
- Model has seen this kind of phrasing structure
- Example: "What is X's Y?" → "When was X born?" (same question structure)

### Out-of-Distribution (OOD)
- **Different syntactic structure**
- **Different word order**
- **Different phrasing style**
- Model has NEVER seen this exact phrasing during training

Examples of OOD variations:
- "Can you tell me..." (indirect question)
- "On what date..." (prepositional phrase first)
- "In which city..." (different question word)
- "What company does..." (different verb structure)

---

## Why This Tests True 1-to-1 Binding

### ❌ Template Memorization (BAD)
If the SAE just memorizes templates:
- Train template: "What is X's birth date?" → Slot 0 ✓
- OOD template: "On what date was X born?" → Random slot ✗
- **Result**: Low OOD accuracy, failed 1-to-1

### ✅ Semantic Concept Learning (GOOD)
If the SAE learns the concept of "asking about birth date":
- Train template: "What is X's birth date?" → Slot 0 ✓
- OOD template: "On what date was X born?" → Slot 0 ✓
- **Result**: High OOD accuracy, true 1-to-1

---

## Complete Example Across All 6 Rules

### Rule 0: Birth Date

| Template | Question | Train/Test | Expected Slot |
|----------|----------|------------|---------------|
| 0 | "What is {NAME}'s birth date?" | TRAIN | 0 |
| 1 | "When was {NAME} born?" | TEST-ID | 0 |
| 2 | "Can you tell me the birth date of {NAME}?" | TEST-OOD | 0 |
| 3 | "On what date was {NAME} born?" | TEST-OOD | 0 |

### Rule 1: Birth City

| Template | Question | Train/Test | Expected Slot |
|----------|----------|------------|---------------|
| 0 | "What is {NAME}'s birth city?" | TRAIN | 1 |
| 1 | "Where was {NAME} born?" | TEST-ID | 1 |
| 2 | "Can you tell me the birth city of {NAME}?" | TEST-OOD | 1 |
| 3 | "In what city was {NAME} born?" | TEST-OOD | 1 |

### Rule 2: University

| Template | Question | Train/Test | Expected Slot |
|----------|----------|------------|---------------|
| 0 | "Which university did {NAME} attend?" | TRAIN | 2 |
| 1 | "Where did {NAME} go to college?" | TEST-ID | 2 |
| 2 | "What is {NAME}'s alma mater?" | TEST-OOD | 2 |
| 3 | "Which college did {NAME} attend?" | TEST-OOD | 2 |

*(Similar for major, employer, work_city)*

---

## The Binding Accuracy Test

### Question → Relation Binding

For each test question, measure:
```python
Does question activate correct slot?

Example:
  Question: "On what date was Alice born?"  (OOD template)
  True rule: birth_date (rule_idx=0)
  SAE output: predicted_slot = ?
  
  Correct if: predicted_slot == 0
```

### Metrics

**Slot Binding Accuracy** = (# questions that activate correct slot) / (# total questions)

Broken down by:
- **Train**: Sanity check (should be ~0.98)
- **Test-ID**: Generalization to new persons, similar templates (~0.90)
- **Test-OOD**: Generalization to new persons, NEW templates (~0.85)

---

## What Good vs Bad Looks Like

### ✅ Successful 1-to-1 SAE

```
Slot Binding Accuracy:
  Train:    0.98  ← Model learned the training data
  Test-ID:  0.92  ← Generalizes to new persons
  Test-OOD: 0.87  ← Generalizes to new phrasings! ✅

Interpretation:
  - Feature 0 (birth_date) activates for ALL birth date questions
  - Even with phrasings never seen during training
  - TRUE semantic concept learned, not template matching
```

### ❌ Failed - Template Memorization

```
Slot Binding Accuracy:
  Train:    0.95  ← Learned training templates fine
  Test-ID:  0.78  ← Some generalization
  Test-OOD: 0.45  ← Fails on new phrasings! ❌

Interpretation:
  - Feature 0 only responds to exact training phrasing
  - Different phrasings activate random/wrong slots
  - Model memorized templates, didn't learn concept
```

---

## Per-Rule OOD Analysis

The evaluation also shows which rules generalize well:

```
Per-Rule OOD Slot Binding:
  birth_date : 0.92  ✅  (Clear semantic concept)
  birth_city : 0.89  ✅  (Clear semantic concept)
  university : 0.85  ✅  (Good generalization)
  major      : 0.72  ⚠️  (Weaker - may need more diverse templates)
  employer   : 0.88  ✅  (Good generalization)
  work_city  : 0.81  ✅  (Good generalization)
```

**If a rule has low OOD accuracy**:
- Add more diverse question templates
- Ensure templates have different syntactic structures
- Check if the concept is ambiguous

---

## Why This Matters for Interpretability

### Traditional SAE Evaluation
- Reconstruction loss: Low ✓
- Sparsity: Good ✓
- **But**: Are features meaningful? Unknown

### With OOD Binding Accuracy
- Reconstruction loss: Low ✓
- Sparsity: Good ✓
- **OOD Binding**: High ✓
- **Therefore**: Features capture true semantic concepts! ✓✓✓

This means:
1. You can **interpret** what each feature means
2. You can **steer** the model by editing features
3. Features are **robust** to surface form variations
4. True **1-to-1 mapping** between concepts and latents

---

## Summary

**OOD Template Test** = The gold standard for 1-to-1 SAE evaluation

**What it tests**: 
- Does each SAE latent capture a semantic concept?
- Or does it just memorize surface patterns?

**How to pass**:
- Train on diverse data
- Use strong supervision (lambda_align)
- Ensure features are independent (lambda_indep)
- Test on completely unseen phrasings

**Target**: OOD Slot Binding Accuracy ≥ 0.85

If you achieve this, you have a **truly interpretable, steerable SAE** where each feature represents exactly one semantic rule! 🎯
