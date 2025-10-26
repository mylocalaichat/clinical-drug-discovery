# Clinical Drug-Disease Association Classification

This document explains how drug-disease associations are automatically classified from clinical notes.

## Overview

When extracting drug-disease pairs from clinical notes, the system now classifies each association into three types:

- **Positive**: Drug helped/improved the condition
- **Negative**: Adverse effects, drug didn't work, or worsened condition
- **Neutral**: Unclear outcome or just mentioned together

## How It Works

### 1. Context Extraction

For each drug-disease pair found in a clinical note, the system:
- Identifies the positions of both entities in the text
- Extracts a **400-character context window** (200 chars before and after)
- This captures the surrounding sentences that describe the clinical outcome

### 2. Keyword-Based Classification

The system looks for indicator keywords in the context:

#### Positive Indicators
Drug helped or improved the condition:
```
improved, resolved, responded, successful, effective,
helped, relief, recovery, better, cured, controlled,
stabilized, decreased, reduced, alleviated, managed,
remission, improvement, benefited, favorable, treated successfully
```

#### Negative Indicators
Adverse effects or drug didn't work:
```
adverse, failed, ineffective, worsened, allergic, allergy,
reaction, side effect, toxicity, discontinued, stopped,
no response, unresponsive, resistant, intolerant, contraindicated,
exacerbated, complications, deteriorated, aggravated, did not respond,
no improvement, no benefit, withdrew, ceased due to
```

#### Neutral
No clear indicators found, or equal positive and negative signals.

### 3. Scoring and Classification

```python
positive_score = count of positive keywords in context
negative_score = count of negative keywords in context

if positive_score > negative_score and positive_score > 0:
    → 'positive'
elif negative_score > positive_score and negative_score > 0:
    → 'negative'
else:
    → 'neutral'
```

## Example Clinical Notes

### Example 1: Positive Association

**Clinical Note:**
```
Patient with hypertension presented with elevated blood pressure readings.
Started on lisinopril 10mg daily. Blood pressure improved significantly
after 2 weeks. Patient responded well to treatment with no adverse effects.
```

**Extracted:**
- Drug: `lisinopril`
- Disease: `hypertension`
- Association: **`positive`**
- Keywords found: `improved`, `responded well`

---

### Example 2: Negative Association

**Clinical Note:**
```
Patient with depression was prescribed sertraline 50mg. After 3 weeks,
patient reported no improvement in symptoms. Additionally, patient
experienced adverse effects including nausea and headache. Medication
was discontinued due to poor tolerance.
```

**Extracted:**
- Drug: `sertraline`
- Disease: `depression`
- Association: **`negative`**
- Keywords found: `no improvement`, `adverse effects`, `discontinued`

---

### Example 3: Neutral Association

**Clinical Note:**
```
Patient has a history of diabetes and hypertension. Currently taking
metformin for blood sugar management and aspirin for cardiovascular
protection. Patient reports medication compliance.
```

**Extracted:**
- Drug: `metformin`
- Disease: `diabetes`
- Association: **`neutral`**
- Keywords found: None (just mentioned together)

---

## Output Data Structure

The extracted data includes detailed association breakdowns:

### CSV Output Columns:

| Column | Type | Description |
|--------|------|-------------|
| `drug_name` | string | Normalized drug name |
| `disease_name` | string | Normalized disease name |
| `association_type` | string | Dominant type: 'positive', 'negative', or 'neutral' |
| `positive_count` | int | # of times mentioned with positive outcome |
| `negative_count` | int | # of times mentioned with negative outcome |
| `neutral_count` | int | # of times mentioned with neutral context |
| `total_frequency` | int | Total mentions across all contexts |

### Example Output:

```
drug_name    disease_name    association_type  positive_count  negative_count  neutral_count  total_frequency
aspirin      heart disease   positive          8               1               2              11
warfarin     bleeding        negative          0               5               1              6
metformin    diabetes        positive          12              0               3              15
```

## Interpretation

### Strong Positive Association
```
aspirin - heart disease
positive_count: 8
negative_count: 0
neutral_count: 2
→ Aspirin consistently helps with heart disease
```

### Mixed Association
```
prednisone - inflammation
positive_count: 5
negative_count: 4
neutral_count: 1
→ Prednisone helps inflammation but has notable adverse effects
```

### Adverse Association
```
penicillin - allergy
positive_count: 0
negative_count: 10
neutral_count: 0
→ Strong adverse reaction pattern
```

## Use Cases

### 1. Drug Safety Monitoring
Identify drugs with high negative association counts for safety review:
```python
# Find drugs with concerning adverse patterns
negative_drugs = df[df['negative_count'] > df['positive_count']]
```

### 2. Treatment Efficacy
Find most effective drug-disease pairs:
```python
# Drugs with strong positive outcomes
effective = df[
    (df['positive_count'] > 5) &
    (df['positive_count'] > df['negative_count'] * 2)
]
```

### 3. Clinical Decision Support
Help clinicians understand:
- Which drugs work well for specific conditions
- Which drugs have common adverse effects
- Mixed results that warrant caution

## Limitations

1. **Context Window**: 400 chars may miss distant relationships
2. **Keyword-Based**: May miss nuanced language or sarcasm
3. **No Causality**: Co-occurrence doesn't prove causation
4. **English Only**: Keywords are English-specific

## Future Improvements

1. **ML Classification**: Train a classifier on labeled clinical notes
2. **Larger Context**: Use sentence boundaries instead of char limits
3. **Temporal Analysis**: Track outcome changes over time
4. **Severity Scoring**: Rate intensity of positive/negative effects
5. **Confidence Scores**: Add probability scores to classifications

## Statistics

The extraction pipeline provides statistics including:

```
Clinical Extraction Statistics:
  total_pairs: 156
  unique_drugs: 45
  unique_diseases: 38
  total_occurrences: 423
  positive_associations: 89 (57%)
  negative_associations: 31 (20%)
  neutral_associations: 36 (23%)
  median_frequency: 3
  max_frequency: 15
```

## Running the Extraction

To materialize clinical extraction with association classification:

1. Go to Dagster UI: http://localhost:3000
2. Navigate to Assets → clinical_extraction group
3. Click "Materialize all"
4. View results in: `data/03_primary/clinical_drug_disease_pairs.csv`

Or via CLI:
```bash
dagster asset materialize -m dagster_definitions --select clinical_drug_disease_pairs
```

## Validation

Check the output to ensure quality:

```python
import pandas as pd

df = pd.read_csv('data/03_primary/clinical_drug_disease_pairs.csv')

# Check distribution
print(df['association_type'].value_counts())

# View top positive associations
print(df[df['association_type'] == 'positive'].head(10))

# View negative associations (for safety review)
print(df[df['association_type'] == 'negative'].head(10))
```
