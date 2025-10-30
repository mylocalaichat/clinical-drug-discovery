# Node Type Relevance Analysis for Clinical Drug Discovery

## Executive Summary

Analyzed **129,375 nodes** across **10 node types** from the Memgraph knowledge graph. Below is the classification of node types by their relevance to drug discovery.

---

## 🟢 HIGHLY USEFUL (Essential for Drug Discovery)

### 1. **drug** (7,957 nodes | 6.15%)
**Relevance**: ⭐⭐⭐⭐⭐ (CRITICAL)

**Why Essential**:
- Primary entity for drug discovery pipeline
- Contains FDA-approved drugs and investigational compounds
- DrugBank IDs (e.g., DB09130, DB00180) enable clinical validation

**Key Relationships**:
- drug → drug: **2,672,628 edges** (drug-drug interactions, similarities)
- drug → disease: **90,542 edges** (known indications)
- drug → effect/phenotype: **87,202 edges** (side effects, efficacy)
- drug → gene/protein: **11,779 edges** (drug targets)

**Use Cases**:
- Primary input for drug repurposing predictions
- Drug-drug interaction analysis
- Drug similarity calculations for embeddings
- Clinical trial validation

---

### 2. **disease** (17,080 nodes | 13.20%)
**Relevance**: ⭐⭐⭐⭐⭐ (CRITICAL)

**Why Essential**:
- Target entity for drug discovery (what we're treating)
- Contains diseases from MONDO, OMIM, Orphanet
- Enables drug repurposing for rare and common diseases

**Key Relationships**:
- disease → disease: **204,523 edges** (disease similarity, co-occurrence)
- disease → gene/protein: **176,078 edges** (disease genes, biomarkers)
- disease → effect/phenotype: **108,382 edges** (disease symptoms)
- disease → anatomy: **133,483 edges** (organ systems affected)

**Use Cases**:
- Target diseases for drug repurposing
- Disease-disease similarity for transfer learning
- Rare disease identification
- Comorbidity analysis

---

### 3. **gene/protein** (27,671 nodes | 21.39%)
**Relevance**: ⭐⭐⭐⭐⭐ (CRITICAL)

**Why Essential**:
- Largest node type (21% of graph)
- Drug targets and mechanism of action
- Disease biomarkers and pathways
- Enables mechanistic understanding

**Key Relationships**:
- gene → gene: **2,467,258 edges** (protein-protein interactions)
- anatomy → gene: **2,023,582 edges** (tissue-specific expression)
- gene → biological_process: **600,425 edges** (functional roles)
- gene → molecular_function: **434,574 edges** (enzymatic activity)

**Use Cases**:
- Drug target identification
- Mechanism of action prediction
- Off-target effect analysis
- Pathway-based drug discovery

---

### 4. **effect/phenotype** (15,311 nodes | 11.83%)
**Relevance**: ⭐⭐⭐⭐ (VERY IMPORTANT)

**Why Important**:
- Human Phenotype Ontology (HPO) terms
- Side effects and adverse events
- Disease symptoms and manifestations
- Clinical outcomes

**Key Relationships**:
- effect → disease: **175,647 edges** (disease phenotypes)
- effect → gene/protein: **99,878 edges** (genetic causes)
- effect → effect: **97,490 edges** (phenotype hierarchies)

**Use Cases**:
- Side effect prediction
- Safety profile analysis
- Symptom-based disease matching
- Clinical outcome prediction
- Adverse event monitoring

---

### 5. **pathway** (2,516 nodes | 1.94%)
**Relevance**: ⭐⭐⭐⭐ (VERY IMPORTANT)

**Why Important**:
- Reactome pathways (e.g., Apoptosis, Hemostasis)
- Systems-level understanding of drug effects
- Pathway-based drug repurposing

**Key Relationships**:
- pathway → disease: **22,784 edges** (disease pathways)
- pathway → effect/phenotype: **19,259 edges** (pathway phenotypes)
- pathway → biological_process: **6,944 edges** (pathway components)
- pathway → pathway: **5,070 edges** (pathway hierarchies)

**Use Cases**:
- Pathway-based drug discovery
- Mechanism of action analysis
- Systems pharmacology
- Drug combination prediction

---

## 🟡 MODERATELY USEFUL (Contextual Value)

### 6. **biological_process** (28,642 nodes | 22.14%)
**Relevance**: ⭐⭐⭐ (MODERATE)

**Why Moderate**:
- Gene Ontology biological processes
- Provides mechanistic context
- Too granular for direct drug discovery
- Useful for understanding pathways

**Key Relationships**:
- biological_process → gene/protein: **612,620 edges**
- biological_process → anatomy: **376,566 edges**
- biological_process → biological_process: **174,381 edges**

**Concerns**:
- 28,642 nodes (22% of graph) - very granular
- Many low-level processes (e.g., "negative regulation of neurotransmitter uptake")
- May add noise to embeddings

**Recommendation**: **Include but downsample or aggregate to higher-level processes**

**Use Cases**:
- Mechanistic validation
- Pathway enrichment
- Drug mechanism hypothesis generation

---

### 7. **molecular_function** (11,169 nodes | 8.63%)
**Relevance**: ⭐⭐⭐ (MODERATE)

**Why Moderate**:
- Gene Ontology molecular functions
- Enzymatic activities and binding properties
- Useful for drug target validation
- Too granular for high-level predictions

**Key Relationships**:
- molecular_function → gene/protein: **429,411 edges**
- molecular_function → anatomy: **282,066 edges**

**Concerns**:
- Very specific (e.g., "catalytic activity, acting on a tRNA")
- May not directly impact drug-disease predictions

**Recommendation**: **Include but aggregate to higher-level functions (e.g., kinase, receptor, transporter)**

**Use Cases**:
- Drug target class identification
- Enzymatic activity prediction
- Binding site analysis

---

### 8. **anatomy** (14,035 nodes | 10.85%)
**Relevance**: ⭐⭐⭐ (MODERATE)

**Why Moderate**:
- Organ systems and tissues (UBERON ontology)
- Tissue-specific gene expression
- Drug distribution and targeting
- 1M+ edges to genes indicate importance

**Key Relationships**:
- anatomy → gene/protein: **2,023,582 edges** (tissue expression)
- anatomy → anatomy: **1,082,212 edges** (anatomical hierarchy)

**Concerns**:
- 14,035 nodes (10.85%) - very granular
- Many specific tissues (e.g., "uterine cervix", "naris")
- May not directly impact drug-disease matching

**Recommendation**: **Include but aggregate to organ systems (e.g., cardiovascular, nervous system)**

**Use Cases**:
- Organ-specific drug effects
- Tissue-specific toxicity
- Drug distribution modeling
- Target tissue expression

---

## 🔴 LESS USEFUL (Potentially Exclude)

### 9. **cellular_component** (4,176 nodes | 3.23%)
**Relevance**: ⭐ (LOW)

**Why Low Relevance**:
- Gene Ontology cellular components
- Subcellular localization (e.g., "COPI-coated vesicle", "intrinsic component of chloroplast inner membrane")
- Too granular and cell biology-focused
- Limited direct relevance to clinical drug discovery

**Key Relationships**:
- cellular_component → gene/protein: **188,736 edges**
- cellular_component → anatomy: **115,638 edges**

**Concerns**:
- Very specific subcellular structures
- Chloroplast components in human drug discovery? (likely from plant data)
- 4,176 nodes add complexity without clear benefit

**Recommendation**: **EXCLUDE from drug discovery embeddings**

**Potential Use Cases** (if included):
- Subcellular drug targeting
- Organelle-specific toxicity
- Cell biology research (not clinical focus)

---

### 10. **exposure** (818 nodes | 0.63%)
**Relevance**: ⭐⭐ (LOW-MODERATE)

**Why Low-Moderate Relevance**:
- Environmental chemical exposures (e.g., "1-hydroxyphenanthrene", "hexachlorobiphenyl")
- Toxicology and environmental health
- Not directly relevant to drug repurposing
- Small node count (818 nodes)

**Key Relationships**:
- exposure → exposure: **4,140 edges**
- exposure → disease: **3,508 edges** (toxicity associations)
- exposure → effect/phenotype: **2,064 edges**

**Concerns**:
- Environmental chemicals, not drugs
- Toxicology focus, not therapeutics
- Pollutants and industrial chemicals

**Recommendation**: **EXCLUDE unless studying drug-environment interactions or toxicology**

**Potential Use Cases** (if included):
- Drug-environment interaction analysis
- Toxicology prediction
- Pollutant-disease associations

---

## Summary Table

| Node Type | Count | % | Relevance | Recommendation | Reason |
|-----------|-------|---|-----------|----------------|--------|
| **drug** | 7,957 | 6.15% | ⭐⭐⭐⭐⭐ | **INCLUDE (Essential)** | Primary entity for drug discovery |
| **disease** | 17,080 | 13.20% | ⭐⭐⭐⭐⭐ | **INCLUDE (Essential)** | Target for drug repurposing |
| **gene/protein** | 27,671 | 21.39% | ⭐⭐⭐⭐⭐ | **INCLUDE (Essential)** | Mechanisms, targets, biomarkers |
| **effect/phenotype** | 15,311 | 11.83% | ⭐⭐⭐⭐ | **INCLUDE (Important)** | Side effects, symptoms, outcomes |
| **pathway** | 2,516 | 1.94% | ⭐⭐⭐⭐ | **INCLUDE (Important)** | Systems-level understanding |
| **biological_process** | 28,642 | 22.14% | ⭐⭐⭐ | **INCLUDE (filtered)** | Aggregate to high-level processes |
| **molecular_function** | 11,169 | 8.63% | ⭐⭐⭐ | **INCLUDE (filtered)** | Aggregate to target classes |
| **anatomy** | 14,035 | 10.85% | ⭐⭐⭐ | **INCLUDE (filtered)** | Aggregate to organ systems |
| **cellular_component** | 4,176 | 3.23% | ⭐ | **EXCLUDE** | Too granular, cell biology focus |
| **exposure** | 818 | 0.63% | ⭐⭐ | **EXCLUDE** | Environmental toxins, not drugs |

---

## Recommended Node Type Filtering Strategy

### Option 1: Core Only (Minimal Graph)
**Include**: drug, disease, gene/protein, effect/phenotype, pathway
- **Total**: 70,535 nodes (54.5% of graph)
- **Pros**: Focused on drug discovery, faster training
- **Cons**: Loses mechanistic context

### Option 2: Extended (Recommended)
**Include**: drug, disease, gene/protein, effect/phenotype, pathway, biological_process, molecular_function, anatomy
- **Total**: 125,381 nodes (96.9% of graph)
- **Exclude**: cellular_component, exposure
- **Pros**: Rich context, mechanistic understanding
- **Cons**: Larger embeddings, slower training

### Option 3: Full Graph (Current)
**Include**: All 10 node types
- **Total**: 129,375 nodes (100%)
- **Pros**: Complete information
- **Cons**: May add noise from non-relevant nodes

---

## Implementation Recommendation

### Approach: Two-Tier Filtering

#### Tier 1: Essential Nodes (Always Include)
```python
ESSENTIAL_TYPES = [
    'drug',           # 7,957
    'disease',        # 17,080
    'gene/protein',   # 27,671
    'effect/phenotype', # 15,311
    'pathway'         # 2,516
]
# Total: 70,535 nodes (54.5%)
```

#### Tier 2: Contextual Nodes (Filter/Aggregate)
```python
CONTEXTUAL_TYPES = {
    'biological_process': {
        'strategy': 'filter_top_level',  # Keep only parent processes
        'threshold': 1000,  # Keep top 1000 most connected
    },
    'molecular_function': {
        'strategy': 'aggregate_by_class',  # Group by enzyme class
        'classes': ['kinase', 'receptor', 'transporter', 'enzyme']
    },
    'anatomy': {
        'strategy': 'aggregate_by_system',  # Organ systems only
        'systems': ['cardiovascular', 'nervous', 'respiratory', etc.]
    }
}
```

#### Tier 3: Excluded Nodes
```python
EXCLUDED_TYPES = [
    'cellular_component',  # Too granular
    'exposure'             # Environmental, not therapeutic
]
# Excluded: 4,994 nodes (3.9%)
```

---

## Updated GNN Implementation

### Current Implementation (loads ALL)
```python
node_query = """
MATCH (n:Node)
RETURN n.node_id as id, n.node_name as name, n.node_type as type
"""
```

### Recommended Implementation (filtered)
```python
node_query = """
MATCH (n:Node)
WHERE n.node_type IN ['drug', 'disease', 'gene/protein', 'effect/phenotype',
                       'pathway', 'biological_process', 'molecular_function', 'anatomy']
RETURN n.node_id as id, n.node_name as name, n.node_type as type
"""
```

This excludes:
- cellular_component (4,176 nodes)
- exposure (818 nodes)
- **Total excluded**: 4,994 nodes (3.9%)
- **Total included**: 124,381 nodes (96.1%)

---

## Rationale by Use Case

### For Drug Repurposing
**Priority**: drug ⭐⭐⭐⭐⭐ > disease ⭐⭐⭐⭐⭐ > gene/protein ⭐⭐⭐⭐⭐ > pathway ⭐⭐⭐⭐ > effect/phenotype ⭐⭐⭐⭐

### For Side Effect Prediction
**Priority**: drug ⭐⭐⭐⭐⭐ > effect/phenotype ⭐⭐⭐⭐⭐ > gene/protein ⭐⭐⭐⭐ > pathway ⭐⭐⭐⭐ > anatomy ⭐⭐⭐

### For Target Identification
**Priority**: gene/protein ⭐⭐⭐⭐⭐ > drug ⭐⭐⭐⭐⭐ > disease ⭐⭐⭐⭐⭐ > molecular_function ⭐⭐⭐⭐ > pathway ⭐⭐⭐⭐

---

## Next Steps

1. **Implement Tier 2 filtering** in `gnn_embeddings.py`
2. **Compare embeddings** with/without filtered nodes
3. **Evaluate XGBoost performance** on filtered vs full graph
4. **Consider hierarchical aggregation** for GO terms (biological_process, molecular_function)
5. **Monitor embedding quality** via visualization (PCA, t-SNE)

---

## Performance Impact

### Current (All 10 types)
- Nodes: 129,375
- Training time: ~15-20 min (MPS)
- Embedding memory: ~250 MB

### Recommended (8 types, exclude 2)
- Nodes: 124,381 (96.1%)
- Training time: ~14-19 min (MPS) - **marginal improvement**
- Embedding memory: ~240 MB - **marginal improvement**

### Core Only (5 types)
- Nodes: 70,535 (54.5%)
- Training time: ~8-10 min (MPS) - **50% faster**
- Embedding memory: ~135 MB - **46% reduction**

---

## Conclusion

**Recommendation**: Use **Option 2 (Extended)** - Include 8 out of 10 node types, excluding:
- ❌ cellular_component (too granular, cell biology)
- ❌ exposure (environmental, not therapeutic)

This provides the best balance of:
- ✅ Comprehensive drug discovery context
- ✅ Mechanistic understanding
- ✅ Minimal noise
- ✅ 96.1% of nodes retained
- ✅ Only 3.9% excluded

For **faster experimentation**, start with **Option 1 (Core Only)** - 5 essential types.
