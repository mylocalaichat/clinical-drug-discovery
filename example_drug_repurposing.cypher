// Drug Repurposing Example Graph
// Demonstrates: Drug A treats Disease B, Drug E treats Disease C
// Since Drug A is similar to Drug E, and Disease D is similar to both B and C,
// We can infer Drug A might treat Disease D

// ===== DRUGS =====
CREATE (drugA:Node {
    node_id: 'EXAMPLE_DRUG_A',
    node_index: 999001,
    node_name: 'Metformin',
    node_source: 'Example',
    node_type: 'drug',
    example_set: 'drug_repurposing',
    is_example: true,
    mechanism: 'AMPK activator',
    drug_class: 'Biguanide',
    fda_approval_year: 1995,
    molecular_weight: 165.62,
    bioavailability: 0.55,
    half_life_hours: 5.0,
    administration_route: 'oral',
    cost_category: 'low',
    daily_dose_mg: 1500,
    side_effects: ['gastrointestinal', 'vitamin_B12_deficiency'],
    pregnancy_category: 'B'
});

CREATE (drugE:Node {
    node_id: 'EXAMPLE_DRUG_E',
    node_index: 999002,
    node_name: 'Aspirin',
    node_source: 'Example',
    node_type: 'drug',
    example_set: 'drug_repurposing',
    is_example: true,
    mechanism: 'COX inhibitor',
    drug_class: 'NSAID',
    fda_approval_year: 1950,
    molecular_weight: 180.16,
    bioavailability: 0.68,
    half_life_hours: 0.3,
    administration_route: 'oral',
    cost_category: 'low',
    daily_dose_mg: 100,
    side_effects: ['bleeding', 'gastric_irritation'],
    pregnancy_category: 'D'
});

// ===== DISEASES =====
CREATE (diseaseB:Node {
    node_id: 'EXAMPLE_DISEASE_B',
    node_index: 999003,
    node_name: 'Type 2 Diabetes',
    node_source: 'Example',
    node_type: 'disease',
    example_set: 'drug_repurposing',
    is_example: true,
    disease_category: 'Metabolic',
    severity: 'Chronic',
    prevalence_per_100k: 8800,
    age_of_onset: 45,
    icd10_code: 'E11',
    genetic_component: true,
    mortality_rate: 0.02,
    healthcare_cost_category: 'high',
    lifestyle_related: true,
    symptoms: ['hyperglycemia', 'fatigue', 'increased_thirst', 'frequent_urination']
});

CREATE (diseaseC:Node {
    node_id: 'EXAMPLE_DISEASE_C',
    node_index: 999004,
    node_name: 'Cardiovascular Disease',
    node_source: 'Example',
    node_type: 'disease',
    example_set: 'drug_repurposing',
    is_example: true,
    disease_category: 'Cardiovascular',
    severity: 'Chronic',
    prevalence_per_100k: 6200,
    age_of_onset: 55,
    icd10_code: 'I51',
    genetic_component: true,
    mortality_rate: 0.05,
    healthcare_cost_category: 'very_high',
    lifestyle_related: true,
    symptoms: ['chest_pain', 'shortness_of_breath', 'fatigue', 'arrhythmia']
});

CREATE (diseaseD:Node {
    node_id: 'EXAMPLE_DISEASE_D',
    node_index: 999005,
    node_name: 'Metabolic Syndrome',
    node_source: 'Example',
    node_type: 'disease',
    example_set: 'drug_repurposing',
    is_example: true,
    disease_category: 'Metabolic',
    severity: 'Chronic',
    prevalence_per_100k: 3400,
    age_of_onset: 50,
    icd10_code: 'E88.81',
    genetic_component: true,
    mortality_rate: 0.03,
    healthcare_cost_category: 'high',
    lifestyle_related: true,
    symptoms: ['obesity', 'hypertension', 'dyslipidemia', 'insulin_resistance']
});

// ===== PROTEINS/TARGETS =====
CREATE (ampk:Node {
    node_id: 'EXAMPLE_PROTEIN_AMPK',
    node_index: 999006,
    node_name: 'AMPK',
    node_source: 'Example',
    node_type: 'gene/protein',
    example_set: 'drug_repurposing',
    is_example: true,
    function: 'Energy sensor and regulator',
    gene_symbol: 'PRKAA1',
    chromosome: '5',
    protein_family: 'Serine/threonine kinases',
    cellular_location: 'cytoplasm',
    expression_level: 'high',
    druggability_score: 0.85,
    tissue_specificity: ['liver', 'muscle', 'brain'],
    is_therapeutic_target: true
});

CREATE (cox2:Node {
    node_id: 'EXAMPLE_PROTEIN_COX2',
    node_index: 999007,
    node_name: 'COX2',
    node_source: 'Example',
    node_type: 'gene/protein',
    example_set: 'drug_repurposing',
    is_example: true,
    function: 'Inflammatory mediator',
    gene_symbol: 'PTGS2',
    chromosome: '1',
    protein_family: 'Cyclooxygenases',
    cellular_location: 'endoplasmic_reticulum',
    expression_level: 'inducible',
    druggability_score: 0.92,
    tissue_specificity: ['immune_cells', 'endothelium', 'brain'],
    is_therapeutic_target: true
});

CREATE (il6:Node {
    node_id: 'EXAMPLE_PROTEIN_IL6',
    node_index: 999008,
    node_name: 'IL6',
    node_source: 'Example',
    node_type: 'gene/protein',
    example_set: 'drug_repurposing',
    is_example: true,
    function: 'Pro-inflammatory cytokine',
    gene_symbol: 'IL6',
    chromosome: '7',
    protein_family: 'Interleukin',
    cellular_location: 'secreted',
    expression_level: 'inducible',
    druggability_score: 0.78,
    tissue_specificity: ['immune_cells', 'adipose', 'muscle'],
    is_therapeutic_target: true
});

// ===== PATHWAYS =====
CREATE (inflam:Node {
    node_id: 'EXAMPLE_PATHWAY_INFLAM',
    node_index: 999009,
    node_name: 'Inflammation Pathway',
    node_source: 'Example',
    node_type: 'pathway',
    example_set: 'drug_repurposing',
    is_example: true,
    pathway_type: 'Inflammatory response',
    pathway_id: 'KEGG:hsa04060',
    organism: 'Homo sapiens',
    complexity_score: 0.72,
    num_proteins: 42,
    is_druggable: true,
    biological_process: 'immune_response',
    tissue_expression: ['immune_tissue', 'vascular', 'adipose']
});

CREATE (metab:Node {
    node_id: 'EXAMPLE_PATHWAY_METAB',
    node_index: 999010,
    node_name: 'Glucose Metabolism Pathway',
    node_source: 'Example',
    node_type: 'pathway',
    example_set: 'drug_repurposing',
    is_example: true,
    pathway_type: 'Metabolic regulation',
    pathway_id: 'KEGG:hsa00010',
    organism: 'Homo sapiens',
    complexity_score: 0.85,
    num_proteins: 67,
    is_druggable: true,
    biological_process: 'glucose_metabolism',
    tissue_expression: ['liver', 'muscle', 'pancreas']
});

// ===== DRUG-TARGET RELATIONSHIPS =====
MATCH (drugA:Node {node_id: 'EXAMPLE_DRUG_A'})
MATCH (ampk:Node {node_id: 'EXAMPLE_PROTEIN_AMPK'})
CREATE (drugA)-[:RELATES {
    relation: 'drug_targets_protein',
    display_relation: 'targets',
    example_set: 'drug_repurposing',
    is_example: true,
    effect: 'activation',
    confidence: 0.95,
    binding_affinity_nm: 12.5,
    selectivity_score: 0.88,
    clinical_significance: 'high',
    interaction_type: 'allosteric_activation',
    evidence_source: 'clinical_trial',
    ki_value: 15.3
}]->(ampk);

MATCH (drugE:Node {node_id: 'EXAMPLE_DRUG_E'})
MATCH (cox2:Node {node_id: 'EXAMPLE_PROTEIN_COX2'})
CREATE (drugE)-[:RELATES {
    relation: 'drug_targets_protein',
    display_relation: 'targets',
    example_set: 'drug_repurposing',
    is_example: true,
    effect: 'inhibition',
    confidence: 0.98,
    binding_affinity_nm: 5.2,
    selectivity_score: 0.65,
    clinical_significance: 'high',
    interaction_type: 'competitive_inhibition',
    evidence_source: 'clinical_trial',
    ki_value: 8.7
}]->(cox2);

// ===== PROTEIN-PATHWAY RELATIONSHIPS =====
MATCH (ampk:Node {node_id: 'EXAMPLE_PROTEIN_AMPK'})
MATCH (metab:Node {node_id: 'EXAMPLE_PATHWAY_METAB'})
CREATE (ampk)-[:RELATES {
    relation: 'protein_participates_pathway',
    display_relation: 'regulates',
    example_set: 'drug_repurposing',
    is_example: true,
    role: 'key_regulator',
    essentiality: 'essential',
    participation_level: 0.95,
    regulatory_effect: 'positive',
    evidence_type: 'experimental'
}]->(metab);

MATCH (cox2:Node {node_id: 'EXAMPLE_PROTEIN_COX2'})
MATCH (inflam:Node {node_id: 'EXAMPLE_PATHWAY_INFLAM'})
CREATE (cox2)-[:RELATES {
    relation: 'protein_participates_pathway',
    display_relation: 'mediates',
    example_set: 'drug_repurposing',
    is_example: true,
    role: 'mediator',
    essentiality: 'important',
    participation_level: 0.82,
    regulatory_effect: 'positive',
    evidence_type: 'experimental'
}]->(inflam);

MATCH (ampk:Node {node_id: 'EXAMPLE_PROTEIN_AMPK'})
MATCH (il6:Node {node_id: 'EXAMPLE_PROTEIN_IL6'})
CREATE (ampk)-[:RELATES {
    relation: 'protein_regulates_protein',
    display_relation: 'inhibits',
    example_set: 'drug_repurposing',
    is_example: true,
    effect: 'downregulation',
    mechanism: 'transcriptional_repression',
    fold_change: -2.3,
    statistical_significance: 0.001,
    interaction_strength: 0.78
}]->(il6);

MATCH (cox2:Node {node_id: 'EXAMPLE_PROTEIN_COX2'})
MATCH (il6:Node {node_id: 'EXAMPLE_PROTEIN_IL6'})
CREATE (cox2)-[:RELATES {
    relation: 'protein_regulates_protein',
    display_relation: 'upregulates',
    example_set: 'drug_repurposing',
    is_example: true,
    effect: 'upregulation',
    mechanism: 'prostaglandin_signaling',
    fold_change: 3.7,
    statistical_significance: 0.0001,
    interaction_strength: 0.85
}]->(il6);

// ===== PROTEIN-DISEASE RELATIONSHIPS =====
MATCH (il6:Node {node_id: 'EXAMPLE_PROTEIN_IL6'})
MATCH (diseaseB:Node {node_id: 'EXAMPLE_DISEASE_B'})
CREATE (il6)-[:RELATES {
    relation: 'protein_disease_association',
    display_relation: 'involved_in',
    example_set: 'drug_repurposing',
    is_example: true,
    association_type: 'pathogenic',
    association_strength: 0.87,
    evidence_level: 'strong',
    biomarker_potential: 'high',
    expression_change: 'upregulated',
    fold_change_in_disease: 4.2
}]->(diseaseB);

MATCH (il6:Node {node_id: 'EXAMPLE_PROTEIN_IL6'})
MATCH (diseaseC:Node {node_id: 'EXAMPLE_DISEASE_C'})
CREATE (il6)-[:RELATES {
    relation: 'protein_disease_association',
    display_relation: 'involved_in',
    example_set: 'drug_repurposing',
    is_example: true,
    association_type: 'pathogenic',
    association_strength: 0.92,
    evidence_level: 'strong',
    biomarker_potential: 'high',
    expression_change: 'upregulated',
    fold_change_in_disease: 5.8
}]->(diseaseC);

MATCH (il6:Node {node_id: 'EXAMPLE_PROTEIN_IL6'})
MATCH (diseaseD:Node {node_id: 'EXAMPLE_DISEASE_D'})
CREATE (il6)-[:RELATES {
    relation: 'protein_disease_association',
    display_relation: 'involved_in',
    example_set: 'drug_repurposing',
    is_example: true,
    association_type: 'pathogenic',
    association_strength: 0.83,
    evidence_level: 'moderate',
    biomarker_potential: 'medium',
    expression_change: 'upregulated',
    fold_change_in_disease: 3.5
}]->(diseaseD);

// ===== PATHWAY-DISEASE RELATIONSHIPS =====
MATCH (metab:Node {node_id: 'EXAMPLE_PATHWAY_METAB'})
MATCH (diseaseB:Node {node_id: 'EXAMPLE_DISEASE_B'})
CREATE (metab)-[:RELATES {
    relation: 'pathway_disease_association',
    display_relation: 'dysregulated_in',
    example_set: 'drug_repurposing',
    is_example: true,
    association_type: 'causal',
    dysregulation_magnitude: 0.91,
    clinical_relevance_score: 0.94,
    pathway_activity_change: 'decreased',
    therapeutic_potential: 'high'
}]->(diseaseB);

MATCH (inflam:Node {node_id: 'EXAMPLE_PATHWAY_INFLAM'})
MATCH (diseaseC:Node {node_id: 'EXAMPLE_DISEASE_C'})
CREATE (inflam)-[:RELATES {
    relation: 'pathway_disease_association',
    display_relation: 'dysregulated_in',
    example_set: 'drug_repurposing',
    is_example: true,
    association_type: 'causal',
    dysregulation_magnitude: 0.88,
    clinical_relevance_score: 0.92,
    pathway_activity_change: 'increased',
    therapeutic_potential: 'high'
}]->(diseaseC);

MATCH (inflam:Node {node_id: 'EXAMPLE_PATHWAY_INFLAM'})
MATCH (diseaseD:Node {node_id: 'EXAMPLE_DISEASE_D'})
CREATE (inflam)-[:RELATES {
    relation: 'pathway_disease_association',
    display_relation: 'dysregulated_in',
    example_set: 'drug_repurposing',
    is_example: true,
    association_type: 'contributing',
    dysregulation_magnitude: 0.72,
    clinical_relevance_score: 0.75,
    pathway_activity_change: 'increased',
    therapeutic_potential: 'medium'
}]->(diseaseD);

MATCH (metab:Node {node_id: 'EXAMPLE_PATHWAY_METAB'})
MATCH (diseaseD:Node {node_id: 'EXAMPLE_DISEASE_D'})
CREATE (metab)-[:RELATES {
    relation: 'pathway_disease_association',
    display_relation: 'dysregulated_in',
    example_set: 'drug_repurposing',
    is_example: true,
    association_type: 'contributing',
    dysregulation_magnitude: 0.79,
    clinical_relevance_score: 0.83,
    pathway_activity_change: 'decreased',
    therapeutic_potential: 'medium'
}]->(diseaseD);

// ===== DISEASE SIMILARITY RELATIONSHIPS =====
MATCH (diseaseB:Node {node_id: 'EXAMPLE_DISEASE_B'})
MATCH (diseaseD:Node {node_id: 'EXAMPLE_DISEASE_D'})
CREATE (diseaseB)-[:RELATES {
    relation: 'disease_similarity',
    display_relation: 'similar_to',
    example_set: 'drug_repurposing',
    is_example: true,
    similarity_score: 0.75,
    shared_features: 'metabolic_dysfunction',
    similarity_method: 'gene_expression_profile',
    comorbidity_rate: 0.42,
    clinical_overlap_score: 0.68,
    shared_pathways: 3,
    shared_genes: 127
}]->(diseaseD);

MATCH (diseaseC:Node {node_id: 'EXAMPLE_DISEASE_C'})
MATCH (diseaseD:Node {node_id: 'EXAMPLE_DISEASE_D'})
CREATE (diseaseC)-[:RELATES {
    relation: 'disease_similarity',
    display_relation: 'similar_to',
    example_set: 'drug_repurposing',
    is_example: true,
    similarity_score: 0.70,
    shared_features: 'cardiovascular_risk',
    similarity_method: 'phenotype_similarity',
    comorbidity_rate: 0.38,
    clinical_overlap_score: 0.61,
    shared_pathways: 2,
    shared_genes: 93
}]->(diseaseD);

// ===== DRUG SIMILARITY RELATIONSHIPS =====
MATCH (drugA:Node {node_id: 'EXAMPLE_DRUG_A'})
MATCH (drugE:Node {node_id: 'EXAMPLE_DRUG_E'})
CREATE (drugA)-[:RELATES {
    relation: 'drug_similarity',
    display_relation: 'similar_to',
    example_set: 'drug_repurposing',
    is_example: true,
    similarity_score: 0.65,
    shared_features: 'anti-inflammatory_effects',
    structural_similarity: 0.42,
    pharmacological_similarity: 0.73,
    mechanism_overlap_score: 0.58,
    side_effect_similarity: 0.55,
    target_overlap: 0.31
}]->(drugE);

// ===== DIRECT DRUG-DISEASE TREATMENT RELATIONSHIPS (Known) =====
MATCH (drugA:Node {node_id: 'EXAMPLE_DRUG_A'})
MATCH (diseaseB:Node {node_id: 'EXAMPLE_DISEASE_B'})
CREATE (drugA)-[:RELATES {
    relation: 'drug_treats_disease',
    display_relation: 'treats',
    example_set: 'drug_repurposing',
    is_example: true,
    evidence: 'FDA_approved',
    efficacy: 'high'
}]->(diseaseB);

MATCH (drugE:Node {node_id: 'EXAMPLE_DRUG_E'})
MATCH (diseaseC:Node {node_id: 'EXAMPLE_DISEASE_C'})
CREATE (drugE)-[:RELATES {
    relation: 'drug_treats_disease',
    display_relation: 'treats',
    example_set: 'drug_repurposing',
    is_example: true,
    evidence: 'FDA_approved',
    efficacy: 'moderate'
}]->(diseaseC);
