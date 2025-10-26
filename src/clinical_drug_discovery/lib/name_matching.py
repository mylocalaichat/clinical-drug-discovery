"""
Name matching service for clinical extraction to PrimeKG database.
Uses built-in PrimeKG synonym information and fuzzy matching.
"""

import re
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

import pandas as pd
from neo4j import GraphDatabase


class PrimeKGNameMatcher:
    """
    Name matching service that maps clinical entity names to PrimeKG database nodes.
    Leverages built-in synonyms in PrimeKG descriptions and fuzzy matching.
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, database: str = "primekg"):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.database = database
        
        # Caches for performance
        self._drug_cache = {}
        self._disease_cache = {}
        self._synonym_cache = {}
        self._loaded = False
        
    def load_database_entities(self):
        """Load all drugs and diseases from database with their synonyms."""
        if self._loaded:
            return
            
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        
        try:
            with driver.session(database=self.database) as session:
                # Load drugs with synonyms
                drug_query = """
                MATCH (n:PrimeKGNode)
                WHERE n.node_type = 'drug'
                RETURN n.node_id as id, n.node_name as name, n.description as description
                """
                
                drug_results = session.run(drug_query)
                for record in drug_results:
                    drug_id = record['id']
                    drug_name = record['name'].lower().strip()
                    description = record['description'] or ""
                    
                    # Store primary name with compound key to ensure uniqueness
                    self._drug_cache[drug_name] = drug_id
                    
                    # Extract synonyms from description
                    synonyms = self._extract_synonyms(description)
                    for synonym in synonyms:
                        synonym_lower = synonym.lower().strip()
                        if synonym_lower and synonym_lower != drug_name:
                            self._drug_cache[synonym_lower] = drug_id
                
                # Load diseases with synonyms - NOTE: only diseases, not other entity types
                disease_query = """
                MATCH (n:PrimeKGNode)
                WHERE n.node_type = 'disease'
                RETURN n.node_id as id, n.node_name as name, n.description as description
                """
                
                disease_results = session.run(disease_query)
                for record in disease_results:
                    disease_id = record['id']
                    disease_name = record['name'].lower().strip()
                    description = record['description'] or ""
                    
                    # Store primary name - only store if not already present or if this is a disease
                    self._disease_cache[disease_name] = disease_id
                    
                    # Extract synonyms from description
                    synonyms = self._extract_synonyms(description)
                    for synonym in synonyms:
                        synonym_lower = synonym.lower().strip()
                        if synonym_lower and synonym_lower != disease_name:
                            self._disease_cache[synonym_lower] = disease_id
                            
        finally:
            driver.close()
            
        self._loaded = True
        print(f"Loaded {len(self._drug_cache)} drug name variants and {len(self._disease_cache)} disease name variants")
    
    def _extract_synonyms(self, description: str) -> List[str]:
        """Extract synonyms from PrimeKG description text."""
        if not description:
            return []
            
        synonyms = []
        description_lower = description.lower()
        
        # Pattern 1: "also known as X" or "commonly known as X"
        patterns = [
            r'also known as[:\s]+([^,\.;]+)',
            r'commonly known as[:\s]+([^,\.;]+)',
            r'known as[:\s]+([^,\.;]+)',
            r'marketed (?:as|under)[:\s]+([^,\.;]+)',
            r'brand names?[:\s]+([^,\.;]+)',
            r'trade names?[:\s]+([^,\.;]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, description_lower, re.IGNORECASE)
            for match in matches:
                # Clean up the match
                cleaned = self._clean_synonym(match)
                if cleaned:
                    synonyms.append(cleaned)
        
        # Pattern 2: Text in underscores (like _Aspirin_)
        underscore_matches = re.findall(r'_([^_]+)_', description)
        for match in underscore_matches:
            cleaned = self._clean_synonym(match)
            if cleaned:
                synonyms.append(cleaned)
                
        # Pattern 3: Text in parentheses after "also known as"
        paren_patterns = [
            r'also known as[^(]*\(([^)]+)\)',
            r'commonly known as[^(]*\(([^)]+)\)',
        ]
        
        for pattern in paren_patterns:
            matches = re.findall(pattern, description_lower, re.IGNORECASE)
            for match in matches:
                cleaned = self._clean_synonym(match)
                if cleaned:
                    synonyms.append(cleaned)
        
        return synonyms
    
    def _clean_synonym(self, text: str) -> Optional[str]:
        """Clean and validate a synonym."""
        if not text:
            return None
            
        # Remove common prefixes/suffixes
        text = text.strip()
        text = re.sub(r'^(the\s+)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Remove marketing terms
        marketing_terms = ['product', 'brand', 'marketed', 'drug', 'medication', 'tablet', 'capsule']
        for term in marketing_terms:
            text = re.sub(rf'\b{term}\b', '', text, flags=re.IGNORECASE).strip()
        
        # Validate length
        if len(text) < 2 or len(text) > 100:
            return None
            
        return text
    
    def match_drug_name(self, clinical_name: str, min_similarity: float = 0.8) -> Optional[str]:
        """
        Match a clinical drug name to a PrimeKG drug node ID.
        
        Args:
            clinical_name: Drug name from clinical extraction
            min_similarity: Minimum similarity score for fuzzy matching
            
        Returns:
            PrimeKG drug node ID if match found, None otherwise
        """
        if not self._loaded:
            self.load_database_entities()
            
        clinical_name_clean = clinical_name.lower().strip()
        
        # 1. Exact match
        if clinical_name_clean in self._drug_cache:
            return self._drug_cache[clinical_name_clean]
        
        # 2. Fuzzy matching
        best_match = None
        best_score = 0
        
        for db_name, drug_id in self._drug_cache.items():
            similarity = SequenceMatcher(None, clinical_name_clean, db_name).ratio()
            if similarity > best_score and similarity >= min_similarity:
                best_score = similarity
                best_match = drug_id
        
        if best_match:
            print(f"Fuzzy matched drug '{clinical_name}' -> '{best_match}' (similarity: {best_score:.3f})")
            
        return best_match
    
    def match_disease_name(self, clinical_name: str, min_similarity: float = 0.6) -> Optional[str]:
        """
        Match a clinical disease name to a PrimeKG disease node ID.
        Uses more flexible matching for diseases since clinical notes often use general terms.
        
        Args:
            clinical_name: Disease name from clinical extraction
            min_similarity: Minimum similarity score for fuzzy matching (lower for diseases)
            
        Returns:
            PrimeKG disease node ID if match found, None otherwise
        """
        if not self._loaded:
            self.load_database_entities()
            
        clinical_name_clean = clinical_name.lower().strip()
        
        # 1. Exact match
        if clinical_name_clean in self._disease_cache:
            return self._disease_cache[clinical_name_clean]
        
        # 2. Partial word matching for diseases (e.g., "diabetes" matches "type 2 diabetes mellitus")
        clinical_words = set(clinical_name_clean.split())
        best_match = None
        best_score = 0
        
        for db_name, disease_id in self._disease_cache.items():
            db_words = set(db_name.split())
            
            # Check for word overlap
            word_overlap = len(clinical_words.intersection(db_words))
            if word_overlap > 0:
                # Calculate similarity based on word overlap and string similarity
                overlap_score = word_overlap / max(len(clinical_words), len(db_words))
                string_similarity = SequenceMatcher(None, clinical_name_clean, db_name).ratio()
                combined_score = (overlap_score * 0.7) + (string_similarity * 0.3)
                
                if combined_score > best_score and combined_score >= min_similarity:
                    best_score = combined_score
                    best_match = disease_id
        
        # 3. Fallback to pure fuzzy matching
        if not best_match:
            for db_name, disease_id in self._disease_cache.items():
                similarity = SequenceMatcher(None, clinical_name_clean, db_name).ratio()
                if similarity > best_score and similarity >= min_similarity:
                    best_score = similarity
                    best_match = disease_id
        
        if best_match:
            print(f"Matched disease '{clinical_name}' -> '{best_match}' (similarity: {best_score:.3f})")
            
        return best_match
    
    def normalize_clinical_pairs(self, clinical_pairs_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Normalize clinical drug-disease pairs to use PrimeKG node IDs.
        
        Args:
            clinical_pairs_df: DataFrame with 'drug' and 'disease' columns
            
        Returns:
            Tuple of (normalized_df, stats_dict)
        """
        if not self._loaded:
            self.load_database_entities()
            
        normalized_pairs = []
        stats = {
            'total_pairs': len(clinical_pairs_df),
            'drug_matches': 0,
            'disease_matches': 0,
            'both_matched': 0,
            'drug_misses': 0,
            'disease_misses': 0,
            'total_dropped': 0
        }
        
        print(f"\nNormalizing {len(clinical_pairs_df)} clinical pairs...")
        
        for idx, row in clinical_pairs_df.iterrows():
            drug_name = row['drug']
            disease_name = row['disease']
            score = row['score']
            
            # Match drug name
            drug_id = self.match_drug_name(drug_name)
            if drug_id:
                stats['drug_matches'] += 1
            else:
                stats['drug_misses'] += 1
                
            # Match disease name  
            disease_id = self.match_disease_name(disease_name)
            if disease_id:
                stats['disease_matches'] += 1
            else:
                stats['disease_misses'] += 1
            
            # Only keep pairs where both drug and disease are matched
            if drug_id and disease_id:
                normalized_pairs.append({
                    'drug_id': drug_id,
                    'disease_id': disease_id,
                    'score': score,
                    'original_drug_name': drug_name,
                    'original_disease_name': disease_name
                })
                stats['both_matched'] += 1
            else:
                stats['total_dropped'] += 1
                print(f"  Dropped: {drug_name} -> {disease_name} (drug_match: {bool(drug_id)}, disease_match: {bool(disease_id)})")
        
        normalized_df = pd.DataFrame(normalized_pairs)
        
        print(f"\nNormalization Results:")
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Drug matches: {stats['drug_matches']}/{stats['total_pairs']} ({stats['drug_matches']/stats['total_pairs']*100:.1f}%)")
        print(f"  Disease matches: {stats['disease_matches']}/{stats['total_pairs']} ({stats['disease_matches']/stats['total_pairs']*100:.1f}%)")
        print(f"  Both matched: {stats['both_matched']}/{stats['total_pairs']} ({stats['both_matched']/stats['total_pairs']*100:.1f}%)")
        print(f"  Pairs dropped: {stats['total_dropped']}")
        
        return normalized_df, stats
    
    def get_database_entity_info(self, entity_id: str, entity_type: str) -> Optional[Dict]:
        """Get detailed information about a database entity."""
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        
        try:
            with driver.session(database=self.database) as session:
                query = """
                MATCH (n:PrimeKGNode {node_id: $entity_id})
                WHERE n.node_type = $entity_type
                RETURN n.node_id as id, n.node_name as name, n.description as description
                """
                
                result = session.run(query, entity_id=entity_id, entity_type=entity_type)
                record = result.single()
                
                if record:
                    return {
                        'id': record['id'],
                        'name': record['name'],
                        'description': record['description']
                    }
                    
        finally:
            driver.close()
            
        return None


def create_name_matcher(neo4j_uri: str, neo4j_user: str, neo4j_password: str, database: str = "primekg") -> PrimeKGNameMatcher:
    """Factory function to create a name matcher instance."""
    return PrimeKGNameMatcher(neo4j_uri, neo4j_user, neo4j_password, database)