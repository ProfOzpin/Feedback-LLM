# phase5_evaluation.py - Phase 5: Comprehensive Model Evaluation Framework
# Processes test datasets and evaluates all diagnostic models with progress saving

import os
import json
import pickle
import itertools
import requests
import time
import re
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold
from openai import OpenAI

warnings.filterwarnings('ignore')

# --- Import Phase 4 Components ---
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Warning: google-generativeai not found. Run: pip install google-generativeai")
    GEMINI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    print("Warning: python-dotenv not found. Run: pip install python-dotenv")
    DOTENV_AVAILABLE = False

try:
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork
    PGMPY_AVAILABLE = True
except ImportError:
    print("Warning: pgmpy not found. Run: pip install pgmpy")
    PGMPY_AVAILABLE = False

class EnhancedASSISTmentsMapper:
    """Enhanced data mapper with expanded specific error pattern detection"""
    
    # In class EnhancedASSISTmentsMapper:
    def __init__(self):
        """
        LOGICAL IMPROVEMENT: The skill mapping is now more granular, breaking down
        broad topics into specific sub-skills for more precise diagnosis.
        """
        self.skill_to_concept_mapping = {
            # --- GRANULAR ALGEBRA ---
            'U_Linear_Equations': [
                'equation solving', 'linear equations', 'solving equations', 
                'solve for x', 'variable', 'isolate variable', 'one-step', 'two-step'
            ],
            'U_Quadratic_Equations': [
                'quadratic formula', 'factoring', 'quadratic', 'polynomial'
            ],
            'U_Inequalities': [
                'inequalities', 'solving inequalities', 'graphing inequalities'
            ],
            # --- GRANULAR GEOMETRY ---
            'U_Area_Perimeter': [
                'area rectangle', 'perimeter of a polygon', 'area triangle', 'area circle',
                'area parallelogram', 'perimeter rectangle', 'radius', 'diameter', 'square area',
                'circumference'
            ],
            'U_Volume_Surface_Area': [
                'volume rectangular prism', 'pythagorean theorem', 'volume sphere', 
                'surface area', 'volume cylinder'
            ],
            # --- GRANULAR FRACTIONS ---
            'U_Fraction_Arithmetic': [
                'addition and subtraction fractions', 'multiplication fractions', 'division fractions'
            ],
            'U_Fraction_Concepts': [
                'equivalent fractions', 'conversion of fraction decimals percents', 'fraction',
                'numerator', 'denominator', 'common denominator', 'reduce fraction', 'simplify fraction',
                'mixed number', 'improper fraction', 'proper fraction'
            ],
            # --- EXISTING CONCEPTS (Unchanged) ---
            'U_Integer_Operations': [
                'addition and subtraction integers', 'multiplication and division integers', 
                'ordering integers', 'integer', 'whole number', 'positive integer', 
                'negative integer', 'add integers', 'subtract integers', 'multiply integers'
            ],
            'U_Percent_Operations': [
                'percent of', 'percents', 'finding percents', 'percent discount', 'percentage',
                'percent change', 'percent increase', 'percent decrease', 'convert to percent'
            ],
            'U_Decimal_Operations': [
                'addition and subtraction positive decimals', 'multiplication and division positive decimals', 
                'ordering positive decimals', 'decimal', 'decimal place', 'round decimal',
                'compare decimals', 'decimal point'
            ],
            'U_Statistics_Data': [
                'box and whisker', 'mean', 'median', 'mode', 'range', 'scatter plot',
                'average', 'data analysis', 'histogram', 'bar graph', 'line graph',
                'frequency table', 'stem and leaf', 'quartile'
            ],
            'U_Linear_Functions': [
                'finding slope', 'intercept', 'write linear equation', 'slope', 'y-intercept',
                'x-intercept', 'linear function', 'graph line', 'point slope form'
            ],
            'U_Probability': [
                'probability of a single event', 'counting methods', 'probability',
                'odds', 'chance', 'likelihood', 'sample space', 'outcome'
            ],
            'U_Order_Of_Operations': [
                'order of operations', 'distributive property', 'pemdas', 'bodmas',
                'parentheses', 'exponent', 'multiply first', 'divide first'
            ],
            'U_Geometry_Properties': [
                'angles - obtuse', 'complementary and supplementary angles', 'congruence',
                'angle', 'acute angle', 'right angle', 'straight angle', 'vertical angles',
                'parallel lines', 'perpendicular', 'triangle', 'quadrilateral'
            ],
            'U_Advanced_Math': [
                'exponents', 'scientific notation', 'square root', 'simplifying expressions',
                'power', 'radical', 'cube root', 'exponential', 'base', 'coefficient'
            ],
            'U_Patterns_Proportions': [
                'proportion', 'rate', 'unit rate', 'pattern finding', 'ratio',
                'cross multiply', 'scale factor', 'similar figures', 'sequence'
            ],
            'U_Transformations': [
                'reflection', 'rotations', 'translations', 'transformation', 'flip',
                'turn', 'slide', 'symmetry', 'coordinate plane'
            ],
            'U_Number_Properties': [
                'prime number', 'greatest common factor', 'least common multiple', 'divisibility rules',
                'composite number', 'factor', 'multiple', 'gcf', 'lcm', 'divisible'
            ],
            'U_Measurement_Tools': [
                'reading a ruler or scale', 'measure', 'ruler', 'scale', 'measurement'
            ],
            'U_Approximation': [
                'estimation', 'rounding', 'estimate', 'round', 'approximate', 'nearest'
            ],
            'U_Unit_Conversion': [
                'unit conversion', 'convert units', 'measurement conversion', 'metric',
                'imperial', 'length conversion', 'weight conversion'
            ]
        }
        
        # Keep everything else exactly the same as your original
        self.error_patterns = {
            'E_Multiple_Attempts': {'weight': 0.4, 'keywords': []},
            'E_Hint_Usage': {'weight': 0.3, 'keywords': []},
            'E_Slow_Response': {'weight': 0.3, 'keywords': []},
            'E_Sign_Error': {'weight': 0.8, 'patterns': [], 'skill_specific': ['U_Integer_Operations', 'U_Linear_Equations', 'U_Inequalities']},
            'E_Fraction_Error': {'weight': 0.7, 'patterns': [], 'skill_specific': ['U_Fraction_Arithmetic', 'U_Fraction_Concepts']},
            'E_Decimal_Error': {'weight': 0.7, 'patterns': [], 'skill_specific': ['U_Decimal_Operations', 'U_Percent_Operations']},
            'E_Order_Error': {'weight': 0.8, 'patterns': [], 'skill_specific': ['U_Order_Of_Operations']},
            'E_Unit_Error': {'weight': 0.7, 'patterns': [], 'skill_specific': ['U_Area_Perimeter', 'U_Volume_Surface_Area', 'U_Unit_Conversion']},
            'E_Calculation_Error': {'weight': 0.3, 'keywords': []},
            'E_Conceptual_Confusion': {'weight': 0.3, 'keywords': []}
        }
        
        self.unmapped_skills = []
        self.semantic_model = None
        self.skill_embeddings = {}


    def _create_embeddings(self):
        # SPEED OPTIMIZATION: Completely disabled to avoid slow embedding creation
        return

    def map_skill_to_concept(self, skill_name):
        if pd.isna(skill_name) or not skill_name or skill_name.strip() == '':
            return None
        
        skill_lower = str(skill_name).lower().strip()
        
        if skill_lower in ['unknown', 'nan']:
            return None
        
        # Direct and fuzzy keyword matching ONLY - no semantic similarity fallback
        for concept_node, skill_keywords in self.skill_to_concept_mapping.items():
            for keyword in skill_keywords:
                if keyword.lower() in skill_lower:
                    return concept_node
        
        for concept_node, skill_keywords in self.skill_to_concept_mapping.items():
            for keyword in skill_keywords:
                keyword_words = set(keyword.lower().split())
                skill_words = set(skill_lower.split())
                overlap = len(keyword_words.intersection(skill_words))
                if overlap >= max(1, len(keyword_words) * 0.6):
                    return concept_node
        
        # No semantic similarity fallback - this was the major bottleneck
        self.unmapped_skills.append(skill_name)
        return None

    def calculate_evidence_strength(self, evidence):
        if not evidence:
            return 0.0
        
        total_weight = 0.0
        for error_type, value in evidence.items():
            if value == 1 and error_type in self.error_patterns:
                total_weight += self.error_patterns[error_type]['weight']
        
        return min(total_weight, 1.0)

    def analyze_answer_text_for_specific_errors(self, answer_text, skill_name):
        # SPEED OPTIMIZATION: Skip expensive text analysis during training
        # Only do basic rule-based error detection based on skill concept
        if not answer_text or pd.isna(answer_text):
            return {}
        
        detected_errors = {}
        skill_concept = self.map_skill_to_concept(skill_name)
        
        # Simple rule-based specific errors (much faster than full text analysis)
        if pd.notna(skill_concept):
            if skill_concept in ['U_Integer_Operations', 'U_Algebraic_Equations']:
                detected_errors['E_Sign_Error'] = 1
            elif skill_concept == 'U_Fraction_Operations':
                detected_errors['E_Fraction_Error'] = 1
            elif skill_concept in ['U_Decimal_Operations', 'U_Percent_Operations']:
                detected_errors['E_Decimal_Error'] = 1
            elif skill_concept == 'U_Order_Of_Operations':
                detected_errors['E_Order_Error'] = 1
            elif skill_concept in ['U_Geometry_Measurement', 'U_Unit_Conversion']:
                detected_errors['E_Unit_Error'] = 1
        
        return detected_errors

    # In class EnhancedASSISTmentsMapper:
    # In class EnhancedASSISTmentsMapper:
    def extract_error_evidence(self, row):
        """
        LOGICAL IMPROVEMENT: This single, authoritative function now generates
        multi-level evidence to capture error severity, providing a much richer
        signal for the Bayesian network to learn from.
        """
        evidence = {}
        # Only generate evidence for incorrect answers
        if row.get('correct', 1) == 0:
            
            # --- DYNAMIC BEHAVIORAL EVIDENCE (Capturing Severity) ---
            
            # 1. Multiple Attempts -> Three states: 0, 1 (Minor), 2 (Severe)
            attempt_count = row.get('attempt_count', 1)
            if attempt_count >= 4:
                evidence['E_Multiple_Attempts'] = 2 # Severe struggling
            elif attempt_count >= 2:
                evidence['E_Multiple_Attempts'] = 1 # Minor slip-up
                
            # 2. Hint Usage -> Three states: 0, 1 (Used hints), 2 (Bottomed out)
            if row.get('bottom_hint', 0) == 1:
                evidence['E_Hint_Usage'] = 2 # Gave up, needed final answer
            elif row.get('hint_count', 0) > 0:
                evidence['E_Hint_Usage'] = 1 # Used some hints
                
            # 3. Slow Response -> Binary is sufficient
            if row.get('ms_first_response', 0) > 25000: # Increased threshold
                evidence['E_Slow_Response'] = 1
                
            # --- RULE-BASED SPECIFIC ERRORS (Unchanged for now) ---
            skill_concept = self.map_skill_to_concept(row.get('skill_name', ''))
            if pd.notna(skill_concept):
                if skill_concept in ['U_Integer_Operations', 'U_Algebraic_Equations']:
                    evidence['E_Sign_Error'] = 1
                elif skill_concept == 'U_Fraction_Operations':
                    evidence['E_Fraction_Error'] = 1
                elif skill_concept in ['U_Decimal_Operations', 'U_Percent_Operations']:
                    evidence['E_Decimal_Error'] = 1
                elif skill_concept == 'U_Order_Of_Operations':
                    evidence['E_Order_Error'] = 1
                elif skill_concept in ['U_Geometry_Measurement', 'U_Unit_Conversion']:
                    evidence['E_Unit_Error'] = 1
                    
        return evidence

    def get_unmapped_skills(self):
        return list(set(self.unmapped_skills))

class ProductionReadyDiagnosticNetwork:
    """Production-ready Bayesian Network with enhanced diagnostics"""

    # In class ProductionReadyDiagnosticNetwork:
    def __init__(self):
        """
        LOGICAL IMPROVEMENT: The model now includes more granular conceptual nodes
        and new latent variables (L_Problem_Solving_Strategy, L_Carelessness) 
        to capture higher-level cognitive patterns.
        """
        self.conceptual_nodes = [
            # New Granular Concepts
            'U_Linear_Equations', 'U_Quadratic_Equations', 'U_Inequalities',
            'U_Area_Perimeter', 'U_Volume_Surface_Area',
            'U_Fraction_Arithmetic', 'U_Fraction_Concepts',
            # Original Concepts
            'U_Integer_Operations', 'U_Percent_Operations', 'U_Decimal_Operations',
            'U_Statistics_Data', 'U_Linear_Functions', 'U_Probability',
            'U_Order_Of_Operations', 'U_Geometry_Properties', 'U_Advanced_Math',
            'U_Patterns_Proportions', 'U_Transformations', 'U_Number_Properties',
            'U_Measurement_Tools', 'U_Approximation', 'U_Unit_Conversion'
        ]
        
        # NEW: Add more descriptive latent nodes
        self.latent_nodes = [
            'L_General_Math_Ability', 
            'L_Problem_Solving_Strategy', # e.g., systematic vs. guessing
            'L_Carelessness' # e.g., making slips vs. conceptual gaps
        ]
        
        self.error_nodes = [
            'E_Multiple_Attempts', 'E_Hint_Usage', 'E_Slow_Response',
            'E_Sign_Error', 'E_Fraction_Error', 'E_Decimal_Error', 
            'E_Order_Error', 'E_Unit_Error',
            'E_Calculation_Error', 'E_Conceptual_Confusion'
        ]
        
        # Update the full node list
        self.all_nodes = self.conceptual_nodes + self.latent_nodes + self.error_nodes
        
        self.model = self._create_enhanced_network_structure()
        self.data_mapper = EnhancedASSISTmentsMapper()
        self.inference = None
        self.trained = False

    def _create_enhanced_network_structure(self):
        """
        LOGICAL IMPROVEMENT: Creates a more sophisticated, hierarchical graph.
        - Connects high-level latent nodes to other latent and behavioral nodes.
        - Maps new, granular concepts to their specific error types.
        """
        edges = []
        
        # --- New Hierarchical Latent Structure ---
        # 1. Problem-solving strategy influences general ability and behavior
        edges.append(('L_Problem_Solving_Strategy', 'L_General_Math_Ability'))
        edges.append(('L_Problem_Solving_Strategy', 'E_Multiple_Attempts')) # Guessing -> more attempts
        edges.append(('L_Problem_Solving_Strategy', 'E_Hint_Usage'))      # Poor strategy -> more hints

        # 2. General ability influences carelessness
        edges.append(('L_General_Math_Ability', 'L_Carelessness')) # Weaker ability can lead to more slips

        # 3. Carelessness directly causes simple mistakes
        edges.append(('L_Carelessness', 'E_Calculation_Error'))
        edges.append(('L_Carelessness', 'E_Sign_Error'))
        
        # --- Connect Latent Ability to All Specific Concepts ---
        for concept in self.conceptual_nodes:
            edges.append(('L_General_Math_Ability', concept))
        
        # --- Behavioral Connections (Same as before) ---
        behavioral_errors = ['E_Multiple_Attempts', 'E_Hint_Usage', 'E_Slow_Response']
        for concept in self.conceptual_nodes:
            for behavior in behavioral_errors:
                edges.append((concept, behavior))
        
        # --- GRANULAR Specific Error Mappings ---
        specific_mappings = {
            # New Mappings
            'U_Linear_Equations': ['E_Sign_Error', 'E_Conceptual_Confusion'],
            'U_Quadratic_Equations': ['E_Calculation_Error', 'E_Conceptual_Confusion'],
            'U_Inequalities': ['E_Sign_Error', 'E_Conceptual_Confusion'],
            'U_Area_Perimeter': ['E_Unit_Error', 'E_Calculation_Error'],
            'U_Volume_Surface_Area': ['E_Unit_Error', 'E_Calculation_Error'],
            'U_Fraction_Arithmetic': ['E_Fraction_Error', 'E_Calculation_Error'],
            'U_Fraction_Concepts': ['E_Fraction_Error'],
            # Original Mappings
            'U_Integer_Operations': ['E_Sign_Error', 'E_Calculation_Error'], 
            'U_Percent_Operations': ['E_Decimal_Error', 'E_Fraction_Error'],
            'U_Decimal_Operations': ['E_Decimal_Error', 'E_Calculation_Error'],
            'U_Order_Of_Operations': ['E_Order_Error', 'E_Conceptual_Confusion'],
            'U_Advanced_Math': ['E_Conceptual_Confusion'],
            'U_Approximation': ['E_Conceptual_Confusion'],
            'U_Unit_Conversion': ['E_Unit_Error']
        }
        
        for concept, specific_errors in specific_mappings.items():
            for error in specific_errors:
                if error in self.error_nodes and concept in self.conceptual_nodes:
                    edges.append((concept, error))
        
        return DiscreteBayesianNetwork(edges)

    def clean_data(self, data, verbose=False):
        """Enhanced data cleaning with quality checks - PRESERVED"""
        original_length = len(data)
        
        # Remove rows with missing skill names or 'Unknown' skills
        clean_data = data[
            data['skill_name'].notna() & 
            (data['skill_name'] != 'Unknown') & 
            (data['skill_name'].str.strip() != '') &
            (data['skill_name'] != 'nan')
        ].copy()
        
        # Remove rows with invalid attempt counts or missing critical fields
        clean_data = clean_data[
            (clean_data['correct'].isin([0, 1])) &
            (clean_data['attempt_count'].notna()) &
            (clean_data['attempt_count'] > 0)
        ]
        
        if verbose:
            removed = original_length - len(clean_data)
            print(f"ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸Ãƒâ€šÃ‚Â§Ãƒâ€šÃ‚Â¹ Data Cleaning:")
            print(f"   Removed {removed:,} problematic examples ({removed/original_length*100:.1f}%)")
            print(f"   Clean dataset: {len(clean_data):,} examples")
        
        return clean_data
    
    def balance_training_data(self, data, verbose=False):
        """NEW: Address data imbalance by concept frequency"""
        if verbose:
            print("Balancing training data...")
        
        # Group by skill concept
        skill_concepts = data['skill_name'].apply(self.data_mapper.map_skill_to_concept)
        data_with_concepts = data.copy()
        data_with_concepts['concept'] = skill_concepts
        
        concept_counts = data_with_concepts['concept'].value_counts()
        
        # Calculate target size (median of all concept counts)
        target_size = int(concept_counts.median())
        max_size = min(target_size * 3, concept_counts.max())  # Cap to avoid explosion
        
        balanced_dfs = []
        
        for concept in concept_counts.index:
            if pd.isna(concept):
                continue
                
            concept_data = data_with_concepts[data_with_concepts['concept'] == concept]
            
            if len(concept_data) < target_size:
                # Upsample underrepresented concepts
                upsampled = resample(concept_data, 
                                   replace=True, 
                                   n_samples=min(target_size, len(concept_data) * 2),
                                   random_state=42)
                balanced_dfs.append(upsampled)
            elif len(concept_data) > max_size:
                # Downsample overrepresented concepts
                downsampled = resample(concept_data, 
                                     replace=False, 
                                     n_samples=max_size,
                                     random_state=42)
                balanced_dfs.append(downsampled)
            else:
                balanced_dfs.append(concept_data)
        
        balanced_data = pd.concat(balanced_dfs, ignore_index=True)
        balanced_data = balanced_data.drop('concept', axis=1)
        
        if verbose:
            print(f"   Original: {len(data):,} examples")
            print(f"   Balanced: {len(balanced_data):,} examples")
            print(f"   Target size per concept: {target_size:,}")
        
        return balanced_data
    
    def preprocess_assistments_data(self, data, verbose=False):
        """VECTORIZED preprocessing - ALL OPTIMIZATIONS PRESERVED"""
        if verbose:
            print(f"Processing {len(data)} ASSISTments examples...")
        
        # Pre-allocate result DataFrame with zeros
        result_df = pd.DataFrame(0, index=data.index, columns=self.all_nodes, dtype=int)
        
        # VECTORIZED skill mapping (PRESERVED)
        skill_concepts = data['skill_name'].apply(self.data_mapper.map_skill_to_concept)
        valid_concepts = skill_concepts[skill_concepts.notna() & skill_concepts.isin(self.conceptual_nodes)]
        
        # Set concept nodes in batch
        for concept_name in valid_concepts.unique():
            concept_mask = (skill_concepts == concept_name)
            result_df.loc[concept_mask, concept_name] = 1
        
        # VECTORIZED error evidence extraction for incorrect answers only (PRESERVED)
        incorrect_mask = (data['correct'] == 0)
        incorrect_data = data[incorrect_mask]
        
        if len(incorrect_data) > 0:
            # Batch process behavioral evidence (PRESERVED)
            multiple_attempts_mask = (incorrect_data['attempt_count'] > 1)
            result_df.loc[incorrect_data[multiple_attempts_mask].index, 'E_Multiple_Attempts'] = 1
            
            hint_usage_mask = (incorrect_data['hint_count'] > 0)
            result_df.loc[incorrect_data[hint_usage_mask].index, 'E_Hint_Usage'] = 1
            
            slow_response_mask = (incorrect_data['ms_first_response'] > 15000)
            result_df.loc[incorrect_data[slow_response_mask].index, 'E_Slow_Response'] = 1
            
            # NEW: Batch process specific errors
            for idx, row in incorrect_data.iterrows():
                specific_errors = self.data_mapper.analyze_answer_text_for_specific_errors(
                    row.get('answer_text', ''), row.get('skill_name', '')
                )
                for error_type, value in specific_errors.items():
                    if error_type in self.error_nodes and value == 1:
                        result_df.loc[idx, error_type] = 1
        
        stats = {
            'mapped_concepts': valid_concepts.count(),
            'extracted_errors': incorrect_mask.sum(),
            'total_incorrect': incorrect_mask.sum()
        }
        
        if verbose:
            print(f"ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒâ€¦  Enhanced Processing Results:")
            print(f"   Concepts mapped: {stats['mapped_concepts']:,}/{len(data):,}")
            print(f"   Errors processed: {stats['extracted_errors']:,}")
        
        return result_df
    
    # In class ProductionReadyDiagnosticNetwork:
    def train_with_data(self, data, use_balancing=True, verbose=False, n_jobs=-1, equivalent_sample_size=12):
        """
        ULTRA-OPTIMIZED training using Bayesian Estimation with controlled smoothing
        to improve model robustness and prevent overfitting from sparse data.
        """
        try:
            if verbose:
                print(f"ðŸš€ Training with Bayesian Smoothing (ESS: {equivalent_sample_size})...")

            # Use pre-cleaned data if available (from cross_validate)
            if hasattr(self, '_data_is_clean') and self._data_is_clean:
                clean_data = data
            else:
                clean_data = self.clean_data(data, verbose=False)
                
            # Balance the training set if enabled and the dataset is large
            if use_balancing and len(clean_data) > 50000:
                clean_data = self.balance_training_data(clean_data, verbose=False)
            
            # Use the fast, simple preprocessor for training
            processed_data = self._ultra_fast_preprocess(clean_data)
            
            # Calculate optimal parallel jobs
            optimal_n_jobs = self._calculate_optimal_n_jobs(processed_data, n_jobs)
            
            if verbose:
                print(f"âš¡ï¸ Data: {len(processed_data):,}, n_jobs: {optimal_n_jobs}")

            # --- LOGICAL IMPROVEMENT: Use BayesianEstimator exclusively ---
            # This estimator is more robust to sparse data than MaximumLikelihoodEstimator
            # by applying smoothing, which prevents zero-probability issues.
            from pgmpy.estimators import BayesianEstimator
            estimator = BayesianEstimator(self.model, processed_data)
            
            cpds = estimator.get_parameters(
                prior_type='K2', 
                equivalent_sample_size=equivalent_sample_size, # Use the new smoothing parameter
                n_jobs=optimal_n_jobs
            )

            self.model.add_cpds(*cpds)
            
            if self.model.check_model():
                self.trained = True
                self.inference = VariableElimination(self.model)
                if verbose:
                    print("âœ… Training complete with Bayesian smoothing.")
                return True
            else:
                if verbose:
                    print("âŒ Model check failed after training.")
                return False
                
        except Exception as e:
            if verbose:
                import traceback
                print(f"â€¼ï¸ Training error: {str(e)}")
                traceback.print_exc()
            return False

    def _fast_preprocess_for_training(self, data):
        """Ultra-optimized preprocessing for training only (minimal operations)"""
        # Pre-allocate with zeros (fastest initialization)
        result_df = pd.DataFrame(0, index=data.index, columns=self.all_nodes, dtype=np.int8)
        
        # VECTORIZED skill mapping (preserved but optimized)
        skill_concepts = data['skill_name'].apply(self.data_mapper.map_skill_to_concept)
        valid_concepts = skill_concepts.dropna()
        
        # Batch set concept nodes (ultra-fast)
        for concept_name in valid_concepts.unique():
            if concept_name in self.conceptual_nodes:
                concept_mask = (skill_concepts == concept_name)
                result_df.loc[concept_mask, concept_name] = 1
        
        # FAST error processing for incorrect answers only
        incorrect_mask = (data['correct'] == 0)
        
        if incorrect_mask.sum() > 0:
            # Vectorized behavioral evidence (ultra-fast)
            incorrect_data = data[incorrect_mask]
            
            # Multiple attempts
            result_df.loc[incorrect_data[incorrect_data['attempt_count'] > 1].index, 'E_Multiple_Attempts'] = 1
            
            # Hint usage  
            result_df.loc[incorrect_data[incorrect_data['hint_count'] > 0].index, 'E_Hint_Usage'] = 1
            
            # Slow response
            result_df.loc[incorrect_data[incorrect_data['ms_first_response'] > 15000].index, 'E_Slow_Response'] = 1
            
            # OPTIMIZATION: Skip specific error analysis for training (major speedup)
            # Only do basic calculation and conceptual errors
            result_df.loc[incorrect_mask, 'E_Calculation_Error'] = 1
            
            # Simple rule-based specific errors (much faster than full text analysis)
            for idx in incorrect_data.index:
                skill_concept = skill_concepts.loc[idx]
                if pd.notna(skill_concept):
                    if skill_concept in ['U_Integer_Operations', 'U_Algebraic_Equations']:
                        result_df.loc[idx, 'E_Sign_Error'] = 1
                    elif skill_concept == 'U_Fraction_Operations':
                        result_df.loc[idx, 'E_Fraction_Error'] = 1
                    elif skill_concept in ['U_Decimal_Operations', 'U_Percent_Operations']:
                        result_df.loc[idx, 'E_Decimal_Error'] = 1
                    elif skill_concept == 'U_Order_Of_Operations':
                        result_df.loc[idx, 'E_Order_Error'] = 1
                    elif skill_concept in ['U_Geometry_Measurement', 'U_Unit_Conversion']:
                        result_df.loc[idx, 'E_Unit_Error'] = 1
        
        return result_df

    # In class ProductionReadyDiagnosticNetwork:
    def _ultra_fast_preprocess(self, data):
        """
        Unified preprocessor that now uses the single, authoritative 
        extract_error_evidence method to ensure consistency between
        training and validation data.
        """
        # Initialize the result DataFrame as before
        result_df = pd.DataFrame(0, index=data.index, columns=self.all_nodes, dtype=np.int8)
        
        # --- Skill Mapping (Unchanged) ---
        if not hasattr(self.data_mapper, '_skill_cache'):
            unique_skills = data['skill_name'].dropna().unique()
            self.data_mapper._skill_cache = {
                skill: self.data_mapper.map_skill_to_concept(skill) for skill in unique_skills
            }
        skill_concepts = data['skill_name'].map(self.data_mapper._skill_cache)
        
        for concept_name in skill_concepts.dropna().unique():
            if concept_name in self.conceptual_nodes:
                result_df.loc[skill_concepts == concept_name, concept_name] = 1
                
        # --- UNIFIED EVIDENCE EXTRACTION ---
        # Process only the incorrect answers for efficiency
        incorrect_mask = (data['correct'] == 0)
        if incorrect_mask.any():
            incorrect_data = data[incorrect_mask]
            
            # Apply the single, authoritative evidence extractor
            evidence_series = incorrect_data.apply(self.data_mapper.extract_error_evidence, axis=1)
            
            # Populate the result_df from the extracted evidence dictionaries
            for idx, evidence_dict in evidence_series.items():
                for error_node, value in evidence_dict.items():
                    if error_node in self.error_nodes:
                        result_df.loc[idx, error_node] = value # Use the value from the dictionary
                        
        return result_df

    def _calculate_optimal_n_jobs(self, data, requested_n_jobs):
        """Calculate optimal number of jobs based on data size and model complexity"""
        data_size = len(data)
        num_nodes = len(self.all_nodes)
        
        # For small datasets or simple models, parallelism adds overhead
        if data_size < 1000 or num_nodes < 10:
            return 1
        
        # For medium datasets, use moderate parallelism
        if data_size < 10000:
            return min(2, requested_n_jobs if requested_n_jobs > 0 else 2)
        
        # For large datasets, use full parallelism
        if requested_n_jobs == -1:
            import multiprocessing
            return multiprocessing.cpu_count()
        
        return max(1, requested_n_jobs)

    def diagnose_misconceptions_with_confidence(self, observed_errors, threshold=0.25):
        """Enhanced diagnosis with confidence assessment - PRESERVED"""
        if not self.trained or not observed_errors:
            return pd.DataFrame(columns=['Misconception', 'Probability', 'Confidence_Level'])
        
        # Calculate evidence strength
        evidence_strength = self.data_mapper.calculate_evidence_strength(observed_errors)
        
        results = {}
        
        for concept in self.conceptual_nodes:
            try:
                posterior = self.inference.query(variables=[concept], evidence=observed_errors)
                prob = posterior.values[1] if len(posterior.values) > 1 else 0.0
                results[concept] = prob
            except:
                results[concept] = 0.0
        
        # Create enhanced results dataframe
        df = pd.DataFrame(list(results.items()), columns=['Misconception', 'Probability'])
        
        # Enhanced confidence calculation based on specific evidence
        df['Confidence_Level'] = df['Probability'].apply(lambda p: 
            'High' if p > 0.6 and evidence_strength > 0.7 else
            'Medium' if p > 0.4 and evidence_strength > 0.4 else
            'Low'
        )
        
        # Filter and sort by probability
        confident_results = df[df['Probability'] >= threshold].sort_values(
            'Probability', ascending=False
        ).reset_index(drop=True)
        
        return confident_results
    
    # In class ProductionReadyDiagnosticNetwork:
    def cross_validate(self, data, k_folds=5, verbose=False):
        """
        Cross-validation with a new, smarter validation logic based on a 
        "Hierarchy of Evidence" to provide a more meaningful accuracy score.
        (This version fixes the frozenset conversion bug).
        """
        if verbose:
            print(f"\nðŸš€ SMART-VALIDATION {k_folds}-Fold Cross-Validation")
            print("=" * 70)

        cv_start_time = time.time()
        
        clean_data = self.clean_data(data, verbose=False)
        self._data_is_clean = True
        base_edges = list(self.model.edges())
        
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_accuracies = []

        # Define which error nodes are considered "specific" vs. "behavioral"
        specific_error_nodes = {
            'E_Sign_Error', 'E_Fraction_Error', 'E_Decimal_Error', 
            'E_Order_Error', 'E_Unit_Error'
        }

        for fold, (train_idx, val_idx) in enumerate(kfold.split(clean_data)):
            fold_start_time = time.time()
            train_fold = clean_data.iloc[train_idx]
            val_fold = clean_data.iloc[val_idx]

            if verbose:
                print(f"FOLD {fold + 1}/{k_folds} (Train: {len(train_fold):,}, Val: {len(val_fold):,})")

            # --- Training (remains unchanged, using _ultra_fast_preprocess) ---
            temp_network = ProductionReadyDiagnosticNetwork()
            temp_network.model = DiscreteBayesianNetwork(base_edges)
            temp_network._data_is_clean = True
            
            success = temp_network.train_with_data(
                train_fold, 
                use_balancing=False,
                verbose=False, 
                n_jobs=2
            )
            
            if not success:
                if verbose: print(f"  âš ï¸ Training failed, skipping fold")
                continue
            
            # --- Enhanced Validation Logic ---
            incorrect_examples = val_fold[val_fold['correct'] == 0].copy()
            
            if incorrect_examples.empty:
                fold_accuracies.append(1.0) # No incorrect examples to be wrong about
                continue
            
            incorrect_examples['evidence'] = incorrect_examples.apply(
                temp_network.data_mapper.extract_error_evidence, axis=1
            )
            incorrect_examples['true_concept'] = incorrect_examples['skill_name'].apply(
                temp_network.data_mapper.map_skill_to_concept
            )
            
            validation_set = incorrect_examples.dropna(subset=['evidence', 'true_concept'])
            validation_set = validation_set[validation_set['evidence'].astype(bool)]

            if validation_set.empty:
                fold_accuracies.append(1.0)
                continue

            correct_predictions = 0
            total_predictions = 0
            
            # --- FIX IS HERE ---
            # Correctly convert the dictionary of evidence to a frozenset of its items
            validation_set['evidence_key'] = validation_set['evidence'].apply(lambda x: frozenset(x.items()))
            
            evidence_groups = validation_set.groupby('evidence_key')
            
            for evidence_key, group in evidence_groups:
                # evidence_key is now a frozenset of tuples, which dict() can handle
                evidence_dict = dict(evidence_key)
                
                predicted_diagnoses = temp_network.diagnose_misconceptions_with_confidence(
                    evidence_dict, threshold=0.1
                )
                
                if predicted_diagnoses.empty:
                    total_predictions += len(group)
                    continue

                # Hierarchy of Evidence Logic
                evidence_nodes = set(evidence_dict.keys())
                has_specific_evidence = any(e in specific_error_nodes for e in evidence_nodes)
                
                if has_specific_evidence:
                    top_prediction = predicted_diagnoses['Misconception'].iloc[0]
                    correct_in_group = (group['true_concept'] == top_prediction).sum()
                else:
                    predicted_concepts = set(predicted_diagnoses['Misconception'].values)
                    correct_in_group = group['true_concept'].isin(predicted_concepts).sum()

                correct_predictions += correct_in_group
                total_predictions += len(group)

            fold_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 1.0
            fold_accuracies.append(fold_accuracy)

            fold_time = time.time() - fold_start_time
            if verbose:
                print(f"  âœ… Fold completed in {fold_time:.1f}s (Smart Accuracy: {fold_accuracy:.1%})")
                
        # Final summary
        total_cv_time = time.time() - cv_start_time
        if verbose:
            print(f"\n" + "=" * 70)
            print(f"ðŸ† SMART CROSS-VALIDATION COMPLETE")
            print(f"=" * 70)
            
        if fold_accuracies:
            avg_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            if verbose:
                print(f"Average Smart Accuracy: {avg_accuracy:.1%} Â± {std_accuracy:.1%}")
                print(f"Total Time: {total_cv_time:.1f}s ({total_cv_time/60:.1f} min)")
            return avg_accuracy, std_accuracy, fold_accuracies
        else:
            if verbose: print("âŒ Cross-validation failed.")
            return 0.0, 0.0, []

# --- Enhanced Hybrid Diagnostic System (from Phase 4 + V3 Implementation) ---
class EnhancedHybridDiagnosticSystem:
    def __init__(self, model_path='diagnostic_model.pkl'):
        print("🚀 Initializing Enhanced Hybrid Diagnostic System for Phase 5...")
        
        self.bayesian_network = self.load_model_robustly(model_path)
        if not self.bayesian_network:
            raise ValueError("Failed to load a valid Bayesian Network model.")
        
        self.load_api_secrets()
        self.setup_llm_clients()
        print("✅ Enhanced Hybrid Diagnostic System initialized successfully!")

    def load_model_robustly(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                model_object = pickle.load(f)
            if model_object and getattr(model_object, 'trained', False):
                print(f"✅ Successfully loaded trained Bayesian Network from '{model_path}'")
                return model_object
            else:
                print(f"❌ FATAL ERROR: The file '{model_path}' is not a valid, trained model object.")
                return None
        except FileNotFoundError:
            print(f"❌ FATAL ERROR: Model file not found at '{model_path}'.")
            return None
        except Exception as e:
            print(f"❌ An unexpected error occurred while loading '{model_path}': {e}")
            return None

    def load_api_secrets(self):
        if DOTENV_AVAILABLE: 
            load_dotenv()
        
        self.gemini_keys = [k.strip() for k in os.getenv("Gemini_Secrets", "").split(',') if k.strip()]
        self.deepseek_keys = [k.strip() for k in os.getenv("Deepseek_Secrets", "").split(',') if k.strip()]
        
        if not self.gemini_keys or not self.deepseek_keys:
            raise ValueError("API keys for Gemini and/or Deepseek not found.")
        
        self.gemini_key_iterator = itertools.cycle(self.gemini_keys)
        self.deepseek_key_iterator = itertools.cycle(self.deepseek_keys)
        print(f"🔑 Loaded {len(self.gemini_keys)} Gemini and {len(self.deepseek_keys)} Deepseek keys.")

    def setup_llm_clients(self):
        if GEMINI_AVAILABLE: 
            self.rotate_gemini_key()
        self.rotate_deepseek_key()

    def rotate_gemini_key(self):
        try:
            self.current_gemini_key = next(self.gemini_key_iterator)
            genai.configure(api_key=self.current_gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
            print("🤖 Gemini 2.5 Pro configured with rotated key.")
        except Exception as e:
            print(f"Failed to configure Gemini: {e}")

    def rotate_deepseek_key(self):
        self.current_deepseek_key = next(self.deepseek_key_iterator)
        print("🧠 Deepseek client configured with rotated key.")

    def get_gemini_hypothesis(self, student_work):
        if not GEMINI_AVAILABLE: 
            raise RuntimeError("Gemini not available - missing google-generativeai package")
        
        prompt = self.create_error_analysis_prompt(student_work)

        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Print the entire response object to see everything
            print(f"FULL GEMINI RESPONSE: {response}") 
            
            # Specifically check for block reasons
            if response.prompt_feedback.block_reason:
                print(f"❌ GEMINI PROMPT BLOCKED: Reason = {response.prompt_feedback.block_reason.name}")
                return {} # Return empty if blocked
                
            print(f"RAW RESPONSE TEXT: {response.text}")
            return self.parse_llm_response(response.text)

        except Exception as e:
            print(f"❌ AN ERROR OCCURRED DURING GEMINI CALL: {e}")
            return {}


    def get_deepseek_hypothesis(self, student_work):
        client = OpenAI(
            api_key=self.current_deepseek_key, 
            base_url="https://api.deepseek.com"
        )
        prompt = self.create_error_analysis_prompt(student_work)
        
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        return self.parse_llm_response(response.choices[0].message.content)

    def run_bn_only(self):
        # ✅ CHANGE 1: Lowered threshold to increase BN sensitivity
        diagnosis = self.bayesian_network.diagnose_misconceptions_with_confidence({}, threshold=0.01)
        return {
            'processed_evidence': {},
            'diagnosis': diagnosis.to_dict('records') if not diagnosis.empty else []
        }

    def run_hybrid_v1_validator(self, llm_hypothesis):
        valid_evidence, contradicted = {}, {}
        
        if llm_hypothesis:
            for var, state in llm_hypothesis.items():
                is_valid, valid_states = self.is_evidence_valid(var, state)
                if is_valid:
                    valid_evidence[var] = state
                else:
                    contradicted[var] = {
                        'provided': state, 
                        'allowed': valid_states, 
                        'action': 'discarded'
                    }
        
        # ✅ CHANGE 1: Lowered threshold for consistency
        diagnosis = self.bayesian_network.diagnose_misconceptions_with_confidence(valid_evidence, threshold=0.01)
        return self.format_result(llm_hypothesis, valid_evidence, contradicted, diagnosis)

    def run_hybrid_v2_synthesizer(self, llm_hypothesis):
        """FINAL Synthesizer: Maps LLM errors to concepts, then combines."""
        
        error_to_concept_map = {
            'E_Sign_Error': ['U_Linear_Equations', 'U_Inequalities', 'U_Integer_Operations'],
            'E_Calculation_Error': ['U_Quadratic_Equations', 'U_Area_Perimeter', 'U_Volume_Surface_Area', 'U_Fraction_Arithmetic', 'U_Decimal_Operations', 'U_Integer_Operations'],
            'E_Conceptual_Confusion': ['U_Linear_Equations', 'U_Quadratic_Equations', 'U_Inequalities', 'U_Order_Of_Operations', 'U_Advanced_Math', 'U_Approximation'],
            'E_Unit_Error': ['U_Area_Perimeter', 'U_Volume_Surface_Area', 'U_Unit_Conversion'],
            'E_Fraction_Error': ['U_Fraction_Arithmetic', 'U_Fraction_Concepts', 'U_Percent_Operations'],
            'E_Decimal_Error': ['U_Percent_Operations', 'U_Decimal_Operations'],
            'E_Order_Error': ['U_Order_Of_Operations']
        }
        
        bn_alone_result = self.run_bn_only()
        bn_diagnoses = bn_alone_result.get('diagnosis', [])
        bn_positive_nodes = {d['Misconception']: 1 for d in bn_diagnoses}
        
        llm_positive_errors = {k: v for k, v in llm_hypothesis.items() if v == 1 and k in error_to_concept_map}
        
        llm_implied_concepts = set()
        for error in llm_positive_errors:
            concepts = error_to_concept_map.get(error, [])
            for concept in concepts:
                llm_implied_concepts.add(concept)
                
        final_diagnosis_set = set()
        for node in self.bayesian_network.conceptual_nodes:
            if bn_positive_nodes.get(node, 0) == 1:
                final_diagnosis_set.add(node)
            if node in llm_implied_concepts:
                final_diagnosis_set.add(node)
                
        final_diagnosis = [{'Misconception': node, 'Probability': 1.0, 'Source': 'BN+LLM Synthesis'} for node in final_diagnosis_set]
        
        return {
            'llm_hypothesis': llm_hypothesis,
            'bn_baseline': bn_positive_nodes,
            'llm_implied_concepts': list(llm_implied_concepts),
            'diagnosis': final_diagnosis
        }

    def run_hybrid_v3_combined(self, llm_hypothesis):
        """
        V3.3 - DYNAMIC FEEDBACK LOOP:
        An intelligent, two-stage inference process that attempts a strict
        diagnosis first, and if the result is weak, it re-runs the diagnosis
        with a relaxed evidence set to avoid discarding potentially useful,
        albeit unverified, signals from the LLM.
        """
        print("🧠 Running V3 Diagnostic: Dynamic Feedback Loop")
        
        # --- STAGE 1: The Strict Pass ---
        # First, run the validator under strict rules. This prioritizes precision.
        print("   - Stage 1: Running with STRICT validation...")
        strict_v1_result = self.run_hybrid_v1_validator(llm_hypothesis)
        strict_diagnosis = strict_v1_result.get('diagnosis', [])
        
        # Define the evidence to be used by the synthesizer. Start with the strict set.
        evidence_for_synthesis = strict_v1_result.get('processed_evidence', {})
        run_mode = "Strict"
        
        # --- STAGE 2: The Feedback Loop and Relaxed Pass ---
        # Check if the strict diagnosis is weak (empty or low confidence).
        # We define "weak" as having no diagnoses.
        if not strict_diagnosis:
            print("   - Stage 1 Result: WEAK. Diagnosis empty.")
            print("   - Stage 2: Relaxing validation. Re-running with FULL LLM hypothesis...")
            # The strict pass failed. Relax the rules and use the ORIGINAL,
            # unfiltered LLM hypothesis as the evidence for the synthesizer.
            # This prioritizes recall to avoid missing a potential diagnosis.
            evidence_for_synthesis = llm_hypothesis
            run_mode = "Relaxed"
        else:
            print("   - Stage 1 Result: STRONG. Using validated evidence.")

        # --- FINAL SYNTHESIS ---
        # Run the V2 synthesizer using the evidence selected by the logic above.
        final_synthesized_result = self.run_hybrid_v2_synthesizer(evidence_for_synthesis)
        
        # --- COMBINE AND RETURN ---
        # The final result includes the output from the initial strict run,
        # the final synthesized diagnosis, and a note on which mode was used.
        combined_result = {
            'llm_hypothesis': llm_hypothesis,
            'strict_v1_result': strict_v1_result, # Always show the initial strict attempt
            'final_run_mode': run_mode,
            'synthesized_diagnosis': final_synthesized_result
        }
        
        # To match the output format of the other models, we'll lift the
        # final diagnosis to the top level.
        combined_result['diagnosis'] = final_synthesized_result.get('diagnosis', [])

        return combined_result

    def is_evidence_valid(self, var, state):
        bn_model = self.bayesian_network.model
        if var not in bn_model.nodes(): 
            return False, []
        try:
            cpd = bn_model.get_cpds(var)
            valid_states = list(range(cpd.variable_card))
            return state in valid_states, valid_states
        except Exception:
            return False, []
    
    def format_result(self, hypothesis, evidence, contradicted, diagnosis):
        return {
            'llm_hypothesis': hypothesis,
            'processed_evidence': evidence,
            'contradicted_evidence': contradicted,
            'diagnosis': diagnosis.to_dict('records') if not diagnosis.empty else []
        }

    def create_error_analysis_prompt(self, student_work):
        error_types = list(self.bayesian_network.data_mapper.error_patterns.keys())
        return f"""
Analyze the student's work for specific error patterns.
Problem: {student_work.get('question', 'N/A')}
Student's Answer: {student_work.get('answer_text', 'N/A')}
Skill: {student_work.get('skill_name', 'N/A')}
Available Error Types: {', '.join(error_types)}
Instructions: Return ONLY a JSON object with error types as keys and 1 as the value.
Example: {{"E_Order_Error": 1, "E_Calculation_Error": 1}}
Your Response:
"""

    def parse_llm_response(self, text):
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group(0)) if match else {}
        except Exception:
            return {}


# --- Phase 5 Evaluator ---
class Phase5Evaluator:
    def __init__(self, model_path='diagnostic_model.pkl'):
        self.diagnostic_system = EnhancedHybridDiagnosticSystem(model_path)

    def process_dataset(self, test_file, results_file, ongoing_file):
        """Process a single dataset with progress saving and error handling"""
        print(f"\n📊 Processing dataset: {test_file}")
        
        if not os.path.exists(test_file):
            print(f"❌ ERROR: Dataset {test_file} not found.")
            return
        
        df = pd.read_csv(test_file)
        print(f"📋 Dataset contains {len(df)} examples")
        
        results = []
        errors = []
        
        for i, row in df.iterrows():
            student_work = {
                'question': row.get('question', ''),
                'answer_text': row.get('answer_text', ''),
                'skill_name': row.get('skill_name', '')
            }
            
            try:
                # Run all models for this student work
                model_outputs = self.run_all_models(student_work)
                true_skill = row.get('skill_name', '')
                
                # Process results for each model
                for model_name, model_results in model_outputs.items():
                    if model_name == 'BN_Alone':
                        # BN alone has single output
                        diagnosis_list = model_results.get('diagnosis', [])
                        predicted_skills = [d['Misconception'] for d in diagnosis_list] if diagnosis_list else []
                        accuracy = any(true_skill == pred for pred in predicted_skills)
                        
                        results.append({
                            'original_index': i,  # NEW: Add original row index as first column
                            'model': model_name,
                            'llm': 'N/A',
                            'true_skill': true_skill,
                            'predicted_skills': predicted_skills,
                            'accuracy': accuracy,
                            'diagnosis_json': json.dumps(diagnosis_list)
                        })
                    else:
                        # LLM-based models have results for each LLM
                        for llm_name, result_set in model_results.items():
                            if model_name == 'LLM_Alone':
                                # LLM alone just returns hypothesis
                                predicted_skills = list(result_set.keys()) if result_set else []
                                accuracy = any(true_skill == pred for pred in predicted_skills)
                                results.append({
                                    'original_index': i,  # NEW: Add original row index as first column
                                    'model': model_name,
                                    'llm': llm_name,
                                    'true_skill': true_skill,
                                    'predicted_skills': predicted_skills,
                                    'accuracy': accuracy,
                                    'diagnosis_json': json.dumps(result_set)
                                })
                            else:
                                # Hybrid models return diagnosis
                                diagnosis_list = result_set.get('diagnosis', [])
                                predicted_skills = [d['Misconception'] for d in diagnosis_list] if diagnosis_list else []
                                accuracy = any(true_skill == pred for pred in predicted_skills)
                                
                                results.append({
                                    'original_index': i,  # NEW: Add original row index as first column
                                    'model': model_name,
                                    'llm': llm_name,
                                    'true_skill': true_skill,
                                    'predicted_skills': predicted_skills,
                                    'accuracy': accuracy,
                                    'diagnosis_json': json.dumps(diagnosis_list)
                                })

                # Progress saving every 20 rows
                if i % 20 == 0:
                    pd.DataFrame(results).to_csv(results_file, index=False)
                    pd.DataFrame(errors).to_csv(ongoing_file, index=False)
                    print(f"💾 Progress saved at row {i}")

            except Exception as e:
                print(f"❌ Error processing row {i}: {e}")
                
                # Rotate keys on API failures
                if "API key not valid" in str(e) or "401" in str(e) or "403" in str(e):
                    try:
                        self.diagnostic_system.rotate_gemini_key()
                        self.diagnostic_system.rotate_deepseek_key()
                    except:
                        pass
                
                # Record the error with timestamp
                errors.append({
                    "original_index": i,  # NEW: Add original row index here too
                    "row_index": i,       # Keep existing for compatibility
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "data": row.to_dict(),
                    "timestamp": pd.Timestamp.now().isoformat()
                })
                
                # Save progress immediately on error
                pd.DataFrame(results).to_csv(results_file, index=False)
                pd.DataFrame(errors).to_csv(ongoing_file, index=False)
                print(f"💾 Error saved to {ongoing_file}")
                
                continue  # Skip to next row



        # Final save
        pd.DataFrame(results).to_csv(results_file, index=False)
        pd.DataFrame(errors).to_csv(ongoing_file, index=False)
        
        print(f"✅ Finished processing {test_file}")
        print(f"📈 Results: {len(results)} records saved to {results_file}")
        print(f"⚠️  Errors: {len(errors)} errors saved to {ongoing_file}")

    def run_all_models(self, student_work):
        """Run all diagnostic models on student work"""
        results = {}
        
        # Get LLM hypotheses
        gemini_hypothesis = self.diagnostic_system.get_gemini_hypothesis(student_work)
        deepseek_hypothesis = self.diagnostic_system.get_deepseek_hypothesis(student_work)
        
        # 1. LLM Alone
        results['LLM_Alone'] = {
            'Gemini-2.5-pro': gemini_hypothesis,
            'Deepseek-reasoner': deepseek_hypothesis
        }
        print(("LLM Alone Results:", results['LLM_Alone']))  # Debugging line
        
        # 2. BN Alone
        results['BN_Alone'] = self.diagnostic_system.run_bn_only()
        print(("BN Alone Results:", results['BN_Alone']))  # Debugging line
        
        # 3. V1 (Validator)
        results['Hybrid_V1_Validator'] = {
            'Gemini-2.5-pro': self.diagnostic_system.run_hybrid_v1_validator(gemini_hypothesis),
            'Deepseek-reason': self.diagnostic_system.run_hybrid_v1_validator(deepseek_hypothesis)
        }
        print(("Hybrid V1 Results:", results['Hybrid_V1_Validator']))  # Debugging line
        
        # 4. V2 (Synthesizer) 
        results['Hybrid_V2_Synthesizer'] = {
            'Gemini-2.5-pro': self.diagnostic_system.run_hybrid_v2_synthesizer(gemini_hypothesis),
            'Deepseek-reason': self.diagnostic_system.run_hybrid_v2_synthesizer(deepseek_hypothesis)
        }
        print(("Hybrid V2 Results:", results['Hybrid_V2_Synthesizer']))  # Debugging line
        
        # 5. V3 (Combined - NEW!)
        results['Hybrid_V3_Combined'] = {
            'Gemini-2.5-pro': self.diagnostic_system.run_hybrid_v3_combined(gemini_hypothesis),
            'Deepseek-reason': self.diagnostic_system.run_hybrid_v3_combined(deepseek_hypothesis)
        }
        print(("Hybrid V3 Results:", results['Hybrid_V3_Combined']))  # Debugging line
        
        return results

# --- Main Execution ---
def main():
    print("=" * 80)
    print("🎓 PHASE 5: COMPREHENSIVE MODEL EVALUATION FRAMEWORK")
    print("=" * 80)
    
    try:
        evaluator = Phase5Evaluator('diagnostic_model.pkl')
        
        # Define datasets to process sequentially
        datasets = [
            ('test_general_benchmark.csv', 'results_general_benchmark.csv', 'ongoing_general_benchmark.csv'),
            ('test_rare_skills.csv', 'results_rare_skills.csv', 'ongoing_rare_skills.csv'),
            ('test_high_challenge.csv', 'results_high_challenge.csv', 'ongoing_high_challenge.csv')
        ]
        
        # Process each dataset one by one
        for test_file, results_file, ongoing_file in datasets:
            evaluator.process_dataset(test_file, results_file, ongoing_file)
            
            # Brief pause between datasets to avoid overwhelming APIs
            print("⏳ Brief pause between datasets...")
            time.sleep(5)
        
        print("\n🎉 Phase 5 Comprehensive Evaluation Complete!")
        print("📊 All results have been saved to their respective CSV files.")
        
    except Exception as e:
        import traceback
        print(f"\n❌ A critical error occurred: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
