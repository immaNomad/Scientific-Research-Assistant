"""
Reinforcement Learning Optimizer for Research Assistant
Learns to optimize queries, select sources, and rank papers
"""

import asyncio
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
import random

from src.rag.enhanced_rag import EnhancedRAG, PaperInfo, ResearchAnalysis

@dataclass
class RLState:
    """State representation for RL agent"""
    query_keywords: List[str]
    query_length: int
    domain: str  # 'cybersecurity', 'healthcare', 'ai', etc.
    sources_available: List[str]
    previous_results_quality: float
    search_iteration: int

@dataclass
class RLAction:
    """Action representation for RL agent"""
    query_modifications: List[str]  # Add/remove keywords
    source_selection: List[str]     # Which sources to use
    search_parameters: Dict         # Max results, filters, etc.

@dataclass
class RLReward:
    """Reward calculation for RL feedback"""
    paper_relevance_score: float    # 0-1 based on relevance
    result_count_score: float       # Penalty for too few/many results
    processing_time_score: float    # Reward for faster processing
    user_feedback_score: float      # Optional user rating
    total_reward: float

class QueryOptimizer:
    """RL Agent for optimizing search queries"""
    
    def __init__(self):
        self.q_table = {}  # State-Action value table
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.3  # Exploration rate
        self.experience_file = "./data/rl_experience.json"
        self.load_experience()
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Simple keyword extraction (can be enhanced with NLP)
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        words = query.lower().replace(',', ' ').replace('.', ' ').split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Limit to top 10 keywords
    
    def detect_domain(self, query: str) -> str:
        """Detect research domain from query"""
        domain_keywords = {
            'cybersecurity': [
                # Core security terms
                'security', 'cybersecurity', 'infosec', 'information security', 'cyber', 'secure', 'safety',
                # Attacks and threats
                'attack', 'threat', 'exploit', 'breach', 'intrusion', 'penetration', 'hacking', 'hack',
                'phishing', 'spear phishing', 'social engineering', 'ransomware', 'malware', 'virus',
                'trojan', 'worm', 'spyware', 'adware', 'rootkit', 'backdoor', 'botnet', 'ddos',
                # Vulnerabilities and risks
                'vulnerability', 'zero-day', 'cve', 'risk', 'exposure', 'weakness', 'flaw', 'bug',
                # Security measures
                'encryption', 'cryptography', 'authentication', 'authorization', 'access control',
                'firewall', 'antivirus', 'ids', 'ips', 'siem', 'endpoint', 'network security',
                'application security', 'web security', 'cloud security', 'mobile security',
                # Security practices
                'incident response', 'forensics', 'compliance', 'governance', 'audit', 'penetration testing',
                'security assessment', 'threat modeling', 'risk management', 'security awareness'
            ],
            'healthcare': [
                # General medical terms
                'medical', 'health', 'patient', 'diagnosis', 'treatment', 'clinical', 'medicine', 'healthcare',
                'hospital', 'clinic', 'physician', 'doctor', 'nurse', 'therapy', 'therapeutic',
                # Diseases and conditions
                'cancer', 'tumor', 'carcinoma', 'oncology', 'metastasis', 'malignant', 'benign',
                'diabetes', 'insulin', 'glucose', 'blood sugar', 'diabetic', 'glycemic',
                'hypertension', 'blood pressure', 'cardiovascular', 'heart', 'cardiac', 'coronary',
                'stroke', 'cerebrovascular', 'brain', 'neurological', 'alzheimer', 'dementia',
                'covid', 'coronavirus', 'pandemic', 'epidemic', 'infectious', 'contagious',
                'pneumonia', 'respiratory', 'lung', 'pulmonary', 'asthma', 'copd',
                'infection', 'bacterial', 'viral', 'fungal', 'pathogen', 'microbe',
                'disease', 'illness', 'condition', 'syndrome', 'disorder', 'pathology',
                'mental health', 'depression', 'anxiety', 'psychiatric', 'psychological',
                'obesity', 'overweight', 'metabolic', 'nutrition', 'diet', 'dietary',
                'arthritis', 'osteoporosis', 'fracture', 'bone', 'joint', 'musculoskeletal',
                'kidney', 'renal', 'liver', 'hepatic', 'gastrointestinal', 'digestive',
                # Treatments and interventions
                'medication', 'drug', 'pharmaceutical', 'medicine', 'prescription', 'dose', 'dosage',
                'therapy', 'treatment', 'intervention', 'procedure', 'surgery', 'surgical', 'operation',
                'immunotherapy', 'chemotherapy', 'radiation', 'radiotherapy', 'vaccine', 'vaccination',
                'antibiotic', 'antiviral', 'antifungal', 'analgesic', 'anesthetic', 'steroid',
                'transplant', 'transfusion', 'dialysis', 'rehabilitation', 'physiotherapy',
                # Medical outcomes and measures
                'efficacy', 'effectiveness', 'outcome', 'result', 'response', 'adverse', 'side effect',
                'mortality', 'death', 'survival', 'morbidity', 'recovery', 'remission', 'relapse',
                'prevention', 'prophylaxis', 'screening', 'early detection', 'diagnostic', 'prognosis',
                'quality of life', 'patient satisfaction', 'safety', 'toxicity', 'contraindication',
                # Medical research and methodology
                'trial', 'clinical trial', 'study', 'research', 'randomized', 'controlled', 'placebo',
                'double-blind', 'single-blind', 'crossover', 'longitudinal', 'retrospective', 'prospective',
                'epidemiology', 'cohort', 'case-control', 'case study', 'systematic review', 'meta-analysis',
                'biomarker', 'biostatistics', 'evidence-based', 'peer-reviewed', 'protocol',
                # Medical specialties
                'cardiology', 'neurology', 'oncology', 'pediatrics', 'geriatrics', 'psychiatry',
                'dermatology', 'ophthalmology', 'orthopedics', 'radiology', 'pathology', 'anesthesiology',
                'emergency medicine', 'family medicine', 'internal medicine', 'surgery'
            ],
            'ai': [
                # Core AI terms
                'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning', 'neural',
                'neural network', 'deep neural network', 'artificial neural network', 'ann', 'dnn',
                'algorithm', 'model', 'training', 'learning', 'intelligence', 'cognitive',
                # ML algorithms and techniques
                'supervised learning', 'unsupervised learning', 'reinforcement learning', 'semi-supervised',
                'classification', 'regression', 'clustering', 'dimensionality reduction', 'feature selection',
                'decision tree', 'random forest', 'support vector machine', 'svm', 'naive bayes',
                'k-means', 'linear regression', 'logistic regression', 'gradient descent', 'backpropagation',
                # Deep learning architectures
                'convolutional neural network', 'cnn', 'recurrent neural network', 'rnn', 'lstm', 'gru',
                'transformer', 'attention', 'self-attention', 'bert', 'gpt', 'autoencoder', 'gan',
                'generative adversarial network', 'variational autoencoder', 'vae', 'resnet', 'inception',
                # NLP and language
                'natural language processing', 'nlp', 'language model', 'text mining', 'sentiment analysis',
                'named entity recognition', 'ner', 'part-of-speech', 'parsing', 'tokenization',
                'word embedding', 'word2vec', 'glove', 'fasttext', 'embedding', 'language generation',
                # Computer vision
                'computer vision', 'image recognition', 'object detection', 'image classification',
                'facial recognition', 'optical character recognition', 'ocr', 'image processing',
                'feature extraction', 'edge detection', 'segmentation', 'tracking',
                # AI applications and concepts
                'automation', 'robotics', 'autonomous', 'expert system', 'knowledge base',
                'fuzzy logic', 'genetic algorithm', 'evolutionary computation', 'swarm intelligence',
                'optimization', 'heuristic', 'search algorithm', 'planning', 'reasoning',
                'knowledge representation', 'ontology', 'semantic web', 'chatbot', 'virtual assistant'
            ],
            'physics': [
                # Classical physics
                'physics', 'mechanics', 'thermodynamics', 'electromagnetism', 'optics', 'acoustics',
                'force', 'energy', 'momentum', 'velocity', 'acceleration', 'gravity', 'mass',
                'wave', 'frequency', 'amplitude', 'electromagnetic', 'radiation', 'spectrum',
                # Quantum physics
                'quantum', 'quantum mechanics', 'quantum physics', 'quantum computing', 'qubit',
                'superposition', 'entanglement', 'quantum field theory', 'quantum state',
                'wave function', 'uncertainty principle', 'schrodinger', 'heisenberg',
                'quantum cryptography', 'quantum information', 'quantum algorithm',
                # Particle physics
                'particle', 'elementary particle', 'subatomic', 'hadron', 'lepton', 'quark',
                'photon', 'electron', 'proton', 'neutron', 'boson', 'fermion', 'antimatter',
                'collider', 'accelerator', 'particle detector', 'standard model',
                # Relativity and cosmology
                'relativity', 'general relativity', 'special relativity', 'spacetime', 'black hole',
                'cosmology', 'universe', 'big bang', 'dark matter', 'dark energy', 'galaxy',
                'astrophysics', 'astronomy', 'stellar', 'planetary', 'gravitational wave'
            ],
            'chemistry': [
                # General chemistry
                'chemistry', 'chemical', 'molecule', 'atom', 'element', 'compound', 'reaction',
                'bond', 'ionic', 'covalent', 'metallic', 'hydrogen bond', 'valence', 'electron',
                'periodic table', 'atomic number', 'molecular weight', 'molar', 'concentration',
                # Organic chemistry
                'organic', 'hydrocarbon', 'alkane', 'alkene', 'alkyne', 'aromatic', 'benzene',
                'functional group', 'alcohol', 'aldehyde', 'ketone', 'carboxylic acid', 'ester',
                'polymer', 'synthesis', 'mechanism', 'stereochemistry', 'isomer', 'chirality',
                # Inorganic chemistry
                'inorganic', 'metal', 'crystal', 'coordination', 'ligand', 'transition metal',
                'oxide', 'salt', 'acid', 'base', 'ph', 'buffer', 'precipitation', 'solubility',
                # Analytical chemistry
                'analytical', 'spectroscopy', 'chromatography', 'mass spectrometry', 'nmr',
                'analysis', 'quantitative', 'qualitative', 'purification', 'separation',
                # Biochemistry
                'biochemistry', 'protein', 'enzyme', 'amino acid', 'dna', 'rna', 'nucleic acid',
                'metabolism', 'catalysis', 'biochemical', 'biological', 'bioorganic'
            ],
            'biology': [
                # General biology
                'biology', 'biological', 'life science', 'organism', 'species', 'evolution',
                'genetics', 'gene', 'dna', 'rna', 'genome', 'chromosome', 'mutation',
                'cell', 'cellular', 'tissue', 'organ', 'organism', 'ecosystem', 'biodiversity',
                # Molecular biology
                'molecular biology', 'protein', 'enzyme', 'amino acid', 'nucleotide', 'codon',
                'transcription', 'translation', 'replication', 'pcr', 'cloning', 'sequencing',
                'crispr', 'gene editing', 'genetic engineering', 'biotechnology', 'bioinformatics',
                # Microbiology
                'microbiology', 'bacteria', 'virus', 'microorganism', 'pathogen', 'antibiotic',
                'culture', 'fermentation', 'probiotic', 'microbiome', 'infectious disease',
                # Ecology and environment
                'ecology', 'environmental', 'conservation', 'habitat', 'population', 'community',
                'food chain', 'nutrient cycle', 'climate change', 'pollution', 'endangered',
                # Other biological fields
                'neurobiology', 'immunology', 'developmental biology', 'marine biology',
                'botany', 'zoology', 'anatomy', 'physiology', 'pharmacology', 'toxicology'
            ],
            'engineering': [
                # General engineering
                'engineering', 'engineer', 'design', 'system', 'optimization', 'efficiency',
                'manufacturing', 'production', 'quality control', 'testing', 'prototype',
                'specification', 'requirement', 'performance', 'reliability', 'safety',
                # Mechanical engineering
                'mechanical', 'thermal', 'fluid dynamics', 'heat transfer', 'combustion',
                'materials', 'stress', 'strain', 'fatigue', 'vibration', 'dynamics',
                'mechanism', 'machine', 'motor', 'engine', 'turbine', 'pump', 'compressor',
                # Electrical engineering
                'electrical', 'electronic', 'circuit', 'semiconductor', 'microprocessor',
                'signal processing', 'power', 'voltage', 'current', 'resistance', 'capacitor',
                'transistor', 'amplifier', 'filter', 'antenna', 'communication', 'wireless',
                # Civil engineering
                'civil', 'structural', 'construction', 'concrete', 'steel', 'foundation',
                'bridge', 'building', 'infrastructure', 'geotechnical', 'earthquake', 'seismic',
                # Chemical engineering
                'chemical engineering', 'process', 'reactor', 'distillation', 'separation',
                'unit operation', 'mass transfer', 'reaction engineering', 'process control',
                # Computer engineering
                'computer engineering', 'hardware', 'software', 'embedded', 'microcontroller',
                'firmware', 'real-time', 'digital signal processing', 'vlsi', 'fpga'
            ],
            'climate': [
                # Climate science
                'climate', 'climate change', 'global warming', 'greenhouse gas', 'carbon dioxide',
                'methane', 'temperature', 'precipitation', 'weather', 'atmospheric', 'ocean',
                'ice', 'glacier', 'sea level', 'polar', 'arctic', 'antarctic', 'permafrost',
                # Environmental science
                'environment', 'environmental', 'ecology', 'ecosystem', 'biodiversity',
                'conservation', 'sustainability', 'pollution', 'contamination', 'toxicity',
                'air quality', 'water quality', 'soil', 'habitat', 'deforestation', 'desertification',
                # Energy and sustainability
                'renewable', 'solar', 'wind', 'hydroelectric', 'geothermal', 'biomass',
                'fossil fuel', 'coal', 'oil', 'natural gas', 'nuclear', 'energy efficiency',
                'carbon footprint', 'emission', 'mitigation', 'adaptation', 'resilience',
                'green technology', 'clean energy', 'electric vehicle', 'battery', 'storage'
            ],
            'robotics': [
                # Core robotics
                'robot', 'robotics', 'robotic', 'automation', 'autonomous', 'control',
                'actuator', 'sensor', 'servo', 'motor', 'joint', 'manipulator', 'gripper',
                'locomotion', 'navigation', 'path planning', 'obstacle avoidance', 'mapping',
                # Robot types
                'humanoid', 'industrial robot', 'service robot', 'mobile robot', 'drone',
                'unmanned aerial vehicle', 'uav', 'autonomous vehicle', 'self-driving',
                'surgical robot', 'rehabilitation robot', 'social robot', 'companion robot',
                # AI in robotics
                'machine learning', 'computer vision', 'artificial intelligence', 'neural network',
                'reinforcement learning', 'imitation learning', 'behavior tree', 'planning',
                'perception', 'recognition', 'tracking', 'localization', 'slam',
                # Control and systems
                'control system', 'feedback', 'pid', 'trajectory', 'kinematics', 'dynamics',
                'inverse kinematics', 'forward kinematics', 'motion control', 'real-time',
                'embedded system', 'microcontroller', 'real-time operating system', 'rtos'
            ],
            'economics': [
                # Core economics
                'economics', 'economic', 'economy', 'market', 'trade', 'commerce', 'business',
                'finance', 'financial', 'investment', 'capital', 'asset', 'stock', 'bond',
                'currency', 'exchange rate', 'inflation', 'deflation', 'recession', 'growth',
                'gdp', 'unemployment', 'employment', 'labor', 'wage', 'salary', 'income',
                'poverty', 'inequality', 'distribution', 'welfare', 'subsidy', 'tax', 'fiscal',
                'monetary', 'central bank', 'interest rate', 'credit', 'debt', 'banking',
                'insurance', 'regulation', 'policy', 'international trade', 'globalization',
                'supply', 'demand', 'price', 'cost', 'profit', 'revenue', 'productivity',
                'competition', 'monopoly', 'oligopoly', 'market structure', 'elasticity'
            ],
            'psychology': [
                # Core psychology
                'psychology', 'psychological', 'behavior', 'cognitive', 'emotion', 'perception',
                'learning', 'memory', 'attention', 'consciousness', 'motivation', 'personality',
                'development', 'social', 'clinical', 'therapy', 'counseling', 'mental health',
                'depression', 'anxiety', 'stress', 'trauma', 'ptsd', 'addiction', 'disorder',
                'neuroscience', 'brain', 'neural', 'psychotherapy', 'behavioral', 'intervention'
            ],
            'mathematics': [
                # Pure mathematics
                'mathematics', 'math', 'algebra', 'calculus', 'geometry', 'topology', 'analysis',
                'number theory', 'combinatorics', 'graph theory', 'set theory', 'logic',
                'theorem', 'proof', 'lemma', 'axiom', 'conjecture', 'formula', 'equation',
                'differential equation', 'partial differential', 'linear algebra', 'matrix',
                'vector', 'eigenvalue', 'eigenvector', 'determinant', 'integral', 'derivative',
                # Applied mathematics
                'statistics', 'probability', 'stochastic', 'bayesian', 'regression', 'correlation',
                'distribution', 'hypothesis testing', 'sampling', 'variance', 'standard deviation',
                'optimization', 'linear programming', 'nonlinear', 'numerical analysis',
                'computational mathematics', 'discrete mathematics', 'cryptography'
            ],
            'computer_science': [
                # Core CS
                'computer science', 'computing', 'programming', 'software', 'algorithm',
                'data structure', 'complexity', 'computational', 'computer', 'programming language',
                'object-oriented', 'functional programming', 'compiler', 'interpreter', 'debugging',
                # Software engineering
                'software engineering', 'software development', 'agile', 'scrum', 'devops',
                'version control', 'git', 'testing', 'unit test', 'integration test', 'deployment',
                'architecture', 'design pattern', 'microservices', 'api', 'rest', 'graphql',
                # Systems and networks
                'operating system', 'linux', 'windows', 'unix', 'kernel', 'process', 'thread',
                'network', 'internet', 'protocol', 'tcp', 'ip', 'http', 'https', 'dns',
                'distributed system', 'cloud computing', 'virtualization', 'container', 'docker',
                # Databases
                'database', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'data mining',
                'big data', 'data warehouse', 'etl', 'data analytics', 'business intelligence',
                # Web development
                'web development', 'html', 'css', 'javascript', 'react', 'angular', 'vue',
                'frontend', 'backend', 'full-stack', 'responsive design', 'user interface'
            ],
            'social_sciences': [
                # Sociology
                'sociology', 'social', 'society', 'culture', 'community', 'social structure',
                'social change', 'social movement', 'inequality', 'class', 'race', 'gender',
                'ethnicity', 'migration', 'urbanization', 'globalization', 'family', 'education',
                # Political science
                'political science', 'politics', 'government', 'democracy', 'election', 'voting',
                'policy', 'governance', 'public administration', 'international relations',
                'diplomacy', 'war', 'peace', 'conflict', 'security', 'terrorism', 'human rights',
                # Anthropology
                'anthropology', 'anthropological', 'cultural', 'ethnography', 'archaeology',
                'linguistic', 'social organization', 'kinship', 'ritual', 'religion', 'belief',
                # Geography
                'geography', 'geographical', 'spatial', 'location', 'mapping', 'gis',
                'urban planning', 'regional', 'landscape', 'demography', 'population'
            ],
            'education': [
                # Educational theory and practice
                'education', 'educational', 'pedagogy', 'teaching', 'learning', 'curriculum',
                'instruction', 'assessment', 'evaluation', 'student', 'teacher', 'classroom',
                'school', 'university', 'college', 'academic', 'literacy', 'numeracy',
                'special education', 'inclusive education', 'distance learning', 'e-learning',
                'educational technology', 'edtech', 'online learning', 'blended learning',
                'educational psychology', 'cognitive load', 'motivation', 'engagement',
                'educational research', 'educational policy', 'educational reform'
            ],
            'business': [
                # Management and strategy
                'business', 'management', 'strategy', 'leadership', 'organization', 'corporate',
                'enterprise', 'company', 'firm', 'startup', 'entrepreneurship', 'innovation',
                'strategic planning', 'competitive advantage', 'market analysis', 'swot',
                # Marketing and sales
                'marketing', 'advertising', 'brand', 'customer', 'consumer', 'sales',
                'digital marketing', 'social media marketing', 'content marketing', 'seo',
                'customer relationship', 'crm', 'market research', 'segmentation', 'targeting',
                # Operations and supply chain
                'operations', 'supply chain', 'logistics', 'procurement', 'inventory',
                'quality management', 'lean', 'six sigma', 'process improvement', 'efficiency',
                # Human resources
                'human resources', 'hr', 'talent management', 'recruitment', 'training',
                'performance management', 'compensation', 'benefits', 'organizational behavior'
            ],
            'law': [
                # Legal fields
                'law', 'legal', 'legislation', 'regulation', 'statute', 'constitutional',
                'criminal law', 'civil law', 'contract', 'tort', 'property law', 'family law',
                'commercial law', 'corporate law', 'intellectual property', 'patent', 'copyright',
                'trademark', 'litigation', 'court', 'judge', 'jury', 'trial', 'appeal',
                'lawyer', 'attorney', 'legal practice', 'jurisprudence', 'legal theory',
                'international law', 'human rights law', 'environmental law', 'tax law',
                'labor law', 'employment law', 'immigration law', 'criminal justice'
            ],
            'agriculture': [
                # Farming and crops
                'agriculture', 'farming', 'crop', 'plant', 'soil', 'fertilizer', 'pesticide',
                'irrigation', 'harvest', 'yield', 'seed', 'cultivation', 'greenhouse', 'organic',
                'sustainable agriculture', 'precision agriculture', 'agricultural technology',
                # Livestock and animal science
                'livestock', 'cattle', 'dairy', 'poultry', 'swine', 'sheep', 'animal husbandry',
                'veterinary', 'animal health', 'breeding', 'genetics', 'nutrition', 'feed',
                # Food science
                'food science', 'food technology', 'food safety', 'food processing',
                'food preservation', 'packaging', 'quality control', 'nutrition', 'diet'
            ],
            'materials_science': [
                # Materials and properties
                'materials science', 'material', 'metal', 'alloy', 'steel', 'aluminum',
                'ceramic', 'polymer', 'plastic', 'composite', 'nanomaterial', 'nanotechnology',
                'crystal', 'crystalline', 'amorphous', 'phase', 'microstructure', 'grain',
                'mechanical properties', 'strength', 'hardness', 'toughness', 'elasticity',
                'thermal properties', 'electrical properties', 'magnetic properties',
                'corrosion', 'oxidation', 'fatigue', 'fracture', 'wear', 'coating',
                'surface treatment', 'heat treatment', 'processing', 'manufacturing'
            ],
            'environmental_science': [
                # Environmental studies
                'environmental science', 'environmental', 'ecology', 'ecosystem', 'biodiversity',
                'conservation', 'preservation', 'habitat', 'species', 'extinction', 'endangered',
                'pollution', 'contamination', 'waste', 'recycling', 'sustainability',
                'air quality', 'water quality', 'soil contamination', 'groundwater',
                'environmental monitoring', 'environmental assessment', 'impact assessment',
                'environmental policy', 'environmental regulation', 'green technology',
                'renewable energy', 'clean energy', 'carbon footprint', 'life cycle assessment'
            ]
        }
        
        query_lower = query.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        return 'general'
    
    def get_state(self, query: str, available_sources: List[str], previous_quality: float = 0.0) -> RLState:
        """Convert current situation to RL state"""
        keywords = self.extract_keywords(query)
        domain = self.detect_domain(query)
        
        return RLState(
            query_keywords=keywords,
            query_length=len(query.split()),
            domain=domain,
            sources_available=available_sources,
            previous_results_quality=previous_quality,
            search_iteration=0
        )
    
    def state_to_key(self, state: RLState) -> str:
        """Convert state to hashable key for Q-table"""
        return f"{state.domain}_{len(state.query_keywords)}_{len(state.sources_available)}"
    
    def get_possible_actions(self, state: RLState) -> List[RLAction]:
        """Generate possible actions for current state"""
        actions = []
        
        # Different query modification strategies
        query_strategies = [
            [],  # No modification
            ['add_synonyms'],  # Add related terms
            ['remove_stopwords'],  # Clean query
            ['add_domain_terms'],  # Add domain-specific terms
            ['make_specific'],  # Make more specific
            ['make_general']   # Make more general
        ]
        
        # Different source selection strategies (include all sources)
        source_strategies = [
            ['pubmed'],  # PubMed only (medical papers)
            ['semantic_scholar'],  # Semantic Scholar only (most reliable)
            ['arxiv'],  # arXiv only (CS/physics papers)
            ['pubmed', 'semantic_scholar'],  # Medical + general academic
            ['semantic_scholar', 'arxiv'],  # General academic + CS/physics
            ['pubmed', 'arxiv'],  # Medical + CS/physics
            ['arxiv', 'pubmed', 'semantic_scholar'],  # All sources
        ]
        
        for query_mod in query_strategies:
            for source_sel in source_strategies:
                if set(source_sel).issubset(set(state.sources_available)):
                    action = RLAction(
                        query_modifications=query_mod,
                        source_selection=source_sel,
                        search_parameters={'max_results': 5}
                    )
                    actions.append(action)
        
        return actions
    
    def select_action(self, state: RLState) -> RLAction:
        """Select action using epsilon-greedy policy"""
        state_key = self.state_to_key(state)
        possible_actions = self.get_possible_actions(state)
        
        if random.random() < self.epsilon or state_key not in self.q_table:
            # Exploration: prefer actions based on query domain
            domain = self.detect_domain(' '.join(state.query_keywords))
            
            if domain == 'healthcare':
                # For medical queries, prefer PubMed first
                pubmed_first_actions = [action for action in possible_actions 
                                      if action.source_selection and action.source_selection[0] == 'pubmed']
                if pubmed_first_actions:
                    return random.choice(pubmed_first_actions)
                
                # Fallback to any action with PubMed
                pubmed_actions = [action for action in possible_actions 
                                if 'pubmed' in action.source_selection]
                if pubmed_actions:
                    return random.choice(pubmed_actions)
            
            # For general/other queries, prefer Semantic Scholar first
            semantic_scholar_first_actions = [action for action in possible_actions 
                                            if action.source_selection and action.source_selection[0] == 'semantic_scholar']
            if semantic_scholar_first_actions:
                return random.choice(semantic_scholar_first_actions)
            
            # Random fallback
            return random.choice(possible_actions)
        else:
            # Exploitation: best known action
            action_values = self.q_table[state_key]
            best_action_hash = max(action_values, key=action_values.get)
            
            # Find the action that corresponds to this hash
            for action in possible_actions:
                action_hash = hash(str(action)) % 1000
                if action_hash == int(best_action_hash):
                    return action
            
            # Fallback if hash not found - prefer domain-appropriate sources
            domain = self.detect_domain(' '.join(state.query_keywords))
            if domain == 'healthcare':
                pubmed_actions = [action for action in possible_actions 
                                if 'pubmed' in action.source_selection]
                if pubmed_actions:
                    return random.choice(pubmed_actions)
            
            # General fallback to semantic scholar actions
            semantic_scholar_actions = [action for action in possible_actions 
                                      if 'semantic_scholar' in action.source_selection]
            if semantic_scholar_actions:
                return random.choice(semantic_scholar_actions)
            else:
                return random.choice(possible_actions)
    
    def apply_action(self, query: str, action: RLAction) -> Tuple[str, List[str]]:
        """Apply action to modify query and select sources"""
        modified_query = query
        
        # Apply query modifications
        if 'add_synonyms' in action.query_modifications:
            # Add domain-specific synonyms
            if 'security' in query.lower():
                modified_query += " cybersecurity protection"
            elif 'machine learning' in query.lower():
                modified_query += " AI artificial intelligence"
        
        if 'add_domain_terms' in action.query_modifications:
            domain = self.detect_domain(query)
            domain_terms = {
                'cybersecurity': ' threat detection prevention',
                'healthcare': ' medical clinical diagnosis',
                'ai': ' algorithm model neural network'
            }
            if domain in domain_terms:
                modified_query += domain_terms[domain]
        
        if 'make_specific' in action.query_modifications:
            modified_query += " recent 2023 2024"
        
        return modified_query, action.source_selection
    
    def calculate_reward(self, papers: List[PaperInfo], query: str, processing_time: float) -> RLReward:
        """Calculate reward based on search results quality"""
        
        # 1. Paper relevance score (based on title/abstract relevance)
        if not papers:
            relevance_score = 0.0
        else:
            query_keywords = set(self.extract_keywords(query))
            total_relevance = 0
            
            for paper in papers:
                title_keywords = set(self.extract_keywords(paper.title))
                # Handle None abstracts
                abstract_text = paper.abstract[:200] if paper.abstract else ""
                abstract_keywords = set(self.extract_keywords(abstract_text))
                paper_keywords = title_keywords.union(abstract_keywords)
                
                # Calculate keyword overlap
                overlap = len(query_keywords.intersection(paper_keywords))
                relevance = overlap / max(len(query_keywords), 1)
                total_relevance += relevance
            
            relevance_score = min(total_relevance / len(papers), 1.0)
        
        # 2. Result count score (optimal around 5 papers)
        result_count = len(papers)
        if result_count == 0:
            count_score = -1.0  # Heavily penalize no results
        elif 3 <= result_count <= 7:
            count_score = 1.0  # Optimal range
        else:
            count_score = max(0.1, 1.0 - abs(result_count - 5) * 0.1)
        
        # 3. Processing time score (faster is better)
        time_score = max(0.1, 1.0 - (processing_time / 30.0))  # Normalize to 30 seconds
        
        # 4. User feedback score (default neutral)
        user_score = 0.5  # Can be updated with actual user feedback
        
        # Calculate total reward
        total = (relevance_score * 0.4 + count_score * 0.3 + time_score * 0.2 + user_score * 0.1)
        
        return RLReward(
            paper_relevance_score=relevance_score,
            result_count_score=count_score,
            processing_time_score=time_score,
            user_feedback_score=user_score,
            total_reward=total
        )
    
    def update_q_table(self, state: RLState, action_idx: int, reward: float, next_state: RLState):
        """Update Q-table using Q-learning"""
        state_key = self.state_to_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        current_q = self.q_table[state_key].get(action_idx, 0.0)
        
        # Get max Q-value for next state
        next_state_key = self.state_to_key(next_state)
        next_max_q = 0.0
        if next_state_key in self.q_table:
            next_max_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state_key][action_idx] = new_q
    
    def save_experience(self):
        """Save Q-table and experience to file"""
        try:
            os.makedirs(os.path.dirname(self.experience_file), exist_ok=True)
            experience_data = {
                'q_table': self.q_table,
                'learning_rate': self.learning_rate,
                'epsilon': self.epsilon,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.experience_file, 'w') as f:
                json.dump(experience_data, f, indent=2)
            logger.info("RL experience saved successfully")
        except Exception as e:
            logger.error(f"Error saving RL experience: {e}")
    
    def load_experience(self):
        """Load Q-table and experience from file"""
        try:
            if os.path.exists(self.experience_file):
                with open(self.experience_file, 'r') as f:
                    experience_data = json.load(f)
                
                self.q_table = experience_data.get('q_table', {})
                self.learning_rate = experience_data.get('learning_rate', 0.1)
                self.epsilon = max(0.1, experience_data.get('epsilon', 0.3) * 0.99)  # Decay epsilon
                logger.info(f"RL experience loaded. Q-table size: {len(self.q_table)}")
            else:
                logger.info("No previous RL experience found. Starting fresh.")
        except Exception as e:
            logger.error(f"Error loading RL experience: {e}")
            self.q_table = {}

class RLEnhancedRAG(EnhancedRAG):
    """Enhanced RAG with Reinforcement Learning optimization"""
    
    def __init__(self):
        super().__init__()
        self.rl_optimizer = QueryOptimizer()
        self.rl_enabled = True
    
    async def research_and_analyze_with_rl(self, query: str, sources: List[str] = None, max_iterations: int = 3) -> ResearchAnalysis:
        """
        Enhanced research analysis with RL optimization
        
        Args:
            query: Original research query
            sources: Available sources
            max_iterations: Maximum RL optimization iterations
            
        Returns:
            Optimized research analysis
        """
        if not self.rl_enabled:
            return await self.research_and_analyze(query, sources)
        
        logger.info(f"🤖 Starting RL-enhanced research analysis")
        
        best_analysis = None
        best_reward = -float('inf')
        
        # Get initial state
        available_sources = sources or ['arxiv', 'pubmed', 'semantic_scholar']
        state = self.rl_optimizer.get_state(query, available_sources)
        
        for iteration in range(max_iterations):
            logger.info(f"🔄 RL Iteration {iteration + 1}/{max_iterations}")
            
            # Select action using RL policy
            action = self.rl_optimizer.select_action(state)
            action_idx = hash(str(action)) % 1000  # Simple action indexing
            
            # Apply action to modify query and sources
            modified_query, selected_sources = self.rl_optimizer.apply_action(query, action)
            
            logger.info(f"🎯 Testing: '{modified_query[:50]}...' with sources: {selected_sources}")
            
            # Perform research with modified parameters
            start_time = datetime.now()
            try:
                analysis = await self.research_and_analyze(modified_query, selected_sources)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Validate analysis
                if not analysis or not hasattr(analysis, 'papers'):
                    logger.error(f"❌ Invalid analysis result in iteration {iteration + 1}")
                    raise Exception("Invalid analysis result")
                
                # Calculate reward
                reward = self.rl_optimizer.calculate_reward(
                    analysis.papers, 
                    query, 
                    processing_time
                )
                
                logger.info(f"📊 Reward: {reward.total_reward:.3f} (relevance: {reward.paper_relevance_score:.3f}, count: {reward.result_count_score:.3f})")
                
                # Update best result
                if reward.total_reward > best_reward:
                    best_reward = reward.total_reward
                    best_analysis = analysis
                    logger.info(f"✅ New best result with reward: {best_reward:.3f}")
                
                # Update Q-table
                next_state = self.rl_optimizer.get_state(
                    modified_query, 
                    selected_sources, 
                    reward.total_reward
                )
                self.rl_optimizer.update_q_table(state, action_idx, reward.total_reward, next_state)
                
                # Update state for next iteration
                state = next_state
                
            except Exception as e:
                logger.error(f"❌ RL iteration {iteration + 1} failed: {e}")
                # Negative reward for failures
                processing_time = (datetime.now() - start_time).total_seconds()
                next_state = self.rl_optimizer.get_state(query, available_sources, -0.5)
                self.rl_optimizer.update_q_table(state, action_idx, -0.5, next_state)
                continue  # Continue to next iteration
        
        # Save RL experience
        self.rl_optimizer.save_experience()
        
        # Return best analysis or fallback to original
        if best_analysis:
            # Add RL metadata
            best_analysis.processing_metadata['rl_enabled'] = True
            best_analysis.processing_metadata['rl_iterations'] = max_iterations
            best_analysis.processing_metadata['best_reward'] = best_reward
            return best_analysis
        else:
            logger.warning("🔄 RL optimization failed, falling back to standard analysis")
            return await self.research_and_analyze(query, sources)
    
    def get_rl_stats(self) -> Dict:
        """Get RL training statistics"""
        return {
            'q_table_size': len(self.rl_optimizer.q_table),
            'epsilon': self.rl_optimizer.epsilon,
            'learning_rate': self.rl_optimizer.learning_rate,
            'total_experiences': sum(len(actions) for actions in self.rl_optimizer.q_table.values())
        } 