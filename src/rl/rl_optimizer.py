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
            'cybersecurity': ['security', 'attack', 'ransomware', 'malware', 'vulnerability', 'encryption'],
            'healthcare': ['medical', 'health', 'patient', 'diagnosis', 'treatment', 'clinical'],
            'ai': ['machine learning', 'deep learning', 'neural', 'artificial intelligence', 'algorithm'],
            'climate': ['climate', 'environment', 'sustainability', 'carbon', 'renewable'],
            'robotics': ['robot', 'automation', 'control', 'robotic', 'autonomous']
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
        
        # Different source selection strategies (prefer working sources)
        source_strategies = [
            ['semantic_scholar'],  # Semantic Scholar only (most reliable)
            ['semantic_scholar', 'arxiv'],  # Semantic Scholar first, then arXiv
            ['arxiv', 'semantic_scholar'],  # Both sources
            ['arxiv'],  # arXiv only (backup option)
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
            # Exploration: strongly prefer actions that use Semantic Scholar first
            semantic_scholar_first_actions = [action for action in possible_actions 
                                            if action.source_selection and action.source_selection[0] == 'semantic_scholar']
            if semantic_scholar_first_actions:
                return random.choice(semantic_scholar_first_actions)
            
            # Fallback to any semantic scholar action
            semantic_scholar_actions = [action for action in possible_actions 
                                      if 'semantic_scholar' in action.source_selection]
            if semantic_scholar_actions:
                return random.choice(semantic_scholar_actions)
            else:
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
            
            # Fallback to semantic scholar actions if hash not found
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
        available_sources = sources or ['arxiv', 'semantic_scholar']
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