"""
This script attempts to understand the model's internal reasoning patterns through
indirect probing methods, using only the API output. Techniques include:
1. Ablation studies (removing parts of the input)
2. Chain-of-thought analysis
3. Feature attribution
4. Response consistency analysis
5. Edge case probing
"""

import json
import pathlib
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class ModelProbe:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.probe_results = defaultdict(dict)
        
    def ablation_study(self, question: str, category: str) -> Dict:
        """
        Test model's response when parts of the question are removed.
        This helps understand which parts are crucial for reasoning.
        """
        words = question.split()
        ablation_results = {}
        
        # Test removing each word
        for i in range(len(words)):
            modified_words = words.copy()
            removed_word = modified_words.pop(i)
            modified_question = " ".join(modified_words)
            
            response = get_llm_response_gemini_curl(modified_question)
            ablation_results[removed_word] = {
                'modified_question': modified_question,
                'response': response
            }
            
        return ablation_results
    
    def chain_of_thought_analysis(self, response: str) -> Dict:
        """
        Analyze the steps in the model's reasoning process by looking
        for linguistic markers of reasoning chains.
        """
        # Look for reasoning markers
        markers = {
            'conditional': ['if', 'then', 'because', 'therefore'],
            'sequential': ['first', 'second', 'finally', 'next'],
            'comparative': ['however', 'although', 'while', 'whereas'],
            'logical': ['thus', 'hence', 'consequently', 'so'],
        }
        
        thought_chain = defaultdict(list)
        sentences = response.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip().lower()
            for marker_type, marker_words in markers.items():
                if any(marker in sentence for marker in marker_words):
                    thought_chain[marker_type].append(sentence)
        
        return dict(thought_chain)
    
    def consistency_check(self, category: str, responses: List[str]) -> Dict:
        """
        Analyze consistency of reasoning patterns within a category
        by comparing response structures and patterns.
        """
        # Vectorize responses
        vectors = self.vectorizer.fit_transform(responses).toarray()
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(vectors)
        
        # Analyze common patterns
        common_phrases = self.extract_common_patterns(responses)
        
        return {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'common_patterns': common_phrases
        }
    
    def extract_common_patterns(self, responses: List[str]) -> Dict:
        """
        Extract common linguistic patterns and phrases across responses.
        """
        # Implement n-gram analysis and frequency counting
        pass
    
    def visualize_category_patterns(self, results_dir: pathlib.Path):
        """
        Create visualizations of reasoning patterns across categories.
        """
        plt.figure(figsize=(15, 10))
        
        # Create heatmap of reasoning pattern similarities
        categories = list(self.probe_results.keys())
        similarity_matrix = np.zeros((len(categories), len(categories)))
        
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                similarity_matrix[i][j] = self.calculate_category_similarity(
                    self.probe_results[cat1],
                    self.probe_results[cat2]
                )
        
        sns.heatmap(similarity_matrix, 
                   xticklabels=categories,
                   yticklabels=categories,
                   cmap='viridis')
        
        plt.title('Cross-Category Reasoning Pattern Similarity')
        plt.tight_layout()
        plt.savefig(results_dir / 'category_patterns.png')
        plt.close()

def load_responses() -> Dict:
    """Load existing responses for analysis"""
    results_dir = pathlib.Path("results")
    if not results_dir.exists():
        return {}
    
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return {}
    
    last_run_dir = max(run_dirs)
    response_file = last_run_dir / "responses_intermediate.json"
    
    try:
        with open(response_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def main():
    # Initialize probe
    probe = ModelProbe()
    
    # Load existing responses
    responses = load_responses()
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = pathlib.Path("probe_results") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each category
    for category, qa_pairs in responses.items():
        print(f"Analyzing {category}...")
        
        # Filter out None responses
        valid_responses = [resp for resp in qa_pairs.values() if resp is not None]
        
        if valid_responses:
            # Run analyses
            consistency_results = probe.consistency_check(category, valid_responses)
            thought_chains = [probe.chain_of_thought_analysis(resp) 
                            for resp in valid_responses]
            
            # Store results
            probe.probe_results[category] = {
                'consistency': consistency_results,
                'thought_chains': thought_chains,
                'response_count': len(valid_responses)
            }
    
    # Generate visualizations
    probe.visualize_category_patterns(results_dir)
    
    # Save analysis results
    with open(results_dir / "probe_analysis.json", 'w') as f:
        json.dump(probe.probe_results, f, indent=4)
    
    print(f"Analysis complete. Results saved in {results_dir}")

if __name__ == "__main__":
    main() 