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
import networkx as nx

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
        # N-gram analysis
        ngram_patterns = defaultdict(int)
        for response in responses:
            # Get 2-4 word phrases
            words = response.lower().split()
            for n in range(2, 5):
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    ngram_patterns[ngram] += 1
        
        # Filter for meaningful patterns
        common_patterns = {
            'reasoning_structures': [],  # e.g., "if...then", "because of", etc.
            'domain_specific': [],      # terms specific to the category
            'transition_phrases': [],   # how the model moves between ideas
            'conclusion_markers': []    # how it signals conclusions
        }
        
        return common_patterns
    
    def analyze_reasoning_flow(self, response: str) -> Dict:
        """
        Analyze how the model builds its reasoning from start to finish.
        """
        flow_analysis = {
            'initial_framing': [],    # How does it start approaching the problem?
            'development': [],        # How does it build on initial ideas?
            'transitions': [],        # How does it connect different parts?
            'conclusion': [],         # How does it arrive at answers?
            'confidence_markers': []  # How certain is it about different parts?
        }
        
        # Analyze sentence structure and connections
        sentences = response.split('.')
        for i, sentence in enumerate(sentences):
            # Analyze position in reasoning chain
            if i == 0:
                flow_analysis['initial_framing'].append(sentence)
            elif i == len(sentences) - 1:
                flow_analysis['conclusion'].append(sentence)
            else:
                # Analyze transition words and logical connectors
                pass
            
        return flow_analysis
    
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

    def calculate_category_similarity(self, category1_data: Dict, category2_data: Dict) -> float:
        """
        Calculate similarity between two categories based on their reasoning patterns.
        Uses both consistency metrics and thought chain patterns.
        """
        # Get thought chains for both categories
        chains1 = category1_data.get('thought_chains', [])
        chains2 = category2_data.get('thought_chains', [])
        
        # Convert thought chains to sets of marker types used
        markers1 = set()
        markers2 = set()
        
        for chain in chains1:
            markers1.update(chain.keys())
        for chain in chains2:
            markers2.update(chain.keys())
        
        # Calculate Jaccard similarity between marker sets
        if not markers1 and not markers2:
            marker_similarity = 0.0
        else:
            marker_similarity = len(markers1.intersection(markers2)) / len(markers1.union(markers2))
        
        # Get consistency metrics
        consistency1 = category1_data.get('consistency', {}).get('mean_similarity', 0.0)
        consistency2 = category2_data.get('consistency', {}).get('mean_similarity', 0.0)
        
        # Calculate consistency similarity
        consistency_similarity = 1.0 - abs(consistency1 - consistency2)
        
        # Combine similarities (you can adjust weights as needed)
        return 0.7 * marker_similarity + 0.3 * consistency_similarity

    def analyze_cross_category_patterns(self) -> Dict:
        """
        Analyze how reasoning patterns differ across categories.
        """
        patterns = {
            'shared_structures': [],      # Reasoning patterns used across all categories
            'category_specific': {},      # Patterns unique to each category
            'reasoning_transitions': {},  # How the model switches reasoning modes
            'complexity_metrics': {}      # Measure of reasoning complexity per category
        }
        
        # Compare patterns across categories
        all_categories = set(self.probe_results.keys())
        for category in all_categories:
            category_chains = self.probe_results[category]['thought_chains']
            # Analyze unique patterns
            # Measure complexity
            # Identify transition points
        
        return patterns

    def visualize_analysis(self, results_dir: pathlib.Path):
        """
        Create multiple visualizations of the reasoning patterns.
        """
        # 1. Reasoning Pattern Distribution
        plt.figure(figsize=(15, 8))
        categories = list(self.probe_results.keys())
        pattern_types = ['conditional', 'logical', 'comparative', 'sequential']
        
        pattern_counts = []
        for category in categories:
            counts = []
            for chain in self.probe_results[category]['thought_chains']:
                type_counts = [len(chain.get(pt, [])) for pt in pattern_types]
                counts.append(type_counts)
            pattern_counts.append(np.mean(counts, axis=0))
        
        pattern_counts = np.array(pattern_counts)
        
        # Create stacked bar chart
        bottom = np.zeros(len(categories))
        for i, pattern in enumerate(pattern_types):
            plt.bar(categories, pattern_counts[:, i], bottom=bottom, label=pattern)
            bottom += pattern_counts[:, i]
        
        plt.title('Distribution of Reasoning Patterns Across Categories')
        plt.xlabel('Category')
        plt.ylabel('Average Count per Response')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(results_dir / 'reasoning_distribution.png', bbox_inches='tight')
        plt.close()
        
        # 2. Reasoning Flow Network
        plt.figure(figsize=(12, 8))
        G = nx.DiGraph()
        
        # Create network of reasoning transitions
        for category in categories:
            for chain in self.probe_results[category]['thought_chains']:
                pattern_sequence = list(chain.keys())
                for i in range(len(pattern_sequence) - 1):
                    G.add_edge(pattern_sequence[i], pattern_sequence[i+1])
        
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=10, font_weight='bold',
                arrows=True, edge_color='gray')
        
        plt.title('Reasoning Flow Network')
        plt.tight_layout()
        plt.savefig(results_dir / 'reasoning_flow.png', bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap of Pattern Co-occurrence
        plt.figure(figsize=(10, 8))
        cooccurrence = np.zeros((len(pattern_types), len(pattern_types)))
        
        for category in categories:
            for chain in self.probe_results[category]['thought_chains']:
                present_patterns = set(chain.keys())
                for i, p1 in enumerate(pattern_types):
                    for j, p2 in enumerate(pattern_types):
                        if p1 in present_patterns and p2 in present_patterns:
                            cooccurrence[i, j] += 1
        
        sns.heatmap(cooccurrence, xticklabels=pattern_types, 
                    yticklabels=pattern_types, annot=True, fmt='.1f',
                    cmap='YlOrRd')
        plt.title('Pattern Co-occurrence Matrix')
        plt.tight_layout()
        plt.savefig(results_dir / 'pattern_cooccurrence.png', bbox_inches='tight')
        plt.close()
        
        # 4. Reasoning Complexity Over Response Length
        plt.figure(figsize=(12, 6))
        lengths = []
        complexities = []
        categories_colors = []
        
        for category in categories:
            for chain in self.probe_results[category]['thought_chains']:
                response_length = sum(len(s) for patterns in chain.values() for s in patterns)
                pattern_complexity = len(chain.keys())  # number of different reasoning types used
                lengths.append(response_length)
                complexities.append(pattern_complexity)
                categories_colors.append(categories.index(category))
        
        scatter = plt.scatter(lengths, complexities, c=categories_colors, 
                             cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label='Category', 
                    ticks=range(len(categories)))
        plt.title('Reasoning Complexity vs Response Length')
        plt.xlabel('Response Length (characters)')
        plt.ylabel('Number of Reasoning Types Used')
        plt.tight_layout()
        plt.savefig(results_dir / 'complexity_analysis.png', bbox_inches='tight')
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
    probe.visualize_analysis(results_dir)
    
    # Save analysis results
    with open(results_dir / "probe_analysis.json", 'w') as f:
        json.dump(probe.probe_results, f, indent=4)
    
    print(f"Analysis complete. Results saved in {results_dir}")

if __name__ == "__main__":
    main() 