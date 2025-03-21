from datasets import load_dataset
import pandas as pd
import numpy as np
from gensim import corpora, models
import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from multiprocessing import Pool, freeze_support
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from functools import partial
import concurrent.futures
import random
import os
from time import time
from gensim.models.ldamulticore import LdaMulticore  # Add this import
# Custom stopwords remain the same
CUSTOM_STOPS = {
    'faid', 'aud', 'iaid', 'ditto', 'fame', 'fold', 'ing', 'con', 
    'hereby', 'said', 'would', 'upon', 'may', 'every', 'next',
    'tie', 'well', 'make', 'made', 'hon'
}

# Enhanced seed topics with more distinctive words
SEED_TOPICS = {
    'mortality_reports': [
        # Core mortality terms
        'mortality', 'deaths', 'deceased', 'casualties', 'fatalities',
        # Statistical indicators
        'weekly', 'monthly', 'annual', 'register', 'statistics',
        # Reporting terms
        'reported', 'recorded', 'certified', 'enumerated', 'total',
        # Demographic specifics
        'infant', 'adult', 'aged', 'children', 'population'
    ],
    
    'public_health': [
        # Health measures
        'vaccination', 'inoculation', 'prevention', 'quarantine', 'isolation',
        # Medical authority
        'physician', 'doctor', 'surgeon', 'hospital', 'board',
        # Public action
        'notice', 'warning', 'advisory', 'proclamation', 'announcement',
        # Health infrastructure
        'clinic', 'dispensary', 'infirmary', 'ward', 'asylum'
    ],
    
    'editorial_commentary': [
        # Analysis terms
        'opinion', 'editorial', 'review', 'observation', 'commentary',
        # Impact assessment
        'impact', 'effect', 'consequence', 'influence', 'result',
        # Social response
        'public', 'community', 'society', 'citizens', 'residents',
        # Evaluation terms
        'consider', 'examine', 'assess', 'investigate', 'debate'
    ],
    
    'disease_context': [
        # Other diseases
        'fever', 'cholera', 'consumption', 'plague', 'influenza',
        # Disease characteristics
        'contagious', 'infectious', 'epidemic', 'outbreak', 'spread',
        # Symptoms
        'symptoms', 'condition', 'affliction', 'illness', 'malady',
        # Treatment
        'treatment', 'remedy', 'cure', 'medicine', 'prescription'
    ],
    
    'historical_reference': [
        # Temporal markers
        'previous', 'former', 'past', 'historical', 'earlier',
        # Reference terms
        'record', 'account', 'document', 'chronicle', 'report',
        # Comparative terms
        'comparison', 'similar', 'pattern', 'trend', 'recurring',
        # Memory terms
        'remembered', 'recalled', 'documented', 'recorded', 'preserved'
    ]
}

def get_top_words_per_topic(model: models.LdaModel, num_words: int = 5) -> Dict[int, List[str]]:
    """
    Extract top N words for each topic from the LDA model.
    
    Args:
        model (models.LdaModel): Trained LDA model
        num_words (int): Number of top words to extract per topic
        
    Returns:
        Dict[int, List[str]]: Dictionary mapping topic indices to their top words
    """
    top_words = {}
    for topic_idx in range(model.num_topics):
        # Get topic terms with probabilities
        topic_terms = model.show_topic(topic_idx, num_words)
        # Extract just the words (without probabilities)
        words = [term[0] for term in topic_terms]
        top_words[topic_idx] = words
    return top_words

class TemporalLDAAnalyzer:
    def __init__(self, window_size: int = 5, num_processes: int = 4, sample_percentage: float = 100.0):
        self.window_size = window_size
        self.num_processes = num_processes
        self.sample_percentage = sample_percentage / 100.0
        self.models = {}
        self.dictionaries = {}
        self.corpora = {}
        self.coherence_scores = {}
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(CUSTOM_STOPS)
        self.lemmatizer = WordNetLemmatizer()
        
    def get_word_topic_distribution(self, word: str, window_start: int) -> Dict[str, Dict[str, float]]:
            """
            Get topic distribution for a single word in a specific time window.
            Returns both raw probabilities and normalized distribution.
            """
            if window_start not in self.models or window_start not in self.dictionaries:
                raise ValueError(f"No model found for window starting at {window_start}")
                
            model = self.models[window_start]
            dictionary = self.dictionaries[window_start]
            
            # Preprocess the word
            processed_word = self.lemmatizer.lemmatize(word.lower())
            
            # Check if word is in dictionary
            if processed_word not in dictionary.token2id:
                return f"Word '{word}' not found in the vocabulary for window {window_start}-{window_start + self.window_size - 1}"
                
            # Get word ID
            word_id = dictionary.token2id[processed_word]
            
            # Get topic distribution for this word
            raw_topic_dist = []
            for topic_idx in range(model.num_topics):
                topic = model.get_topic_terms(topic_idx, topn=None)  # Get all terms
                for term_id, prob in topic:
                    if term_id == word_id:
                        raw_topic_dist.append((topic_idx, prob))
                        break
                else:
                    raw_topic_dist.append((topic_idx, 0.0))
            
            # Convert to dictionary with topic names
            raw_probs = {}
            total_prob = 0.0
            for topic_idx, prob in raw_topic_dist:
                topic_name = list(SEED_TOPICS.keys())[topic_idx]
                raw_probs[topic_name] = prob
                total_prob += prob
                
            # Calculate normalized distribution
            normalized_probs = {}
            if total_prob > 0:
                for topic_name, prob in raw_probs.items():
                    normalized_probs[topic_name] = prob / total_prob
                    
            return {
                'raw_probabilities': raw_probs,
                'normalized_distribution': normalized_probs if total_prob > 0 else raw_probs
            }

    def print_word_topic_distribution(self, word: str, window_start: int):
        """
        Print both raw probabilities and normalized distribution for a word.
        """
        try:
            distribution = self.get_word_topic_distribution(word, window_start)
            
            if isinstance(distribution, str):  # Error message
                print(distribution)
                return
                
            print(f"\nTopic distribution for '{word}' in window {window_start}-{window_start + self.window_size - 1}:")
            print("-" * 60)
            
            # Print raw probabilities
            print("\nRaw probabilities (word's probability within each topic):")
            for topic_name, prob in sorted(distribution['raw_probabilities'].items(), key=lambda x: x[1], reverse=True):
                if prob > 0:
                    print(f"{topic_name:<30} {prob:.4f}")
            
            # Print normalized distribution
            print("\nNormalized distribution (how the word is distributed across topics):")
            for topic_name, prob in sorted(distribution['normalized_distribution'].items(), key=lambda x: x[1], reverse=True):
                if prob > 0:
                    print(f"{topic_name:<30} {prob:.4f}")
                    
            # Get the most relevant topic from normalized distribution
            top_topic = max(distribution['normalized_distribution'].items(), key=lambda x: x[1])
            print(f"\nMost relevant topic: {top_topic[0]} (normalized probability: {top_topic[1]:.4f})")
            
        except Exception as e:
            print(f"Error analyzing word '{word}': {str(e)}")
        
    def preprocess_text(self, text: str) -> List[str]:
        """Optimized preprocessing function"""
        # Combine regex operations
        text = re.sub(r'[^a-zA-Z\s]|\s+', ' ', text.lower()).strip()
        
        # Use list comprehension with combined conditions
        tokens = word_tokenize(text)
        return [self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if len(token) > 3 
                and token.isalpha() 
                and token not in self.stop_words]

    def process_texts_batch(self, texts: List[str]) -> List[List[str]]:
        """Process a batch of texts in parallel"""
        with Pool(self.num_processes) as pool:
            return pool.map(self.preprocess_text, texts)

    def train_window_model(self, texts: List[List[str]], dictionary: corpora.Dictionary, 
                          corpus: List[Any], window_start: int) -> models.LdaModel:
        """Train LDA model for a specific time window"""
        num_topics = len(SEED_TOPICS)
        vocab_size = len(dictionary)
        eta = np.full((num_topics, vocab_size), 0.001)
        
        # Set up eta matrix with seed words
        for topic_idx, (topic_name, seed_words) in enumerate(SEED_TOPICS.items()):
            for word in seed_words:
                if word in dictionary.token2id:
                    word_id = dictionary.token2id[word]
                    eta[topic_idx, word_id] = 0.5
        
        # Train LDA model with parallel processing
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,
            alpha='asymmetric',
            eta=eta,
            random_state=42,
            chunksize=2000,
            iterations=50,
            workers=20  # Use 20 cores
        )
        
        # Calculate coherence
        coherence_scores = self.evaluate_coherence(model, texts, dictionary)
        self.coherence_scores[window_start] = coherence_scores
        
        return model

    def evaluate_coherence(self, model: models.LdaModel, texts: List[List[str]], 
                          dictionary: corpora.Dictionary) -> Dict[str, float]:
        """Calculate multiple coherence metrics"""
        coherence_cv = models.CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        ).get_coherence()
        
        coherence_umass = models.CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='u_mass'
        ).get_coherence()
        
        return {
            'c_v': coherence_cv,
            'u_mass': coherence_umass
        }

    def process_temporal_window(self, df: pd.DataFrame, window_start: int) -> Tuple[List[List[str]], corpora.Dictionary, List[Any]]:
        """Process window using pre-sampled data"""
        if len(df) == 0:
            print(f"Warning: No documents found in window {window_start}-{window_start + self.window_size - 1}")
            return None, None, None
    
        texts = self.process_texts_batch(df['article'].tolist())
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=50000)
        corpus = [dictionary.doc2bow(text) for text in texts]
    
        return texts, dictionary, corpus
        
def analyze_cooccurrences_multi(text: str, target_words: List[str], window_size: int = 10) -> Dict[str, int]:
    # Convert to lowercase and tokenize once
    words = word_tokenize(text.lower())
    target_words_set = set(target_words)
    
    # Find smallpox positions
    smallpox_positions = [i for i, word in enumerate(words) if word == 'smallpox']
    
    # Initialize counts
    cooccurrences = {word: 0 for word in target_words}
    cooccurrences['death'] = 0  # Special case for death/deaths
    
    # Process each window
    for sp_pos in smallpox_positions:
        window_start = max(0, sp_pos - window_size)
        window_end = min(len(words), sp_pos + window_size + 1)
        window_set = set(words[window_start:window_end])
        
        if 'death' in window_set or 'deaths' in window_set:
            cooccurrences['death'] += 1
            
        # Update counts for matching target words
        for word in target_words_set.intersection(window_set):
            cooccurrences[word] += 1
                
    return cooccurrences

def process_article(args: Tuple[Dict, models.LdaModel, corpora.Dictionary, TemporalLDAAnalyzer, Dict[int, List[str]]]):
    article, model, dictionary, analyzer, top_words_dict = args
    text = article['article']
    article_id = article['article_id']
    
    # Pre-calculate topic words once
    topic_words = set()
    for words in top_words_dict.values():
        topic_words.update(words)
    
    # Single text preprocessing
    word_count = len(word_tokenize(text))
    preprocessed_text = analyzer.preprocess_text(text)
    bow = dictionary.doc2bow(preprocessed_text)
    
    # Single co-occurrence analysis
    cooccurrences = analyze_cooccurrences_multi(text, list(topic_words))
    
    # Build result
    result = {
        'id': article_id,
        'word_count': word_count,
        'smallpox_death_cooccurrences': cooccurrences['death']
    }
    
    # Add co-occurrence counts
    for topic_idx, topic_words in top_words_dict.items():
        for word in topic_words:
            result[f'cooccur_topic{topic_idx+1}_{word}'] = cooccurrences[word]
    
    # Get topic distribution once
    topic_dist = model.get_document_topics(bow)
    for topic_idx in range(len(SEED_TOPICS)):
        prob = 0.0
        for t_idx, t_prob in topic_dist:
            if t_idx == topic_idx:
                prob = t_prob
                break
        result[f'topic_{topic_idx + 1}_prob'] = prob
        
    return result

def process_yearly_data(dataset, year: str, model: models.LdaModel, dictionary: corpora.Dictionary, 
                      analyzer: TemporalLDAAnalyzer, window: int, top_words_cache: Dict = None):
    results = []
    batch_size = 50
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        process_args = [(article, model, dictionary, analyzer, top_words_cache) for article in batch]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=analyzer.num_processes) as executor:
            batch_results = list(executor.map(process_article, process_args))
            results.extend(batch_results)
    
    df = pd.DataFrame(results)
    
    if not os.path.exists(f'yearly_occurrence_data/window_{window}'):
        os.makedirs(f'yearly_occurrence_data/window_{window}')
    
    output_file = f'{year}_cooccurrence_analysis.csv'
    df.to_csv(f'yearly_occurrence_data/window_{window}/{output_file}', index=False)
    
    return df


def process_window(start_year: int, window_size: int, analyzer: TemporalLDAAnalyzer):
    window_start_time = time()
    print(f"\n{'='*80}")
    print(f"Starting window: {start_year}-{start_year + window_size - 1}")
    print(f"{'='*80}\n")
    
    # Load data
    load_start = time()
    window_data = {}
    total_articles = 0
    for year in range(start_year, start_year + window_size):
        year_str = str(year)
        year_load_start = time()
        print(f"\nLoading data for year {year_str}...")
        
        year_dataset = load_dataset("dell-research-harvard/AmericanStories",
                                  "subset_years",
                                  year_list=[year_str],
                                  trust_remote_code=True)
        
        year_data = year_dataset[year_str].to_pandas()
        sample_size = int(len(year_data) * analyzer.sample_percentage)
        if sample_size > 0:
            window_data[year] = year_data.sample(n=sample_size, random_state=42)
            total_articles += sample_size
            print(f"Sampled {sample_size} articles from {len(year_data)} total articles")
            print(f"Year load time: {time() - year_load_start:.2f}s")
        
        del year_dataset
        del year_data
    
    print(f"\nTotal data load time: {time() - load_start:.2f}s")
    
    # Preprocess
    preprocess_start = time()
    print("\nConcatenating window data...")
    window_df = pd.concat([window_data[year] for year in window_data.keys()])
    print(f"Total articles in window: {total_articles}")
    
    print("\nPreprocessing texts and building dictionary...")
    texts, dictionary, corpus = analyzer.process_temporal_window(window_df, start_year)
    print(f"Preprocessing time: {time() - preprocess_start:.2f}s")
    
    if texts is not None:
        # Train LDA
        lda_start = time()
        print("\nTraining LDA model for window...")
        model = analyzer.train_window_model(texts, dictionary, corpus, start_year)
        print(f"LDA training time: {time() - lda_start:.2f}s")
        
        # Process co-occurrences
        cooccur_start = time()
        print("\nGenerating top words cache...")
        top_words_cache = get_top_words_per_topic(model)
        
        print("\nProcessing co-occurrences for each year...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=analyzer.num_processes) as executor:
            futures = []
            for year in window_data.keys():
                print(f"\nSubmitting articles for year {year}...")
                sampled_articles = list(window_data[year].to_dict('records'))
                future = executor.submit(
                    process_yearly_data,
                    sampled_articles,
                    str(year),
                    model,
                    dictionary,
                    analyzer,
                    window_size,
                    top_words_cache
                )
                futures.append((str(year), future))
            
            print("\nProcessing submitted articles...")
            for year_str, future in futures:
                try:
                    year_df = future.result()
                    print(f"✓ Year {year_str}: Processed {len(year_df)} articles")
                except Exception as e:
                    print(f"✗ Error processing {year_str}: {e}")
        
        print(f"Co-occurrence processing time: {time() - cooccur_start:.2f}s")
        
        # Cleanup
        cleanup_start = time()
        print("\nCleaning up window data...")
        del window_data
        del window_df
        del texts
        del dictionary
        del corpus
        del model
        print(f"Cleanup time: {time() - cleanup_start:.2f}s")
        
        total_window_time = time() - window_start_time
        print(f"\nWindow {start_year}-{start_year + window_size - 1} complete")
        print(f"Total window processing time: {total_window_time:.2f}s")
        print(f"Articles processed per second: {total_articles/total_window_time:.2f}")

def main():
    total_start = time()
    print("Initializing NLTK resources...")
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

    window_size = 5
    num_processes = 8
    sample_percentage = 10.0
    
    print(f"\nInitializing analyzer with:")
    print(f"- Window size: {window_size} years")
    print(f"- Processes: {num_processes}")
    print(f"- Sample percentage: {sample_percentage}%")
    
    analyzer = TemporalLDAAnalyzer(
        window_size=window_size,
        num_processes=num_processes,
        sample_percentage=sample_percentage
    )
    
    start_year = 1890
    end_year = 1900
    total_windows = (end_year - start_year) // window_size
    
    print(f"\nProcessing {total_windows} windows from {start_year} to {end_year}")
    
    for window_start in range(start_year, end_year, window_size):
        process_window(window_start, window_size, analyzer)
    
    print(f"\nTotal execution time: {time() - total_start:.2f}s")

if __name__ == '__main__':
    freeze_support()
    main()
