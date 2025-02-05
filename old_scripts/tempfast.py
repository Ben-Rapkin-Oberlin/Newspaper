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
        self.sample_percentage = sample_percentage / 100.0  # Convert to decimal
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
        
        # Train LDA model with optimized chunk size
        model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,
            alpha='asymmetric',
            eta=eta,
            random_state=42,
            chunksize=4000,  # Increased chunk size for better performance
            iterations=50
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
        """Process window with sampling"""
        window_end = window_start + self.window_size - 1
        
        # Group by year and sample
        dates = pd.to_datetime(df['date']).dt.year
        df['year'] = dates
        
        sampled_dfs = []
        for year in df['year'].unique():
            year_df = df[df['year'] == year]
            sample_size = int(len(year_df) * self.sample_percentage)
            if sample_size > 0:
                sampled_df = year_df.sample(n=sample_size, random_state=42)
                sampled_dfs.append(sampled_df)
        
        window_df = pd.concat(sampled_dfs) if sampled_dfs else pd.DataFrame()
        
        if len(window_df) == 0:
            print(f"Warning: No documents found in window {window_start}-{window_end}")
            return None, None, None
        
        texts = self.process_texts_batch(window_df['article'].tolist())
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=50000)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        return texts, dictionary, corpus

def analyze_cooccurrences_multi(text: str, target_words: List[str], window_size: int = 10) -> Dict[str, int]:
    """Optimized co-occurrence analysis"""
    words = word_tokenize(text.lower())
    
    # Use set for faster lookups
    target_words_set = set(target_words)
    
    # Find smallpox positions more efficiently
    smallpox_positions = [i for i, word in enumerate(words) if word == 'smallpox']
    
    # Initialize counts
    cooccurrences = {word: 0 for word in target_words}
    cooccurrences['death'] = 0
    
    # Process windows more efficiently
    for sp_pos in smallpox_positions:
        window_start = max(0, sp_pos - window_size)
        window_end = min(len(words), sp_pos + window_size + 1)
        
        # Create window set for faster lookups
        window_set = set(words[window_start:window_end])
        
        # Check death/deaths
        if 'death' in window_set or 'deaths' in window_set:
            cooccurrences['death'] += 1
        
        # Update counts for all matching target words
        for word in target_words_set.intersection(window_set):
            cooccurrences[word] += 1
                
    return cooccurrences

def process_article(args: Tuple[Dict, models.LdaModel, corpora.Dictionary, TemporalLDAAnalyzer, Dict[int, List[str]]]) -> Dict:
    """Process a single article"""
    article, model, dictionary, analyzer, top_words_dict = args
    
    text = article['article']
    article_id = article['article_id']
    
    # Count words
    word_count = len(word_tokenize(text))
    
    # Get all top words
    all_top_words = [word for words in top_words_dict.values() for word in words]
    
    # Get co-occurrences
    cooccurrences = analyze_cooccurrences_multi(text, all_top_words)
    
    # Create result dictionary
    result = {
        'id': article_id,
        'word_count': word_count,
        'smallpox_death_cooccurrences': cooccurrences['death']
    }
    
    # Add co-occurrence counts
    for topic_idx, topic_words in top_words_dict.items():
        for word in topic_words:
            result[f'cooccur_topic{topic_idx+1}_{word}'] = cooccurrences[word]
    
    # Get topic distribution
    preprocessed_text = analyzer.preprocess_text(text)
    bow = dictionary.doc2bow(preprocessed_text)
    topic_dist = model.get_document_topics(bow)
    
    # Add topic probabilities
    for topic_idx in range(len(SEED_TOPICS)):
        prob = 0.0
        for t_idx, t_prob in topic_dist:
            if t_idx == topic_idx:
                prob = t_prob
                break
        result[f'topic_{topic_idx + 1}_prob'] = prob
    
    return result

def process_yearly_data(dataset, year: str, model: models.LdaModel, 
    dictionary: corpora.Dictionary, analyzer: TemporalLDAAnalyzer, window: int, 
    top_words_cache: Dict = None) -> pd.DataFrame:
    """Process yearly data with sampling"""
    if top_words_cache is None:
        top_words_dict = get_top_words_per_topic(model)
    else:
        top_words_dict = top_words_cache
    
    # Sample articles
    articles = dataset
    sample_size = int(len(articles) * analyzer.sample_percentage)
    if sample_size > 0:
        articles = articles.sample(n=sample_size, random_state=42)
    
    process_args = [(article, model, dictionary, analyzer, top_words_dict) 
                   for article in articles]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=analyzer.num_processes) as executor:
        results = list(executor.map(process_article, process_args))
    
    df = pd.DataFrame(results)
    output_file = f'cooccurrence_analysis_{year}_{window}.csv'
    df.to_csv('yearly_results/{window}/'+output_file, index=False)
    
    return df

def main():
    # Download NLTK data
    for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

    window_size = 5
    num_processes = 8
    sample_percentage = 10.0  # Adjust this value to sample different percentages
    analyzer = TemporalLDAAnalyzer(
        window_size=window_size, 
        num_processes=num_processes,
        sample_percentage=sample_percentage
    )
    years = list(range(1830, 1839))
    dataset = load_dataset("dell-research-harvard/AmericanStories",
                          "subset_years",
                          year_list=[str(year) for year in years],
                          trust_remote_code=True)

    # Process windows in parallel
    for window_start in range(min(years), max(years), analyzer.window_size):
        print(f"\nProcessing window: {window_start}-{window_start + analyzer.window_size - 1}")
        
        # Combine data for window
        df = pd.concat([dataset[str(year)].to_pandas() 
                       for year in range(window_start, 
                                      min(window_start + analyzer.window_size, max(years) + 1))])
        
        # Process window
        texts, dictionary, corpus = analyzer.process_temporal_window(df, window_start)
        
        if texts is not None:
            # Train model
            model = analyzer.train_window_model(texts, dictionary, corpus, window_start)
            
            # Store results
            analyzer.models[window_start] = model
            analyzer.dictionaries[window_start] = dictionary
            analyzer.corpora[window_start] = corpus
            
            words_to_analyze = []#['liver','breast','expired','duell','victim']  # Add any words you're interested in
            print(f"\nAnalyzing specific words for window {window_start}-{window_start+window_size}:")
            for word in words_to_analyze:
                analyzer.print_word_topic_distribution(word, window_start)
                
    # Process each year's data in parallel with caching
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for year in years:
            year_str = str(year)
            window_start = (year // analyzer.window_size) * analyzer.window_size
            model = analyzer.models[window_start]
            dictionary = analyzer.dictionaries[window_start]
            
            # Cache top words for each model
            top_words_cache = get_top_words_per_topic(model)
            
            future = executor.submit(process_yearly_data, 
                                   dataset[year_str], 
                                   year_str, 
                                   model, 
                                   dictionary, 
                                   analyzer, 
                                   window_size,
                                   top_words_cache)
            futures.append((year_str, future))
        
        # Get results as they complete
        for year_str, future in futures:
            try:
                year_df = future.result()
                print(f"Processed {len(year_df)} articles for {year_str}")
            except Exception as e:
                print(f"Error processing {year_str}: {e}")
                
        

if __name__ == '__main__':
    freeze_support()
    main()
