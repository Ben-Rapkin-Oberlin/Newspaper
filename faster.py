from datasets import load_dataset
import pandas as pd
import numpy as np
from gensim import corpora, models
import gensim
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from multiprocessing import Pool, freeze_support, cpu_count
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from functools import partial
import spacy

# Load spaCy model once
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Compile regex patterns once
word_pattern = re.compile(r'[^a-zA-Z\s]')
space_pattern = re.compile(r'\s+')

# Custom stopwords for newspaper-specific terms
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

CUSTOM_STOPS = {
    'faid', 'aud', 'iaid', 'ditto', 'fame', 'fold', 'ing', 'con', 
    'hereby', 'said', 'would', 'upon', 'may', 'every', 'next',
    'tie', 'well', 'make', 'made', 'hon'
}

# Pre-compute stopwords set
STOP_WORDS = set(stopwords.words('english')).union(CUSTOM_STOPS)

# Initialize tokenizer once
tokenizer = RegexpTokenizer(r'\w+')

class TemporalLDAAnalyzer:
    def __init__(self, window_size: int = 5, num_processes: int = None):
        self.window_size = window_size
        self.models = {}
        self.dictionaries = {}
        self.corpora = {}
        self.coherence_scores = {}
        self.num_processes = num_processes or max(1, cpu_count() - 1)
        
        # Initialize lemmatizer once
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text_batch(self, texts: List[str]) -> List[List[str]]:
        """Batch process multiple texts using spaCy's pipe"""
        processed_texts = []
        
        # Process texts in batches using spaCy
        for doc in nlp.pipe(texts, batch_size=1000):
            # Extract lemmatized tokens that meet our criteria
            tokens = [token.lemma_ for token in doc 
                     if token.lemma_.isalpha() and 
                     len(token.lemma_) > 3 and 
                     token.lemma_.lower() not in STOP_WORDS]
            processed_texts.append(tokens)
            
        return processed_texts

    def process_temporal_window(self, df: pd.DataFrame, window_start: int) -> Tuple[List[List[str]], corpora.Dictionary, List[Any]]:
        """Process documents within a specific time window"""
        window_end = window_start + self.window_size - 1
        
        # Filter documents for time window
        mask = (pd.to_datetime(df['date']).dt.year >= window_start) & \
               (pd.to_datetime(df['date']).dt.year <= window_end)
        window_df = df[mask]
        
        if len(window_df) == 0:
            print(f"Warning: No documents found in window {window_start}-{window_end}")
            return None, None, None
        
        # Batch process texts
        texts = self.preprocess_text_batch(window_df['article'].tolist())
        
        # Create dictionary and corpus for window using worker processes
        with Pool(self.num_processes) as pool:
            dictionary = corpora.Dictionary(texts)
            dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=50000)
            
            # Process corpus creation in parallel
            chunk_size = max(1, len(texts) // (self.num_processes * 4))
            corpus = list(pool.imap(
                partial(self._create_bow, dictionary=dictionary),
                texts,
                chunksize=chunk_size
            ))
        
        return texts, dictionary, corpus

    @staticmethod
    def _create_bow(text: List[str], dictionary: corpora.Dictionary) -> List[Tuple[int, int]]:
        """Helper function for parallel corpus creation"""
        return dictionary.doc2bow(text)

    def train_window_model(self, texts: List[List[str]], dictionary: corpora.Dictionary, 
                          corpus: List[Any], window_start: int) -> models.LdaMulticore:
        """Train LDA model for a specific time window using multiple cores"""
        num_topics = len(SEED_TOPICS)
        vocab_size = len(dictionary)
        eta = np.full((num_topics, vocab_size), 0.001)
        
        # Set up eta matrix with seed words
        for topic_idx, (topic_name, seed_words) in enumerate(SEED_TOPICS.items()):
            for word in seed_words:
                if word in dictionary.token2id:
                    word_id = dictionary.token2id[word]
                    eta[topic_idx, word_id] = 0.5
        
        # Train LDA model using multiple cores and seeded topics
        model = models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            workers=self.num_processes,
            passes=100,
            alpha='asymmetric',
            eta=eta,  # Using the seeded eta matrix
            random_state=42,
            chunksize=2000,
            iterations=500,
            callbacks=[
                # Add callback to track topic convergence
                models.CoherenceMetric(corpus=corpus, dictionary=dictionary)
            ]
        )
        
        # Calculate coherence in parallel
        coherence_scores = self.evaluate_coherence(model, texts, dictionary)
        self.coherence_scores[window_start] = coherence_scores
        
        return model

def parallel_cooccurrence_analysis(text: str, window_size: int = 10) -> int:
    """Optimized co-occurrence analysis using regex"""
    # Convert to lowercase and tokenize in one pass using pre-compiled regex
    words = tokenizer.tokenize(text.lower())
    
    # Find positions using list comprehension
    smallpox_positions = [i for i, word in enumerate(words) if word == 'smallpox']
    death_positions = set(i for i, word in enumerate(words) if word in ['death', 'deaths'])
    
    # Count co-occurrences using set operations for faster lookup
    return sum(1 for pos in smallpox_positions
              for death_pos in death_positions
              if abs(pos - death_pos) <= window_size)

def process_yearly_data_parallel(args: Tuple) -> pd.DataFrame:
    """Parallel processing function for yearly data"""
    dataset, year, model, dictionary, analyzer = args
    
    results = []
    chunk_size = 1000  # Process in chunks to avoid memory issues
    
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i + chunk_size]
        
        # Process chunk in parallel
        with Pool(analyzer.num_processes) as pool:
            # Parallel word count and co-occurrence analysis
            texts = [article['article'] for article in chunk]
            word_counts = pool.map(lambda x: len(tokenizer.tokenize(x)), texts)
            cooccurrences = pool.map(parallel_cooccurrence_analysis, texts)
            
            # Process topic distributions
            preprocessed_texts = analyzer.preprocess_text_batch(texts)
            bows = [dictionary.doc2bow(text) for text in preprocessed_texts]
            topic_dists = [model.get_document_topics(bow) for bow in bows]
            
            # Combine results
            for article, word_count, cooc, topic_dist in zip(chunk, word_counts, cooccurrences, topic_dists):
                result = {
                    'id': article['article_id'],
                    'word_count': word_count,
                    'cooccurrences': cooc
                }
                
                # Add topic probabilities
                for topic_idx in range(len(SEED_TOPICS)):
                    prob = 0.0
                    for t_idx, t_prob in topic_dist:
                        if t_idx == topic_idx:
                            prob = t_prob
                            break
                    result[f'topic_{topic_idx + 1}_prob'] = prob
                
                results.append(result)
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    output_file = f'cooccurrence_analysis_{year}.csv'
    df.to_csv(output_file, index=False)
    
    return df

def main():
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # Initialize analyzer with parallel processing
    analyzer = TemporalLDAAnalyzer(window_size=5)

    # Load the dataset for a range of years
    years = list(range(1800, 1830))
    dataset = load_dataset("dell-research-harvard/AmericanStories",
                          "subset_years",
                          year_list=[str(year) for year in years])

    # Combine all years into one DataFrame
    df = pd.concat([dataset[str(year)].to_pandas() for year in years])
    
    # Process windows in parallel
    for window_start in range(min(years), max(years), analyzer.window_size):
        print(f"\nProcessing window: {window_start}-{window_start + analyzer.window_size - 1}")
        
        texts, dictionary, corpus = analyzer.process_temporal_window(df, window_start)
        
        if texts is not None:
            model = analyzer.train_window_model(texts, dictionary, corpus, window_start)
            
            analyzer.models[window_start] = model
            analyzer.dictionaries[window_start] = dictionary
            analyzer.corpora[window_start] = corpus
    
    # Process yearly data in parallel
    with Pool(analyzer.num_processes) as pool:
        process_args = [(dataset[str(year)], str(year), 
                        analyzer.models[year], 
                        analyzer.dictionaries[year], 
                        analyzer) 
                       for year in years]
        
        results = pool.map(process_yearly_data_parallel, process_args)
    
    # Save models
    for window_start, model in analyzer.models.items():
        model.save(f'lda_model_{window_start}_{window_start + analyzer.window_size - 1}')
        analyzer.dictionaries[window_start].save(f'dictionary_{window_start}_{window_start + analyzer.window_size - 1}')

if __name__ == '__main__':
    freeze_support()
    main()