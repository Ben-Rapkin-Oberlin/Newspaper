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
from multiprocessing import freeze_support
from multiprocessing import freeze_support
from collections import defaultdict
from typing import Dict, List, Tuple, Any
# Custom stopwords for newspaper-specific terms
CUSTOM_STOPS = {
    'faid', 'aud', 'iaid', 'ditto', 'fame', 'fold', 'ing', 'con', 
    'hereby', 'said', 'would', 'upon', 'may', 'every', 'next',
    'tie', 'well', 'make', 'made', 'hon'
}

# Enhanced seed topics with more distinctive words
SEED_TOPICS = {
    'mortality_reports': [
        'death', 'died', 'mortality', 'deceased', 'fatality',
        'casualties', 'deaths', 'toll', 'register', 'record',
        'statistics', 'weekly', 'monthly', 'reported', 'total'
    ],
    'public_health': [
        'health', 'disease', 'epidemic', 'outbreak', 'infection',
        'prevention', 'sanitary', 'quarantine', 'vaccination', 'hospital',
        'physician', 'medical', 'treatment', 'cure', 'board'
    ],
    'vital_statistics': [
        'birth', 'marriage', 'census', 'population', 'registry',
        'records', 'rate', 'increase', 'decrease', 'annual',
        'estimate', 'official', 'report', 'survey', 'count'
    ],
    'death_notices': [
        'funeral', 'burial', 'cemetery', 'survived', 'bereaved',
        'mourning', 'memorial', 'obituary', 'passed', 'age',
        'resident', 'family', 'leaves', 'services', 'arrangements'
    ],
    'disease_specific': [
        'consumption', 'fever', 'smallpox', 'cholera', 'typhoid',
        'tuberculosis', 'influenza', 'diphtheria', 'plague', 'pneumonia',
        'measles', 'scarlet', 'dysentery', 'whooping', 'malaria'
    ]
}

import gensim.models.callbacks as callbacks

class PerplexityLogger:
    """Simple callback class for tracking perplexity during LDA training"""
    def __init__(self):
        self.epoch = 0
        self.model = None
        self.corpus = None
    
    def set_model(self, model):
        """Called when training starts - store the model reference"""
        self.model = model
        self.corpus = model.corpus
    
    def on_epoch_end(self, model):
        """Called after each epoch"""
        self.epoch += 1
        if self.epoch % 10 == 0:
            if hasattr(model, 'log_perplexity'):
                perplexity = model.log_perplexity(self.corpus)
                print(f"    Pass {self.epoch}/100 - Perplexity: {perplexity:.2f}")


class TemporalLDAAnalyzer:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.models = {}
        self.dictionaries = {}
        self.corpora = {}
        self.coherence_scores = {}
        
    def preprocess_text(self, text: str) -> List[str]:
        """Enhanced preprocessing function"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        tokens = word_tokenize(text)
        
        stop_words = set(stopwords.words('english'))
        stop_words.update(CUSTOM_STOPS)
        tokens = [token for token in tokens if token not in stop_words]
        
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        tokens = [token for token in tokens if len(token) > 3 and token.isalpha()]
        
        return tokens

    def process_temporal_window(self, df: pd.DataFrame, window_start: int) -> Tuple[List[List[str]], corpora.Dictionary, List[Any]]:
        """Process documents within a specific time window"""
        window_end = window_start + self.window_size - 1
        print(f"\nProcessing window {window_start}-{window_end}")
        
        # Filter documents for time window
        print("  Filtering documents for time window...")
        mask = (pd.to_datetime(df['date']).dt.year >= window_start) & \
               (pd.to_datetime(df['date']).dt.year <= window_end)
        window_df = df[mask]
        
        print(f"  Found {len(window_df)} documents in window")
        if len(window_df) == 0:
            print(f"  Warning: No documents found in window {window_start}-{window_end}")
            return None, None, None
        
        # Process texts in window
        print("  Preprocessing texts...")
        texts = []
        for idx, article in enumerate(window_df['article']):
            if (idx + 1) % 100 == 0:
                print(f"    Processed {idx + 1}/{len(window_df)} articles")
            texts.append(self.preprocess_text(article))
        
        # Create dictionary
        print("  Creating dictionary...")
        dictionary = corpora.Dictionary(texts)
        initial_tokens = len(dictionary)
        print(f"    Initial vocabulary size: {initial_tokens}")
        
        # Filter dictionary
        print("  Filtering dictionary...")
        dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=50000)
        final_tokens = len(dictionary)
        print(f"    Final vocabulary size: {final_tokens}")
        print(f"    Removed {initial_tokens - final_tokens} tokens")
        
        # Create corpus
        print("  Creating corpus...")
        corpus = [dictionary.doc2bow(text) for text in texts]
        print(f"    Corpus size: {len(corpus)} documents")
        
        return texts, dictionary, corpus

    def train_window_model(self, texts: List[List[str]], dictionary: corpora.Dictionary, 
                      corpus: List[Any], window_start: int) -> models.LdaModel:
        window_end = window_start + self.window_size - 1
        print(f"\nTraining LDA model for window {window_start}-{window_end}")
        
        print("  Initializing eta matrix...")
        num_topics = len(SEED_TOPICS)
        vocab_size = len(dictionary)
        eta = np.full((num_topics, vocab_size), 0.001)
        
        # Set up eta matrix
        print("  Setting up seed words...")
        seed_word_counts = defaultdict(int)
        for topic_idx, (topic_name, seed_words) in enumerate(SEED_TOPICS.items()):
            for word in seed_words:
                if word in dictionary.token2id:
                    word_id = dictionary.token2id[word]
                    eta[topic_idx, word_id] = 0.5
                    seed_word_counts[topic_name] += 1
        
        # Report seed word coverage
        print("  Seed word coverage:")
        for topic, count in seed_word_counts.items():
            print(f"    {topic}: {count}/{len(SEED_TOPICS[topic])} words found in vocabulary")
        
        # Train LDA model with proper callback
        print(f"  Training model with {num_topics} topics...")
        callback = PerplexityLogger()
        
        model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=100,
            alpha='asymmetric',
            eta=eta,
            random_state=42,
            chunksize=2000,
            iterations=500,
            callbacks=[callback]
        )
        
        # Calculate coherence
        print("  Calculating coherence scores...")
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
        
        coherence_scores = {
            'c_v': coherence_cv,
            'u_mass': coherence_umass
        }
        
        self.coherence_scores[window_start] = coherence_scores
        print(f"  Coherence scores: CV={coherence_scores['c_v']:.4f}, UMass={coherence_scores['u_mass']:.4f}")
        
        return model

    #@staticmethod
    #def perplexity_logger(model, iteration, pass_):
    #    """Callback to log perplexity during training"""
    #    if pass_ % 10 == 0:
    #        print(f"    Pass {pass_}/100 - Perplexity: {model.log_perplexity(model.corpus):.2f}")

    def evaluate_coherence(self, model: models.LdaModel, texts: List[List[str]], 
                          dictionary: corpora.Dictionary) -> Dict[str, float]:
        """Calculate multiple coherence metrics"""
        # [Previous coherence code remains the same]
        return {
            'c_v': coherence_cv,
            'u_mass': coherence_umass
        }

    def analyze_topic_evolution(self):
        """Analyze how topics evolve across time windows"""
        print("\nAnalyzing topic evolution across time windows")
        topic_evolution = defaultdict(list)
        
        for window_start in sorted(self.models.keys()):
            window_end = window_start + self.window_size - 1
            print(f"\nTime Window: {window_start}-{window_end}")
            
            # [Previous analysis code remains the same]
        
        return topic_evolution

def main():
    print("Initializing NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

    print("\nInitializing Temporal LDA Analyzer...")
    analyzer = TemporalLDAAnalyzer(window_size=5)

    print("\nLoading dataset...")
    years = list(range(1800, 1810))
    print(f"Processing years: {min(years)} to {max(years)}")
    
    dataset = load_dataset("dell-research-harvard/AmericanStories",
                          "subset_years",
                          year_list=[str(year) for year in years])

    print("Combining data into DataFrame...")
    df = pd.concat([dataset[str(year)].to_pandas() for year in years])
    print(f"Total documents loaded: {len(df)}")
    
    # Process each time window
    for window_start in range(min(years), max(years), analyzer.window_size):
        window_end = window_start + analyzer.window_size - 1
        print(f"\n{'='*80}")
        print(f"Processing window: {window_start}-{window_end}")
        print(f"{'='*80}")
        
        texts, dictionary, corpus = analyzer.process_temporal_window(df, window_start)
        
        if texts is not None:
            model = analyzer.train_window_model(texts, dictionary, corpus, window_start)
            
            analyzer.models[window_start] = model
            analyzer.dictionaries[window_start] = dictionary
            analyzer.corpora[window_start] = corpus
            
            print(f"\nSaving model and dictionary for window {window_start}-{window_end}...")
            model.save(f'lda_model_{window_start}_{window_end}')
            dictionary.save(f'dictionary_{window_start}_{window_end}')

    print("\nAnalyzing topic evolution...")
    topic_evolution = analyzer.analyze_topic_evolution()
    
    print("\nProcessing complete!")

if __name__ == '__main__':
    freeze_support()
    main()