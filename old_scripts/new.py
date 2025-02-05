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
        
        # Filter documents for time window
        mask = (pd.to_datetime(df['date']).dt.year >= window_start) & \
               (pd.to_datetime(df['date']).dt.year <= window_end)
        window_df = df[mask]
        
        if len(window_df) == 0:
            print(f"Warning: No documents found in window {window_start}-{window_end}")
            return None, None, None
        
        # Process texts in window
        texts = window_df['article'].apply(self.preprocess_text).tolist()
        
        # Create dictionary and corpus for window
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=50000)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        return texts, dictionary, corpus

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
        
        # Train LDA model
        model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,
            alpha='asymmetric',
            eta=eta,
            random_state=42,
            chunksize=2000,
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

    def analyze_topic_evolution(self):
        """Analyze how topics evolve across time windows"""
        topic_evolution = defaultdict(list)
        
        for window_start in sorted(self.models.keys()):
            window_end = window_start + self.window_size - 1
            model = self.models[window_start]
            
            print(f"\nTime Window: {window_start}-{window_end}")
            print("Coherence Scores:", self.coherence_scores[window_start])
            #print("\nTop 10 words in each topic:")
            
            for idx, topic in enumerate(model.print_topics(num_words=10)):
                topic_name = list(SEED_TOPICS.keys())[idx]
                #print(f"\nTopic {idx + 1} ({topic_name}):")
                #print(topic)
                
                # Store evolution data
                topic_evolution[topic_name].append({
                    'window': f"{window_start}-{window_end}",
                    'words': topic[1],
                    'coherence': self.coherence_scores[window_start]
                })
        
        return topic_evolution

def analyze_cooccurrences_multi(text: str, target_words: List[str], window_size: int = 10) -> Dict[str, int]:
    """
    Count co-occurrences of 'smallpox' with multiple target words AND 'death' within a specified window.
    
    Args:
        text (str): Input text to analyze
        target_words (List[str]): List of words to check for co-occurrence with 'smallpox'
        window_size (int): Number of words to consider for co-occurrence window
        
    Returns:
        Dict[str, int]: Dictionary mapping target words to their co-occurrence counts, plus 'death' count
    """
    # Tokenize the text
    words = word_tokenize(text.lower())
    
    # Find all positions of smallpox
    smallpox_positions = [i for i, word in enumerate(words) if word == 'smallpox']
    
    # Initialize co-occurrence counts for all target words plus 'death'
    cooccurrences = {word: 0 for word in target_words}
    cooccurrences['death'] = 0  # Add specific death counter
    
    # Count co-occurrences for each target word and death
    for sp_pos in smallpox_positions:
        window_start = max(0, sp_pos - window_size)
        window_end = min(len(words), sp_pos + window_size + 1)
        
        # Check window for each target word
        window_text = words[window_start:window_end]
        
        # Check for death/deaths specifically
        if 'death' in window_text or 'deaths' in window_text:
            cooccurrences['death'] += 1
            
        # Check for other target words
        for target_word in target_words:
            if target_word in window_text:
                cooccurrences[target_word] += 1
                
    return cooccurrences
    
    
def process_yearly_data(dataset, year: str, model: models.LdaModel, 
    dictionary: corpora.Dictionary, analyzer: TemporalLDAAnalyzer, window: int) -> pd.DataFrame:
    """
    Process articles for a given year and create a DataFrame with co-occurrences and topic distributions.
    """
    # Get top words for each topic
    top_words_dict = get_top_words_per_topic(model)
    
    # Create flat list of all top words
    all_top_words = []
    for topic_words in top_words_dict.values():
        all_top_words.extend(topic_words)
    
    # Initialize lists to store results
    results = []
    
    # Process each article
    for article in dataset:
        text = article['article']
        article_id = article['article_id']
        
        # Count words
        word_count = len(word_tokenize(text))
        
        # Get co-occurrences with all top words and death
        cooccurrences = analyze_cooccurrences_multi(text, all_top_words)
        
        # Create result dictionary
        result = {
            'id': article_id,
            'word_count': word_count,
            'smallpox_death_cooccurrences': cooccurrences['death']  # Add the death co-occurrence count
        }
        
        # Add co-occurrence counts for topic words
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
        
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = f'cooccurrence_analysis_{year}_{window}.csv'
    df.to_csv(output_file, index=False)
    
    return df


def main():
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Initialize analyzer
    window_size=5
    analyzer = TemporalLDAAnalyzer(window_size=window_size)

    # Load the dataset for a range of years
    years = list(range(1800, 1809))  # Modify year range as needed
    dataset = load_dataset("dell-research-harvard/AmericanStories",
                          "subset_years",
                          year_list=[str(year) for year in years],
                           trust_remote_code=True)

    # Combine all years into one DataFrame
    df = pd.concat([dataset[str(year)].to_pandas() for year in years])
    
    # Process each time window
    for window_start in range(min(years), max(years), analyzer.window_size):
        print(f"\nProcessing window: {window_start}-{window_start + analyzer.window_size - 1}")
        
        # Process window
        texts, dictionary, corpus = analyzer.process_temporal_window(df, window_start)
        
        if texts is not None:
            # Train model for window
            model = analyzer.train_window_model(texts, dictionary, corpus, window_start)
            
            # Store results
            analyzer.models[window_start] = model
            analyzer.dictionaries[window_start] = dictionary
            analyzer.corpora[window_start] = corpus

    # Analyze topic evolution
    topic_evolution = analyzer.analyze_topic_evolution()
    
    # Save models
    for window_start, model in analyzer.models.items():
        model.save(f'lda_model_{window_start}_{window_start + analyzer.window_size - 1}')
        analyzer.dictionaries[window_start].save(f'dictionary_{window_start}_{window_start + analyzer.window_size - 1}')
    # Process each year's data with co-occurrence analysis
    for year in years:
        year_str = str(year)
        print(f"\nProcessing year: {year_str}")
        
        # Find the appropriate window for this year
        window_start = (year // analyzer.window_size) * analyzer.window_size
        
        # Get model and dictionary for this window
        model = analyzer.models[window_start]
        dictionary = analyzer.dictionaries[window_start]
        
        # Process the year's data
        year_df = process_yearly_data(dataset[year_str], year_str, model, dictionary, analyzer, window_size)
        print(f"Processed {len(year_df)} articles for {year_str}")
        
if __name__ == '__main__':
    freeze_support()
    main()