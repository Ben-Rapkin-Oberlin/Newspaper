from datasets import load_dataset
import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models.ldamulticore import LdaMulticore
import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from multiprocessing import Pool, freeze_support
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import concurrent.futures
import random
import os
from time import time

# Custom stopwords remain the same
CUSTOM_STOPS = {
    'faid', 'aud', 'iaid', 'ditto', 'fame', 'fold', 'ing', 'con', 
    'hereby', 'said', 'would', 'upon', 'may', 'every', 'next',
    'tie', 'well', 'make', 'made', 'hon'
}

# Enhanced seed topics with more distinctive words
SEED_TOPICS = {
    'mortality_reports': [
        'mortality', 'deaths', 'deceased', 'casualties', 'fatalities',
        'weekly', 'monthly', 'annual', 'register', 'statistics',
        'reported', 'recorded', 'certified', 'enumerated', 'total',
        'infant', 'adult', 'aged', 'children', 'population'
    ],
    'public_health': [
        'vaccination', 'inoculation', 'prevention', 'quarantine', 'isolation',
        'physician', 'doctor', 'surgeon', 'hospital', 'board',
        'notice', 'warning', 'advisory', 'proclamation', 'announcement',
        'clinic', 'dispensary', 'infirmary', 'ward', 'asylum'
    ],
    'editorial_commentary': [
        'opinion', 'editorial', 'review', 'observation', 'commentary',
        'impact', 'effect', 'consequence', 'influence', 'result',
        'public', 'community', 'society', 'citizens', 'residents',
        'consider', 'examine', 'assess', 'investigate', 'debate'
    ],
    'disease_context': [
        'fever', 'cholera', 'consumption', 'plague', 'influenza',
        'contagious', 'infectious', 'epidemic', 'outbreak', 'spread',
        'symptoms', 'condition', 'affliction', 'illness', 'malady',
        'treatment', 'remedy', 'cure', 'medicine', 'prescription'
    ],
    'historical_reference': [
        'previous', 'former', 'past', 'historical', 'earlier',
        'record', 'account', 'document', 'chronicle', 'report',
        'comparison', 'similar', 'pattern', 'trend', 'recurring',
        'remembered', 'recalled', 'documented', 'recorded', 'preserved'
    ]
}


def get_top_words_per_topic(model: models.LdaModel, num_words: int = 5) -> Dict[int, List[str]]:
    """
    Extract top N words for each topic from the LDA model.
    """
    top_words = {}
    for topic_idx in range(model.num_topics):
        topic_terms = model.show_topic(topic_idx, num_words)
        words = [term[0] for term in topic_terms]
        top_words[topic_idx] = words
    return top_words


def analyze_cooccurrences_bulk(
    token_lists: List[List[str]],
    topic_words: List[str],
    window_size: int = 10
) -> List[Dict[str, int]]:
    """
    Analyze co-occurrences with 'smallpox' in a bulk manner.
    For each list of tokens, returns a dict with co-occurrence counts
    for each word in `topic_words` + a special 'death' key that increments
    if 'death' or 'deaths' appear near 'smallpox'.
    """
    topic_words_set = set(topic_words)
    results = []

    for tokens in token_lists:
        # Find positions of "smallpox"
        smallpox_positions = [i for i, w in enumerate(tokens) if w == 'smallpox']

        # Initialize cooccurrences
        cooccurrences = {w: 0 for w in topic_words_set}
        cooccurrences['death'] = 0  # We'll store 'death' & 'deaths' hits here

        for pos in smallpox_positions:
            win_start = max(pos - window_size, 0)
            win_end = min(pos + window_size + 1, len(tokens))
            window_slice = tokens[win_start:win_end]

            # If "death" or "deaths" appear in that window, increment
            if 'death' in window_slice or 'deaths' in window_slice:
                cooccurrences['death'] += 1

            # Update counts for the topic words
            for w in topic_words_set:
                if w in window_slice:
                    cooccurrences[w] += 1

        results.append(cooccurrences)
    return results


def batch_inference(model: models.LdaModel, bows: List[List[Tuple[int,int]]], chunk_size=1000):
    """
    Perform topic inference in batches to avoid overhead 
    from calling model.get_document_topics() in a tight loop.
    """
    doc_topic_distributions = []
    for start_idx in range(0, len(bows), chunk_size):
        chunk = bows[start_idx : start_idx + chunk_size]
        gamma, _ = model.inference(chunk)  # gamma shape: (num_docs_in_chunk, num_topics)
        # Normalize each row to get topic distributions
        for row in gamma:
            row_sum = np.sum(row)
            if row_sum > 0:
                normalized = row / row_sum
            else:
                normalized = row
            doc_topic_distributions.append(normalized)
    return doc_topic_distributions


class TemporalLDAAnalyzer:
    def __init__(self, window_size: int = 5, num_processes: int = 4, sample_percentage: float = 100.0):
        self.window_size = window_size
        self.num_processes = num_processes
        self.sample_percentage = sample_percentage / 100.0
        self.models = {}
        self.dictionaries = {}
        self.corpora = {}
        self.coherence_scores = {}
        
        # Stopwords and lemmatizer
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(CUSTOM_STOPS)
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text: str) -> List[str]:
        """Optimized preprocessing function."""
        # Combine regex operations
        text = re.sub(r'[^a-zA-Z\s]+|\s+', ' ', text.lower()).strip()
        tokens = word_tokenize(text)

        # Lemmatize & filter
        return [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if len(t) > 3
               and t not in self.stop_words
               and t.isalpha()
        ]

    def process_temporal_window(self, df: pd.DataFrame, window_start: int):
        """
        Preprocess the entire window's data, build dictionary, etc.
        """
        if len(df) == 0:
            print(f"Warning: No documents found in window {window_start}-{window_start + self.window_size - 1}")
            return None, None, None

        # Preprocess in parallel or single-threaded (shown single-threaded here)
        texts = [self.preprocess_text(article) for article in df['article']]
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=50000)
        corpus = [dictionary.doc2bow(txt) for txt in texts]

        return texts, dictionary, corpus

    def train_window_model(
        self, 
        texts: List[List[str]],
        dictionary: corpora.Dictionary,
        corpus: List[Any],
        window_start: int
    ) -> models.LdaModel:
        """
        Train an LDA model for this window using seed topics.
        """
        num_topics = len(SEED_TOPICS)
        vocab_size = len(dictionary)
        eta = np.full((num_topics, vocab_size), 0.001)

        # Fill ETA with seed words
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
            workers=self.num_processes  # Use multiple cores
        )

        # Evaluate coherence
        coherence_scores = self.evaluate_coherence(model, texts, dictionary)
        self.coherence_scores[window_start] = coherence_scores
        return model

    def evaluate_coherence(
        self,
        model: models.LdaModel,
        texts: List[List[str]],
        dictionary: corpora.Dictionary
    ) -> Dict[str, float]:
        """Calculate multiple coherence metrics."""
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


def process_yearly_data_bulk(
    dataset: List[Dict[str, Any]],
    year: str,
    model: models.LdaModel,
    dictionary: corpora.Dictionary,
    analyzer: TemporalLDAAnalyzer,
    window: int,
    top_words_dict: Dict[int, List[str]]
) -> pd.DataFrame:
    """
    Process an entire year's data in a single/bulk pass:
      1. Preprocess (already done at window level? Or do it here if needed)
      2. doc2bow (already done or do it here)
      3. Bulk co-occurrence
      4. Batch inference
      5. Save to CSV
    """

    # 1) We assume articles are not preprocessed yet. 
    #    If you *already* preprocessed at the window level, you'd pass that in.
    texts = []
    bows = []
    token_lists_for_coocc = []

    # Prepare a combined set of top words from all topics for co-occurrence
    all_topic_words = set()
    for tw in top_words_dict.values():
        all_topic_words.update(tw)
    all_topic_words = list(all_topic_words)  # final list of unique words

    # Single-pass to preprocess everything
    for article in dataset:
        text = article['article']
        # Preprocess
        preproc = analyzer.preprocess_text(text)
        texts.append(preproc)

        # doc2bow
        bow = dictionary.doc2bow(preproc)
        bows.append(bow)

        # For co-occurrence, we can do a simpler tokenization (or reuse "preproc" if you want).
        # But if your "window_size" means 10 tokens in raw text, you need the raw tokenization:
        # We'll do a simpler approach with .split(), or a quick word_tokenize if needed.
        raw_tokens = text.lower().split()
        token_lists_for_coocc.append(raw_tokens)

    # 2) Bulk co-occurrence
    cooccur_results = analyze_cooccurrences_bulk(
        token_lists_for_coocc, 
        all_topic_words, 
        window_size=10
    )

    # 3) Batch topic inference
    topic_dists = batch_inference(model, bows, chunk_size=1000)

    # 4) Build final DataFrame
    records = []
    for i, article in enumerate(dataset):
        rec = {}
        rec['year'] = year
        rec['id'] = article['article_id']
        rec['word_count'] = len(token_lists_for_coocc[i])

        # Co-occurrence results
        for k, v in cooccur_results[i].items():
            if k == 'death':
                rec['smallpox_death_cooccurrences'] = v
            else:
                # For each topic word, store cooccurrence
                # e.g. rec[f'cooccur_{k}'] = v
                rec[f'cooccur_{k}'] = v

        # Topic distribution results
        dist = topic_dists[i]
        for topic_idx in range(len(SEED_TOPICS)):
            rec[f'topic_{topic_idx + 1}_prob'] = dist[topic_idx]

        records.append(rec)

    df = pd.DataFrame(records)

    # 5) Save CSV
    output_dir = f'yearly_occurrence_data/window_{window}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f'{year}_cooccurrence_analysis.csv'
    df.to_csv(os.path.join(output_dir, output_file), index=False)

    return df


def process_window(start_year: int, window_size: int, analyzer: TemporalLDAAnalyzer):
    """
    A single window: load data for each year, sample, combine -> build dict, train LDA -> 
    then process each year in a bulk manner (co-occurrence & doc topic dist).
    """
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
        
        year_dataset = load_dataset(
            "dell-research-harvard/AmericanStories",
            "subset_years",
            year_list=[year_str],
            trust_remote_code=True
        )
        
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
    
    # Combine for dictionary building & LDA
    preprocess_start = time()
    if not window_data:
        print(f"No data for window {start_year}-{start_year + window_size - 1}")
        return
    window_df = pd.concat([window_data[yr] for yr in window_data])
    print(f"\nTotal articles in window: {len(window_df)}")

    # Preprocess & build dictionary/corpus
    print("\nPreprocessing & building dictionary for entire window...")
    texts, dictionary, corpus = analyzer.process_temporal_window(window_df, start_year)
    print(f"Preprocessing time: {time() - preprocess_start:.2f}s")

    if texts is None:
        print("No texts found, skipping this window.")
        return

    # Train LDA model
    lda_start = time()
    print("\nTraining LDA model for window...")
    model = analyzer.train_window_model(texts, dictionary, corpus, start_year)
    print(f"LDA training time: {time() - lda_start:.2f}s")

    # Get top words for co-occurrence
    top_words_cache = get_top_words_per_topic(model)

    # Process each year in bulk
    cooccur_start = time()
    print("\nProcessing co-occurrences & topic inference for each year in bulk...")
    futures = []
    results = {}

    # We can do year-level parallelism using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=analyzer.num_processes) as executor:
        for year in window_data:
            sampled_articles = list(window_data[year].to_dict('records'))
            year_str = str(year)
            future = executor.submit(
                process_yearly_data_bulk,
                sampled_articles,
                year_str,
                model,
                dictionary,
                analyzer,
                window_size,
                top_words_cache
            )
            futures.append((year_str, future))

        for year_str, fut in futures:
            try:
                df_result = fut.result()
                results[year_str] = df_result
                print(f"✓ Year {year_str}: Processed {len(df_result)} articles")
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
    freeze_support()  # Needed on Windows if you use multiprocessing
    total_start = time()

    print("Initializing NLTK resources...")
    nltk.download('punkt')
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

    start_year = 1810
    end_year = 1820
    total_windows = (end_year - start_year) // window_size

    print(f"\nProcessing {total_windows} windows from {start_year} to {end_year}")

    for window_start in range(start_year, end_year, window_size):
        process_window(window_start, window_size, analyzer)

    print(f"\nTotal execution time: {time() - total_start:.2f}s")


if __name__ == '__main__':
    main()
