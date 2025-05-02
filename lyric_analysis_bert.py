import pandas as pd
import numpy as np
from transformers import pipeline
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import logging # Added for cleaner output

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else f'tokenizers/{resource}')
    except LookupError:
        logging.info(f"Downloading NLTK resource '{resource}'...")
        nltk.download(resource)

# Initialize BERT sentiment analysis pipeline
# Using a specific revision for consistency
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", revision="af0f99b")
logging.info("Sentiment analysis pipeline initialized.")

# --- Preprocessing Setup ---
lemmatizer = WordNetLemmatizer()
# Extend standard English stopwords with common lyric words/contractions/interjections
custom_stopwords = set(stopwords.words('english')).union({
    'yeah', 'oh', 'ooh', 'uh', 'ah', 'hmm', 'mmm',
    'like', 'got', 'get', 'know', 'dont', 'im', 'youre', 'hes', 'shes', 'its', 'theyre', 'ive', 'cant', 'wont', 'thats',
    'gon', 'wan', 'na', 'da', 'la', 'hey', 'yo', 'em', 'ya', 'huh'
    # Add more domain-specific words if you observe them frequently
})
logging.info(f"Using {len(custom_stopwords)} stopwords.")
# --- End Preprocessing Setup ---


def preprocess_lyrics_for_lda(lyrics_list):
    """Preprocess lyrics for topic modeling: clean, tokenize, remove stopwords, lemmatize."""
    processed_docs = []
    logging.info("Starting lyric preprocessing for LDA...")
    for i, doc in enumerate(lyrics_list):
        if not isinstance(doc, str): # Handle potential non-string entries
             logging.warning(f"Skipping non-string lyric at index {i}")
             processed_docs.append([]) # Append empty list to maintain structure
             continue

        # 1. Lowercase and remove non-alphanumeric characters (keeping spaces)
        text = re.sub(r'[^a-z\s]', '', doc.lower())
        # 2. Tokenize
        tokens = word_tokenize(text)
        # 3. Lemmatize, remove stopwords and short words
        processed_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in custom_stopwords and len(token) > 2 # Keep words longer than 2 chars
        ]
        processed_docs.append(processed_tokens)
        if (i + 1) % 100 == 0: # Log progress
            logging.info(f"Processed {i+1}/{len(lyrics_list)} lyrics for LDA...")

    logging.info("Finished lyric preprocessing for LDA.")
    # Filter out empty documents that might result from preprocessing
    processed_docs = [doc for doc in processed_docs if doc]
    return processed_docs

def get_sentiment(lyrics_list):
    """Perform sentiment analysis using BERT."""
    sentiments = []
    scores = []
    logging.info("Starting sentiment analysis...")
    # Using batch processing for potentially faster inference (if supported and beneficial)
    # However, transformers pipeline handles batching internally often, so simple loop is fine too.
    for i, lyric in enumerate(lyrics_list):
         if not isinstance(lyric, str): # Handle potential non-string entries
             logging.warning(f"Skipping non-string lyric for sentiment at index {i}")
             sentiments.append('UNKNOWN')
             scores.append(0.0)
             continue
         if not lyric.strip(): # Handle empty strings
             logging.warning(f"Skipping empty lyric for sentiment at index {i}")
             sentiments.append('UNKNOWN')
             scores.append(0.0)
             continue

         try:
            # Truncate lyrics safely
            truncated_lyric = lyric[:510] # Leave space for special tokens
            result = sentiment_analyzer(truncated_lyric)[0]
            label = result['label']
            score = result['score']

            # Refined sentiment mapping (optional, keep simple POS/NEG/NEUTRAL)
            # A score close to 0.5 from either label could be Neutral.
            # Using a threshold like 0.6 is a heuristic. Adjust if needed.
            if score < 0.6: # Example threshold for neutrality
                 sentiment_label = 'NEUTRAL'
            elif label == 'POSITIVE':
                 sentiment_label = 'POSITIVE'
            else: # label == 'NEGATIVE'
                 sentiment_label = 'NEGATIVE'

            sentiments.append(sentiment_label)
            scores.append(score if sentiment_label != 'NEUTRAL' else 1.0 - score) # Store original confidence or derived neutral score

         except Exception as e:
            logging.error(f"Error processing lyric for sentiment: {lyric[:50]}... Error: {e}")
            sentiments.append('ERROR')
            scores.append(0.0)

         if (i + 1) % 100 == 0: # Log progress
            logging.info(f"Processed {i+1}/{len(lyrics_list)} lyrics for sentiment...")

    logging.info("Finished sentiment analysis.")
    return sentiments, scores

def get_topics(processed_docs, num_topics=10, num_words=7, passes=20):
    """Perform topic modeling using LDA on preprocessed documents."""
    if not processed_docs:
        logging.warning("No documents available for topic modeling after preprocessing.")
        return [], None # Return empty list and None for model if no docs

    logging.info(f"Starting LDA topic modeling with num_topics={num_topics}, passes={passes}...")
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_docs)
    # Optional: Filter extremes (remove tokens that appear in less than 5 documents or more than 50% of documents)
    # dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in processed_docs]

    if not corpus:
        logging.warning("Corpus is empty after creating dictionary and BoW representation.")
        return [], None # Return empty list and None for model

    # Train LDA model
    # Added alpha='auto' and eta='auto' for potentially better topic separation
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes,        # Increased passes
        alpha='auto',       # Let Gensim learn asymmetric alpha
        eta='auto',         # Let Gensim learn asymmetric eta
        eval_every=None     # Disable perplexity evaluation during training for speed
    )
    logging.info("Finished LDA model training.")

    # --- Extract topics for original documents (Handle potential index mismatches) ---
    # This part needs care if preprocess_lyrics_for_lda filtered out empty docs.
    # We will assign topics based on the *processed* docs and map back if needed,
    # but for simplicity here, we'll assume the output length matches the input `lyrics_list` initially.
    # A more robust approach would map indices.

    # For now, let's just get the topics for the documents that were actually processed
    doc_topics = []
    logging.info("Assigning topics to documents...")
    corpus_for_inference = [dictionary.doc2bow(text) for text in processed_docs] # Use the same corpus used for training
    topic_distributions = lda_model.get_document_topics(corpus_for_inference, minimum_probability=0.0)

    for i, dist in enumerate(topic_distributions):
        if not dist: # Handle cases where a document might not strongly belong to any topic
            top_topic_words = ["N/A"]
        else:
            # Get the dominant topic
            dominant_topic = max(dist, key=lambda x: x[1])
            topic_id = dominant_topic[0]
            # Get the top words for that topic
            topic_words_probs = lda_model.show_topic(topic_id, topn=num_words)
            top_topic_words = [word for word, prob in topic_words_probs]
        doc_topics.append(", ".join(top_topic_words))
        if (i + 1) % 100 == 0: # Log progress
             logging.info(f"Assigned topics to {i+1}/{len(processed_docs)} documents...")

    logging.info("Finished assigning topics.")
    # Note: The length of doc_topics will match len(processed_docs),
    # which might be less than the original number of lyrics if some were empty/non-string.
    # For this example, we'll proceed, but a production system might need index mapping.
    return doc_topics, lda_model # Return topics and the trained model

def display_topics(lda_model, num_words=10):
    """Prints the keywords for each topic discovered by the LDA model."""
    if not lda_model:
        logging.warning("LDA model not available for displaying topics.")
        return
    logging.info("\n--- Discovered Topics ---")
    topics = lda_model.show_topics(num_topics=lda_model.num_topics, num_words=num_words, formatted=False)
    for i, topic in topics:
        words = [word for word, prob in topic]
        logging.info(f"Topic {i}: {', '.join(words)}")
    logging.info("--- End Discovered Topics ---")


def main():
    """Main function to perform lyric analysis and store results."""
    input_csv = "all_lyrics.csv"
    output_csv = 'lyric_analysis_results.csv'
    num_lda_topics = 10  # Tunable: Increased number of topics
    num_lda_words = 7   # Tunable: Number of words per topic
    num_lda_passes = 25 # Tunable: Increased number of passes

    try:
        df = pd.read_csv(input_csv)
        # Ensure 'lyrics' column exists and handle missing values
        if 'lyrics' not in df.columns:
            logging.error(f"'lyrics' column not found in {input_csv}")
            return None
        # Fill NaN values with empty strings BEFORE processing
        df['lyrics'] = df['lyrics'].fillna('')
        lyrics_list = df['lyrics'].tolist()
        logging.info(f"Loaded {len(lyrics_list)} lyrics from {input_csv}")

    except FileNotFoundError:
        logging.error(f"Input file not found: {input_csv}")
        return None
    except Exception as e:
        logging.error(f"Error loading or processing CSV: {e}")
        return None

    # --- Sentiment Analysis ---
    sentiments, scores = get_sentiment(lyrics_list)

    # --- Topic Modeling ---
    # 1. Preprocess specifically for LDA
    processed_lyrics_lda = preprocess_lyrics_for_lda(lyrics_list)

    # 2. Perform LDA on processed lyrics
    # Note: get_topics now returns topics for the *processed* documents
    topics_list, trained_lda_model = get_topics(
        processed_lyrics_lda,
        num_topics=num_lda_topics,
        num_words=num_lda_words,
        passes=num_lda_passes
    )

    # Display the discovered topics and their keywords
    display_topics(trained_lda_model, num_words=10)


    # --- Create Results DataFrame ---
    # !! Crucial alignment step !!
    # Since preprocessing might remove some docs, we need to align results.
    # A simple way is to create a temporary df from processed docs and merge,
    # but for this example, we'll assume a mapping or assign 'N/A' to skipped ones.
    # We need an identifier that survives preprocessing (like original index or title/artist if unique)
    # For simplicity here, we'll pad the topics list if it's shorter than the original df.
    # This is NOT robust if the order changed significantly due to filtering.
    # A safer method involves tracking original indices during preprocessing.

    # Let's try a slightly safer approach using the original DataFrame index
    results_data = {
        'title': df['title'],
        'artist': df['artist'],
        'year': df['year'],
        'source': df['source'],
        'lyrics': df['lyrics'], # Keep original lyrics
        'sentiment': sentiments, # Length should match original df
        'sentiment_score': scores, # Length should match original df
        'topics': ["Preprocessing Skipped/Failed"] * len(df) # Default value
    }

    # Create an index mapping from the original dataframe for lyrics that were processed
    processed_indices = [i for i, doc in enumerate(df['lyrics'].tolist()) if isinstance(doc, str) and doc.strip() and preprocess_lyrics_for_lda([doc])[0]] # Re-run preprocessing logic carefully
    
    if len(processed_indices) == len(topics_list):
        for original_idx, topic_str in zip(processed_indices, topics_list):
            results_data['topics'][original_idx] = topic_str
    else:
        logging.warning(f"Mismatch between processed indices ({len(processed_indices)}) and topics generated ({len(topics_list)}). Topic assignment might be inaccurate.")
        # Fallback: assign topics sequentially, acknowledging potential inaccuracy
        min_len = min(len(results_data['topics']), len(topics_list))
        for i in range(min_len):
             # This assumes the first `min_len` entries correspond. Risky.
             # Only use if the index mapping above failed or seems wrong.
             # results_data['topics'][i] = topics_list[i] 
             pass # Keep default "Preprocessing Skipped/Failed" if mapping failed


    results_df = pd.DataFrame(results_data)


    # --- Save and Print Results ---
    try:
        results_df.drop(columns=['lyrics']).to_csv(output_csv, index=False, encoding='utf-8')
        logging.info(f"Results saved to {output_csv}")
    except Exception as e:
        logging.error(f"Error saving results to CSV: {e}")

    logging.info("\nLyric Analysis Results (first 5 rows):")
    print(results_df.drop(columns=['lyrics']).head().to_string())

    return results_df

if __name__ == "__main__":
    main()
