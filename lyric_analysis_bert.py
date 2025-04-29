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

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize BERT sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize NLTK tools for topic modeling
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_lyrics(lyrics):
    """Preprocess lyrics for topic modeling: tokenize, remove stopwords, lemmatize."""
    processed_lyrics = []
    for lyric in lyrics:
        # Convert to lowercase and remove special characters
        lyric = re.sub(r'[^a-zA-Z\s]', '', lyric.lower())
        # Tokenize
        tokens = word_tokenize(lyric)
        # Remove stopwords and short words, lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
        processed_lyrics.append(tokens)
    return processed_lyrics

def get_sentiment(lyrics):
    """Perform sentiment analysis using BERT."""
    sentiments = []
    scores = []
    for lyric in lyrics:
        try:
            # Truncate lyrics to 512 tokens (BERT's max length)
            result = sentiment_analyzer(lyric[:512])[0]
            label = result['label']
            score = result['score']
            # Approximate neutral sentiment for low confidence scores (e.g., < 0.6)
            if score < 0.6:
                label = 'NEUTRAL'
            else:
                label = 'POSITIVE' if label == 'POSITIVE' else 'NEGATIVE'
            sentiments.append(label)
            scores.append(score)
        except Exception as e:
            print(f"Error processing lyric: {lyric[:50]}... Error: {e}")
            sentiments.append('UNKNOWN')
            scores.append(0.0)
    return sentiments, scores

def get_topics(lyrics, num_topics=3, num_words=5):
    """Perform topic modeling using LDA."""
    # Preprocess lyrics
    processed_lyrics = preprocess_lyrics(lyrics)
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_lyrics)
    corpus = [dictionary.doc2bow(text) for text in processed_lyrics]
    
    # Train LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
    
    # Extract topics for each lyric
    topics = []
    for bow in corpus:
        topic_dist = lda_model[bow]
        # Get the dominant topic and its keywords
        dominant_topic = max(topic_dist, key=lambda x: x[1], default=(0, 0.0))
        topic_id = dominant_topic[0]
        topic_words = [word for word, _ in lda_model.show_topic(topic_id, topn=num_words)]
        topics.append(topic_words)
    
    return topics

def main():
    """Main function to perform lyric analysis and store results."""
    # Load lyrics from .csv
    input_csv = "all_lyrics.csv"
    df = pd.read_csv(input_csv)
    lyrics = df['lyrics'].tolist()
    
    # Perform sentiment analysis
    sentiments, scores = get_sentiment(lyrics)
    
    # Perform topic modeling
    topics = get_topics(lyrics, num_topics=3, num_words=5)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'title': df['title'],
        'artist': df['artist'],
        'year': df['year'],
        'source': df['source'],
        'lyrics': lyrics,
        'sentiment': sentiments,
        'sentiment_score': scores,
        'topics': [', '.join(topic) for topic in topics]
    })
    
    # Save results to CSV
    results_df.to_csv('lyric_analysis_results.csv', index=False, encoding='utf-8')
    
    # Print results
    print("\nLyric Analysis Results:")
    print(results_df.head())
    
    return results_df

if __name__ == "__main__":
    main()