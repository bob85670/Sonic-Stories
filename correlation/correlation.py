import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(lyric_file="../lyric_analysis_results.csv", audio_file="../audio_features_results.csv"):
    """Load and merge lyric and audio analysis results."""
    try:
        lyric_df = pd.read_csv(lyric_file)
        audio_df = pd.read_csv(audio_file)
        
        # Convert sentiment to numeric values for correlation
        sentiment_map = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
        lyric_df['sentiment_numeric'] = lyric_df['sentiment'].map(sentiment_map)
        
        # Merge datasets
        merged_df = pd.merge(lyric_df, audio_df, left_on='title', right_on='title', how='inner')
        logging.info(f"Loaded {len(merged_df)} songs with both lyric and audio data")
        
        return merged_df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def calculate_correlations(df):
    """Calculate correlations between sentiment and audio features."""
    audio_features = ['tempo', 'vocal_tone_quality', 'sound_brightness', 
                     'musical_note_strength', 'sound_loudness']
    
    correlations = {}
    for feature in audio_features:
        correlation, p_value = stats.pearsonr(df['sentiment_numeric'], df[feature])
        correlations[feature] = {
            'correlation': correlation,
            'p_value': p_value
        }
    
    # Convert to DataFrame for easier viewing
    corr_df = pd.DataFrame.from_dict(correlations, orient='index')
    corr_df = corr_df.sort_values('correlation', ascending=False)
    
    return corr_df

def plot_correlations(df, output_dir="correlation_plots"):
    """Create visualization plots for correlations."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the style
    plt.style.use('seaborn')
    
    # Key features to visualize
    key_features = ['tempo', 'vocal_tone_quality', 'sound_brightness', 
                   'musical_note_strength', 'sound_loudness']
    
    # Create scatter plots for key features
    for feature in key_features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=feature, y='sentiment_numeric', alpha=0.6)
        sns.regplot(data=df, x=feature, y='sentiment_numeric', scatter=False, color='red')
        
        plt.title(f'Sentiment vs {feature.replace("_", " ").title()}')
        plt.xlabel(feature.replace("_", " ").title())
        plt.ylabel('Sentiment (Negative → Positive)')
        
        # Add correlation coefficient to plot
        corr = df['sentiment_numeric'].corr(df[feature])
        plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.savefig(os.path.join(output_dir, f'sentiment_vs_{feature}.png'))
        plt.close()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 8))
    features_for_heatmap = ['sentiment_numeric'] + key_features
    correlation_matrix = df[features_for_heatmap].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', center=0, 
                vmin=-1, vmax=1, square=True)
    plt.title('Correlation Heatmap: Sentiment vs Audio Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

def compare_ai_human_songs(df):
    """Compare correlation patterns between AI and human songs."""
    # Assuming 'source' column indicates AI vs human (e.g., 'AI', 'Human')
    if 'source' not in df.columns:
        logging.warning("Source column not found. Cannot compare AI vs human songs.")
        return None
    
    results = {}
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        correlations = calculate_correlations(source_df)
        results[source] = correlations
    
    # Compare correlations
    comparison = pd.DataFrame()
    for source, corr_df in results.items():
        comparison[f'{source}_correlation'] = corr_df['correlation']
        comparison[f'{source}_p_value'] = corr_df['p_value']
    
    return comparison

def main():
    """Main function to perform correlation analysis."""
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Calculate overall correlations
    logging.info("Calculating correlations...")
    correlations = calculate_correlations(df)
    print("\nOverall Correlations:")
    print(correlations)
    
    # Create visualization plots
    logging.info("Creating visualization plots...")
    plot_correlations(df)
    
    # Compare AI vs human songs
    # logging.info("Comparing AI vs human correlations...")
    # comparison = compare_ai_human_songs(df)
    # if comparison is not None:
    #     print("\nAI vs Human Comparison:")
    #     print(comparison)
    
    # Save results
    correlations.to_csv('correlation_results.csv')
    # if comparison is not None:
    #     comparison.to_csv('ai_human_comparison.csv')
    
    logging.info("Analysis complete. Results saved to CSV files.")

if __name__ == "__main__":
    main()