import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def choose_sentiment_column(df):
    """Return the sentiment column to use for correlation: numeric if it varies, else score."""
    if df['sentiment_numeric'].nunique() == 1:
        logging.info(
            f"All sentiment labels are the same ('{df['sentiment'].iloc[0]}'). Using sentiment_score for correlation."
        )
        return 'sentiment_score'
    else:
        logging.info("Mixed sentiment labels. Using sentiment_numeric for correlation.")
        return 'sentiment_numeric'

def load_data(lyric_file="lyric_analysis_results.csv", audio_file="audio_features_results.csv"):
    """Load and merge lyric and audio analysis results."""
    try:
        lyric_df = pd.read_csv(lyric_file)
        audio_df = pd.read_csv(audio_file)
        
        # Convert sentiment to numeric values for correlation
        sentiment_map = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
        lyric_df['sentiment_numeric'] = lyric_df['sentiment'].map(sentiment_map)

        def canonical_title(series):
            return series.str.replace(" ", "_").str.lower().str.strip()

        lyric_df['title'] = canonical_title(lyric_df['title'])
        audio_df['title'] = canonical_title(audio_df['title'])
        
        # Merge datasets
        merged_df = pd.merge(lyric_df, audio_df, left_on='title', right_on='title', how='inner')
        logging.info(f"Loaded {len(merged_df)} songs with both lyric and audio data")
        
        return merged_df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def calculate_correlations(df):
    """Calculate correlations between sentiment and audio features."""
    sentiment_column = choose_sentiment_column(df)
    audio_features = ['tempo', 'vocal_tone_quality', 'sound_brightness',
                     'musical_note_strength', 'sound_loudness']
    
    correlations = {}
    for feature in audio_features:
        try:
            correlation, p_value = stats.pearsonr(df[sentiment_column], df[feature])
            
            # Add sample size warning if needed
            if len(df) < 10:
                logging.warning(f"Small sample size ({len(df)} samples) may affect correlation reliability")
                
            correlations[feature] = {
                'correlation': correlation,
                'p_value': p_value,
                'sample_size': len(df)
            }
        except Exception as e:
            logging.error(f"Error calculating correlation for {feature}: {str(e)}")
            correlations[feature] = {
                'correlation': np.nan,
                'p_value': np.nan,
                'sample_size': len(df)
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
    plt.style.use('seaborn-v0_8')  # Updated style name
    
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
        # Calculate correlation with error handling
        try:
            if df['sentiment_numeric'].std() == 0 or df[feature].std() == 0:
                corr = np.nan
            else:
                corr = df['sentiment_numeric'].corr(df[feature])
        except Exception:
            corr = np.nan
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
    
    print(f"\nCorrelation Analysis Results (Sample Size: {len(df)} songs)")
    print("-" * 60)
    
    # Format and display results
    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x) if not pd.isna(x) else 'N/A')
    formatted_corr = correlations.copy()
    formatted_corr['significance'] = formatted_corr.apply(
        lambda row: '***' if row['p_value'] < 0.001 else
                   '**' if row['p_value'] < 0.01 else
                   '*' if row['p_value'] < 0.05 else
                   'ns', axis=1
    )
    
    print(formatted_corr[['correlation', 'p_value', 'significance']])
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant")
    
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