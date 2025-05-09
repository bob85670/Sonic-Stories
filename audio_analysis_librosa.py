import librosa
import numpy as np
import pandas as pd
import os
import warnings

def extract_audio_features(df, audio_folder="wav_input", output_csv="audio_features.csv"):
    """
    Extract audio features using Librosa for .wav files corresponding to song titles in df.
    Save flattened features to a CSV file and print summary statistics.
    
    Args:
        df (pd.DataFrame): DataFrame with 'title' column.
        audio_folder (str): Path to folder containing .wav files.
        output_csv (str): Path to save the output CSV.
    
    Returns:
        pd.DataFrame: DataFrame with extracted audio features.
    """
    audio_features = {
        "title": [],
        "tempo": [],
        "vocal_tone_quality": [],  # First MFCC coefficient mean
        "sound_brightness": [],    # Spectral centroid mean
        "musical_note_strength": [], # First chroma coefficient mean
        "sound_loudness": []       # RMS mean
    }

    processed_count = 0
    for title in df["title"]:
        audio_path = os.path.join(audio_folder, f"{title}.wav")
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found for {title} at {audio_path}")
            continue

        try:
            # Load audio
            y, sr = librosa.load(audio_path)

            # Extract features
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)

            # Store features
            audio_features["title"].append(title)
            audio_features["tempo"].append(tempo[0])  # tempo is returned as a 1-element array
            audio_features["vocal_tone_quality"].append(np.mean(mfcc, axis=1)[0])  # First MFCC coefficient mean
            audio_features["sound_brightness"].append(np.mean(spectral_centroid))
            audio_features["musical_note_strength"].append(np.mean(chroma, axis=1)[0])  # First chroma coefficient mean
            audio_features["sound_loudness"].append(np.mean(rms))

            processed_count += 1
            print(f"Processed {title}")

        except Exception as e:
            print(f"Error processing {title}: {str(e)}")
            continue

    # Create DataFrame
    audio_df = pd.DataFrame(audio_features)

    # Validate and print summary statistics
    if audio_df.empty:
        warnings.warn("No audio features extracted. Check audio files and paths.")
        return audio_df

    print("\nSummary Statistics:")
    print(audio_df[["tempo", "vocal_tone_quality", "sound_brightness", "musical_note_strength", "sound_loudness"]].describe())
    print(f"Total songs processed: {processed_count}/{len(df)}")

    # Save to CSV
    audio_df.to_csv(output_csv, index=False)
    print(f"Saved audio features to {output_csv}")

    return audio_df

def main():
    """
    Main function to convert .mp3 to .wav (if needed) and extract audio features.
    """
    # Configuration
    wav_folder = "wav_input"
    lyric_csv = "lyric_analysis_results.csv"
    output_csv = "audio_features_results.csv"

    # Step 2: Load lyric DataFrame
    try:
        df = pd.read_csv(lyric_csv)
        if "title" not in df.columns:
            raise ValueError("DataFrame must contain 'title' column")
    except FileNotFoundError:
        print(f"Error: {lyric_csv} not found.")
        return
    except Exception as e:
        print(f"Error loading DataFrame: {str(e)}")
        return

    # Step 3: Ensure title matches .wav filenames (e.g., replace spaces with underscores)
    df["title"] = df["title"].str.replace(" ", "_").str.strip()

    # Step 4: Extract audio features
    extract_audio_features(df, audio_folder=wav_folder, output_csv=output_csv)

if __name__ == "__main__":
    main()