import librosa
import numpy as np
import pandas as pd
import os
import warnings

def extract_audio_features(df, audio_folder="audio_files_wav", output_csv="audio_features.csv"):
    """
    Extract audio features using Librosa for .wav files corresponding to song titles in df.
    Save flattened features to a CSV file and print summary statistics.
    
    Args:
        df (pd.DataFrame): DataFrame with 'song_title' column.
        audio_folder (str): Path to folder containing .wav files.
        output_csv (str): Path to save the output CSV.
    
    Returns:
        pd.DataFrame: DataFrame with extracted audio features.
    """
    audio_features = {
        "song_title": [],
        "tempo": [],
        "spectral_centroid": [],
        "rms_mean": [],
    }
    # Add columns for 13 MFCC coefficients and 12 chroma features
    for i in range(13):
        audio_features[f"mfcc_{i+1}"] = []
    for i in range(12):
        audio_features[f"chroma_{i+1}"] = []

    processed_count = 0
    for song_title in df["song_title"]:
        audio_path = os.path.join(audio_folder, f"{song_title}.wav")
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found for {song_title} at {audio_path}")
            continue

        try:
            # Load audio
            y, sr = librosa.load(audio_path)

            # Extract features
            tempo, _ = librosa.beat.tempo(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)

            # Store features
            audio_features["song_title"].append(song_title)
            audio_features["tempo"].append(tempo)
            audio_features["spectral_centroid"].append(np.mean(spectral_centroid))
            audio_features["rms_mean"].append(np.mean(rms))

            # Store individual MFCC and chroma coefficients
            mfcc_mean = np.mean(mfcc, axis=1)
            chroma_mean = np.mean(chroma, axis=1)
            for i in range(13):
                audio_features[f"mfcc_{i+1}"].append(mfcc_mean[i])
            for i in range(12):
                audio_features[f"chroma_{i+1}"].append(chroma_mean[i])

            processed_count += 1
            print(f"Processed {song_title}")

        except Exception as e:
            print(f"Error processing {song_title}: {str(e)}")
            continue

    # Create DataFrame
    audio_df = pd.DataFrame(audio_features)

    # Validate and print summary statistics
    if audio_df.empty:
        warnings.warn("No audio features extracted. Check audio files and paths.")
        return audio_df

    print("\nSummary Statistics:")
    print(audio_df[["tempo", "spectral_centroid", "rms_mean"]].describe())
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
    wav_folder = "audio_files_wav"
    lyric_csv = "song_lyrics.csv"
    output_csv = "audio_features.csv"

    # Step 2: Load lyric DataFrame
    try:
        df = pd.read_csv(lyric_csv)
        if "song_title" not in df.columns:
            raise ValueError("DataFrame must contain 'song_title' column")
    except FileNotFoundError:
        print(f"Error: {lyric_csv} not found.")
        return
    except Exception as e:
        print(f"Error loading DataFrame: {str(e)}")
        return

    # Step 3: Ensure song_title matches .wav filenames (e.g., replace spaces with underscores)
    df["song_title"] = df["song_title"].str.replace(" ", "_").str.strip()

    # Step 4: Extract audio features
    audio_df = extract_audio_features(df, audio_folder=wav_folder, output_csv=output_csv)

    # # Step 5: Optional - Merge with lyric DataFrame
    # if not audio_df.empty:
    #     combined_df = pd.merge(df, audio_df, on="song_title", how="left")
    #     combined_df.to_csv("combined_analysis.csv", index=False)
    #     print("Saved combined lyric and audio features to combined_analysis.csv")
    # else:
    #     print("No audio features extracted, skipping merge.")

if __name__ == "__main__":
    main()