import librosa
import numpy as np
import os

def test_librosa_features(audio_path):
    """
    Test Librosa's audio feature extraction on a single .wav file.
    Prints tempo, MFCC, spectral centroid, chroma, and RMS features to the terminal.
    
    Args:
        audio_path (str): Path to the input .wav file.
    """
    # Check if audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    try:
        # Load audio file
        print(f"Loading {audio_path}...")
        y, sr = librosa.load(audio_path)
        print(f"Audio loaded: duration {librosa.get_duration(y=y, sr=sr):.2f} seconds, sample rate {sr} Hz")

        # Extract and print features
        # Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        print(f"Tempo: {tempo[0]:.2f} BPM")  # tempo is returned as a 1-element array

        # MFCC (timbre/vocal qualities)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        print(f"Vocal Tone Quality: {mfcc_mean[0]:.2f}")

        # Spectral Centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        print(f"Sound Brightness: {np.mean(spectral_centroid):.2f} Hz")

        # Chroma (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        print(f"Musical Note Strength: {chroma_mean[0]:.3f}")

        # RMS (loudness)
        rms = librosa.feature.rms(y=y)
        print(f"Sound Loudness: {np.mean(rms):.3f}")

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")

if __name__ == "__main__":
    # Use your specific audio file
    audio_file = "audio_data/Justin Bieber - Off My Face (Live from Paris).wav"
    test_librosa_features(audio_file)