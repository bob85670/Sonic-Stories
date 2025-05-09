# Sonic-Stories

Sonic-Stories is a data-driven Python project analyzing the relationship between song lyrics, audio features, and sentiment. Using a combination of lyric sentiment analysis (BERT model) and audio signal processing (`librosa`), the project quantifies and correlates lyrical mood with measurable audio features (e.g., tempo, loudness). The workflow is designed for extensibility—new mp3 files, lyrics, or analysis types can be added easily.

## Project Flow

1. **Collect Data**
   - Place MP3 song files in the `audio_data/` directory.
   - Place lyrics text files in the `lyics_data/` directory.

2. **Convert lyric Files**
   - `convert_txt_to_csv.py`: Converts `.txt` files in `lyrics_data/` to one `.csv`

3. **Analyze Lyrics**
   - `lyric_analysis_bert.py`: Analyzes song lyrics for sentiment (via BERT), assigns topics, and outputs results to `lyric_analysis_results.csv`.

4. **Convert Audio Files**
   - `convert_mp3_to_wav.py`: Converts `.mp3` files in `audio_data/` to standardized `.wav` files in `wav_input/` using FFmpeg.

5. **Extract Audio Features**
   - `audio_analysis_librosa.py`: Loads `.wav` files and uses `librosa` to extract various audio features (tempo, tone quality, brightness, note strength, loudness), outputting results to `audio_features_results.csv`.

6. **Correlate and Visualize**
   - `correlation/correlation.py`: Merges lyric and audio features, computes correlations between sentiment and audio aspects, and visualizes the results.

## File & Folder Overview

- **convert_mp3_to_wav.py**  
  Converts all mp3 files to wav format for consistent audio analysis. Standardizes to mono, 44.1kHz using FFmpeg.
- **audio_analysis_librosa.py**  
  Extracts detailed audio features from wav files using `librosa`. Output is stored as a CSV.
- **audio_features_test.py**  
  Provides testing functionality for verifying audio feature extraction accuracy.
- **convert_txt_to_csv.py**  
  Helper to parse TXT-formatted lyrics/transcripts into CSV format.
- **lyric_analysis_bert.py**  
  Uses a BERT-based model to produce sentiment scores and topic analysis for lyrics.
- **correlation/**
  - **correlation.py**  
    Merges analysis results, computes Pearson correlations, and creates visualizations. Main analysis logic for the project.
- **.python-version**  
  Pin Python version to 3.11.8 for reproducibility.

## Environment Setup

1. **Python Version**
   - Python 3.11.8 is required (see `.python-version`)
   - Create and activate the virtual environment (recommended: Python 3.11):
   
     **Mac/Linux:**
     ```bash
     python3.11 -m venv venv
     source venv/bin/activate
     ```

2. **Dependencies**
   - FFmpeg must be installed and available on your `$PATH` for audio conversion
   - Install Python packages:
     ```bash
     pip install -r requirements.txt
     ```

## Typical Workflow

1. **Copy audio_data(.mp3) and lyrics_data(.txt files) folder to this repo.**

2. **Convert Lyrics:**
   ```bash
   python convert_txt_to_csv.py
   ```
   This will produce all_lyrics.csv

3. **Analyze Lyrics:**
   ```bash
   python lyric_analysis_bert.py
   ```
   This will produce lyric_analysis_results.csv

4. **Convert MP3s:**
   ```bash
   python convert_mp3_to_wav.py
   ```
   This will produce create .wav files in mp3_input folder

5. **Extract Audio Features:**
   ```bash
   python audio_analysis_librosa.py
   ```
   This will produce audio_features_results.csv

6. **Correlate & Visualize:**
   ```bash
   python correlation/correlation.py
   ```
   See results!

## Project Vision

Sonic-Stories combines Natural Language Processing (NLP) and Digital Signal Processing (DSP) to explore how the feel of music (through audio features) relates to the written emotion in lyrics. It's designed for music analytics, emotional mapping, or even new AI-assisted songwriting tools. The structure makes it easy to extend with new models or analytical angles.