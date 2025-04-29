import os
import pandas as pd
import re

# Directory containing .txt files
txt_dir = "lyrics_data/"

# List to store data
data = []

# Regex to extract title and artist from --- 'Title' by Artist --- line
title_artist_pattern = re.compile(r"--- '(.+?)' by (.+?) ---")

def parse_txt_file(filepath):
    """Parse a single .txt file and extract songs with metadata."""
    # Extract year from filename (e.g., lyrics_2022.txt -> 2022)
    year = re.search(r'\d{4}', os.path.basename(filepath)).group()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by song separator (--- to ==================================================)
    songs = content.split('==================================================')
    songs = [song.strip() for song in songs if song.strip()]  # Remove empty entries
    
    parsed_songs = []
    for song in songs:
        # Skip header if present (e.g., LYRICS COLLECTION)
        if song.startswith('LYRICS COLLECTION'):
            continue
        
        # Extract title and artist
        title_artist_match = title_artist_pattern.search(song)
        if not title_artist_match:
            print(f"Warning: Could not parse title/artist in {filepath}: {song[:50]}...")
            continue
        
        title = title_artist_match.group(1)
        artist = title_artist_match.group(2)
        
        # Extract lyrics (everything after the --- line)
        lyrics_start = song.find('---', 3) + 3  # Skip the --- 'Title' by Artist --- line
        lyrics = song[lyrics_start:].strip()
        
        # Clean lyrics (remove leading/trailing newlines)
        lyrics = lyrics.strip()
        
        # Add to parsed songs
        parsed_songs.append({
            'title': title,
            'artist': artist,
            'year': year,
            'source': 'human',  # Default to human; update for AI songs
            'lyrics': lyrics
        })
    
    return parsed_songs

# Iterate over .txt files
for txt_file in os.listdir(txt_dir):
    if txt_file.endswith(".txt"):
        filepath = os.path.join(txt_dir, txt_file)
        songs = parse_txt_file(filepath)
        data.extend(songs)

# Create DataFrame
df = pd.DataFrame(data)

# Save to .csv
output_csv = "all_lyrics.csv"
df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"Combined {len(df)} songs into {output_csv}")
print(df.head())