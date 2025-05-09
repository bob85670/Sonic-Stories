import os
import subprocess

def convert_mp3_to_wav(input_folder, output_folder):
    """
    Convert all .mp3 files in input_folder to .wav files and save them in output_folder.
    Uses FFmpeg to standardize to 44.1kHz sample rate and mono channel.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through files in input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".mp3"):
            input_path = os.path.join(input_folder, filename)
            output_filename = filename.replace(".mp3", ".wav")
            output_path = os.path.join(output_folder, output_filename)

            # FFmpeg command to convert to .wav (44.1kHz, mono)
            cmd = [
                "ffmpeg",
                "-i", input_path,           # Input file
                "-ar", "44100",            # Sample rate: 44.1kHz
                "-ac", "1",                # Mono channel
                "-y",                      # Overwrite output if exists
                output_path                # Output file
            ]

            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print(f"Converted {filename} to {output_filename}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {filename}: {e.stderr}")

if __name__ == "__main__":
    input_folder = "audio_data" 
    output_folder = "wav_input" 
    convert_mp3_to_wav(input_folder, output_folder)