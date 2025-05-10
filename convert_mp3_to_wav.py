import os
import subprocess

def convert_mp3_to_wav(input_folder, output_folder):
    """
    Recursively convert all .mp3 files in input_folder (and subfolders) to .wav files,
    saving them in output_folder while replicating the directory structure.
    Uses FFmpeg to standardize to 44.1kHz sample rate and mono channel.
    """
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(".mp3"):
                input_path = os.path.join(root, filename)
                rel_path = os.path.relpath(root, input_folder)
                target_dir = os.path.join(output_folder, rel_path)
                os.makedirs(target_dir, exist_ok=True)
                output_filename = filename.replace(".mp3", ".wav")
                output_path = os.path.join(target_dir, output_filename)

                # FFmpeg command to convert to .wav (44.1kHz, mono)
                cmd = [
                    "ffmpeg",
                    "-i", input_path,
                    "-ar", "44100",
                    "-ac", "1",
                    "-y",
                    output_path
                ]

                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    print(f"Converted {input_path} to {output_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error converting {input_path}: {e.stderr}")

if __name__ == "__main__":
    input_folder = "audio_data" 
    output_folder = "wav_input" 
    convert_mp3_to_wav(input_folder, output_folder)