import os
import pandas as pd
import librosa

# Function to extract features from audio files
def extract_features(audio_path):
    # Load audio file
    audio, _ = librosa.load(audio_path, sr=None)
    # Return audio features
    return audio

# Function to process directory recursively
def process_directory1(directory):
    data = []
    # Iterate through files and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file is an audio file
            if file.endswith('.wav'):
                # Extract features from the audio file
                audio_path = os.path.join(root, file)
                audio_features = extract_features(audio_path)
                # Append to the data list
                data.append({'eng_audio': audio_features, 'hindi_audio': None})
    return data

def process_directory2(directory):
    data = []
    # Iterate through files and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file is an audio file
            if file.endswith('.wav'):
                # Extract features from the audio file
                audio_path = os.path.join(root, file)
                audio_features = extract_features(audio_path)
                # Append to the data list
                data.append({'eng_audio': audio_features, 'hindi_audio': None})
    return data

# Path to the directories containing the audio files
eng_directory = 'labels_audio'
hindi_directory = 'hindi_audio'

# Process English audio files
eng_data = process_directory1(eng_directory)
# Process Hindi audio files
hindi_data = process_directory2(hindi_directory)

# Combine English and Hindi data
data = [{'eng_audio': eng['eng_audio'], 'hindi_audio': hindi['hindi_audio']} for eng, hindi in zip(eng_data, hindi_data)]

# Create a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('audio_data.csv', index=False)

print("DataFrame saved to audio_data.csv")
