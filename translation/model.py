import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to load audio samples from directory
def load_audio_samples(directory, max_length=None, num_samples=100):
    audio_samples = []
    filenames = os.listdir(directory)
    if num_samples is not None:
        filenames = np.random.choice(filenames, num_samples, replace=False)
    for filename in filenames:
        if filename.endswith(".npy"):
            file_path = os.path.join(directory, filename)
            audio = np.load(file_path)
            if max_length is not None:
                # Pad or truncate audio samples to a fixed length
                audio = pad_or_truncate(audio, max_length)
            audio_samples.append(audio)
    return np.array(audio_samples)

# Function to pad or truncate audio samples to a fixed length
def pad_or_truncate(audio, max_length):
    if len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
    elif len(audio) > max_length:
        audio = audio[:max_length]
    return audio

# Load audio samples from directories
eng_audio_directory = "eng_audios"
hindi_audio_directory = "hin_audios"

# Find the maximum length of audio samples
max_length_eng = max(len(np.load(os.path.join(eng_audio_directory, filename))) for filename in os.listdir(eng_audio_directory))
max_length_hindi = max(len(np.load(os.path.join(hindi_audio_directory, filename))) for filename in os.listdir(hindi_audio_directory))
max_length = max(max_length_eng, max_length_hindi)

X_audio = load_audio_samples(eng_audio_directory, max_length=max_length)
print(X_audio.shape)
y_audio = load_audio_samples(hindi_audio_directory, max_length=max_length)
print(y_audio.shape)

# Define the encoder
encoder_input = layers.Input(shape=(None, 1))  # Input shape is (timesteps, features)
encoder_lstm = layers.LSTM(256, return_state=True)
_, encoder_state_h, encoder_state_c = encoder_lstm(encoder_input)
encoder_states = [encoder_state_h, encoder_state_c]

# Define the decoder
decoder_input = layers.Input(shape=(None, 1))  # Input shape is (timesteps, features)
decoder_lstm = layers.LSTM(256, return_sequences=True, return_state=True)
decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
decoder_dense = layers.Dense(1, activation='linear')
decoder_output = decoder_dense(decoder_output)

# Define the seq2seq model
model = models.Model([encoder_input, decoder_input], decoder_output)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit([X_audio[:, :, np.newaxis], y_audio[:, :, np.newaxis]], y_audio[:, :, np.newaxis], batch_size=32, epochs=10, validation_split=0.2)

# Save the model
model.save('speech_translation_model.h5')
