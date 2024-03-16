import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to load audio samples from directory
def load_audio_samples(directory):
    audio_samples = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            audio, _ = librosa.load(file_path, sr=None)  # Load audio file
            audio_samples.append(audio)
    return np.array(audio_samples)

# Load audio samples from directories
eng_audio_directory = "eng_audio"
hindi_audio_directory = "hindi_audio"

X_audio = load_audio_samples(eng_audio_directory)
y_audio = load_audio_samples(hindi_audio_directory)

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
