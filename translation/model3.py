import os
import numpy as np
import tensorflow as tf

eng_audio_directory = "eng_audios"
hindi_audio_directory = "hin_audios"

# Function to load audio samples from a directory
def load_audio_samples(directory):
    audio_data = []
    for file in os.listdir(directory):
        if file.endswith(".npy"):
            audio = np.load(os.path.join(directory, file))
            audio_data.append(audio)
    return audio_data

# Load audio samples from directories
eng_audio_data = load_audio_samples(eng_audio_directory)
hindi_audio_data = load_audio_samples(hindi_audio_directory)



# Padding sequences to make them of equal length
max_input_length = max(len(seq) for seq in eng_audio_data)
max_output_length = max(len(seq) for seq in hindi_audio_data)

# Pad sequences using numpy
padded_input_data = np.zeros((len(eng_audio_data), max_input_length))
for i, seq in enumerate(eng_audio_data):
    padded_input_data[i, :len(seq)] = seq

padded_output_data = np.zeros((len(hindi_audio_data), max_output_length))
for i, seq in enumerate(hindi_audio_data):
    padded_output_data[i, :len(seq)] = seq


# Convert to NumPy arrays
padded_input_data = np.array(padded_input_data)
padded_output_data = np.array(padded_output_data)
print(padded_output_data)
print(padded_input_data)

# Define the Seq2Seq model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=100, output_dim=256, input_length=max_input_length),
    tf.keras.layers.LSTM(256),
    tf.keras.layers.RepeatVector(max_output_length),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation='softmax'))
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(padded_input_data, padded_output_data, epochs=10)

# Make predictions
predictions = model.predict(padded_input_data)

# Save the model
model.save('speech_translation_model2.h5')