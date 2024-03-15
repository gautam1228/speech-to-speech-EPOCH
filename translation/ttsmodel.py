import librosa
import numpy as np
import noisereduce as nr
import webrtcvad
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import os

def reduce_noise(audio_data):
    stft = librosa.stft(audio_data)
    stft_db = librosa.amplitude_to_db(np.abs(stft))
    mean_noise = np.mean(stft_db[:, :200], axis=1)
    stft_db -= mean_noise[:, np.newaxis]
    noise_reduced_audio = librosa.griffinlim(librosa.db_to_amplitude(stft_db))
    return noise_reduced_audio


# Load the WAV audio file
audio_file = os.path.join(os.path.dirname(__file__), 'output_file.wav')
y, sr = librosa.load(audio_file, sr=None)

# Resample the audio to a common sampling rate (e.g., 16 kHz) - Preprocessing 1
target_sr = 16000
y_resampled = librosa.resample(y, orig_sr=44100, target_sr=16000)

# Normalize the audio to ensure consistent amplitude levels - Preprocessing 2
y_normalized = librosa.util.normalize(y_resampled)


# Remove silence using a threshold (e.g., -40 dB) - Preprocessing 3
y_trimmed, _ = librosa.effects.trim(y_normalized, top_db=40)

# print(y_trimmed.shape)

# Perform noise reduction using the NoiseReduce library - Preprocessing 4
denoised_audio = reduce_noise(y_trimmed) 


# Initialize VAD with aggressiveness level (0-3) - Preprocessing 5
vad = webrtcvad.Vad(2)

# Segment the audio into frames and perform VAD
frame_duration = 30  # Frame duration in milliseconds
samples_per_frame = int(sr * frame_duration / 1000)
segments = []
for i in range(0, len(y), samples_per_frame):
    segment = y_normalized[i:i+samples_per_frame]
    if vad.is_speech(segment.tobytes(), sr):
        segments.append(segment)

# Concatenate the speech segments into a single NumPy array
preprocessed_y5 = np.concatenate(segments)
# Now segments contain the speech segments detected by VAD

# Find peaks in the audio signal - Preprocessing 6
peaks, _ = find_peaks(np.abs(preprocessed_y5), height=0.5)

# Apply compression by reducing the amplitude of peaks
compression_factor = 0.5
y_compressed = np.copy(preprocessed_y5)
y_compressed[peaks] *= compression_factor
# Now y_compressed contains the audio with dynamic range compression applied


# Initialize StandardScaler for feature scaling - Preprocessing 7
scaler = StandardScaler()

# Reshape y_compressed to a 2D array (assuming it's a 1D array representing audio signal)
y_reshaped = y_compressed.reshape(-1, 1)

# Scale the features (audio samples) using the scaler
scaled_audio = scaler.fit_transform(y_reshaped)

# Reshape scaled_audio back to 1D array (if needed)
scaled_audio = scaled_audio.ravel()

# Now scaled_audio contains the scaled audio signal with zero mean and unit variance
