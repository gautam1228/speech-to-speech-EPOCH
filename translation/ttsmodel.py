import librosa
import numpy as np
import noisereduce as nr
import webrtcvad
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

# Load the WAV audio file
audio_file = 'audio.wav'
y, sr = librosa.load(audio_file, sr=None)

# Resample the audio to a common sampling rate (e.g., 16 kHz)
target_sr = 16000
y_resampled = librosa.resample(y, sr, target_sr)

# Normalize the audio to ensure consistent amplitude levels
y_normalized = librosa.util.normalize(y_resampled)


# Remove silence using a threshold (e.g., -40 dB)
y_trimmed, _ = librosa.effects.trim(y_normalized, top_db=40)


# Perform noise reduction using the NoiseReduce library
noisy_part = y_normalized[:10000]  # Example: Consider only the first 10 seconds for noise reduction
reduced_noise = nr.reduce_noise(y_normalized, noisy_part)


# Initialize VAD with aggressiveness level (0-3)
vad = webrtcvad.Vad(2)

# Segment the audio into frames and perform VAD
frame_duration = 30  # Frame duration in milliseconds
samples_per_frame = int(sr * frame_duration / 1000)
segments = []
for i in range(0, len(y), samples_per_frame):
    segment = y_normalized[i:i+samples_per_frame]
    if vad.is_speech(segment.tobytes(), sr):
        segments.append(segment)

# Now segments contain the speech segments detected by VAD



# Find peaks in the audio signal
peaks, _ = find_peaks(np.abs(y), height=0.5)

# Apply compression by reducing the amplitude of peaks
compression_factor = 0.5
y_compressed = np.copy(y)
y_compressed[peaks] *= compression_factor

# Now y_compressed contains the audio with dynamic range compression applied


# Initialize StandardScaler for feature scaling
scaler = StandardScaler()

# Scale the features (e.g., MFCCs) using the scaler
scaled_features = scaler.fit_transform(features)

# Now scaled_features contain the scaled features with zero mean and unit variance
