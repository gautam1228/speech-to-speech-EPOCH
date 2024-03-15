from moviepy.editor import VideoFileClip
import noisereduce as nr
import soundfile as sf

def extract_audio(video_path, audio_output_path):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(audio_output_path)

# Example usage
video_path = "inputVideo.mp4"
output_path = "inputAudio.mp3"
extract_audio(video_path, output_path)


from pydub import AudioSegment
from pydub.playback import play
import noisereduce as nr
import numpy as np
# Load the audio file
audio = AudioSegment.from_mp3("inputAudio.mp3")

# Convert to numpy array for processing
audio_data = np.array(audio.get_array_of_samples())

# Apply noise reduction
reduced_noise = nr.reduce_noise(y=audio_data, sr=audio.frame_rate)

# Convert numpy array back to AudioSegment
reduced_audio = AudioSegment(
    reduced_noise.tobytes(), 
    frame_rate=audio.frame_rate, 
    sample_width=reduced_noise.dtype.itemsize, 
    channels=1
)

# Play or save the reduced audio
reduced_audio.export("reduced_audio.mp3", format="mp3")




