import numpy as np
import librosa
from scipy.io.wavfile import write


import numpy as np
import librosa

import numpy as np
from scipy.io.wavfile import write

def generateOutputAudioFile(outputDestination, inputArray):
    # Scale the values in the array to the range [-32768, 32767] (for 16-bit PCM audio)
    scaled_array = np.int16(inputArray * 32767)

    # Write the array to a WAV file
    write(outputDestination, 16000, scaled_array)

inputAudioFile = "C:\\Users\\mehul\\Downloads\\NumpyDataset\\eng_audios\\audio_0.npy"
inputArray = np.load(inputAudioFile)
generateOutputAudioFile('output.wav', inputArray)

