audio_file = 'output_file.wav'
y, sr = librosa.load(audio_file, sr=None)

def generateOutputAudioFile(outputDestination, inputArray):

    # Scale the values in the array to the range [-32768, 32767] (for 16-bit PCM audio)
    scaled_array = np.int16(inputArray*32767)

    # Write the array to a WAV file
    write(outputDestination, sr, scaled_array)