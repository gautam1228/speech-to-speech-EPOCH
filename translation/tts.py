from gtts import gTTS

def text_to_speech(text, lang='hi', speed=1.0, output_file="translation/hindi_audio/tts_output.mp3"):
    """
    Convert text to speech using gTTS and save the audio to a file.

    Args:
        text (str): The text to be converted to speech.
        lang (str, optional): The language of the text (default is 'en' for English).
        slow (bool, optional): Whether to generate speech at a slower speed (default is False).
        output_file (str, optional): The filename to save the audio to (default is "tts_output.mp3").

    Returns:
        None
    """
    # Create a gTTS object with the desired text, language, and speed
    speech = gTTS(text=text, lang=lang, slow=(speed < 1.0))

    # Save the audio to the specified file
    speech.save(output_file)

# # Example usage
# text = "शैक्षणिक संस्थान सुरक्षा पर पहले से कहीं अधिक ध्यान दे रहे हैं।"
# text_to_speech(text)
