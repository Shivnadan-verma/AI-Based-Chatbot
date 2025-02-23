import speech_recognition as sr

# Speech to Text Conversion
def speech_to_text():
    # Recognizer instance
    recognizer = sr.Recognizer()

    # Use microphone for audio input
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... Speak now!")

        try:
            # Capture audio input
            audio = recognizer.listen(source)
            print("Recognizing speech...")

            # Convert speech to text using Google's API
            text = recognizer.recognize_google(audio, language="en-IN")  # For Hindi use "hi-IN"
            print("You said:", text)
            return text

        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
        except sr.RequestError as e:
            print(f"Request error from Google Speech Recognition service: {e}")

# Run the function
if __name__ == "__main__":
    result = speech_to_text()
