import speech_recognition as sr
import numpy as np
import librosa
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import os
import pyaudio
import pyttsx3


def list_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get("deviceCount")
    for i in range(0, numdevices):
        if p.get_device_info_by_host_api_device_index(0, i).get("maxInputChannels") > 0:
            print(
                f"Input Device id {i} - {p.get_device_info_by_host_api_device_index(0, i).get('name')}"
            )


def record_voice(sample_duration=5, device_index=None):
    recognizer = sr.Recognizer()
    print("Available audio devices:")
    list_audio_devices()
    try:
        with sr.Microphone(device_index=device_index) as source:
            print("Recording...")
            audio = recognizer.listen(source, phrase_time_limit=sample_duration)
            print("Recording finished.")
        return audio
    except AssertionError as e:
        print(f"Assertion Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


def record_continuous_audio(device_index=None):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone(device_index=device_index) as source:
            print("Start speaking and press Enter to stop recording...")
            audio_data = []
            stop_recording = False
            while not stop_recording:
                try:
                    audio = recognizer.listen(source, timeout=5)
                    audio_data.append(audio)
                except sr.WaitTimeoutError:
                    pass
                if input("Press Enter to stop recording...") == "":
                    stop_recording = True
            return audio_data
    except AssertionError as e:
        print(f"Assertion Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


def extract_features(audio):
    audio_data = np.frombuffer(audio.get_wav_data(), np.int16)
    sample_rate = 16000  # Assuming 16kHz sample rate
    mfccs = librosa.feature.mfcc(y=audio_data.astype(float), sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs


def load_model(model_path="voice_auth_model.h5"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"The model file {model_path} does not exist. Please train a model first."
        )
    model = keras.models.load_model(model_path)
    return model


def predict_user(model, features, label_encoder):
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction, axis=1)
    user = label_encoder.inverse_transform(predicted_label)
    return user[0]


def speech_to_text(audio):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"


def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def main():
    print(f"Current working directory: {os.getcwd()}")

    model_path = "voice_auth_model.h5"
    classes_path = "classes.npy"

    # Check and print existence of model and classes files
    if not os.path.exists(model_path):
        print(
            f"Error: The model file {model_path} does not exist. Please train a model first."
        )
        return

    if not os.path.exists(classes_path):
        print(
            f"Error: The classes file {classes_path} does not exist. Please ensure it is available."
        )
        return

    # Record voice for authentication
    audio = record_voice(device_index=None)

    # Continue only if recording was successful
    if audio is not None:
        # Extract features and predict user
        features = extract_features(audio)
        model = load_model(model_path)

        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load(
            classes_path
        )  # Assuming classes.npy contains the saved classes
        user = predict_user(model, features, label_encoder)
        text_to_speech(f"Authenticated user: {user}")

        # Perform speech-to-text operation
        command_text = speech_to_text(audio)
        print(f"Recognized command: {command_text}")

        # Write the recognized text to a file
        with open("output.txt", "w") as file:
            file.write(command_text)
        print("Text written to output.txt")

        # Ask if user wants to record more audio
        text_to_speech("Do you want to record more audio? Say yes or no.")
        response_audio = record_voice(sample_duration=5)
        response_text = speech_to_text(response_audio).lower()

        if "yes" in response_text:
            continuous_audio_data = record_continuous_audio()
            if continuous_audio_data:
                combined_audio = sr.AudioData(
                    b"".join([chunk.get_raw_data() for chunk in continuous_audio_data]),
                    continuous_audio_data[0].sample_rate,
                    continuous_audio_data[0].sample_width,
                )
                continuous_text = speech_to_text(combined_audio)
                print(f"Recognized continuous text: {continuous_text}")

                # Write the recognized continuous text to a file
                with open("output.txt", "w") as file:
                    file.write(continuous_text)
                print("Continuous text written to output.txt")
            else:
                print("Continuous recording failed. Please check the audio device.")
        else:
            text_to_speech("No additional recording will be performed.")
    else:
        print("Recording failed. Please check the audio device.")


if __name__ == "__main__":
    main()
