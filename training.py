import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import soundfile as sf


def extract_features(file_path):
    try:
        # Try to load using soundfile first
        audio_data, sample_rate = sf.read(file_path)
        if len(audio_data.shape) > 1:  # Stereo to mono
            audio_data = np.mean(audio_data, axis=1)
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    except Exception as e:
        print(f"soundfile failed: {e}")
        try:
            # Fallback to librosa's default loader
            audio_data, sample_rate = librosa.load(file_path, sr=16000)
        except Exception as e:
            print(f"librosa failed: {e}")
            return None

    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error extracting MFCCs from {file_path}: {e}")
        return None


def load_data(data_dir):
    labels = []
    features = []

    for user_dir in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user_dir)
        if os.path.isdir(user_path):
            for audio_file in os.listdir(user_path):
                if audio_file.endswith(".wav"):
                    file_path = os.path.join(user_path, audio_file)
                    print(f"Processing file: {file_path}")
                    mfccs = extract_features(file_path)
                    if mfccs is not None:
                        features.append(mfccs)
                        labels.append(user_dir)
                    else:
                        print(f"Failed to extract features from {file_path}")

    if len(features) == 0:
        raise ValueError(
            "No features were extracted. Please check your data directory and audio files."
        )

    return np.array(features), np.array(labels)


def train_model(
    data_dir="data", model_path="voice_auth_model.h5", classes_path="classes.npy"
):
    features, labels = load_data(data_dir)

    if len(labels) == 0:
        raise ValueError(
            "No labels found. Please ensure that the data directory contains subdirectories with audio files."
        )

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_categorical, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(Dense(256, input_shape=(40,), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(len(label_encoder.classes_), activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test)
    )

    model.save(model_path)
    np.save(classes_path, label_encoder.classes_)


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred: {e}")
