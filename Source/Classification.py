import joblib
import sklearn

from Source.Utilities import *
from config import Test, Model, CreateDataset

PATH = librosa.util.find_files(Test.DATA_PATH)


def main():
    songs = []

    # Load Test Files
    for p in PATH:
        song, sr = librosa.load(p, sr=SAMPLING_RATE, duration=5.0)
        songs.append(song)

    # Extract Features of Test Files and Save Them in Array
    data = numpy.array([extract_features(song) for song in songs])

    # Scale ALL Variables Between -1 to 1
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)

    # Predict Genres
    svm = joblib.load(Model.NAME)
    predicts = svm.predict(data)

    print(predicts)
