import os

import pandas
import sklearn

from Source.Utilities import *
from config import CreateDataset

DATASET_DIR = CreateDataset.DIRECTORY
DATASET_NAME = CreateDataset.NAME


# Read Subdirectory of Files
def get_subdirectories(directory):
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]


# Get Array of Songs from Specific Folder (Genre)
def get_sample_arrays(folder_name):
    audios = []
    path_of_audios = librosa.util.find_files(DATASET_DIR + "/" + folder_name)
    for audio in path_of_audios:
        x, sr = librosa.load(audio, sr=SAMPLING_RATE, duration=5.0)
        audios.append(x)
    audios_numpy = numpy.array(audios)
    return audios_numpy


def main():
    labels = []
    dataset_numpy = None
    genres = get_subdirectories(DATASET_DIR)
    for genre in genres:
        signals = get_sample_arrays(genre)
        for signal in signals:
            row = extract_features(signal)
            if dataset_numpy is None:
                dataset_numpy = numpy.array(row)
            else:
                dataset_numpy = numpy.vstack((dataset_numpy, row))

            labels.append(genre)

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    dataset_numpy = scaler.fit_transform(dataset_numpy)

    dataset_pandas = pandas.DataFrame(dataset_numpy, columns=CreateDataset.Feature_Names)

    dataset_pandas["genre"] = labels
    dataset_pandas.to_csv(DATASET_NAME, index=False)


if __name__ == '__main__':
    main()
