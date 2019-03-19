import librosa
import numpy
from config import CreateDataset

SAMPLING_RATE = CreateDataset.SAMPLING_RATE
FRAME_SIZE = CreateDataset.FRAME_SIZE
HOP_SIZE = CreateDataset.HOP_SIZE
MFCC_SIZE = CreateDataset.MFCC_SIZE
CHROMA_SIZE = CreateDataset.CHROMA_SIZE
MEL_SCALE_SIZE = CreateDataset.MEL_SCALE_SIZE
TONNETZ_SIZE = CreateDataset.TONNETZ_SIZE


def extract_features(signal):

    result = []
    features = []

    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    spectral_rollof = librosa.feature.spectral_rolloff(y=signal, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    spectral_flux = librosa.onset.onset_strength(y=signal, sr=SAMPLING_RATE, center=True)
    zero_crossing = librosa.feature.zero_crossing_rate(y=signal, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)
    tempo = librosa.beat.tempo(y=signal, sr=SAMPLING_RATE, hop_length=HOP_SIZE)
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLING_RATE, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    chroma = librosa.feature.chroma_stft(y=signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, sr=SAMPLING_RATE, n_chroma=CHROMA_SIZE)
    mel_scale = librosa.feature.melspectrogram(y=signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, sr=SAMPLING_RATE)
    mel_scale = librosa.power_to_db(mel_scale)
    tonal_centroid = librosa.feature.tonnetz(y=signal, sr=SAMPLING_RATE)

    features.append(spectral_centroid)
    features.append(spectral_bandwidth)
    features.append(spectral_contrast)
    features.append(spectral_rollof)
    features.append(spectral_flux)
    features.append(zero_crossing)
    features.append(tempo)

    for feature in features:
        result.append(numpy.mean(feature))
        result.append(numpy.std(feature))

    for i in range(0, MFCC_SIZE):
        result.append(numpy.mean(mfcc[i, :]))
        result.append(numpy.std(mfcc[i, :]))

    for i in range(0, CHROMA_SIZE):
        result.append(numpy.mean(chroma[i, :]))
        result.append(numpy.std(chroma[i, :]))

    for i in range(0, MEL_SCALE_SIZE):
        result.append(numpy.mean(mel_scale[i, :]))
        result.append(numpy.std(mel_scale[i, :]))

    for i in range(0, TONNETZ_SIZE):
        result.append(numpy.mean(tonal_centroid[i, :]))
        result.append(numpy.std(tonal_centroid[i, :]))

    return result

