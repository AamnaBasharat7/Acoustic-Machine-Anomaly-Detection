

import librosa
import numpy as np


def extract_features(audio, sr):

    # Extract 13 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # RMS features
    rms = librosa.feature.rms(y=audio)[0]
    rms_features = [np.mean(rms), np.std(rms)]

    # Zero Crossing Rate features
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    zcr_features = [np.mean(zcr), np.std(zcr)]

    # Spectral Centroid features
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )[0]
    centroid_features = [np.mean(centroid), np.std(centroid)]

    # Spectral Bandwidth features
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )[0]
    bandwidth_features = [np.mean(bandwidth), np.std(bandwidth)]

    # Combine all 34 features
    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        rms_features,
        zcr_features,
        centroid_features,
        bandwidth_features
    ])

    return features
