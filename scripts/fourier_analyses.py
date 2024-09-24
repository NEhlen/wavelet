import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import torch

from sklearn.model_selection import train_test_split

# data folders
data_folder_ok = "/home/niels/dev/wavelet/data/GutMeissel"
data_folder_nok1 = "/home/niels/dev/wavelet/data/DreiMeisselDiagonal"
data_folder_nok2 = "/home/niels/dev/wavelet/data/DreiMeisselHorizontal"
data_folder_nok3 = "/home/niels/dev/wavelet/data/DreiMeisselVertikal"
data_folder_nok4 = "/home/niels/dev/wavelet/data/EinEckmeissel"
data_folder_nok5 = "/home/niels/dev/wavelet/data/EinMeissel"


# define sensor
sensor = "a_x_23"
# sensor = "Sens1_Acc_X"


# get ok data
ok_fourier = []
ok_files = []
for file in glob.glob(data_folder_ok + f"/MeisselsatzNeu/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    # take slice of data
    time = f.iloc[:, 0].to_numpy()[0:400000]
    data = f.iloc[:, 1].to_numpy()[0:400000]
    dt = time[1] - time[0]
    # fourier transform
    fourier = np.absolute(np.fft.rfft(data))
    fourier /= np.linalg.norm(fourier)  #  normalize
    # get frequencies of fourier components
    fourier_x = np.fft.rfftfreq(len(data), dt)
    # find the index where frequencies > 4.5 Hz
    cutoff_index = np.where(fourier_x > 4.5)[0][0]

    # append fourier
    ok_files.append(file)
    ok_fourier.append(fourier[1:cutoff_index])


# get nok data, see above
nok_fourier = []
nok_files = []
for file in glob.glob(data_folder_nok1 + f"/*/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    time = f.iloc[:, 0].to_numpy()[:400000]
    data = f.iloc[:, 1].to_numpy()[:400000]

    fourier = np.absolute(np.fft.rfft(data))
    fourier /= np.linalg.norm(fourier)
    fourier_x = np.fft.rfftfreq(len(data), time[1] - time[0])
    cutoff_index = np.where(fourier_x > 4.5)[0][0]
    nok_files.append(file)
    nok_fourier.append(fourier[1:cutoff_index])

for file in glob.glob(data_folder_nok2 + f"/*/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    time = f.iloc[:, 0].to_numpy()[:400000]
    data = f.iloc[:, 1].to_numpy()[:400000]

    fourier = np.absolute(np.fft.rfft(data))
    fourier /= np.linalg.norm(fourier)
    fourier_x = np.fft.rfftfreq(len(data), time[1] - time[0])
    cutoff_index = np.where(fourier_x > 4.5)[0][0]
    nok_files.append(file)
    nok_fourier.append(fourier[1:cutoff_index])

for file in glob.glob(data_folder_nok3 + f"/*/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    time = f.iloc[:, 0].to_numpy()[:400000]
    data = f.iloc[:, 1].to_numpy()[:400000]

    fourier = np.absolute(np.fft.rfft(data))
    fourier /= np.linalg.norm(fourier)
    fourier_x = np.fft.rfftfreq(len(data), time[1] - time[0])
    cutoff_index = np.where(fourier_x > 4.5)[0][0]
    nok_files.append(file)
    nok_fourier.append(fourier[1:cutoff_index])

for file in glob.glob(data_folder_nok4 + f"/*/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    time = f.iloc[:, 0].to_numpy()[:400000]
    data = f.iloc[:, 1].to_numpy()[:400000]

    fourier = np.absolute(np.fft.rfft(data))
    fourier /= np.linalg.norm(fourier)
    fourier_x = np.fft.rfftfreq(len(data), time[1] - time[0])
    cutoff_index = np.where(fourier_x > 4.5)[0][0]
    nok_files.append(file)
    nok_fourier.append(fourier[1:cutoff_index])

for file in glob.glob(data_folder_nok5 + f"/*/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    time = f.iloc[:, 0].to_numpy()[:400000]
    data = f.iloc[:, 1].to_numpy()[:400000]

    fourier = np.absolute(np.fft.rfft(data))
    fourier /= np.linalg.norm(fourier)
    fourier_x = np.fft.rfftfreq(len(data), time[1] - time[0])
    cutoff_index = np.where(fourier_x > 4.5)[0][0]
    nok_files.append(file)
    nok_fourier.append(fourier[1:cutoff_index])

# turn into numpy arrays
ok_fourier = np.vstack(ok_fourier)
nok_fourier = np.vstack(nok_fourier)

# plot mean fourier
plt.figure()
plt.plot(fourier_x[1 : ok_fourier.shape[1] + 1], np.mean(ok_fourier, axis=0))
plt.plot(fourier_x[1 : nok_fourier.shape[1] + 1], np.mean(nok_fourier, axis=0))
plt.xlim(0, 5)

# normalize
ok_fourier = (ok_fourier.T / np.linalg.norm(ok_fourier, axis=1)).T
nok_fourier = (nok_fourier.T / np.linalg.norm(nok_fourier, axis=1)).T

plt.figure()
for i in range(len(ok_fourier)):
    plt.plot(fourier_x[1 : ok_fourier.shape[1] + 1], ok_fourier[i] + 0.1 * i)

plt.figure()
for i in [-1, -2, -3]:
    plt.plot(
        fourier_x[1 : nok_fourier.shape[1] + 1],
        nok_fourier[i] + 0.1 * np.abs(i),
    )

# Notes:
# The GutMeissel with MeisselsatzGebraucht seem to have large fourier components at very low wavelengths
# The two main components are the vibration at ~1.8 Hz and the overtone at ~3.6 Hz
# the difference in intensity between these two could be influenced by a broken Meißel
