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


# define function to sample fourier transforms of consecutive time slices
def sample_df(
    df,  # df containing data
    samples=50,  # how many samples should be taken
    # if a cutoff index is taken, the fourier transform will be cut at that index,
    # needed to avoid off-by-one problems due to sampling
    cutoff_index=None,
    # how many timesteps should be taken per slice for the fourier transform
    num_sampling=50000,
    cutoff_frequency=4.5,  # frequency after which fourier transform should be cut off
):
    # randomly generate a starting position in the dataset so that the full
    # slice will be within the range
    start = np.random.randint(0, len(df) - num_sampling - 1, samples)
    # generate fourier samples
    fourier_list = []
    for s in start:
        # take slices
        time = df.iloc[:, 0].to_numpy()[s : s + num_sampling]
        data = df.iloc[:, 1].to_numpy()[s : s + num_sampling]
        # calculate sampling dt
        dt = time[1] - time[0]

        # calculate real fourier transform
        fourier = np.absolute(np.fft.rfft(data))
        fourier /= np.linalg.norm(fourier)  # normalize
        fourier_x = np.fft.rfftfreq(len(data), dt)  # get frequencies
        # find index of cutoff point of cutoff_frequency
        if cutoff_index is None:
            cutoff_index = np.where(fourier_x > cutoff_frequency)[0][0]

        # normalize cut frequency domain
        fourier_list.append(
            fourier[1:cutoff_index] / np.linalg.norm(fourier[1:cutoff_index])
        )
    return fourier_list, cutoff_index


# get ok data
X_train = []
y_train = []
ok_fourier = []
ok_files = []
for file in glob.glob(data_folder_ok + f"/*/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    X_train.extend(sample_df(f))
    # start = np.random.randint(0, len(f) - 200000, 100)
    time = f.iloc[:, 0].to_numpy()[0:400000]
    data = f.iloc[:, 1].to_numpy()[0:400000]
    dt = time[1] - time[0]

    fourier = np.absolute(np.fft.rfft(data))
    fourier /= np.linalg.norm(fourier)
    fourier_x = np.fft.rfftfreq(len(data), dt)
    cutoff_index = np.where(fourier_x > 4.5)[0][0]

    ok_files.append(file)
    ok_fourier.append(fourier[1:cutoff_index])


# get nok data
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

ok_fourier = np.vstack(ok_fourier)
nok_fourier = np.vstack(nok_fourier)

plt.figure()
plt.plot(fourier_x[1 : ok_fourier.shape[1] + 1], np.mean(ok_fourier, axis=0))
plt.plot(fourier_x[1 : nok_fourier.shape[1] + 1], np.mean(nok_fourier, axis=0))
plt.xlim(0, 5)

ok_fourier = (ok_fourier.T / np.linalg.norm(ok_fourier, axis=1)).T
nok_fourier = (nok_fourier.T / np.linalg.norm(nok_fourier, axis=1)).T

plt.figure()
plt.plot(
    [fourier_x[1 : ok_fourier.shape[1] + 1]] * ok_fourier.shape[0], ok_fourier, axis=0
)
plt.figure()
plt.plot(
    [fourier_x[1 : nok_fourier.shape[1] + 1]] * nok_fourier.shape[0], ok_fourier, axis=0
)


##########

# get ok data
X_train = []
y_train = []
num_samples = 200
index = None
for file in glob.glob(data_folder_ok + f"/MeisselsatzNeu/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    samples, index = sample_df(f, num_samples, index, num_sampling=400000)
    print(index)
    X_train.extend(samples)
    y_train.extend([1] * num_samples)
for file in glob.glob(data_folder_ok + f"/MeisselsatzGebraucht/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    samples, index = sample_df(f, num_samples // 4, index, num_sampling=400000)
    print(index)
    X_train.extend(samples)
    y_train.extend([1] * (num_samples // 4))
for file in glob.glob(data_folder_nok1 + f"/*/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    samples, index = sample_df(f, (num_samples // 4), index, num_sampling=400000)
    print(index)
    X_train.extend(samples)
    y_train.extend([0] * (num_samples // 4))
for file in glob.glob(data_folder_nok4 + f"/*/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    samples, index = sample_df(f, (num_samples // 4), index, num_sampling=400000)
    print(index)
    X_train.extend(samples)
    y_train.extend([0] * (num_samples // 4))
for file in glob.glob(data_folder_nok4 + f"/*/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    samples, index = sample_df(f, (num_samples // 4), index, num_sampling=400000)
    print(index)
    X_train.extend(samples)
    y_train.extend([0] * (num_samples // 4))
for file in glob.glob(data_folder_nok4 + f"/*/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    samples, index = sample_df(f, (num_samples // 4), index, num_sampling=400000)
    print(index)
    X_train.extend(samples)
    y_train.extend([0] * (num_samples // 4))
for file in glob.glob(data_folder_nok2 + f"/*/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
    samples, index = sample_df(f, (num_samples // 4), index, num_sampling=400000)
    print(index)
    X_train.extend(samples)
    y_train.extend([0] * (num_samples // 4))

import torch.nn as nn
import torch.nn.functional as F

X_train = np.vstack(X_train)
y_train = np.vstack(y_train)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, train_size=0.8, shuffle=True
)

X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc1 = nn.Linear(n, 1)
        # self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        x = F.sigmoid(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(index - 1)
model = model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
import torch.optim as optim


# loss function and optimizer
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 256  # number of epochs to run
batch_size = 12  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_acc = -np.inf  # init to negative infinity
best_weights = None
model.train()
for epoch in range(n_epochs):
    print(epoch)
    # X is a torch Variable
    permutation = torch.randperm(X_train.size()[0])

    for i in range(0, X_train.size()[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[i : i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        # in case you wanted a semi-full example
        outputs = model(batch_x)
        loss = loss_fn(outputs, batch_y)

        loss.backward()
        optimizer.step()
# evaluate accuracy at end of each epoch
model.eval()
(y_train == model(X_train).round()).sum()
from sklearn.metrics import (
    PrecisionRecallDisplay,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

prec, recall, _ = precision_recall_curve(
    y_test,
    model(torch.tensor(X_test, dtype=torch.float).to(device)).detach().cpu().numpy(),
)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

cm = confusion_matrix(
    y_test,
    model(torch.tensor(X_test, dtype=torch.float).to(device))
    .round()
    .detach()
    .cpu()
    .numpy()
    .astype(int),
)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.figure()
plt.plot(
    fourier_x[1 : ok_fourier.shape[1] + 1], model.fc1.weight.detach().cpu().numpy()[0]
)
