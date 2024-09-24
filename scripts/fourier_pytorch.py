import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    PrecisionRecallDisplay,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
num_samples = 200
index = None
for file in glob.glob(data_folder_ok + f"/MeisselsatzNeu/*/{sensor}.CSV"):
    f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)  # load data
    samples, index = sample_df(
        f, num_samples, index, num_sampling=400000
    )  # sample fourier
    print(index)
    # generate training data
    X_train.extend(samples)
    y_train.extend([1] * num_samples)
# # see above, the num_samples are cut in 4 to avoid imbalanced ok dataset
# # use index from above as cutoff_index to avoid off-by-one problems
# for file in glob.glob(data_folder_ok + f"/MeisselsatzGebraucht/*/{sensor}.CSV"):
#     f = pd.read_csv(file, sep=";", decimal=",", skiprows=1)
#     samples, index = sample_df(f, num_samples // 4, index, num_sampling=400000)
#     print(index)
#     X_train.extend(samples)
#     y_train.extend([1] * (num_samples // 4))
# get nok data, see above, the num_samples are cut by 4 to avoid imbalanced nok dataset
# vs ok dataset
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


# dataset to numpy
X_train = np.vstack(X_train)
y_train = np.vstack(y_train)

# train test splitting
# note: this does not ensure there is no overlap between train
# and test data. We're taking random cuts of the same time series. Especially with
# large num_sampling there is probably a huge overlap between train and test data
# even without overlap in time between both, they are all the same machines in runs
# around the same time, this needs to be checked in more detail
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, train_size=0.8, shuffle=True
)

X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)


# define super simple neural net
# just linear layer with sigmoid for BCELoss
# this allows for interpretability of the fc1 kernel
# a high value means that part of the fourier spectrum
# is a good predictor for a working meißel
# a value around zero means that part does not have
# any predictive power
# a very low value below zero means that part is a good
# predictor for a broken meißel
class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc1 = nn.Linear(n, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x


# move to cuda if cuda available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(index - 1)
model = model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)


# loss function and optimizer
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # use adam optimizer

n_epochs = 256  # number of epochs to run
batch_size = 12  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# very simple training code without scheduler
model.train()
for epoch in range(n_epochs):
    print(epoch)
    # get a random permutation to shuffle train data batches
    permutation = torch.randperm(X_train.size()[0])

    for i in range(0, X_train.size()[0], batch_size):
        optimizer.zero_grad()
        # get batch
        indices = permutation[i : i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        # predict
        outputs = model(batch_x)
        # get loss
        loss = loss_fn(outputs, batch_y)

        # autodiff and backpropagation
        loss.backward()
        optimizer.step()

# evaluate accuracy at end of each epoch
model.eval()

# get the precision recall curve
prec, recall, _ = precision_recall_curve(
    y_test,
    model(torch.tensor(X_test, dtype=torch.float).to(device)).detach().cpu().numpy(),
)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()


# get the confusion matrix. Here every value >0.5 in predicted class is assumed to be a 1.
# If you want to get the confusion matrix for a different cutoff determined via the precision-
# recall-curve, this needs to be changed
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

# plot kernel of model
plt.figure()
plt.plot(
    np.linspace(0, 4.5, len(model.fc1.weight.detach().cpu().numpy()[0])),
    model.fc1.weight.detach().cpu().numpy()[0],
)
