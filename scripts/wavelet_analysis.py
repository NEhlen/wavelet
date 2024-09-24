# %%
import pywt
import numpy as np
import matplotlib.pyplot as plt
import resampy

file = "/home/niels/dev/wavelet/data/GutMeissel/MeisselsatzNeu/2023-10-06 15-12-58/a_x_23.CSV"
# file = "/home/niels/dev/wavelet/data/DreiMeisselHorizontal/Fahrtrichtung/2023-10-05 15-54-56/a_x_23.CSV"

# load file
with open(file, "r") as f:
    data = []
    time = []
    for line in f.readlines():
        t_, entry = line.split(";")
        entry = entry.replace(",", ".")
        t_ = t_.replace(",", ".")
        try:
            data.append(float(entry))
            time.append(float(t_))
        except Exception:
            pass

time, data = np.array(time), np.array(data)

# calculate fourier transform
fourier_components = np.absolute(np.fft.rfft(data))  # calculate fourier components
fourier_x = np.fft.rfftfreq(
    len(time), time[1] - time[0]
)  # get corresponding frequencies
# %%
# resample data, we don't lose information because the frequencies we're interested in
# are much larger than our resampling
new_sampling_rate = 100  # in Hz
data = resampy.resample(
    data,  # data
    1 / (time[1] - time[0]),  # sampling rate of data
    new_sampling_rate,  # new sampling rate in Hz
)
# new sampling rate
time = np.array([n * (1 / new_sampling_rate) for n in range(len(data))])


dt = time[1] - time[0]  # sampling timesteps
fs = 1.0 / dt  # sampling frequency
# %%
wavelet = "cmor20-1.8"
frequencies = np.linspace(
    10, 0.6, 400
)  # we're interested in low frequencies between 5 and 1 Hz
# normalize frequencies relative to sampling frequency, needed because pywt frequency2scale expects it
frequencies /= fs
# get corresponding scales to frequencies of interest
scale = pywt.frequency2scale(wavelet, frequencies)

# rename, if you want to only use part of the data, you can cut off here
x = time
y = data

# do cwt transform
coef, freqs = pywt.cwt(y, scale, wavelet, sampling_period=dt)

# %%
# plot result
# to plot the "power"/"amplitude", we need to take the absolute value of the complex coefficients
# this is equivalent to sqrt(c.c*)
# use terrain as colormap because the values span several magnitudes
fig, axarr = plt.subplots(1, 2, sharey=True)
im = axarr[0].imshow(
    np.absolute(coef),
    extent=[x[0], x[-1], freqs[-1], freqs[0]],
    aspect="auto",
    cmap=plt.get_cmap("terrain"),
    vmin=0,
    # set the maximum in the colormap to the mean + 5 * standard deviation
    # !Attention! this makes the colormap data dependent, comparisons of absolute intensity
    # between datasets is thus not valid
    vmax=np.absolute(coef).mean() + np.absolute(coef).std() * 5,
)
plt.colorbar(im)
plt.ylim(np.amin(frequencies * fs), np.amax(frequencies * fs))
plt.xlabel("Zeit [s]")
plt.ylabel("Frequenz [Hz]")
# axarr[0].set_xlim(209, 220)

# plot normalized fourier transform
axarr[1].plot(
    fourier_components / np.linalg.norm(fourier_components),
    fourier_x,
)
plt.xlim(0, 0.02)
plt.xlabel("Amplitude [a.u.]")
plt.ylabel("Frequenz [Hz]")
plt.tight_layout()
# plt.yscale("log")
# plt.show()


# same as above but rather than fourier transform show the mean of the left
# wavelet transform instead
fig, axarr = plt.subplots(1, 2, sharey=True)
im = axarr[0].imshow(
    np.absolute(coef),
    extent=[x[0], x[-1], freqs[-1], freqs[0]],
    aspect="auto",
    cmap=plt.get_cmap("terrain"),
    vmin=0,
    # set the maximum in the colormap to the mean + 5 * standard deviation
    # !Attention! this makes the colormap data dependent, comparisons of absolute intensity
    # between datasets is thus not valid
    vmax=np.absolute(coef).mean() + np.absolute(coef).std() * 5,
)
plt.colorbar(im)
plt.ylim(np.amin(frequencies * fs), np.amax(frequencies * fs))
plt.xlabel("Zeit [s]")
plt.ylabel("Frequenz [Hz]")
# axarr[0].set_xlim(209, 220)

# plot mean of wavelet transform along time axis
axarr[1].plot(
    np.absolute(coef).mean(axis=1), np.linspace(freqs[0], freqs[-1], coef.shape[0])
)
plt.xlabel("Amplitude [a.u.]")
plt.ylabel("Frequenz [Hz]")
plt.tight_layout()
# plt.yscale("log")
# plt.show()

# plot phase of the wavelet coefficients. Only plot a small part of the x-axis
# because the details will otherwise be lost and look like noise
# phase information is probably not really useful
plt.figure()
plt.imshow(
    np.arctan2(coef.imag, coef.real),
    aspect="auto",
    extent=[x[0], x[-1], freqs[-1], freqs[0]],
    cmap=plt.get_cmap("hsv"),
    vmin=-np.pi,
    vmax=np.pi,
)
plt.xlabel("Zeit [s]")
plt.ylabel("Frequenz [Hz]")
plt.xlim(100, 110)

# plot real component of the wavelet coefficients. As example
plt.figure()
plt.imshow(
    coef.real,
    aspect="auto",
    vmin=0,
    vmax=0.01,
    extent=[x[0], x[-1], freqs[-1], freqs[0]],
)
plt.xlabel("Zeit [s]")
plt.ylabel("Frequenz [Hz]")

# %%


def cmor(b, c, t):
    cm = 1.0 / np.sqrt(np.pi * b)
    cm *= np.exp(-(t**2) / b)
    cm *= np.exp(1j * 2 * np.pi * c * t)
    return cm


xvals = np.linspace(-10, 10, 1000)
plt.plot(xvals, [(cmor(1, 1.8, t)).real for t in xvals])
# %%
