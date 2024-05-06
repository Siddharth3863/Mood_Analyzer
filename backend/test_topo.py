import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

# from utils import nextpow2
# import utils  # Our own utility functions
# import muselsl

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3
    Gamma = 4

# def nextpow2(i):
#     """
#     Find the next power of 2 for number i
#     """
#     n = 1
#     while n < i:
#         n *= 2
#     return n


# def compute_band_powers(eegdata, fs):
#     """Extract the features (band powers) from the EEG.

#     Args:
#         eegdata (numpy.ndarray): array of dimension [number of samples,
#                 number of channels]
#         fs (float): sampling frequency of eegdata

#     Returns:
#         (numpy.ndarray): feature matrix of shape [number of feature points,
#             number of different features]
#     """
#     eegdata = np.array([eegdata])
#     # 1. Compute the PSD
#     winSampleLength, nChd = eegdata.shape

#     # Apply Hamming window
#     print(eegdata)
#     print(winSampleLength)

#     w = np.hamming(winSampleLength)
#     dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
#     dataWinCenteredHam = (dataWinCentered.T * w).T

#     NFFT = nextpow2(winSampleLength)
#     Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
#     PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])
#     f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))

#     # SPECTRAL FEATURES
#     # Average of band powers
#     # Delta <4
#     ind_delta, = np.where(f < 4)
#     meanDelta = np.mean(PSD[ind_delta, :], axis=0)
#     # Theta 4-8
#     ind_theta, = np.where((f >= 4) & (f <= 8))
#     meanTheta = np.mean(PSD[ind_theta, :], axis=0)
#     # Alpha 8-12
#     ind_alpha, = np.where((f >= 8) & (f <= 12))
#     meanAlpha = np.mean(PSD[ind_alpha, :], axis=0)
#     # Beta 12-30
#     ind_beta, = np.where((f >= 12) & (f < 30))
#     meanBeta = np.mean(PSD[ind_beta, :], axis=0)
#     # Gamma 30-45
#     ind_gamma, = np.where((f >= 30) & (f < 45))
#     meanGamma = np.mean(PSD[ind_gamma, :], axis=0)

#     feature_vector = np.concatenate((meanDelta, meanTheta, meanAlpha,
#                                      meanBeta, meanGamma), axis=0)

#     feature_vector = np.log10(feature_vector)
#     return feature_vector

def plot_topo():
    # with open('./static/realtime_csv.csv') as f:
    #     data=f.readlines()
    #     data=[i.strip().split(',') for i in data]
    #     # print(data[0])
    data=pd.read_csv('./static/realtime_csv.csv')
    # print(data.columns)
    data = data[['TP9', 'AF7', 'AF8', 'TP10']]

    # data_epoch1 = data['TP9']
    # data_epoch2 = data['AF7']
    # data_epoch3 = data['AF8']
    # data_epoch4 = data['TP10']
    # fs = 256
    # data_epoch1 = pd.to_numeric(data_epoch1)
    # data_epoch2 = pd.to_numeric(data_epoch2)
    # data_epoch3 = pd.to_numeric(data_epoch3)
    # data_epoch4 = pd.to_numeric(data_epoch4)
    # # data_epoch1 = [data_epoch1.to_numpy(float)]
    # # data_epoch1 = np.array(data_epoch1)
    # # data_epoch2 = [data_epoch2.to_numpy(float)]
    # # data_epoch2 = np.array(data_epoch2)
    # # data_epoch3 = [data_epoch3.to_numpy(float)]
    # # data_epoch3 = np.array(data_epoch3)
    # # data_epoch4 = [data_epoch4.to_numpy(float)]
    # # data_epoch4 = np.array(data_epoch4)
    # # print(data_epoch1)
    # band_powers1 = compute_band_powers(data_epoch1, fs)
    # band_powers2 = compute_band_powers(data_epoch2, fs)
    # band_powers3 = compute_band_powers(data_epoch3, fs)
    # band_powers4 = compute_band_powers(data_epoch4, fs)
    # # print(data.compute_psd())
    # x2_1 = [band_powers1[Band.Delta], band_powers1[Band.Theta], band_powers1[Band.Alpha], band_powers1[Band.Beta], band_powers1[Band.Gamma]]
    # x2_2 = [band_powers2[Band.Delta], band_powers2[Band.Theta], band_powers2[Band.Alpha], band_powers2[Band.Beta], band_powers1[Band.Gamma]]
    # x2_3 = [band_powers3[Band.Delta], band_powers3[Band.Theta], band_powers3[Band.Alpha], band_powers3[Band.Beta], band_powers1[Band.Gamma]]
    # x2_4 = [band_powers4[Band.Delta], band_powers4[Band.Theta], band_powers4[Band.Alpha], band_powers4[Band.Beta], band_powers1[Band.Gamma]] 
    
    # x2_mean = [(x2_1[0]+x2_2[0]+x2_3[0]+x2_4[0])/4, (x2_1[1]+x2_2[1]+x2_3[1]+x2_4[1])/4, (x2_1[2]+x2_2[2]+x2_3[2]+x2_4[2])/4, (x2_1[3]+x2_2[3]+x2_3[3]+x2_4[3])/4, (x2_1[4]+x2_2[4]+x2_3[4]+x2_4[4])/4]
    # y_bar = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    # print(x2_mean)
    # plt.bar( y_bar, x2_mean, color = ['red', 'blue', 'yellow', 'green', 'purple'])
    # plt.show()
    # print(data)
    montage = mne.channels.make_standard_montage('standard_1005')
    info = mne.create_info(ch_names=['TP9', 'AF7', 'AF8', 'TP10'], sfreq=125, ch_types=['eeg']*4)
    raw = mne.io.RawArray(data.T, info)
    # info.set_montage(montage)
    # montage=info.get_montage()
    raw.set_montage(montage)
    bands = {'Delta': (0, 4), 'Theta': (4, 8),
         'Alpha': (8, 12), 'Beta': (12, 30),
         'Gamma': (30, 45)}
    psd_combined = {band: 0 for band in bands}
    # print(raw.compute_psd())
    channels = ['TP9', 'AF7', 'AF8', 'TP10']
    for channel in channels:
        data_channel = data[channel].values  # Extract data for the channel
        freqs, psd = welch(data_channel, fs=1000)  # Assuming a sampling frequency of 1000 Hz
        for band, (low, high) in bands.items():
            band_indices = np.where((freqs >= low) & (freqs < high))[0]
            psd_combined[band] += np.mean(band_indices)
    
    num_channels = len(channels)
    for band in bands:
        psd_combined[band] /= num_channels

    # Plotting
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    plt.figure(figsize=(8, 6))
    plt.bar(psd_combined.keys(), psd_combined.values(), color=colors)
    plt.xlabel('Frequency Band')
    plt.ylabel('Average Power Spectral Density')
    plt.title('Average Power Spectral Density Across Frequency Bands (All Channels)')
    plt.savefig('../src/components/bar_graph.png')
    plt.clf()
    raw.plot_psd(show = False).savefig('../src/components/psd_plot.png')
    # save to templates/2.jpg
    try:
        raw.compute_psd().plot_topomap(bands,ch_type='eeg', normalize=True, show = False).savefig('../src/components/psd_topomap.png')

        return 
    except IndexError:
        return
    # save elctrode position to templates/3.jpg
    # raw.plot_sensors(show = True)
    # data=data[['TP9', 'AF7', 'AF8', 'TP10']]
    # data=data.to_numpy(float).transpose()
    # evk_array = mne.EvokedArray(data.T, info, baseline=None)
    # print(data)
    # t = 0.1  # time in seconds
    # print(info.ch_names)
    # fig, ax = plt.subplots()
    # print(data.shape[1])
    # # j = np.random.randint(500, data.shape[1])
    # times = np.linspace(0.05, 0.13, 5)
    # evk_array.plot_topomap(ch_type="eeg", times=evk_array.times[::500], colorbar=True, 
    #                             #    show=False, 
    #                             #    axes =ax
                                #    )
    # fig, anim = evk_array.animate_topomap(times=evk_array.times[::100], blit=False,ch_type='eeg')
    # anim.save('Brainmation.gif', writer='imagemagick', fps=10)
    # input()
    # plt.show()
# plot_topo()
