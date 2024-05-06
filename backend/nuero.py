# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers

This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it shows how ratios of the band powers can be used to estimate
mental state for neurofeedback.

The neurofeedback protocols described here are inspired by
*Neurofeedback: A Comprehensive Review on System Design, Methodology and Clinical Applications* by Marzbani et. al

Adapted from https://github.com/NeuroTechX/bci-workshop
"""

import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
import muselsl
from muselsl import record
from flask_cors import CORS
from flask import Flask, jsonify
import pandas as pd
import mne
import matplotlib.pyplot as plt
import numpy as np
from test_topo import plot_topo
from predict import predict

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3
    Gamma = 4


""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]

app = Flask(__name__)
CORS(app)
eeg_buffer1=None
eeg_buffer2=None
eeg_buffer3=None
eeg_buffer4=None
filter_state=None
inlet=None
eeg_time_correction=None
info=None
description=None
fs=None
filter_state=None
band_buffer=None
n_win_test=None
@app.route('/startstream')

# Handy little enum to make code more readable

def start_stream():

    global eeg_buffer1
    global eeg_buffer2
    global eeg_buffer3
    global eeg_buffer4
    global filter_state
    global inlet
    global eeg_time_correction
    global info
    global description
    global fs
    global filter_state
    global band_buffer
    global n_win_test


    if __name__ == "__main__":

        """ 1. CONNECT TO EEG STREAM """
        muselsl.stream(None,backend="bluemuse")
        # Search for active LSL streams
        print('Looking for an EEG stream...')
        streams = resolve_byprop('type', 'EEG', timeout=2)
        if len(streams) == 0:
            raise RuntimeError('Can\'t find EEG stream.')

        # Set active EEG stream to inlet and apply time correction
        print("Start acquiring data")
        inlet = StreamInlet(streams[0], max_chunklen=12)
        eeg_time_correction = inlet.time_correction()

        # Get the stream info and description
        info = inlet.info()
        description = info.desc()

        # Get the sampling frequency
        # This is an important value that represents how many EEG data points are
        # collected in a second. This influences our frequency band calculation.
        # for the Muse 2016, this should always be 256
        fs = int(info.nominal_srate())

        """ 2. INITIALIZE BUFFERS """

        # Initialize raw EEG data buffer
        eeg_buffer1 = np.zeros((int(fs * BUFFER_LENGTH), 1))
        eeg_buffer2 = np.zeros((int(fs * BUFFER_LENGTH), 1))
        eeg_buffer3 = np.zeros((int(fs * BUFFER_LENGTH), 1))
        eeg_buffer4 = np.zeros((int(fs * BUFFER_LENGTH), 1))
        filter_state = None  # for use with the notch filter

        # Compute the number of epochs in "buffer_length"
        n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                                SHIFT_LENGTH + 1))

        # Initialize the band power buffer (for plotting)
        # bands will be ordered: [delta, theta, alpha, beta]
        band_buffer = np.zeros((n_win_test, 4))
        # import subprocess
        # subprocess.call('muselsl view',shell=True)
        return "starting"

@app.route('/fetchstream')

# Handy little enum to make code more readable

def fetch_stream():
    global filter_state
    global inlet
    global eeg_time_correction
    global info
    global description
    global fs
    global filter_state
    global band_buffer
    global n_win_test
    global eeg_buffer1
    global eeg_buffer2
    global eeg_buffer3
    global eeg_buffer4
        
    if __name__ == "__main__":

        """ 1. CONNECT TO EEG STREAM """
        # muselsl.stream(None,backend="bluemuse")
        # # Search for active LSL streams
        # print('Looking for an EEG stream...')
        # streams = resolve_byprop('type', 'EEG', timeout=2)
        # if len(streams) == 0:
        #     raise RuntimeError('Can\'t find EEG stream.')

        # # Set active EEG stream to inlet and apply time correction
        # print("Start acquiring data")
        # inlet = StreamInlet(streams[0], max_chunklen=12)
        # eeg_time_correction = inlet.time_correction()

        # # Get the stream info and description
        # info = inlet.info()
        # description = info.desc()

        # # Get the sampling frequency
        # # This is an important value that represents how many EEG data points are
        # # collected in a second. This influences our frequency band calculation.
        # # for the Muse 2016, this should always be 256
        # fs = int(info.nominal_srate())


        # """ 2. INITIALIZE BUFFERS """

        # # Initialize raw EEG data buffer
        # eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
        # filter_state = None  # for use with the notch filter

        # # Compute the number of epochs in "buffer_length"
        # n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
        #                         SHIFT_LENGTH + 1))

        # # Initialize the band power buffer (for plotting)
        # # bands will be ordered: [delta, theta, alpha, beta]
        # band_buffer = np.zeros((n_win_test, 4))
        # import subprocess
        # subprocess.call('muselsl view',shell=True)
        # return "starting"



        """ 3. GET DATA """

        # The try/except structure allows to quit the while loop by aborting the
        # script with <Ctrl-C>
        print('Press Ctrl-C in the console to break the while loop.')
        
        try:
            # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
            while True:

                """ 3.1 ACQUIRE DATA """
                # Obtain EEG data from the LSL stream
                eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(SHIFT_LENGTH * fs))

                # Only keep the channel we're interested in
                ch_data1 = np.array(eeg_data)[:, [0]]
                ch_data2 = np.array(eeg_data)[:, [1]]
                ch_data3 = np.array(eeg_data)[:, [2]]
                ch_data4 = np.array(eeg_data)[:, [3]]

                # Update EEG buffer with the new data
                eeg_buffer1, filter_state = utils.update_buffer(
                    eeg_buffer1, ch_data1, notch=True,
                    filter_state=filter_state)
                eeg_buffer2, filter_state = utils.update_buffer(
                    eeg_buffer2, ch_data2, notch=True,
                    filter_state=filter_state)
                eeg_buffer3, filter_state = utils.update_buffer(
                    eeg_buffer3, ch_data3, notch=True,
                    filter_state=filter_state)
                eeg_buffer4, filter_state = utils.update_buffer(
                    eeg_buffer4, ch_data4, notch=True,
                    filter_state=filter_state)

                """ 3.2 COMPUTE BAND POWERS """
                # Get newest samples from the buffer
                data_epoch1 = utils.get_last_data(eeg_buffer1,
                                                EPOCH_LENGTH * fs)
                data_epoch2 = utils.get_last_data(eeg_buffer2,
                                                EPOCH_LENGTH * fs)
                data_epoch3 = utils.get_last_data(eeg_buffer3,
                                                EPOCH_LENGTH * fs)
                data_epoch4 = utils.get_last_data(eeg_buffer4,
                                                EPOCH_LENGTH * fs)

                # Compute band powers
                band_powers1 = utils.compute_band_powers(data_epoch1, fs)
                band_powers2 = utils.compute_band_powers(data_epoch2, fs)
                band_powers3 = utils.compute_band_powers(data_epoch3, fs)
                band_powers4 = utils.compute_band_powers(data_epoch4, fs)

                # band_buffer, _ = utils.update_buffer(band_buffer,
                #                                     np.asarray([band_powers]))
                # # Compute the average band powers for all epochs in buffer
                # # This helps to smooth out noise
                # smooth_band_powers = np.mean(band_buffer, axis=0)

                
                # df = pd.DataFrame()
                # df['TP9'] = data_epoch1[0]
                # df['AF7'] = data_epoch2[0]
                # df['AF8'] = data_epoch3[0]
                # df['TP10'] = data_epoch4[0]

                x2_1 = [band_powers1[Band.Delta], band_powers1[Band.Theta], band_powers1[Band.Alpha], band_powers1[Band.Beta], band_powers1[Band.Gamma]]
                x2_2 = [band_powers2[Band.Delta], band_powers2[Band.Theta], band_powers2[Band.Alpha], band_powers2[Band.Beta], band_powers1[Band.Gamma]]
                x2_3 = [band_powers3[Band.Delta], band_powers3[Band.Theta], band_powers3[Band.Alpha], band_powers3[Band.Beta], band_powers1[Band.Gamma]]
                x2_4 = [band_powers4[Band.Delta], band_powers4[Band.Theta], band_powers4[Band.Alpha], band_powers4[Band.Beta], band_powers1[Band.Gamma]] 
                # print(x2_1)
                # print(x2_2)
                # print(x2_3)
                # print(x2_4)
                x2_mean = [(x2_1[0]+x2_2[0]+x2_3[0]+x2_4[0])/4, (x2_1[1]+x2_2[1]+x2_3[1]+x2_4[1])/4, (x2_1[2]+x2_2[2]+x2_3[2]+x2_4[2])/4, (x2_1[3]+x2_2[3]+x2_3[3]+x2_4[3])/4, (x2_1[4]+x2_2[4]+x2_3[4]+x2_4[4])/4]
                y_bar = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
                # print(data_epoch1,'hi')
                # if (data_epoch1):
                #     df.to_csv('static/data_epoch.csv', index=False)
                #     plot_topo()
                x1 = list(range(1,len(data_epoch1)+1))
                y1 = [y[0] for y in data_epoch1]
                data_list1 = {'x' : x1, 'y': y1, 'type': 'scatter', 'mode' : 'lines', 'name' : 'TP9', 'line':{'color': 'red'}}
                x1 = list(range(1,len(data_epoch2)+1))
                y2 = [y[0] for y in data_epoch2]
                data_list2 = {'x' : x1, 'y': y2, 'type': 'scatter', 'mode' : 'lines', 'name' : 'AF7', 'line':{'color': 'blue'}}
                x1 = list(range(1,len(data_epoch3)+1))
                y3 = [y[0] for y in data_epoch3]
                data_list3 = {'x' : x1, 'y': y3, 'type': 'scatter', 'mode' : 'lines', 'name' : 'AF8', 'line':{'color': 'yellow'}}
                x1 = list(range(1,len(data_epoch4)+1))
                y4 = [y[0] for y in data_epoch4]
                data_list4 = {'x' : x1, 'y': y4, 'type': 'scatter', 'mode' : 'lines', 'name' : 'TP10', 'line':{'color': 'green'}}
                data_list_main = [data_list1, data_list2, data_list3, data_list4]
                bargraph = [{'x': y_bar, 'y': x2_mean, 'type': 'bar', 'marker': {'color': ['red', 'blue', 'yellow', 'green', 'purple']}}]
                data_list = [
                    {'data': data_list_main, 'layout': {'width': 800, 'height': 400, 'title': 'EEG Signals in Real Time', 'plot_bgcolor': 'rgba(240, 240, 240, 0.9)','xaxis': {'title': ' Chanel Wise Distribution '}, 'yaxis': {'title': ' '}}},
                    {'data': [data_list1], 'layout': {'width': 800, 'height': 300, 'plot_bgcolor': 'rgba(240, 240, 240, 0.9)','xaxis': {'title': ' '}, 'yaxis': {'title': 'TP9'}}},
                    {'data': [data_list2], 'layout': {'width': 800, 'height': 300, 'plot_bgcolor': 'rgba(240, 240, 240, 0.9)','xaxis': {'title': ' '}, 'yaxis': {'title': 'AF7'}}},
                    {'data': [data_list3], 'layout': {'width': 800, 'height': 300, 'plot_bgcolor': 'rgba(240, 240, 240, 0.9)','xaxis': {'title': ' '}, 'yaxis': {'title': 'AF8'}}},
                    {'data': [data_list4], 'layout': {'width': 800, 'height': 300, 'plot_bgcolor': 'rgba(240, 240, 240, 0.9)','xaxis': {'title': ' '}, 'yaxis': {'title': 'TP10'}}},
                    {'data': bargraph, 'layout': {'width': 400, 'height': 400, 'title': 'Frequency Band Distribution', 'xaxis': {'title': ' Frequency Bands '}, 'yaxis': {'title': ' '}}}
                ]


                # df = {'TP9': y1, 'AF7': y2, 'AF8': y3, 'TP10': y4}
                # df = pd.DataFrame(data=df)
                # # print(df)
                # df.to_csv('static/test.csv')
                # plot_topo()
                return jsonify({'data': data_list})

                """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
                # These metrics could also be used to drive brain-computer interfaces

                # Alpha Protocol:
                # Simple redout of alpha power, divided by delta waves in order to rule out noise
                # alpha_metric = smooth_band_powers[Band.Alpha] / \
                #     smooth_band_powers[Band.Delta]
                # print('Alpha Relaxation: ', alpha_metric)

                # # Beta Protocol:
                # # Beta waves have been used as a measure of mental activity and concentration
                # # This beta over theta ratio is commonly used as neurofeedback for ADHD
                # beta_metric = smooth_band_powers[Band.Beta] / \
                #     smooth_band_powers[Band.Theta]
                # print('Beta Concentration: ', beta_metric)

                # # Alpha/Theta Protocol:
                # # This is another popular neurofeedback metric for stress reduction
                # # Higher theta over alpha is supposedly associated with reduced anxiety
                # theta_metric = smooth_band_powers[Band.Theta] / \
                #     smooth_band_powers[Band.Alpha]
                # print('Theta Relaxation: ', theta_metric)

        except KeyboardInterrupt:
            print('Closing!')




@app.route('/recordstream')

    # Note: an existing Muse LSL stream is required
def record_stream():
    if __name__ == "__main__":
        record(30,filename='./static/realtime_csv.csv')
        predict()
        plot_topo()
        # Note: Recording is synchronous, so code here will not execute until the stream has been closed
        print('Recording has ended')
        return 'hi'


if __name__ == '__main__':
    app.run(debug=True, port = 5000)