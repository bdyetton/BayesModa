import numpy as np
import tkinter as tk
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch
root = tk.Tk()
screen_width = 1600
screen_height = 900
dpi = round(root.winfo_fpixels('1i'))

sample_rate = 100
seconds_per_epoch = 25
samples_per_epoch = seconds_per_epoch*sample_rate

conf_color_map = {'low': 'r', 'med':'y', 'high': 'g', 'Gold Standard': 'k'}
cmap = LinearSegmentedColormap.from_list('mycmap', [(234/255, 234/255, 242/255), (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)])
conf_cmap = LinearSegmentedColormap.from_list('confcmap', ['r','y','g'], 6)



#%% Features
def plot_epoch(signal):
    plt.plot(np.array(range(0, samples_per_epoch))/sample_rate, signal, linewidth=0.5)
    plt.gca()
    plt.xlim((0, 25))
    plt.xlabel('')
    plt.xticks([])
    plt.ylabel('uV')


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def plot_bandpassed_epoch(signal):
    signal = butter_bandpass_filter(signal, 10, 16, 100)
    plt.plot(np.array(range(0, samples_per_epoch)) / sample_rate, signal, linewidth=0.5)
    plt.xlim((0, 25))
    plt.ylabel('uV')
    plt.xlabel('Seconds')


def plot_marker(marker, y_to_plot_at, fat=False):
    color = conf_cmap(marker['conf'])
    width = {True:3.0, False:1.0}
    if marker['theta'] == 1 or marker['rater_i'] == 'matlab_gs':
        style = '-'
    else:
        style = '--'
    plt.plot([marker['s'], marker['e']],
             [y_to_plot_at, y_to_plot_at],
             color=color, linewidth=width[fat], linestyle=style)
    plt.title('25 second Epoch of Stage 2, C3')
    plt.xlabel('')
    plt.xlim([0, max(25, y_to_plot_at)])
    plt.xticks([])
    plt.ylabel('')
    plt.yticks([])
    plt.grid(False)
    #plt.legend(['High Confidence','Med Confidence','Low Confidence'])


#%% Plot
def plot_an_epoch(epoch_signal, predicted_markers, real_markers, rater_to_plot_at_bottom='matlab_gs'):
    fig = plt.figure(figsize=(round(0.75 * screen_width / dpi), round(0.4 * screen_height / dpi)), dpi=dpi)

    ax2 = fig.add_subplot(3, 1, 2)
    plot_epoch(epoch_signal)
    for idx, row in predicted_markers.iterrows():
        plt.axvspan(row['s'], row['s'] + row['d'], color='red', alpha=0.3)

    ax1 = fig.add_subplot(3, 1, 1)
    rater = 0
    for marker_idxs, markers_slices in real_markers.groupby(['rater_i']):
        for idx, marker in markers_slices.iterrows():
            if marker_idxs == rater_to_plot_at_bottom:
                y_plot = -3  # plot GS at the bottom
                fat_line = True
            else:
                y_plot = idx * 1
                fat_line = False
            plot_marker(marker, y_plot, fat=fat_line)
            if marker['theta'] == 1:
                con = ConnectionPatch(xyA=((marker['s']+marker['e'])/2, y_plot),
                                      xyB=((predicted_markers.loc[marker['w'], 's'] +
                                            predicted_markers.loc[marker['w'], 'e'])/2, ax2.get_ylim()[1]),
                                      coordsA="data",
                                      coordsB="data",
                                      axesA=ax1,
                                      axesB=ax2,
                                      color="grey", alpha=0.3)
                ax1.add_artist(con)
            rater = rater + 1

    ax3 = fig.add_subplot(3, 1, 3)
    plot_bandpassed_epoch(epoch_signal)
    plt.show()
