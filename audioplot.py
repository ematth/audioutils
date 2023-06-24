import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import spectrogram
from matplotlib import cm

## FUNCTIONS

def plot_range(dist: list[int or float] = (0, 1), bins: int = 0, divisor: int = 1, decimal: int = 2):
    """
    Given a range to mark on an axis, returns a list of evenly-spaced values for the axis, with approximated labels.
    """

    if bins <= 0: # If empty, provide no markers or labels
        return [], []
    elif bins == 1:
        result = [dist[0], dist[1]]
    else:
        result = [i for i in range(dist[0], dist[1], dist[1] // (bins - 1))] + [dist[1]]
    labels = [f'${round(i / divisor, decimal)}$' for i in result]
    return result, labels

def wavread(path):
    sr, s = read(path)
    return sr, s.astype('float32')/32768


# Plot waveform of an audio file
def wave_plot(plot, sample, sr=48000, xbins=2, ybins=2,
              xbinlabels=[], ybinlabels=['$-\\alpha$', '$0.0$', '$\\alpha$'],
              xlabel='samples', ylabel='Amplitude', title='Waveform',
              grid=False, color='C0'):
    plot.plot(sample, color=color)

    # Axis markers and labels
    xrange, xlabels = plot_range([0, len(sample)], bins=xbins, divisor=sr, decimal=2)
    plot.set_xticks(xrange, labels=xbinlabels if (len(xbinlabels) == len(xrange)) else xlabels)
    yrange, ylabels = plot_range([round(np.min(sample)), round(np.max(sample))], bins=2, decimal=2)
    plot.set_yticks(yrange, labels=ybinlabels if (len(ybinlabels) == len(yrange)) else ylabels)

    # Plot appearance
    plot.tick_params(axis='both', which='major', labelsize=15)
    plot.set_frame_on(False)
    plot.grid(grid, axis='y')
    plot.set_xlabel(xlabel, fontsize=20)
    plot.set_ylabel(ylabel, fontsize=20)
    plot.set_title(title)


# Plot waveform of an audio file
def spec_plot(plot, sample, sr, nfft=2048, xbins=2, ybins=2,
              xbinlabels=[], ybinlabels=[],
              xlabel='time (s)', ylabel='Frequency (Hz)', title='Spectrogram', cmap=cm.viridis):
    
    f, t, s = spectrogram(sample, sr, axis=0, nfft=nfft)

    print(f.shape)
    print(t.shape)
    print(sample.shape[0]/sr)

    plot.pcolormesh(t, f, 10.0*np.log10(s), shading="auto", cmap=cmap)
    plot.locator_params(axis='x', nbins=xbins*2)
    # plot.set_xlabel('time (s)')
    freq_range = range(0, sr//2, 1000*ybins)
   
   # Plot appearance
    plot.tick_params(axis='both', which='major', labelsize=15)
    plot.set_frame_on(False)
    plot.set_xlabel(xlabel, fontsize=20)
    plot.set_ylabel(ylabel, fontsize=20)
    plot.set_title(title)


def spec3d_plot(plot, sample, sr, interval, nfft=2048):
    f, t, s = spectrogram(sample, sr, axis=0, nfft=nfft)
    # I don't know why this line works, it just does
    plot.plot_surface(f[:, None], t[None, :], 10.0*np.log(s), cmap=cm.viridis)
    # plot.set_title('3D Spectrogram')
    freq_range = range(0, sr//2, 1000*interval)
    plot.set_xticks([])
    plot.set_yticks([])
    plot.set_zticks([])
    # plot.set_xticks([i for i in freq_range], [f'{i//1000}' for i in freq_range])
    # plot.set_xlabel('Frequency (kHz)')
    # plot.set_ylabel('time(s)')


# def audioplot(sample, sr):

#     fig = plt.figure()

#     ax1 = fig.add_subplot(1,3,1); ax1.set_yticks([])
#     ax2 = fig.add_subplot(1,3,2)
#     ax3 = fig.add_subplot(1,3,3, projection='3d')
#     fig.set_size_inches(12, 6)
#     fig.suptitle("Chopin Op.28 No.1")

#     wave_plot(ax1, sample1, sr1, 5)
#     spec_plot(ax2, sample1, sr1, 5)
#     spec3d_plot(ax3, sample1, sr1, 5)
#     sm = cm.ScalarMappable(cmap=cm.viridis)
#     fig.colorbar(sm, ax=ax3, pad=0.15)
#     fig.tight_layout(pad=2.0)
#     plt.show()

# if __name__ == '__main__':
#     sr1, sample1 = wavread('./reunion.wav')
#     sample1 = np.sum(sample1, axis=1)
#     audioplot(sample1, sr1)