import numpy as np
import matplotlib.pyplot as plt

REAL = int or float or complex


def plot_range(dist: list[REAL] = (0, 1), 
               bins: int = 0, 
               divisor: int = 1, 
               decimal: int = 2) -> tuple[list[int or float], list[str]]:
    """ Returns a range of evenly-spaced values in a desired range.

    Args:
        dist (list[REAL], optional): List of two values denoting the range of values to plot. Defaults to (0, 1).
        bins (int, optional): Number of markers to plot in the range. Defaults to 0.
        divisor (int, optional): Integer scalar for range; 
            setting divisor to the sample rate of the waveform scales the axis to seconds. Defaults to 1.
        decimal (int, optional): Number of decimal places to round marker values to. Defaults to 2.

    Returns:
        tuple[list[int or float], list[str]]: list of evenly-spaced values.
    """

    if bins <= 0: # If empty, provide no markers or labels
        return [], []
    elif bins == 1:
        result = [dist[0], dist[1]]
    else:
        result = [i for i in range(dist[0], dist[1], dist[1] // (bins - 1))] + [dist[1]]
    labels = [f'${round(i / divisor, decimal)}$' for i in result]
    return result, labels


from scipy.io.wavfile import read
def wavread(path: str) -> tuple[int, list[int or float]]:
    """Read in .wav files as sample rate (integer), and array of values;
    uses import "read" from scipy.io.wavfile.

    Args:
        path (str): file path to .wav file

    Returns:
        tuple[int, list[int or float]]: n-D array of len(array) samples and n channels.
    """

    sr, s = read(path)
    return sr, s.astype('float32')/32768


def wave_plot(plot: plt.Axes, 
              sample: np.ndarray or list[REAL], 
              sr: int or float = 48000, 
              xbins: int = 2, 
              ybins: int = 2,
              xbinlabels: list[str] = [], 
              ybinlabels: list[str] = ['$-\\alpha$', '$0.0$', '$\\alpha$'],
              xlabel: str = 'samples', 
              ylabel: str = 'Amplitude', 
              title: str = 'Waveform',
              grid: bool = False, 
              color: str ='C0') -> None:
    """Generates a waveform graph into the given plot.

    Args:
        plot (plt.Axes): plot to graph the waveform to.
        sample (np.ndarrayorlist[REAL]): sample array to be loaded as a waveform.
        sr (intorfloat, optional): sample rate of the waveform. Defaults to 48000.
        xbins (int, optional): number of markers to plot in the x-axis. Defaults to 2.
        ybins (int, optional): number of markers to plot in the y-axis Defaults to 2.
        xbinlabels (list[str], optional): List of marker strings to replace default x-axis labels. Defaults to [].
        ybinlabels (list[str], optional): List of marker strings to replace default y-axis labels. Defaults to ['$-\alpha$', '.0$', '$\alpha$'].
        xlabel (str, optional): label string for x-axis. Defaults to 'samples'.
        ylabel (str, optional): label string for y-axis. Defaults to 'Amplitude'.
        title (str, optional): title for waveform. Defaults to 'Waveform'.
        grid (bool, optional): boolean option to show or hide grid lines. Defaults to False.
        color (str, optional): color of the waveform. Defaults to 'C0'.
    """

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


DFT_SIZE = 2048
from scipy import fft
def stft( input_sound: np.ndarray or list[int or float], 
         dft_size: int = DFT_SIZE, 
         hop_size: int = 256, 
         zero_pad: int = 0, 
         window: np.ndarray = np.hanning(DFT_SIZE)) -> np.ndarray:

    # zero-padding for equally sized frames
    frames = []
    input_sound = np.append(input_sound, np.zeros(len(input_sound) % dft_size))

    for i in range(0, len(input_sound) - dft_size, hop_size):
        frame = input_sound[i : i + dft_size] * window
        frames.append(frame)

    spectrogram = np.array([(fft.rfft(f, dft_size + zero_pad)) for f in frames], dtype=np.complex64)
    # Return a complex-valued spectrogram (frequencies x time)
    return spectrogram.T 


# Plot waveform of an audio file
def spec_plot(plot: plt.Axes,
            sample: np.ndarray or list[REAL],
            sr: int, 
            nfft: int = 2048,
            xbins: int = 2,
            ybins: int = 2,
            xbinlabels: list[str] = [], 
            ybinlabels: list[str] = [],
            xlabel: str = 'time (s)', 
            ylabel: str = 'Frequency (kHz)', 
            title: str = 'Spectrogram', 
            cmap: str = 'viridis') -> None:
    
    sample_stft = stft(sample, dft_size=nfft, hop_size=nfft//16)

    plot.pcolormesh(10.0*np.log(np.absolute(sample_stft)), shading="auto", cmap=cmap)

    plot.set_xticks(np.linspace(0, sample_stft.shape[1], xbins))
    plot.set_xticklabels(xbinlabels if xbinlabels else [np.round(i * len(sample) / (xbins - 1) / sr, 2) for i in range(xbins)])
    plot.set_yticks(np.linspace(0, sample_stft.shape[0], ybins))
    plot.set_yticklabels(ybinlabels if ybinlabels else np.linspace(0, sr/2000, ybins))
   
   # Plot appearance
    plot.tick_params(axis='both', which='major', labelsize=15)
    if xbins == 0:
        plot.set_frame_on(False)
    plot.set_xlabel(xlabel, fontsize=20)
    plot.set_ylabel(ylabel, fontsize=20)
    plot.set_title(title)


from scipy.signal import spectrogram
def spec3d_plot(plot: plt.Axes, 
                sample: np.ndarray or list[int or float or complex], 
                sr: int, 
                nfft: int = 2048, 
                xbins: int = 2, 
                ybins: int = 2, 
                zbins: int = 2,
                cmap: str = 'viridis'):
    
    f, t, s = spectrogram(sample, sr, axis=0, nfft=nfft)
    s = 10.0 * np.log(s) # log scale for spectrum

    plot.plot_surface(t[None, :], f[:, None], s, cmap=cmap)

    plot.set_xticks(np.linspace(0, len(sample)//sr, xbins))
    plot.set_yticks(np.linspace(0, sr/2, ybins))
    plot.set_zticks(np.linspace(np.min(s), 0, zbins))
    plot.set_ylabel('Frequency (kHz)')
    plot.set_xlabel('time (s)')
    plot.set_zlabel('Volume (dB)')