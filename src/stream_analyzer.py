import numpy as np
import pyaudio
import time, sys, math, cv2
from time import perf_counter
from math import log
from collections import deque
import scipy
from scipy.signal import savgol_filter

from src.fft import getFFT

def round_up_to_even(f):
    return int(math.ceil(f / 2.) * 2)

def round_to_nearest_power_of_two(f, base=2):
    l = math.log(f,base)
    rounded = int(np.round(l,0))
    return base**rounded

def get_frequency_bins(start, stop, n):
    octaves = np.logspace(log(start)/log(2), log(stop)/log(2), n, endpoint=True, base=2, dtype=None)
    return np.insert(octaves, 0, 0)

def get_smoothing_filter(FFT_window_size_ms, filter_length_ms, verbose = 0):
    buffer_length = round_up_to_even(filter_length_ms / FFT_window_size_ms)+1
    filter_sigma = buffer_length/4
    filter_weights = cv2.getGaussianKernel(buffer_length * 2, filter_sigma)
    max_index = np.argmax(filter_weights)
    filter_weights = filter_weights[:max_index+1]
    filter_weights = filter_weights / np.mean(filter_weights)

    if verbose:
        min_fraction = 100*np.min(filter_weights)/np.max(filter_weights)
        print('\nApplying temporal smoothing to the FFT features...')
        print("Smoothing buffer contains %d FFT windows (sigma: %d) --> min_contribution: %.3f%%" %(buffer_length, filter_sigma, min_fraction))
        print("Filter weights:")
        for i, w in enumerate(filter_weights):
            print("%02d: %.3f" %(len(filter_weights)-i, w))

    return filter_weights

class numpy_data_buffer:
    """
    A fast, circular FIFO buffer in numpy with minimal memory interactions by using an array of index pointers
    """

    def __init__(self, n_windows, samples_per_window, dtype = np.int32, start_value = 0, data_dimensions = 1):
        self.n_windows = n_windows
        self.data_dimensions = data_dimensions
        self.samples_per_window = samples_per_window
        self.data = start_value * np.ones((self.n_windows, self.samples_per_window), dtype = dtype)

        if self.data_dimensions == 1:
            self.total_samples = self.n_windows * self.samples_per_window
        else:
            self.total_samples = self.n_windows

        self.elements_in_buffer = 0
        self.overwrite_index = 0

        self.indices = np.arange(self.n_windows, dtype=np.int32)
        self.last_window_id = np.max(self.indices)
        self.index_order = np.argsort(self.indices)

    def append_data(self, data_window):
        self.data[self.overwrite_index, :] = data_window

        self.last_window_id += 1
        self.indices[self.overwrite_index] = self.last_window_id
        self.index_order = np.argsort(self.indices)

        self.overwrite_index += 1
        self.overwrite_index = self.overwrite_index % self.n_windows

        self.elements_in_buffer += 1
        self.elements_in_buffer = min(self.n_windows, self.elements_in_buffer)

    def get_most_recent(self, window_size):
        ordered_dataframe = self.data[self.index_order]
        if self.data_dimensions == 1:
            ordered_dataframe = np.hstack(ordered_dataframe)
        return ordered_dataframe[self.total_samples - window_size:]

    def get_buffer_data(self):
        return self.data[:self.elements_in_buffer]


class Audio_Analyzer:
    """
    The Audio_Analyzer class provides access to continuously recorded
    (and mathematically processed) microphone data.

    Arguments:

        device - the number of the sound card input to use.
        rate - sample rate to use. Defaults to something supported.
        FFT_window_size_ms - time window size (in ms) to use for the FFT transform
        updatesPerSecond - how fast to record new data. Note that smaller
        numbers allow more data to be accessed and therefore high
        frequencies to be analyzed if using a FFT later
    """

    ### https://github.com/mwickert/scikit-dsp-comm/tree/master/sk_dsp_comm

    def __init__(self, device = None, rate = None, 
        FFT_window_size_ms = 50, 
        updates_per_second = 100, 
        smoothing_length_ms = 50, 
        n_frequency_bins = 51, 
        visualize = True,
        plot_audio_history = False,
        verbose = False):

        self.n_frequency_bins = n_frequency_bins
        self.plot_audio_history = plot_audio_history
        self.rate = rate
        self.verbose = verbose
        self.visualize = visualize
        self.pa = pyaudio.PyAudio()

        self.log_features = False
        self.rolling_stats_window_s = 60    #The axis range of the FFT features will adapt dynamically using a window of N seconds
        self.equalizer_strength = 0.2       #[0-1] --> gradually rescales all FFT features to have the same mean

        self.apply_frequency_smoothing = True  #Apply a postprocessing smoothing filter over the FFT outputs
        if self.apply_frequency_smoothing:
            self.filter_width = round_up_to_even(0.03*self.n_frequency_bins) - 1
        
        if self.visualize:
            from src.visualizer import spectrum_visualizer

        self.delays = deque(maxlen=20)
        self.data_capture_delays = deque(maxlen=20)
        self.num_data_captures = 0
        self.num_ffts = 0
        self.update_window_n_frames = 1024 #Don't remove this, needed for device testing!
        
        self.strongest_frequency = 0
        self.device = device
        if self.device is None:
            self.device = self.input_device()
        if self.rate is None:
            self.rate = self.valid_low_rate(self.device)

        self.update_window_n_frames = round_up_to_even(self.rate / updates_per_second)
        self.updates_per_second = self.rate / self.update_window_n_frames

        print("\n#############################\nDefaulted to using first working mic, Running on:")
        self.print_mic_info(self.device)

        #self.FFT_window_size = round_to_nearest_power_of_two(self.rate * FFT_window_size_ms / 1000)
        self.FFT_window_size = round_up_to_even(self.rate * FFT_window_size_ms / 1000)

        self.FFT_window_size_ms = 1000 * self.FFT_window_size / self.rate
        self.ideal_fps = self.rate / self.FFT_window_size
        self.datax = np.arange(self.FFT_window_size) / float(self.rate)
        self.info = self.pa.get_device_info_by_index(self.device)

        self.new_data = False
        self.fft  = np.ones(int(self.FFT_window_size/2), dtype=float)
        self.fftx = np.arange(int(self.FFT_window_size/2), dtype=float) * self.rate / self.FFT_window_size


        # Temporal smoothing:
        # Currently the buffer acts on the FFT_features (which are computed only occasionally eg 25 fps)
        self.smoothing_length_ms = smoothing_length_ms
        if self.smoothing_length_ms > 0:
            self.smoothing_kernel = get_smoothing_filter(self.FFT_window_size_ms, self.smoothing_length_ms, verbose=1)
            #self.feature_buffer = deque(maxlen=len(self.smoothing_kernel))
            self.feature_buffer = numpy_data_buffer(len(self.smoothing_kernel), len(self.fft), dtype = np.float32, data_dimensions = 2)


        self.data_windows_to_buffer = math.ceil(self.FFT_window_size / self.update_window_n_frames)
        self.data_windows_to_buffer = max(1,self.data_windows_to_buffer)
        self.data_buffer = numpy_data_buffer(self.data_windows_to_buffer, self.update_window_n_frames)
        self.features_up_to_date = False

        self.blocking = False
        self.stream = self.pa.open(
            format = pyaudio.paInt16,
            channels = 1,
            rate = self.rate,
            input=True,
            frames_per_buffer = self.update_window_n_frames,
            stream_callback=self.non_blocking_stream_read)

        self.stream_start()
        self.stream_start_time = time.time()

        #This can probably be done more elegantly...
        self.fftx_bin_indices = np.logspace(np.log2(len(self.fftx)), 0, len(self.fftx), endpoint=True, base=2, dtype=None) - 1
        self.fftx_bin_indices = np.round(((self.fftx_bin_indices - np.max(self.fftx_bin_indices))*-1) / (len(self.fftx) / self.n_frequency_bins),0).astype(int)
        self.fftx_bin_indices = np.minimum(np.arange(len(self.fftx_bin_indices)), self.fftx_bin_indices - np.min(self.fftx_bin_indices))

        self.frequency_bin_energies = np.zeros(self.n_frequency_bins)
        self.frequency_bin_centres  = np.zeros(self.n_frequency_bins)
        self.fftx_indices_per_bin   = []
        for bin_index in range(self.n_frequency_bins):
            bin_frequency_indices = np.where(self.fftx_bin_indices == bin_index)
            self.fftx_indices_per_bin.append(bin_frequency_indices)
            fftx_frequencies_this_bin = self.fftx[bin_frequency_indices]
            self.frequency_bin_centres[bin_index] = np.mean(fftx_frequencies_this_bin)

        #Hardcoded parameters:
        self.fft_fps = 30

        #Assume the incoming sound follows a pink noise spectrum:
        self.power_normalization_coefficients = np.logspace(np.log2(1), np.log2(np.log2(self.rate/2)), len(self.fftx), endpoint=True, base=2, dtype=None)
        self.rolling_stats_window_n = self.rolling_stats_window_s * self.fft_fps #Assumes ~30 FFT features per second
        
        self.rolling_bin_values = numpy_data_buffer(self.rolling_stats_window_n, self.n_frequency_bins, start_value = 25000)
        self.bin_max_values  = np.ones(self.n_frequency_bins)
        self.bin_mean_values = np.ones(self.n_frequency_bins)

        if self.visualize:
            self.visualizer = spectrum_visualizer(self, plot_audio_history = plot_audio_history)
            self.visualizer.start()

        print("###########################################################################")
        print('Recording from %s at %d Hz with (non-overlapping) data-windows of %d samples (updating at %.2ffps)' 
            %(self.info["name"],self.rate, self.update_window_n_frames, self.updates_per_second))
        print("Using FFT_window_size length of %d for FFT ---> window_size = %dms (ideal fps: %.2f)"
            %(self.FFT_window_size, self.FFT_window_size_ms, self.ideal_fps))
        if 0:
            for i in range(self.n_frequency_bins):
                print("Frequency bin %d: %d Hz " %(i, self.frequency_bin_centres[i]))
        print("###########################################################################")

        '''
        import aubio
        #self.a_tempo = aubio.tempo("default", self.FFT_window_size, int(self.rate/self.updates_per_second), self.rate)
        self.a_tempo = aubio.tempo("default", 2*2048, int(2048), self.rate)
        '''


    def update_features(self, n_bins = 3):

        latest_data_window = self.data_buffer.get_most_recent(self.FFT_window_size)

        '''
        is_beat = self.a_tempo(latest_data_window[:2048].astype(np.float32))
        if is_beat: print("\nBEAT!")
        '''

        self.fft = getFFT(latest_data_window, self.rate, self.FFT_window_size, log_scale = self.log_features)

        #Equalize pink noise spectrum falloff:
        self.fft = self.fft * self.power_normalization_coefficients

        if self.smoothing_length_ms > 0:
            self.feature_buffer.append_data(self.fft)
            buffered_features = self.feature_buffer.get_most_recent(len(self.smoothing_kernel))

            if len(buffered_features) == len(self.smoothing_kernel):
                buffered_features = self.smoothing_kernel * buffered_features
                self.fft = np.mean(buffered_features, axis=0)

        self.strongest_frequency = self.fftx[np.argmax(self.fft)]

        #ToDo: replace this for-loop with pure numpy code
        for bin_index in range(self.n_frequency_bins):
            self.frequency_bin_energies[bin_index] = np.mean(self.fft[self.fftx_indices_per_bin[bin_index]])

        return

    def update_rolling_stats(self):
        self.rolling_bin_values.append_data(self.frequency_bin_energies)
        self.bin_mean_values  = np.mean(self.rolling_bin_values.get_buffer_data(), axis=0)
        self.bin_mean_values  = np.maximum((1-self.equalizer_strength)*np.mean(self.bin_mean_values), self.bin_mean_values)

    def get_audio_features(self):

        if not self.features_up_to_date:
            if self.verbose:
                start = time.time()

            self.update_features()
            self.update_rolling_stats()

            self.frequency_bin_energies = np.nan_to_num(self.frequency_bin_energies, copy=True)

            if self.filter_width > 3 and self.apply_frequency_smoothing:
                self.frequency_bin_energies = savgol_filter(self.frequency_bin_energies, self.filter_width, 3)

            self.frequency_bin_energies[self.frequency_bin_energies < 0] = 0

            #Beat detection ToDo:
            #https://www.parallelcube.com/2018/03/30/beat-detection-algorithm/
            #https://github.com/shunfu/python-beat-detector
            #https://pypi.org/project/vamp/

            self.features_up_to_date = True
            self.num_ffts += 1
            self.fft_fps  = self.num_ffts / (time.time() - self.stream_start_time)

            if self.verbose:
                self.delays.append(time.time() - start)
                avg_fft_delay = 1000.*np.mean(np.array(self.delays))
                avg_data_capture_delay = 1000.*np.mean(np.array(self.data_capture_delays))
                data_fps = self.num_data_captures / (time.time() - self.stream_start_time)
                print("\nAvg fft  delay: %.2fms  -- avg data delay: %.2fms" %(avg_fft_delay, avg_data_capture_delay))
                print("Num data captures: %d (%.2ffps)-- num fft computations: %d (%.2ffps)" %(self.num_data_captures, data_fps, self.num_ffts, self.fft_fps))

            if self.visualize and self.visualizer._is_running:
                self.visualizer.update()

        return self.fftx, self.fft, self.frequency_bin_centres, self.frequency_bin_energies

    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        if self.verbose:
            start = time.time()

        self.data_buffer.append_data(np.frombuffer(in_data, dtype=np.int16))
        self.features_up_to_date = False

        if self.verbose:
            self.num_data_captures += 1
            self.data_capture_delays.append(time.time() - start)

        return in_data, pyaudio.paContinue

    def stream_start(self):
        print("\n--🎙  -- Starting live audio stream...\n")
        self.Recording = True
        self.stream.start_stream()

        if self.blocking:
            self.stream_thread_new()

    def terminate(self):
        print("👋  Sending stream termination command...")
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def valid_low_rate(self, device, test_rates = [44100, 22050]):
        """Set the rate to the lowest supported audio rate."""
        for testrate in test_rates:
            if self.test_device(device, rate=testrate):
                return testrate

        #If none of the test_rates worked, try the default rate:
        self.info = self.pa.get_device_info_by_index(device)
        default_rate = int(self.info["defaultSampleRate"])

        if self.test_device(device, rate=default_rate):
            return default_rate
            
        print("SOMETHING'S WRONG! I can't figure out a good sample-rate for DEVICE =>", device)
        return default_rate

    def test_device(self, device, rate=None):
        """given a device ID and a rate, return True/False if it's valid."""
        try:
            self.info = self.pa.get_device_info_by_index(device)
            if not self.info["maxInputChannels"] > 0:
                return False

            if rate is None:
                rate = int(self.info["defaultSampleRate"])

            stream = self.pa.open(
                format = pyaudio.paInt16,
                channels = 1,
                input_device_index=device,
                frames_per_buffer=self.update_window_n_frames,
                rate = rate, 
                input = True)
            stream.close()
            return True
        except Exception as e:
            #print(e)
            return False

    def input_device(self):
        """
        See which devices can be opened for microphone input.
        Return the first valid device
        """
        mics=[]
        for device in range(self.pa.get_device_count()):
            if self.test_device(device):
                mics.append(device)

        if len(mics) == 0:
            print("No working microphone devices found!")
            sys.exit()

        print("Found %d working microphone device(s): " % len(mics))
        for mic in mics:
            self.print_mic_info(mic)

        return mics[0]

    def print_mic_info(self, mic):
        mic_info = self.pa.get_device_info_by_index(mic)
        print('\nMIC %s:' %(str(mic)))
        for k, v in sorted(mic_info.items()):
            print("%s: %s" %(k, v))