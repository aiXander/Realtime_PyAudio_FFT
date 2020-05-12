import time
from src.stream_analyzer import Audio_Analyzer

ear = Audio_Analyzer(
                    device = None, 
                    rate = None,
                    FFT_window_size_ms  = 60,     # Window size used for the FFT transform
                    updates_per_second  = 1000,   # How often to read the audio stream for new data
                    smoothing_length_ms = 100,    # Apply some temporal smoothing to reduce noisy features
                    n_frequency_bins    = 200,
                    visualize = 1,                # Visualize the FFT features with PyGame
                    plot_audio_history = 1,       # Plot a fading history trace of the FFT signal
                    verbose   = 0                 # Print running statistics (latency, fps, ...)
                )


fps = 75   #How often to update the FFT features + display
last_update = 0
while True:
    if (time.time() - last_update) > (1./fps):
        last_update = time.time()
        raw_fftx, raw_fft, binned_frequencies, binned_energies = ear.get_audio_features()
        

        

