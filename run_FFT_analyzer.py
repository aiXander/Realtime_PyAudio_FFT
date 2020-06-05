import time
from src.stream_analyzer import Stream_Analyzer

ear = Stream_Analyzer(
                device = None,               # Manually play with this (int) if you don't see anything
                rate   = None,               # Audio samplerate, None uses the default source settings
                FFT_window_size_ms  = 60,    # Window size used for the FFT transform
                updates_per_second  = 1000,  # How often to read the audio stream for new data
                smoothing_length_ms = 50,    # Apply some temporal smoothing to reduce noisy features
                n_frequency_bins    = 400,   # The FFT features are grouped in bins
                visualize = 1,               # Visualize the FFT features with PyGame
                verbose   = 1                # Print running statistics (latency, fps, ...)
                )

fps = 60  #How often to update the FFT features + display
last_update = time.time()
while True:
    if (time.time() - last_update) > (1./fps):
        last_update = time.time()
        raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()