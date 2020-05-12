# Realtime_PyAudio_FFT
A simple package to do realtime audio analysis in Python, using PyAudio and Numpy to extract and visualize FFT features from streaming audio.

The basic pipeline:
- Starts a PyAudio stream that pulls live audio data from any source (soundcard, microphone, ...)
- Read data from this stream many times per second (eg 1000 updates per second) and store that data in a fifo buffer
- When triggered, applies FFT to the latest audio segment
