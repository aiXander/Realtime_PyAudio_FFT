import numpy as np

def getFFT(data, rate, chunk_size, log_scale=False):
    data = data * np.hamming(len(data))
    try:
        FFT = np.abs(np.fft.rfft(data)[1:])
    except:
        FFT = np.fft.fft(data)
        left, right = np.split(np.abs(FFT), 2)
        FFT = np.add(left, right[::-1])

    #fftx = np.fft.fftfreq(chunk_size, d=1.0/rate)
    #fftx = np.split(np.abs(fftx), 2)[0]
    
    if log_scale:
        try:
            FFT = np.multiply(20, np.log10(FFT))
        except Exception as e:
            print('Log(FFT) failed: %s' %str(e))

    return FFT


## TODO: Realtime Harmonic/Percussive decomposition

'''
from scipy import signal
def median_filter_horizontal(x, filter_len):
    return signal.medfilt(x, [1, filter_len])

def median_filter_vertical(x, filter_len):
    return signal.medfilt(x, [filter_len, 1])

def harmonic_percussive_decomposition(FFT_features, Fs):
    # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S1_HPS.html
    
    N, H = 1024, 512
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann', center=True, pad_mode='constant')
    Y = np.abs(X)**2

    L_set = np.array([[5,5],[23,9],[87,47]])
    num = L_set.shape[0]
    for m in range(num):
        L_h = L_set[m,0]
        L_p = L_set[m,1]
        Y_h = median_filter_horizontal(Y, L_h)
        Y_p = median_filter_vertical(Y, L_p)
        title_h = r'Horizontal filtering ($L^h=%d$)'%L_h
        title_p = r'Vertical filtering ($L^p=%d$)'%L_p
        plot_spectrogram_hp(Y_h, Y_p, Fs=Fs, N=N, H=H, title_h=title_h, title_p=title_p, ylim=[0, 3000], log=True)
'''