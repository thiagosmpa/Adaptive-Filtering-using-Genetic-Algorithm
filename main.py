#7 Backup

import scipy
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import numpy as np
from scipy.fft import fft, ifft,fftfreq, fftshift
from scipy import signal
import lms
import training as af
import pickle

# =============================================================================
# Generating an ECG signal
# =============================================================================

rr = [1.0, 1.0, 0.5, 1.5, 1.0, 1.0] # rr time in seconds
fs = 800.0 # sampling rate
pqrst = sig.wavelets.daub(10) # just to simulate a signal, whatever

ecg = scipy.concatenate([sig.resample(pqrst, int(r*fs)) for r in rr])
# ecg = ecg[:2000]
ecg = ecg*10
t = scipy.arange(len(ecg))/fs








# =============================================================================
# Generating a low band noise 
# =============================================================================

time = np.arange(0, 6, 0.00125)

# Finding noise at each time
noise = 3*np.sin(time)





# =============================================================================
# Corrupting with the low band noise
# =============================================================================

x = ecg + noise





# =============================================================================
# Fourier Transform
# =============================================================================

freqs = fftfreq(4800)
mask = freqs > 0
fft_vals = fft(x)       #Com ruido
fft_vals2 = fft(ecg)    #Sem ruido

# plt.figure(2)
# plt.plot(freqs, fft_vals)
# plt.title('FFT of X (With Noise)')
# plt.xlim([-0.01, 0.01])
# # plt.ylim([0, 2])
# plt.show()

# plt.figure(3)
# plt.plot(freqs, fft_vals2)
# plt.xlim([-0.01, 0.01])
# plt.title('FFT of ECG (Without noise)')
# # plt.ylim([0, 2])
# plt.show()

fft_theo = 2*np.abs(fft_vals/4800)  #Com ruido
fft_theo2 = 2*np.abs(fft_vals2/4800)#Sem ruido





# =============================================================================
# Butterworth Filter In Order to Compare
# =============================================================================

# def butter_highpass(cutoff, fs, order=5):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = sig.butter(order, normal_cutoff, btype='high', analog=False)
#     return b, a

# def butter_highpass_filter(data, cutoff, fs, order=5):
#     b, a = butter_highpass(cutoff, fs, order=order)
#     yfiltered_butterworth = sig.filtfilt(b, a, data)
#     return yfiltered_butterworth

def bandPassFilter (signal):
    lowcut = 0.33
    highcut = 70
    
    nyq = 0.5*fs
    low = lowcut / nyq
    high = highcut / nyq
    
    order = 2
    
    b, a = scipy.signal.butter(order, [low, high], 'bandpass', analog=False)
    y = scipy.signal.filtfilt(b, a, signal, axis=0)
    
    return (y)

yfiltered_butterworth = bandPassFilter(x)


# =============================================================================
# Implementing the Adaptive Filter
# =============================================================================

# Loading coeffs file
# file = open('myvars.rtf', 'rb')
# dict = pickle.load(file)
# w_m = dict ['w_m']



N = len(x)                     #Numero de amostras do sinal
M = 4                                #Tamanho do filtro FIR
y_filtered_w = np.zeros(N)
w = [1, 1, 1, 1]


y_adaptive_filtered = np.zeros(N)

afilter = af.adaptiveFilt(x, ecg, w)


# Uncomment this part if you loaded the coeff's file and comment the next two lines: "run = af.Adaptive... y_adaptive_filtered..."
# x_m = afilter.matrixzation()
# for i in range (N):
#     y_adaptive_filtered[i] = np.sum(x_m[i] * w_m[i])


run = af.AdaptiveRun(x, ecg, w)
y_adaptive_filtered , w_m = run.run()

# Saving coeffs in a file
# dict = {'w_m': w_m}
# file = open('myvars.rtf', 'wb')
# pickle.dump(dict, file)
# file.close()



# =============================================================================
# Plots
# =============================================================================

# =============================================================================
# Signals
# =============================================================================

plt.figure()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
fig.suptitle('ECG Não contaminado / Contaminado / Filtrado / Adaptativo')
ax1.plot(t, ecg)

ax2.plot(t, x)

ax3.plot(t, yfiltered_butterworth)

ax4.plot(t, y_adaptive_filtered)
         
plt.show()

# =============================================================================
# FFTs
# =============================================================================

fft_vals3 = fft(yfiltered_butterworth)
fft_theo3 = 2*np.abs(fft_vals3/4800)

fft_vals4 = fft(y_adaptive_filtered)
fft_theo4 = 2*np.abs(fft_vals4/4800)

plt.figure()
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('FFTs - Não Contaminado / Contaminado / Filtrado / Adaptativo (*625Hz)')

ax1.plot(freqs[mask], fft_theo2[mask])
ax1.set_xlim(0, 0.01)
ax1.set_ylim(0, 2.5)

ax2.plot(freqs[mask], fft_theo[mask])
ax2.set_xlim(0, 0.01)
ax2.set_ylim(0, 2.5)

ax3.plot(freqs[mask], fft_theo3[mask])
ax3.set_xlim(0, 0.01)
ax3.set_ylim(0, 2.5)
plt.show()

ax4.plot(freqs[mask], fft_theo4[mask])
ax4.set_xlim(0, 0.5)
ax4.set_ylim(0, 1.2)
plt.show()




# =============================================================================
# Calculating Signal to Noise Ratio (SNR)
# =============================================================================

# SNR = P(signal)/P(noise)
# SNRdb = P(signal - db) - P(noise - db)
# SNRdb = 10*log10*(Psignal)

psignal_ecg = ecg ** 2
# plt.figure()
# plt.plot(t, psignal_ecg)
# plt.title('Psignal_ECG')
# plt.show()
psignal_medio_ecg = np.mean(psignal_ecg)

psignaldb_ecg = 10*np.log10(psignal_ecg)
# plt.figure()
# plt.plot(t, psignaldb_ecg)
# plt.title('PsignalDB_ECG')
# plt.show()
psignaldb_medio_ecg = 10*np.log10(psignal_medio_ecg)

#--------------------------

psignal_noise = noise ** 2
# plt.figure()
# plt.plot(t, psignal_noise)
# plt.title('Psignal_Noise')
# plt.show()
psignal_medio_noise = np.mean(psignal_noise)

psignaldb_noise = 10*np.log10(psignal_noise)
# plt.figure()
# plt.plot(t, psignaldb_noise)
# plt.title('PsignalDB_Noise')
# plt.show()
psignaldb_medio_noise = 10*np.log10(psignal_medio_noise)

#--------------------------

psignal_x = x ** 2
# plt.figure()
# plt.plot(t, psignal_x)
# plt.title('Psignal_X')
# plt.show()
psignal_medio_x = np.mean(psignal_x)

psignaldb_x = 10*np.log10(psignal_x)
# plt.figure()
# plt.plot(t, psignaldb_x)
# plt.title('PsignalDB_X')
# plt.show()
psignaldb_medio_x = 10*np.log10(psignal_medio_x)

#--------------------------

psignal_yfiltered_butterworth = yfiltered_butterworth ** 2
# plt.figure()
# plt.plot(t, psignal_yfiltered_butterworth)
# plt.title('Psignal_yfiltered_butterworth')
# plt.show()
psignal_medio_yfiltered_butterworth = np.mean(psignal_yfiltered_butterworth)

psignaldb_yfiltered_butterworth = 10*np.log10(psignal_yfiltered_butterworth)
# plt.figure()
# plt.plot(t, psignaldb_yfiltered_butterworth)
# plt.title('PsignalDB_yfiltered_butterworth')
# plt.show()
psignaldb_medio_yfiltered_butterworth = 10*np.log10(psignal_medio_yfiltered_butterworth)

psignal_y_adaptativo = y_adaptive_filtered ** 2
# plt.figure()
# plt.plot(t, psignal_ecg)
# plt.title('Psignal_ECG')
# plt.show()
psignal_medio_y_adaptativo = np.mean(psignal_y_adaptativo)

psignaldb_y_adaptativo = 10*np.log10(psignal_y_adaptativo)
# plt.figure()
# plt.plot(t, psignaldb_ecg)
# plt.title('PsignalDB_ECG')
# plt.show()
psignaldb_medio_y_adaptativo = 10*np.log10(psignal_medio_y_adaptativo)

# -----------------------------------------------------------------------------

snr_ecg = psignal_ecg / psignal_noise
snrdb_ecg = psignaldb_ecg / psignaldb_noise
snr_medio_ecg = np.mean(psignal_ecg) / np.mean(psignal_noise)

snr_x = psignal_x / psignal_noise
snrdb_x = psignaldb_x / psignaldb_noise
snr_medio_x = np.mean(psignal_x) / np.mean(psignal_noise)

snr_yfiltered_butterworth = psignal_yfiltered_butterworth / psignal_noise
snrdb_yfiltered_butterworth = psignaldb_yfiltered_butterworth / psignaldb_noise
snr_medio_yfiltered_butterworth = np.mean(psignal_yfiltered_butterworth) / np.mean(psignal_noise)

snr_y_adaptativo = psignal_y_adaptativo / psignal_noise
snrdb_y_adaptativo = psignaldb_y_adaptativo / psignaldb_noise
snr_medio_y_adaptativo = np.mean(psignal_y_adaptativo) / np.mean(psignal_noise)

plt.figure()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
ax1.plot(t ,snr_ecg)
ax2.plot(t, snr_x)
ax3.plot(t, snr_yfiltered_butterworth)
ax4.plot(t, snr_y_adaptativo)
fig.suptitle('SNRdb - ECG Não Contaminado / Contaminado / Filtrado / Adaptativo')
plt.show()


print ("SNR medio ECG: ", snr_medio_ecg)
print ('SNR medio X: ', snr_medio_x)
print ('SNR medio Y Filtrado (ButterWorth): ', snr_medio_yfiltered_butterworth)
print ('SNR medio Y Filtrado (Adaptativo): ', snr_medio_y_adaptativo)



plt.plot(freqs[mask], fft_theo4[mask])
plt.title('Transformada de fourier de Y usando o filtro adaptativo')
plt.show()


