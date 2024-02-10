import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from reservoirpy.datasets import mackey_glass, lorenz
from scamp import *
import numpy.fft as fft
from scipy.signal import find_peaks

leak_rate = 0.9

def update_lr():
    print("calledd")
    leak_rate = 1.9
    #reseau()
    
rpy.verbosity(0)  # no need to be too verbose here
rpy.set_seed(42)  # make everything reproducible !

# ======= RESERVOIR ==========

# Parameters
fs = 8000 
#fs = 44100  # Sampling rate, 44.1 kHz
duration = 2  # Duration in seconds
f_carrier = 440  # Carrier frequency in Hz (A4 note)
#f_modulator = 5  # Modulator frequency in Hz
modulation_index = 1.0  # Modulation index determines the depth of modulation

reservoir = Reservoir(100, lr=0.9, sr=0.9,input_scaling=0.3)
X = lorenz(duration*fs)
#X = mackey_glass(88200)
#X = np.sin(np.linspace(0, 6*np.pi, 88200)).reshape(-1, 1)
""" note = np.sin(np.linspace(0, 440*np.pi, 32000))
plt.plot(note)
plt.show()
write('note.wav', 16000, note) """

print(X[0].shape)
print(X[0].reshape(1, -1).shape)
#print('old',reservoir(X[0]))
s = reservoir(X[0].reshape(1, -1))
print(reservoir)

print("New state vector shape: ", s.shape)
print(X.shape)

s = reservoir.state()
print(s)

states = np.empty((len(X), reservoir.output_dim))
print("states \n", states)
for i in range(len(X)):
    states[i] = reservoir(X[i].reshape(1, -1))
    
print("states \n", states)

y = states[:, 25]
x = states[:, 60]


#========INITIALISATION DES PARAMETRES POUR LE RESERVOIR========
units = 100               # - number of neurons
leak_rate = 0.3           # - leaking rate
spectral_radius = 1.25    # - spectral radius of W
input_scaling = 1.0       # - input scaling
connectivity = 0.1        # - density of reservoir internal matrix
input_connectivity = 0.2  # and of reservoir input matrix
regularization = 1e-8     # - regularization coefficient for ridge regression
seed = 1234 

#=========INITIALISATION INPUT X ===================
nb_steps = fs*duration
X = mackey_glass(nb_steps)

# rescale between -1 and 1
X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

#========CHANGEMENT DES PARAMETRES SPECTRAL RADIUS==========
states0 = []
radii = [0.1, 1.25, 10.0]
for sr in radii:
    reservoir = Reservoir(units,
                          sr=sr,
                          input_scaling=0.1,
                          lr=leak_rate,
                          rc_connectivity=connectivity,
                          input_connectivity=input_connectivity,
                          seed=seed)

    s = reservoir.run(X[:int(nb_steps/10)])
    states0.append(s)

#======AFFICHAGE DE L ACTIVITE D UN NEURONE POUR DIFFERENT SPECTRAL RADIUS==========
units_nb = 20

plt.figure()
plt.plot(X)

plt.figure(figsize=(15, 8))

for i, s in enumerate(states0):
    plt.subplot(len(radii)*100+10+i+1)
    plt.plot(s[:, 0:100], alpha=0.6)
    plt.ylabel(f"$sr={radii[i]}$")
plt.xlabel(f"Activations ({units_nb} neurons)")
plt.show()

#======= MODULATION ========

#modulator = y
modulator = states[0][:,25]

# Time array
t = np.arange(0, duration, 1/fs)

# Carrier and Modulator signals
carrier = np.sin(2 * np.pi * f_carrier * t)  # Carrier signal
#modulator = (modulation_index * np.sin(2 * np.pi * f_modulator * t)) + 1  # Modulator signal; +1 to ensure carrier amplitude is always positive

# Amplitude Modulated signal
am_signal = carrier * modulator

# Plotting
plt.figure(figsize=(10, 8))

# Carrier Signal
plt.subplot(3, 1, 1)
plt.plot(t, carrier)
plt.title('Carrier Signal')
plt.xlim(0, 0.3)  # Limiting time axis for better visibility

# Modulator Signal
plt.subplot(3, 1, 2)
plt.plot(t, modulator)
plt.title('Modulator Signal')
plt.xlim(0,  0.3)  # Limiting time axis for better visibility

# AM Signal
plt.subplot(3, 1, 3)
plt.plot(t, am_signal)
plt.title('AM Signal')
plt.xlim(0,  0.3)  # Limiting time axis for better visibility

plt.tight_layout()
plt.show()

# Saving the AM signal as a WAV file
am_signal_normalized = np.int16((am_signal / am_signal.max()) * 32767)  # Normalize the signal
write('am_signal_neurone.wav', fs, am_signal_normalized)
    
    
""" newY = y
for i in range(1000):
    newY += y """
    
""" print(y)
plt.plot(y)
plt.show() """

""" spectrum = fft.fft(y)
freq = fft.fftfreq(len(spectrum))
plt.plot(freq, abs(spectrum))
plt.show() """

""" threshold = 0.5 * max(abs(spectrum))
mask = abs(spectrum) > threshold
peaks = freq[mask]

t = np.arange(4e6) / 1e6 # sampling times in seconds
data = np.sin(2 * np.pi * 1000 * t)

print(data)
    """
    
"""plt.plot(y)
plt.show()
# amplitudes = 2 / n_samples * np.abs()
print("amp ", find_peaks(x, height=0)) """

#write('noise.wav', len(y), y)
#write('noise.wav', 16000, y)
#print("leny = ", len(y))
