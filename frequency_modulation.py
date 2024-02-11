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

#array_decale = np.array([y[1:len(y)]]).reshape(len(y)-1, 0)
#print(array_decale.shape)

print(y.shape)
#np.insert(array_decale,len(array_decale)-1, 2*y[len(y)-1])

 
print("y : ", y)
#print("array_decale : ", array_decale)

#derive_neurone = abs(y-array_decale)
#plt.plot(y)
#plt.show()

##print(derive_neurone.shape)
#print("Y shape ",y.shape)

#print("decale shape ",array_decale.shape)

#print(derive_neurone)
#write('derive.wav', 16000, derive_neurone)

derive_neurone = []
for i in range(1, len(y)):
    derive_neurone.append(abs(y[i] - y[i-1]))

plt.plot(derive_neurone)
plt.show()
write('derive.wav', 16000, np.array(derive_neurone))


