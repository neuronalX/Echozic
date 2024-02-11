import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import hilbert
from scipy.io.wavfile import write
from reservoirpy.datasets import mackey_glass, lorenz
from scamp import *
import numpy.fft as fft
from scipy.signal import find_peaks
import pandas as pd
from mingus.core import chords
import librosa


#==============

chord_progression = ["Cmaj7", "Cmaj7", "Fmaj7", "Gdom7"]

NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)

errors = {
    'notes': 'Bad input, please refer this spec-\n'
}


def swap_accidentals(note):
    if note == 'Db':
        return 'C#'
    if note == 'D#':
        return 'Eb'
    if note == 'E#':
        return 'F'
    if note == 'Gb':
        return 'F#'
    if note == 'G#':
        return 'Ab'
    if note == 'A#':
        return 'Bb'
    if note == 'B#':
        return 'C'

    return note


def note_to_number(note: str, octave: int) -> int:
    note = swap_accidentals(note)
    assert note in NOTES, errors['notes']
    assert octave in OCTAVES, errors['notes']

    note = NOTES.index(note)
    note += (NOTES_IN_OCTAVE * octave)

    assert 0 <= note <= 127, errors['notes']

    return note

#======
rpy.verbosity(0)  # no need to be too verbose here
rpy.set_seed(42)  # make everything reproducible !

def IsSature(array):
    isSature = False
    
    #Seuillage
    print(max(array))
    print(min(array))
    seuil = 0.1
    #result = (y > seuil) * y
    #result = (y > seuil) * 1.7
    result = array[(array > seuil)]

    print(result)
    
    return isSature


scamp = Session()

fs = 8000 
#fs = 44100  # Sampling rate, 44.1 kHz
duration = 2  # Duration in seconds
f_carrier = 440  # Carrier frequency in Hz (A4 note)
#f_modulator = 5  # Modulator frequency in Hz
modulation_index = 1.0  # Modulation index determines the depth of modulation


# ======= RESERVOIR ==========

# 5 neurones
reservoir = Reservoir(100, lr=0.9, sr=0.9,input_scaling=0.3)
X = lorenz(duration*fs, h=0.01)
#X = np.sin(np.linspace(0, 6*np.pi,fs*duration )).reshape(-1, 1)

s = reservoir(X[0].reshape(1, -1))

s = reservoir.state()

states = np.empty((len(X), reservoir.output_dim))
for i in range(len(X)):
    states[i] = reservoir(X[i].reshape(1, -1))
    
y = states[:, 0]
x = states[:, 1]

clarinet = scamp.new_part("clarinet")

noteToPlay = librosa.note_to_midi('A')
clarinet.play_note(noteToPlay, 0.8, 50) 

for i in range(5):
    isSature = IsSature(states[:, i])
    if(isSature):
        match i:
            case 1:
                clarinet.play_note(librosa.note_to_midi('A'), 0.8, 0.5) 
            case 2:
               clarinet.play_note(librosa.note_to_midi('C'), 0.8, 0.5)
            case 3:
                clarinet.play_note(librosa.note_to_midi('E'), 0.8, 0.5)
            case 4:
                clarinet.play_note(librosa.note_to_midi('A'), 0.8, 0.5)
            case 5:
                clarinet.play_note(librosa.note_to_midi('A'), 0.8, 0.5)


