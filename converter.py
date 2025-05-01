#Converts the Hz to gp* or mid

import numpy as np
import librosa

bass_tunings = {
    #standard E for now
    4: 28, #E1
    3: 33, #A1
    2: 38, #D2
    1: 43 #G2
}

max_frets = 24 #maximum 24 frets

#load our pitches text file
def load_pitches(path="pitches.txt"): #in the final c# gui, this will be an array of sorts cached in
    return np.loadtxt(path)

def hz_to_midi(hz):
    return int(round(librosa.hz_to_midi(hz))) #rounded hz

def match_to_string_fret(midi_note): #match the hz to the string and fret
    possibility = []
    for string, open_midi in bass_tunings.items():
        fret = midi_note - open_midi
        if 0 <= fret <= max_frets:
            possibility.append((string, fret))
    return possibility or [("?","?")]   #or question mark if cant find


