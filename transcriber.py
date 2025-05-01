#Right now only trying bass since it's simpler


import librosa
import numpy as np


class BassTranscribe:
    def __init__(self, y, sr):
        self.y = y #  Waveform 
        self.sr = sr # sample rate (Default is 22050 Hz)


#Using librosa to detect the onsets and guestimate the pitches
    def detect_onsets(self):
        frames = librosa.onset.onset_detect(y=self.y, sr=self.sr)
        return librosa.frames_to_time(frames, sr=self.sr)
    
    def estimate_pitch(self, onsets, window=0.2): #Want a window of 0.2 to balance time resolution for slightly better pitch
        pitches_list = []
        for t in onsets:
            start = int((t - window / 2 ) * self.sr)
            end = int((t + window / 2 ) * self.sr)
            y_slice = self.y[max(0, start):min(len(self.y), end)]

            pitches, mags = librosa.piptrack(y=y_slice, sr=self.sr)
            if pitches.any():
                idx = mags.argmax(axis=0)
                pitch = np.max(pitches[idx, range(pitches.shape[1])])
                if 40 < pitch < 400: #bass range - change accordingly later
                    pitches_list.append((t, pitch))



        return pitches_list           





#Next we can estimate what string and fret combo the pitch is (another class)      