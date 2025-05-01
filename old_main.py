import os
from spleeter.separator import Separator
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


# firstly, seperate the vocal and instrumental parts of the audio file
def separate_vocals(input_path: str, output_dir: str) -> str:
    print(f"Seperating {input_path}.") #debug purposes
    seperator = Separator('spleeter:5stems')
    seperator.separate_to_file(input_path, output_dir)

    # make expected vocal path
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    vocal_path = os.path.join(output_dir, file_name, 'vocals.wav')
    return vocal_path

#get audio
def load_audio(file_path: str):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

#detect the onsets
def detect_onset(y,sr_):
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times

def estimate_pitches(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    detected_pitches = []

    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            detected_pitches.append(pitch)
    
    return detected_pitches

if __name__ == "__main__":
    input_audio = "song.mp3"
    output_dir = "output"

    vocals_path = separate_vocals(input_audio, output_dir)
    y, sr = load_audio(vocals_path)
    
    onsets = detect_onset(y, sr)
    print(f"Detected {len(onsets)} onsets.")
    
    pitches = estimate_pitches(y, sr)
    print("First 10 estimated pitches (Hz):", pitches[:10])

    # visiualisations
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.vlines(onsets, -1, 1, color='r', label='Onsets')
    plt.title("Vocal Waveform with Onsets")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("onset_plot.png")
    np.savetxt("onsets.txt", onsets, fmt="%.5f")
    np.savetxt("pitches.txt", pitches, fmt="%.2f")
    plt.show()
