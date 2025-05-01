import os
import shlex
import demucs.separate
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from transcriber import BassTranscribe


def separate_audio_with_demucs(input_path: str, output_dir: str, model="htdemucs_6s") -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # Demucs used here (CLI)
    args = f"-n {model} --out {output_dir} \"{input_path}\""
    demucs.separate.main(shlex.split(args))


    base_name = os.path.splitext(os.path.basename(input_path))[0]
    stem_folder = os.path.join(output_dir, model, base_name)

    stems = {
        "drums": os.path.join(stem_folder, "drums.wav"),
        "bass": os.path.join(stem_folder, "bass.wav"),
        "other": os.path.join(stem_folder, "other.wav"),
        "vocals": os.path.join(stem_folder, "vocals.wav"),
        "guitar": os.path.join(stem_folder, "guitar.wav"),
        "piano": os.path.join(stem_folder, "piano.wav")
    }
    return stems


#load audio
def load_audio(file_path: str):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


#detect the onsets
#def detect_onset(y,sr_):
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times

#def estimate_pitches(y, sr):
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

    # Separate stems
    stems = separate_audio_with_demucs(input_audio, output_dir)
    # Test steam to pick for transcription is bass
    bass_path = stems["bass"]
    y, sr = load_audio(bass_path)
    transcriber = BassTranscribe(y,sr)

    # Transcription logic
    onsets = transcriber.detect_onsets()
    pitches = transcriber.estimate_pitch(onsets)

    print(f"Detected {len(pitches)} onsets.")
    print("First 10 estimated pitches (Hz):", pitches[:10])

    # Optional visualisationS
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.vlines([t for t, _ in pitches], -1, 1, color='r', label='Onsets')
    plt.title("Bass Waveform with Onsets")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("onset_plot.png")
    np.savetxt("onsets.txt", onsets, fmt="%.5f")
    np.savetxt("pitches.txt", pitches, fmt="%.2f")
    plt.show()