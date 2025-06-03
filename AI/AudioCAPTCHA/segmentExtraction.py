import librosa
import soundfile as sf
import numpy as np

audio_file_path = 'AudioCaptchas/00249.wav'
audio_samples, sample_rate = librosa.load(audio_file_path, sr=22050)

audio_samples = librosa.effects.preemphasis(audio_samples)
amplitude_threshold = 0.35

segments = []
start_time = None

print(f"Loaded audio: {audio_file_path} | Sample rate: {sample_rate} | Duration: {len(audio_samples) / sample_rate:.2f}s")

for i, sample in enumerate(audio_samples):
    time = i / sample_rate
    if sample > amplitude_threshold and start_time is None:
        start_time = max(time - (5 / sample_rate), 0)
    
    elif sample <= amplitude_threshold and start_time is not None:
        sample_threshold = np.where(audio_samples[i:i+5] > amplitude_threshold)[0]
    if sample_threshold.size > 0:
        continue

    else:
        if segments and (time - segments[-1][1] < 0.1):
            continue

        segments.append((start_time, time + 0.03))
        start_time = None

print(f"Total segments detected: {len(segments)}")
