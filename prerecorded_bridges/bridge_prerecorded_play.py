import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import mido
import time
import os

# -------------------------------
# 1. Set MIDI output port to ECG_MIDI 3
# -------------------------------
midi_port_name = "ECG_MIDI 1"

available_ports = mido.get_output_names()
print("Available MIDI output ports:", available_ports)

if midi_port_name not in available_ports:
    raise RuntimeError(f"MIDI port '{midi_port_name}' not found. Make sure loopMIDI is running.")

outport = mido.open_output(midi_port_name)
print(f"Opened MIDI output: {midi_port_name}")

# -------------------------------
# 2. Load ECG CSV
# -------------------------------
csv_file = 'realistic_ecg.csv'
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file '{csv_file}' not found.")

df = pd.read_csv(csv_file)
print("CSV columns detected:", df.columns.tolist())

fs = 1000  # sampling rate
time_col = df['Time_s'].values if 'Time_s' in df.columns else np.arange(len(df)) / fs
ecg_signal = df['ECG'].values if 'ECG' in df.columns else df.iloc[:, 0].values

# -------------------------------
# 3. Detect R-peaks
# -------------------------------
max_bpm = 200
min_rr_seconds = 60 / max_bpm
distance_samples = int(min_rr_seconds * fs)

r_peaks, _ = find_peaks(ecg_signal, height=0.8, distance=distance_samples)

if len(r_peaks) < 2:
    raise ValueError("Not enough R-peaks detected. Try adjusting the threshold.")

peak_times = time_col[r_peaks]
rr_intervals = np.diff(peak_times)
print(f"Detected {len(peak_times)} R-peaks with amplitude > 0.8 and minimum spacing {min_rr_seconds}s.")

# -------------------------------
# 4. Smooth RR intervals with moving median
# -------------------------------
def moving_median(data, window_size=3):
    padded = np.pad(data, (window_size//2, window_size-1-window_size//2), mode='edge')
    return np.array([np.median(padded[i:i+window_size]) for i in range(len(data))])

smoothed_rr = moving_median(rr_intervals, window_size=3)
smoothed_bpm = 60 / smoothed_rr
min_bpm, max_bpm = np.min(smoothed_bpm), np.max(smoothed_bpm)

# -------------------------------
# 5. Define note and velocity mapping
# -------------------------------
min_note = 48  # C3
max_note = 84  # C6
min_velocity = 50
max_velocity = 70

def bpm_to_note_quantized(bpm):
    bpm = np.clip(bpm, min_bpm, max_bpm)
    norm = (bpm - min_bpm) / (max_bpm - min_bpm)
    return int(round(min_note + norm * (max_note - min_note)))

def bpm_to_velocity(bpm):
    bpm = np.clip(bpm, min_bpm, max_bpm)
    norm = (bpm - min_bpm) / (max_bpm - min_bpm)
    return int(min_velocity + (max_velocity - min_velocity) * np.sqrt(norm))

# -------------------------------
# 6. Send MIDI notes to Ableton (legato style)
# -------------------------------
current_note = None
last_bpm = None
bpm_change_threshold = 1
note_glide_speed = 0.2
midi_channel = 0  # Python channel 0 = MIDI channel 1

for i, peak_time in enumerate(peak_times):
    bpm_value = smoothed_bpm[i] if i < len(smoothed_rr) else smoothed_bpm[-1]
    rr_interval = smoothed_rr[i] if i < len(smoothed_rr) else smoothed_rr[-1]

    if last_bpm is None or abs(bpm_value - last_bpm) >= bpm_change_threshold:
        target_note = bpm_to_note_quantized(bpm_value)
        velocity = bpm_to_velocity(bpm_value)

        if current_note is None:
            current_note = target_note
            # First note, just send note_on
            msg_on = mido.Message('note_on', note=current_note, velocity=velocity, channel=midi_channel)
            outport.send(msg_on)
            print(f"Sent first note_on: {msg_on} at {peak_time:.3f}s")
        else:
            # Turn off previous note before sending the new one
            msg_off = mido.Message('note_off', note=current_note, velocity=0, channel=midi_channel)
            outport.send(msg_off)
            print(f"Sent note_off: {msg_off} at {peak_time:.3f}s")

            # Glide calculation
            note_distance = target_note - current_note
            glide_step = int(round(note_distance * note_glide_speed))
            if glide_step == 0 and note_distance != 0:
                glide_step = np.sign(note_distance)
            note_to_play = current_note + glide_step
            current_note = note_to_play

            # Send new note_on
            msg_on = mido.Message('note_on', note=current_note, velocity=velocity, channel=midi_channel)
            outport.send(msg_on)
            print(f"Sent note_on: {msg_on} at {peak_time:.3f}s")

        last_bpm = bpm_value

    # Maintain timing between peaks
    time.sleep(rr_interval)

# Optional: turn off the last note at the end
if current_note is not None:
    msg_off = mido.Message('note_off', note=current_note, velocity=0, channel=midi_channel)
    outport.send(msg_off)
    print(f"Sent final note_off: {msg_off}")

print("Finished sending legato MIDI notes from ECG.")