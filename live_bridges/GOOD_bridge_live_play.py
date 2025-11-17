"""
ECG -> MIDI bridge.
- Reads ECG from an LSL stream (type 'ECG' preferably)
- Detects R-peaks with a real-time detection algorithm (bandpass + adaptive threshold + peak search)
- Sends MIDI note_on / note_off messages to a specified loopMIDI port (Ableton reads that port)
- Optionally sends BPM as a MIDI CC
"""

import argparse
import logging
import threading
import time
from collections import deque

import numpy as np
from pylsl import resolve_streams, StreamInlet, resolve_byprop
from scipy.signal import butter, filtfilt, find_peaks
import mido

# -------------------------
# Configuration / Defaults
# -------------------------
available_ports = mido.get_output_names()
print("Available MIDI output ports:", available_ports)

DEFAULT_MIDI_PORT = "ECG_MIDI 1"
DEFAULT_NOTE = 60            # MIDI note (C4)
DEFAULT_VELOCITY = 100
NOTE_DURATION_MS = 100       # duration of each triggered note
MIN_BPM = 30
MAX_BPM = 220
REFRACTORY_PERIOD_SEC = 0.25  # minimum spacing between R-peaks



# -------------------------
# Filtering
# -------------------------
def bandpass_filter(signal, fs, low=5.0, high=30.0, order=2):
    """
    Butterworth bandpass filter used for QRS emphasis.
    """
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    lowcut = max(lowcut, 1e-6)
    highcut = min(highcut, 0.999999)
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, signal)

# -------------------------
# MIDI
# -------------------------
class MidiOut:
    def __init__(self, port_name):
        try:
            self.outport = mido.open_output(port_name)
        except IOError:
            raise RuntimeError(f"Could not open MIDI output '{port_name}'. Check loopMIDI port name.")

    def note_on(self, note, velocity):
        self.outport.send(mido.Message('note_on', note=int(note), velocity=int(velocity)))

    def note_off(self, note):
        self.outport.send(mido.Message('note_off', note=int(note)))

    def cc(self, control, value):
        self.outport.send(mido.Message('control_change', control=int(control), value=int(value)))

# -------------------------
# R-peak detector
# -------------------------
class RealTimeRPeakDetector:
    def __init__(self, fs, window_sec=6.0):
        self.fs = fs
        self.window_len = int(window_sec * fs)
        self.buffer = deque(maxlen=self.window_len)
        self.time_buffer = deque(maxlen=self.window_len)
        self.last_peak_time = -999.0

    def add(self, sample, timestamp):
        """
        Add one sample. Returns:
        (is_peak, peak_time, snr, filtered_array, peak_index)
        """
        self.buffer.append(float(sample))
        self.time_buffer.append(float(timestamp))

        if len(self.buffer) < int(0.6 * self.window_len):
            return False, None, None, None, None

        buf = np.array(self.buffer)
        tbuf = np.array(self.time_buffer)

        # Filtering
        try:
            filtered = bandpass_filter(buf, self.fs)
        except:
            filtered = buf - np.mean(buf)

        mean = np.mean(filtered)
        std = np.std(filtered)
        threshold = mean + 0.9 * std

        min_distance = int(0.25 * self.fs)
        peaks, props = find_peaks(
            filtered,
            height=threshold,
            distance=min_distance,
            prominence=0.3 * std if std > 0 else None
        )

        if len(peaks) == 0:
            return False, None, None, filtered, None

        peak_idx = peaks[-1]
        peak_time = tbuf[peak_idx]

        # Refractory
        if peak_time - self.last_peak_time < REFRACTORY_PERIOD_SEC:
            return False, None, None, filtered, peak_idx

        # Reject peaks too far in the past window
        if peak_idx < int(0.3 * len(filtered)):
            return False, None, None, filtered, peak_idx

        self.last_peak_time = peak_time

        snr = (filtered[peak_idx] - mean) / (std + 1e-9)

        return True, peak_time, snr, filtered, peak_idx

# -------------------------
# BPM helpers
# -------------------------
def bpm_from_rr(rr):
    if rr is None or rr <= 0:
        return None
    return 60.0 / rr

def map_bpm_to_note(bpm, base_note=60, span=24, lo=MIN_BPM, hi=MAX_BPM):
    bpm = np.clip(bpm, lo, hi)
    frac = (bpm - lo) / (hi - lo)
    note = int(base_note - span//2 + frac * span)
    return np.clip(note, 0, 127)

def map_bpm_to_cc(bpm, lo=MIN_BPM, hi=MAX_BPM):
    bpm = np.clip(bpm, lo, hi)
    frac = (bpm - lo) / (hi - lo)
    return int(round(frac * 127))

# -------------------------
# Main loop
# -------------------------
def run_bridge(args):
    logging.info("Resolving LSL ECG stream...")

    # 1. Try resolve_byprop first
    streams = resolve_byprop('type', 'ECG', timeout=3.0)

    # 2. Fallback: list all & find ECG-ish names
    if not streams:
        all_streams = resolve_streams()  # no timeout arg allowed in older pylsl
        streams = [s for s in all_streams if 'ECG' in (s.name() or '')]

    # 3. Final fallback: just use first stream
    if not streams:
        all_streams = resolve_streams()
        if not all_streams:
            raise RuntimeError("No LSL streams found at all. Start OpenSignals/LabRecorder.")
        streams = [all_streams[0]]

    info = streams[0]

    logging.info(
        f"Using LSL stream '{info.name()}', type='{info.type()}', "
        f"channels={info.channel_count()}, fs={info.nominal_srate()}"
    )

    inlet = StreamInlet(info, max_chunklen=1024)

    fs = info.nominal_srate()
    if not fs or fs == 0 or np.isnan(fs):
        logging.warning("Stream has no nominal sampling rate; estimating from timestamps...")

        timestamps = []
        for _ in range(60):
            s, t = inlet.pull_sample(timeout=1.0)
            if s is not None:
                timestamps.append(t)

        if len(timestamps) >= 2:
            diffs = np.diff(timestamps)
            fs = 1.0 / np.median(diffs)
            logging.info(f"Estimated fs = {fs:.1f} Hz")
        else:
            fs = 250.0
            logging.warning("Failed to estimate fs; defaulting to 250 Hz")

    detector = RealTimeRPeakDetector(fs)
    midi = MidiOut(args.midi_port)

    last_peak_time = None

    def send_note(note):
        midi.note_on(note, args.velocity)
        threading.Timer(
            args.note_duration_ms / 1000.0,
            lambda: midi.note_off(note)
        ).start()

    logging.info("Bridge running. Press Ctrl+C to stop.")

    try:
        while True:
            sample, ts = inlet.pull_sample(timeout=1.0)
            if sample is None:
                continue

            # Get chosen channel
            x = sample[args.channel]

            hit, pk_time, snr, filt, idx = detector.add(x, ts)

            if hit:
                if last_peak_time is None:
                    rr = None
                    bpm = None
                else:
                    rr = pk_time - last_peak_time
                    bpm = bpm_from_rr(rr)

                last_peak_time = pk_time

                # choose pitch
                if args.map_by_bpm and bpm is not None:
                    note = map_bpm_to_note(
                        bpm,
                        base_note=args.base_note,
                        span=args.scale_span
                    )
                else:
                    note = args.note

                send_note(note)

                if args.send_bpm_cc and bpm is not None:
                    midi.cc(args.bpm_cc, map_bpm_to_cc(bpm))

                if bpm:
                    logging.info(f"R-peak! BPM={bpm:.1f} note={note} snr={snr:.2f}")
                else:
                    logging.info(f"R-peak! note={note} snr={snr:.2f}")

    except KeyboardInterrupt:
        logging.info("Stopping bridge...")

# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ECG → MIDI bridge")
    p.add_argument("--midi-port", default=DEFAULT_MIDI_PORT)
    p.add_argument("--note", type=int, default=DEFAULT_NOTE)
    p.add_argument("--velocity", type=int, default=DEFAULT_VELOCITY)
    p.add_argument("--note-duration-ms", type=int, default=NOTE_DURATION_MS)
    p.add_argument("--map-by-bpm", action="store_true")
    p.add_argument("--base-note", type=int, default=60)
    p.add_argument("--scale-span", type=int, default=24)
    p.add_argument("--send-bpm-cc", action="store_true")
    p.add_argument("--bpm-cc", type=int, default=1)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    run_bridge(args)
