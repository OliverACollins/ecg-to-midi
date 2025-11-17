"""
ECG -> MIDI CC113 modulation bridge with LIVE PLOT.
- Detects R-peaks from ECG LSL stream
- Computes SNR/amplitude around each peak
- Smooths & adaptively normalizes the modulation signal
- Sends MIDI CC113 continuously
- Live plot shows smoothed value and CC output
"""

import argparse
import logging
import time
from collections import deque

import numpy as np
import mido
from pylsl import resolve_streams, StreamInlet, resolve_byprop
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

# ---------------------------------------
# CONFIG
# ---------------------------------------
DEFAULT_MIDI_PORT = "ECG_MIDI 1"
MOD_CC = 113

SMOOTH_FACTOR = 0.3
ROLLING_NORM_SEC = 5.0
SEND_INTERVAL = 0.02          # 50 Hz update rate

MIN_CC = 0
MAX_CC = 127
REFRACTORY_PERIOD_SEC = 0.25

PLOT_LENGTH = 300             # sliding window for live plot


# ---------------------------------------
# FILTERING
# ---------------------------------------
def bandpass_filter(signal, fs, low=5.0, high=30.0, order=2):
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, signal)


# ---------------------------------------
# MIDI
# ---------------------------------------
class MidiOut:
    def __init__(self, port_name):
        self.outport = mido.open_output(port_name)

    def cc(self, control, value):
        self.outport.send(
            mido.Message('control_change', control=control, value=int(value))
        )


# ---------------------------------------
# R-PEAK DETECTOR
# ---------------------------------------
class RealTimeRPeakDetector:
    def __init__(self, fs, window_sec=6.0):
        self.fs = fs
        self.window_len = int(window_sec * fs)
        self.buffer = deque(maxlen=self.window_len)
        self.time_buffer = deque(maxlen=self.window_len)
        self.last_peak_time = -999

    def add(self, sample, timestamp):
        self.buffer.append(float(sample))
        self.time_buffer.append(float(timestamp))

        if len(self.buffer) < int(0.6 * self.window_len):
            return False, None, None, None, None

        buf = np.array(self.buffer)
        tbuf = np.array(self.time_buffer)

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

        if peak_idx < int(0.3 * len(filtered)):
            return False, None, None, filtered, peak_idx

        self.last_peak_time = peak_time
        snr = (filtered[peak_idx] - mean) / (std + 1e-9)

        return True, peak_time, snr, filtered, peak_idx


# ---------------------------------------
# MAIN LOOP
# ---------------------------------------
def run_bridge(args):
    logging.info("Resolving LSL ECG stream...")

    streams = resolve_byprop('type', 'ECG', timeout=3)
    if not streams:
        all_streams = resolve_streams()
        streams = [s for s in all_streams if "ECG" in (s.name() or "")]

    if not streams:
        raise RuntimeError("No ECG stream found.")

    info = streams[0]
    inlet = StreamInlet(info, max_chunklen=1024)

    fs = info.nominal_srate()
    if not fs or fs == 0 or np.isnan(fs):
        fs = 250.0
        logging.warning("Stream missing nominal rate; defaulting to fs=250Hz")

    detector = RealTimeRPeakDetector(fs)
    midi = MidiOut(args.midi_port)

    # Adaptive normalization
    smoothed_value = None
    history = deque(maxlen=int(ROLLING_NORM_SEC * fs))
    last_send_time = 0

    # ---------------------------------------
    # LIVE PLOT SETUP
    # ---------------------------------------
    plt.ion()
    fig, ax = plt.subplots()
    line_mod, = ax.plot([], [], label="Smoothed Value")
    line_cc, = ax.plot([], [], label="CC113 Output")

    ax.set_ylim(0, 130)
    ax.set_xlim(0, PLOT_LENGTH)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Value")
    ax.legend()

    mod_data = []
    cc_data = []

    logging.info("ECG → MIDI CC113 modulation running... (Ctrl+C to stop)")

    try:
        while True:
            sample, ts = inlet.pull_sample(timeout=1.0)
            if sample is None:
                continue

            x = sample[args.channel]
            hit, pk_time, snr, filtered, idx = detector.add(x, ts)

            if hit:
                raw_value = snr if snr else 0.0

                # Smooth
                if smoothed_value is None:
                    smoothed_value = raw_value
                else:
                    smoothed_value = (
                        SMOOTH_FACTOR * raw_value
                        + (1 - SMOOTH_FACTOR) * smoothed_value
                    )

                # Adaptive normalize
                history.append(smoothed_value)
                vmin = min(history)
                vmax = max(history)
                norm = (smoothed_value - vmin) / max(1e-9, (vmax - vmin))
                norm = float(np.clip(norm, 0, 1))

                cc_value = int(MIN_CC + norm * (MAX_CC - MIN_CC))

                # Rate limit & send CC113
                now = time.time()
                if now - last_send_time >= SEND_INTERVAL:
                    midi.cc(MOD_CC, cc_value)
                    last_send_time = now

                # -----------------------------
                # UPDATE PLOT
                # -----------------------------
                mod_data.append(smoothed_value)
                cc_data.append(cc_value)

                if len(mod_data) > PLOT_LENGTH:
                    mod_data = mod_data[-PLOT_LENGTH:]
                    cc_data = cc_data[-PLOT_LENGTH:]

                line_mod.set_ydata(mod_data)
                line_mod.set_xdata(range(len(mod_data)))

                line_cc.set_ydata(cc_data)
                line_cc.set_xdata(range(len(cc_data)))

                ax.set_xlim(0, max(PLOT_LENGTH, len(mod_data)))
                fig.canvas.draw()
                fig.canvas.flush_events()

                logging.info(f"Peak → CC113={cc_value} (raw={raw_value:.3f})")

            time.sleep(0.001)

    except KeyboardInterrupt:
        logging.info("Stopping bridge...")
        plt.ioff()
        plt.close()


# ---------------------------------------
# ARGS
# ---------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ECG → MIDI CC113 modulation bridge w/ live plot")
    p.add_argument("--midi-port", default=DEFAULT_MIDI_PORT)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_bridge(args)
