import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
from matplotlib.animation import FuncAnimation
import threading
import scipy.fft as sp_fft

# Optional FFTW import
try:
    import pyfftw
    FFTW_AVAILABLE = True
except ImportError:
    FFTW_AVAILABLE = False

# ==== Configuration ====
CHUNK = 1024
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16

# ==== Buffer / State ====
BUFFER_SECONDS = 2.0                     # how many seconds to keep in the circular buffer
BUFFER_LEN = int(RATE * BUFFER_SECONDS)  # buffer size in samples
ring_buffer = np.zeros(BUFFER_LEN, dtype=np.int16)
write_ptr = 0

running = False
selected_device_index = None
selected_fft = "NumPy FFT"
fft_selector_visible = True
amplitude_scale = 10 ** -5  # FFT amplitude scale (log slider controls this)
input_sensitivity = 1.0     # input gain
time_window_s = CHUNK / RATE  # default time window (seconds) shown in time plot
lock = threading.Lock()

# ==== Audio Handling ====
p = pyaudio.PyAudio()

def list_input_devices():
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            devices.append((i, info['name']))
    return devices

def audio_thread_func():
    global write_ptr, ring_buffer, running
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=selected_device_index,
                    frames_per_buffer=CHUNK)
    while running:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        with lock:
            n = len(data)
            end = write_ptr + n
            if end <= BUFFER_LEN:
                ring_buffer[write_ptr:end] = data
            else:
                first = BUFFER_LEN - write_ptr
                ring_buffer[write_ptr:] = data[:first]
                ring_buffer[:end % BUFFER_LEN] = data[first:]
            write_ptr = end % BUFFER_LEN
    stream.stop_stream()
    stream.close()

# ==== Plot Setup ====
fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 6))
plt.subplots_adjust(left=0.32, bottom=0.33, hspace=0.5)  # room for sliders

# --- Time-domain plot (init empty) ---
line_time, = ax_time.plot([], [])
ax_time.set_ylim([-32768, 32767])
ax_time.set_title("Live Audio Input (Time Domain)")
ax_time.set_xlabel("Time [s]")
ax_time.set_ylabel("Amplitude")

# --- Frequency-domain plot (init empty) ---
line_freq, = ax_freq.plot([], [])
ax_freq.set_xlim([0, RATE / 2])
ax_freq.set_ylim([-120, 0])
ax_freq.set_title("Frequency Spectrum")
ax_freq.set_xlabel("Frequency [Hz]")
ax_freq.set_ylabel("Magnitude [dB]")

# ==== Controls ====
ax_button = plt.axes([0.4, 0.05, 0.2, 0.05])
button = Button(ax_button, 'Start')

devices = list_input_devices()
device_names_full = [name for _, name in devices]
device_names = [name[:40] + "..." if len(name) > 25 else name for name in device_names_full]
device_indices = [i for i, _ in devices]

ax_radio = plt.axes([0.05, 0.35, 0.20, 0.6])
radio = RadioButtons(ax_radio, device_names, activecolor='blue')
for lbl in radio.labels:
    lbl.set_fontsize(8)

# ==== FFT Selection ====
fft_options = ["NumPy FFT", "SciPy FFT"]
if FFTW_AVAILABLE:
    fft_options.append("PyFFTW FFT")

ax_fft_button = plt.axes([0.05, 0.18, 0.20, 0.07])
fft_button = Button(ax_fft_button, f"FFT: {selected_fft}")

ax_fft_radio = plt.axes([0.05, 0.05, 0.20, 0.1])
fft_radio = RadioButtons(ax_fft_radio, fft_options, activecolor='blue')
for lbl in fft_radio.labels:
    lbl.set_fontsize(8)
ax_fft_radio.set_visible(False)

def toggle_fft_menu(event):
    global fft_selector_visible
    fft_selector_visible = not fft_selector_visible
    ax_fft_radio.set_visible(fft_selector_visible)
    plt.draw()

def select_fft_method(label):
    global selected_fft, fft_selector_visible
    selected_fft = label
    fft_button.label.set_text(f"FFT: {selected_fft}")
    fft_selector_visible = False
    ax_fft_radio.set_visible(False)
    plt.draw()
    print(f"Selected FFT method: {selected_fft}")

fft_button.on_clicked(toggle_fft_menu)
fft_radio.on_clicked(select_fft_method)

# ==== Sliders ====
# FFT amplitude (log)
ax_slider_fftamp = plt.axes([0.4, 0.24, 0.3, 0.03])
slider_fftamp = Slider(ax_slider_fftamp, "FFT Amp (log10)", -6, -2, valinit=np.log10(amplitude_scale), valstep=0.01)

# Input sensitivity (gain)
ax_slider_gain = plt.axes([0.4, 0.19, 0.3, 0.03])
slider_gain = Slider(ax_slider_gain, "Input Sensitivity", 0.001, 2.0, valinit=input_sensitivity, valstep=0.00001)

# Time window (seconds)
ax_slider_time = plt.axes([0.4, 0.14, 0.3, 0.03])
slider_time = Slider(ax_slider_time, "Time Window (s)", 0.0001, min(0.2, BUFFER_SECONDS), valinit=CHUNK / RATE, valstep=0.0001)

# ==== Metrics Display ====
thd_text = ax_freq.text(0.02, 0.9, "", transform=ax_freq.transAxes, fontsize=10, color='green')
duty_text = ax_time.text(0.02, 0.9, "", transform=ax_time.transAxes, fontsize=10, color='purple')

# ==== Slider callbacks ====
def update_amplitude(val):
    global amplitude_scale
    amplitude_scale = 10 ** slider_fftamp.val
    # no plt.draw() needed; update happens in animation
    # print for debugging
    print(f"FFT amplitude scale set to: {amplitude_scale:.6e}")

def update_gain(val):
    global input_sensitivity
    input_sensitivity = slider_gain.val
    print(f"Input sensitivity set to: {input_sensitivity:.2f}x")

def update_time_window(val):
    global time_window_s
    time_window_s = float(val)
    print(f"Time window set to: {time_window_s:.4f} s")

slider_fftamp.on_changed(update_amplitude)
slider_gain.on_changed(update_gain)
slider_time.on_changed(update_time_window)

# ==== Control Functions ====
def toggle_run(event):
    global running, audio_thread
    if not running:
        if selected_device_index is None:
            print("Please select an input device first.")
            return
        running = True
        button.label.set_text("Stop")
        audio_thread = threading.Thread(target=audio_thread_func, daemon=True)
        audio_thread.start()
    else:
        running = False
        button.label.set_text("Start")

def select_device(label):
    global selected_device_index
    idx = device_names.index(label)
    selected_device_index = device_indices[idx]
    print(f"Selected input device: {device_names_full[idx]}")

button.on_clicked(toggle_run)
radio.on_clicked(select_device)

# ==== Helper metrics/functions ====
def calculate_thd_from_fft(fft_mag, fund_idx):
    # fft_mag corresponds to magnitudes (not dB)
    if fund_idx <= 0 or fund_idx >= len(fft_mag):
        return 0.0
    fund_mag = fft_mag[fund_idx]
    # harmonics starting from 2*fund_idx (approx)
    harmonics = fft_mag[2*fund_idx:] if 2*fund_idx < len(fft_mag) else np.array([])
    thd = np.sqrt(np.sum(harmonics**2)) / fund_mag * 100 if fund_mag > 0 else 0.0
    return thd

def calculate_duty_cycle(signal):
    sig = signal - np.mean(signal)
    peak = np.max(np.abs(sig)) + 1e-12
    if peak == 0:
        return None
    sig = sig / peak
    zero_crossings = np.where(np.diff(np.sign(sig)))[0]
    if len(zero_crossings) < 2:
        return None
    high = np.sum(sig > 0)
    duty_cycle = (high / len(sig)) * 100
    return duty_cycle

# ==== Animation update ====
def update(frame):
    global ring_buffer, write_ptr
    try:
        with lock:
            buf_copy = ring_buffer.copy()
            ptr = write_ptr

        # determine how many samples to show
        window_samples = int(np.clip(time_window_s * RATE, 1, BUFFER_LEN))

        # extract last `window_samples` from circular buffer
        if ptr >= window_samples:
            window = buf_copy[ptr - window_samples:ptr]
        else:
            part1 = buf_copy[BUFFER_LEN - (window_samples - ptr):]
            part2 = buf_copy[:ptr]
            window = np.concatenate((part1, part2))

        # convert to float and apply input sensitivity
        y = window.astype(np.float32) * input_sensitivity

        # time axis (seconds): show from -time_window_s .. 0
        x = np.linspace(-window_samples / RATE, 0.0, window_samples)

        # update time plot
        line_time.set_data(x, y)
        ax_time.set_xlim(x[0], x[-1])

        # optional auto vertical scaling (comment/uncomment)
        # ax_time.set_ylim(np.min(y) * 1.1, np.max(y) * 1.1)

        # --- FFT using a sensible nfft (power of two >= max(CHUNK, window_samples)) ---
        nfft = int(2 ** np.ceil(np.log2(max(CHUNK, window_samples))))
        # Window the signal to reduce leakage
        win = np.hanning(window_samples)
        padded = (y * win)
        # If nfft > window_samples pad with zeros automatically in rfft via `n` argument
        if selected_fft == "SciPy FFT":
            fft_vals = sp_fft.rfft(padded, n=nfft)
        elif selected_fft == "PyFFTW FFT" and FFTW_AVAILABLE:
            fft_vals = pyfftw.interfaces.numpy_fft.rfft(padded, n=nfft)
        else:
            fft_vals = np.fft.rfft(padded, n=nfft)

        mag = np.abs(fft_vals) / (window_samples if window_samples > 0 else 1)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / RATE)

        mag_db = 20 * np.log10(np.maximum(mag * amplitude_scale, 1e-10))
        line_freq.set_data(freqs, mag_db)
        ax_freq.set_xlim(0, RATE / 2)

        # THD estimate (use mag, find fundamental)
        mag_copy = mag.copy()
        if len(mag_copy) > 1:
            mag_copy[0] = 0
            fund_idx = np.argmax(mag_copy)
            thd_val = calculate_thd_from_fft(mag_copy, fund_idx)
        else:
            thd_val = 0.0
        thd_text.set_text(f"THD: {thd_val:.2f} %")

        # Duty cycle (on current window)
        duty = calculate_duty_cycle(y)
        duty_text.set_text(f"Duty Cycle: {duty:.1f} %" if duty is not None else "")

        # return artists for animation (blit=False, but still okay)
        return line_time, line_freq, thd_text, duty_text
    except Exception as e:
        # catch exceptions so animation doesn't crash silently
        print("Update error:", e)
        return line_time, line_freq, thd_text, duty_text

# Use blit=False for dynamic-length lines for stability
ani = FuncAnimation(fig, update, interval=30, blit=False)

# ==== Cleanup ====
def on_close(event):
    global running
    running = False
    p.terminate()

fig.canvas.mpl_connect('close_event', on_close)
plt.show()
