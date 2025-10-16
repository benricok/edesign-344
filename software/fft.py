import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
from matplotlib.animation import FuncAnimation
import threading
import scipy.fft as sp_fft

# ==== Configuration ====
CHUNK = 1024
RATE = 44100
CHANNELS = 1

freq_presets = [100, 500, 1000, 5000, 10000, 15000]  # Hz
preset_axes = []
preset_buttons = []

# ==== Buffer / State ====
BUFFER_SECONDS = 1.0
BUFFER_LEN = int(RATE * BUFFER_SECONDS)
ring_buffer = np.zeros(BUFFER_LEN, dtype=np.float32)
write_ptr = 0

running = False
selected_device_index = None
amplitude_scale = 10 ** -0.2
input_sensitivity = 2.0
time_window_s = 0.15 #CHUNK / RATE
lock = threading.Lock()


# ==== Audio Handling ====
def list_input_devices():
    devices = sd.query_devices()
    input_devices = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            input_devices.append((i, d['name']))
    return input_devices


def audio_callback(indata, frames, time, status):
    """Called automatically by sounddevice when audio input arrives."""
    global ring_buffer, write_ptr
    if not running:
        return
    if status:
        print(status)

    data = indata[:, 0]  # single channel
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


def start_audio_stream():
    return sd.InputStream(
        samplerate=RATE,
        channels=CHANNELS,
        dtype='float32',
        device=selected_device_index,
        callback=audio_callback,
        blocksize=CHUNK
    )


# ==== Plot Setup ====
fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 6))
plt.subplots_adjust(left=0.3, bottom=0.33, hspace=0.3, right=0.97, top=0.95)

line_time, = ax_time.plot([], [])
ax_time.set_ylim([-1, 1])
ax_time.set_title("Live Audio Input (Time Domain)")
ax_time.set_xlabel("Time [s]")
ax_time.set_ylabel("Amplitude")

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

ax_radio = plt.axes([0.01, 0.35, 0.20, 0.6])
radio = RadioButtons(ax_radio, device_names, activecolor='blue')
for lbl in radio.labels:
    lbl.set_fontsize(7)

# ==== Preset buttons ====
x0, y0 = 0.75, 0.05
dx, dy = 0.05, 0.04

def make_preset_callback(freq):
    def callback(event):
        global time_window_s
        tw = 30.0 / freq
        time_window_s = tw
        slider_time.set_val(time_window_s)
        print(f"Time window set to 30 periods of {freq} Hz: {time_window_s:.6f}s")
    return callback

for i, f in enumerate(freq_presets):
    ax = plt.axes([x0 + (i % 2) * dx, y0 + (i // 2) * dy, dx, dy])
    btn = Button(ax, f"{f} Hz")
    btn.on_clicked(make_preset_callback(f))
    preset_axes.append(ax)
    preset_buttons.append(btn)

# ==== Sliders ====
ax_slider_fftamp = plt.axes([0.4, 0.24, 0.3, 0.03])
slider_fftamp = Slider(ax_slider_fftamp, "FFT Amp (log10)", -3, 1, valinit=np.log10(amplitude_scale), valstep=0.001)

ax_slider_gain = plt.axes([0.4, 0.19, 0.3, 0.03])
slider_gain = Slider(ax_slider_gain, "Input Sensitivity", 0.01, 4.0, valinit=input_sensitivity, valstep=0.00001)

ax_slider_time = plt.axes([0.4, 0.14, 0.3, 0.03])
slider_time = Slider(ax_slider_time, "Time Window (s)", 0.000001, min(0.3, BUFFER_SECONDS), valinit=time_window_s, valstep=0.000001)

# ==== Metrics ====
thd_text = ax_freq.text(0.02, 0.9, "", transform=ax_freq.transAxes, fontsize=10, color='green')
duty_text = ax_time.text(0.02, 0.9, "", transform=ax_time.transAxes, fontsize=10, color='purple')


# ==== Slider callbacks ====
def update_amplitude(val):
    global amplitude_scale
    amplitude_scale = 10 ** slider_fftamp.val
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
stream = None

def toggle_run(event):
    global running, stream
    if not running:
        if selected_device_index is None:
            print("Please select an input device first.")
            return
        print(f"Starting stream on device index {selected_device_index}")
        running = True
        stream = start_audio_stream()
        stream.start()
        button.label.set_text("Stop")
    else:
        print("Stopping stream...")
        running = False
        if stream:
            stream.stop()
            stream.close()
        button.label.set_text("Start")

def select_device(label):
    global selected_device_index
    idx = device_names.index(label)
    selected_device_index = device_indices[idx]
    print(f"Selected input device: {device_names_full[idx]}")

button.on_clicked(toggle_run)
radio.on_clicked(select_device)


# ==== Helper Metrics ====
def calculate_thd_from_fft(fft_mag, fund_idx):
    if fund_idx <= 0 or fund_idx >= len(fft_mag):
        return 0.0
    fund_mag = fft_mag[fund_idx]
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


# ==== Animation Update ====
def update(frame):
    global ring_buffer, write_ptr
    try:
        with lock:
            buf_copy = ring_buffer.copy()
            ptr = write_ptr

        window_samples = int(np.clip(time_window_s * RATE, 1, BUFFER_LEN))
        if ptr >= window_samples:
            window = buf_copy[ptr - window_samples:ptr]
        else:
            part1 = buf_copy[BUFFER_LEN - (window_samples - ptr):]
            part2 = buf_copy[:ptr]
            window = np.concatenate((part1, part2))

        y = window * input_sensitivity
        x = np.linspace(-window_samples / RATE, 0.0, window_samples)
        line_time.set_data(x, y)
        ax_time.set_xlim(x[0], x[-1])

        nfft = int(2 ** np.ceil(np.log2(window_samples)))
        win = np.hanning(window_samples)
        fft_vals = np.fft.rfft(y * win, n=nfft)
        mag = np.abs(fft_vals) / (window_samples if window_samples > 0 else 1)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / RATE)
        mag_db = 20 * np.log10(np.maximum(mag * amplitude_scale, 1e-10))
        line_freq.set_data(freqs, mag_db)

        mag_copy = mag.copy()
        if len(mag_copy) > 1:
            mag_copy[0] = 0
            fund_idx = np.argmax(mag_copy)
            thd_val = calculate_thd_from_fft(mag_copy, fund_idx)
        else:
            thd_val = 0.0
        thd_text.set_text(f"THD: {thd_val:.2f} %")

        duty = calculate_duty_cycle(y)
        duty_text.set_text(f"Duty Cycle: {duty:.1f} %" if duty is not None else "")

        return line_time, line_freq, thd_text, duty_text
    except Exception as e:
        print("Update error:", e)
        return line_time, line_freq, thd_text, duty_text


ani = FuncAnimation(fig, update, interval=30, blit=False)

def on_close(event):
    global running, stream
    running = False
    if stream:
        stream.stop()
        stream.close()
    sd.stop()

fig.canvas.mpl_connect('close_event', on_close)
plt.show()
