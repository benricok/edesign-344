import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
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

# ==== Global State ====
running = False
selected_device_index = None
audio_buffer = np.zeros(CHUNK)
selected_fft = "NumPy FFT"
fft_selector_visible = False
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
    global running, audio_buffer
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=selected_device_index,
                    frames_per_buffer=CHUNK)
    while running:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        with lock:
            audio_buffer = data.copy()
    stream.stop_stream()
    stream.close()

# ==== Plot Setup ====
fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 6))
plt.subplots_adjust(left=0.32, bottom=0.25, hspace=0.5)

# --- Time-domain plot ---
line_time, = ax_time.plot(np.zeros(CHUNK))
ax_time.set_ylim([-32768, 32767])
ax_time.set_xlim([0, CHUNK])
ax_time.set_title("Live Audio Input (Time Domain)")
ax_time.set_xlabel("Sample")
ax_time.set_ylabel("Amplitude")

# --- Frequency-domain plot ---
freqs = np.fft.rfftfreq(CHUNK, 1 / RATE)
line_freq, = ax_freq.plot(freqs, np.zeros(len(freqs)))
ax_freq.set_xlim([0, RATE / 2])
ax_freq.set_ylim([-120, 0])
ax_freq.set_title("Frequency Spectrum")
ax_freq.set_xlabel("Frequency [Hz]")
ax_freq.set_ylabel("Magnitude [dB]")

# ==== Controls ====
ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
button = Button(ax_button, 'Start')

devices = list_input_devices()
device_names_full = [name for _, name in devices]
device_names = [name[:25] + "..." if len(name) > 25 else name for name in device_names_full]
device_indices = [i for i, _ in devices]

ax_radio = plt.axes([0.05, 0.35, 0.25, 0.6])
radio = RadioButtons(ax_radio, device_names, activecolor='blue')
for lbl in radio.labels:
    lbl.set_fontsize(8)

# ==== FFT Selection ====
fft_options = ["NumPy FFT", "SciPy FFT"]
if FFTW_AVAILABLE:
    fft_options.append("PyFFTW FFT")

ax_fft_button = plt.axes([0.05, 0.18, 0.25, 0.07])
fft_button = Button(ax_fft_button, f"FFT: {selected_fft}")

ax_fft_radio = plt.axes([0.05, 0.05, 0.25, 0.1])
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

# ==== FFT Calculation ====
def compute_fft(signal):
    if selected_fft == "SciPy FFT":
        fft_vals = sp_fft.rfft(signal)
    elif selected_fft == "PyFFTW FFT" and FFTW_AVAILABLE:
        fft_vals = pyfftw.interfaces.numpy_fft.rfft(signal)
    else:
        fft_vals = np.fft.rfft(signal)

    magnitude = np.abs(fft_vals) / len(signal)
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
    return magnitude_db

# ==== Live Animation ====
def update(frame):
    with lock:
        ydata = audio_buffer.copy()

    # Update time-domain plot
    line_time.set_ydata(ydata)

    # Compute and update FFT plot
    magnitude_db = compute_fft(ydata)
    line_freq.set_ydata(magnitude_db)

    return line_time, line_freq

ani = FuncAnimation(fig, update, interval=30, blit=True)

# ==== Cleanup ====
def on_close(event):
    global running
    running = False
    p.terminate()

fig.canvas.mpl_connect('close_event', on_close)
plt.show()
