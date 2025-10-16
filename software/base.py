import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.animation import FuncAnimation
import threading

# ==== Configuration ====
CHUNK = 1024
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16

# ==== Global State ====
running = False
selected_device_index = None
audio_buffer = np.zeros(CHUNK)
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
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.32, bottom=0.25)
line, = ax.plot(np.zeros(CHUNK))
ax.set_ylim([-32768, 32767])
ax.set_xlim([0, CHUNK])
ax.set_title("Live Audio Input")
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")

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

# ==== Live Animation ====
def update(frame):
    with lock:
        ydata = audio_buffer.copy()
    line.set_ydata(ydata)
    return line,

ani = FuncAnimation(fig, update, interval=30, blit=True)  # ~33 FPS

# ==== Cleanup ====
def on_close(event):
    global running
    running = False
    p.terminate()

fig.canvas.mpl_connect('close_event', on_close)
plt.show()
