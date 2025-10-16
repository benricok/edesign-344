import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
from matplotlib.animation import FuncAnimation
import threading
import scipy.fft as sp_fft
from scipy.signal import find_peaks
import matplotlib.image as mpimg

# ==== Configuration ====
CHUNK = 1024
RATE = 48000
CHANNELS = 1
AMP_TO_VOLT_RATIO = 10.0 

# ==== Buffer / State ====
BUFFER_SECONDS = 1.0
BUFFER_LEN = int(RATE * BUFFER_SECONDS)
ring_buffer = np.zeros(BUFFER_LEN, dtype=np.float32)
write_ptr = 0

running = False
selected_device_index = 1
amplitude_scale = 10 ** -0.2
input_sensitivity = 2.0
time_window_s = 0.15  # for FFT
time_display_s = 0.15  # for time-domain graph
lock = threading.Lock()

# ==== Auto Time Scale ====
auto_time_scale = True
AUTO_PERIODS = 20
smoothed_fund_freq = 0.0
SMOOTH_ALPHA = 0.2  # smoothing factor (0.0–1.0)


# ==== Audio Handling ====
def list_input_devices():
    devices = sd.query_devices()
    input_devices = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            input_devices.append((i, d['name']))
    return input_devices


def audio_callback(indata, frames, time, status):
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
plt.subplots_adjust(left=0.3, bottom=0.4, hspace=0.3, right=0.97, top=0.95)

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
ax_button = plt.axes([0.4, 0.05, 0.14, 0.04])
button = Button(ax_button, 'Start')

devices = list_input_devices()
device_names_full = [name for _, name in devices]
device_names = [name[:40] + "..." if len(name) > 25 else name for name in device_names_full]
device_indices = [i for i, _ in devices]

ax_radio = plt.axes([0.01, 0.35, 0.20, 0.6])
radio = RadioButtons(ax_radio, device_names, activecolor='blue')
for lbl in radio.labels:
    lbl.set_fontsize(7)

# === Load and insert logo ===
try:
    logo = mpimg.imread("splash.png")  # replace with your filename/path
    ax_logo = fig.add_axes([0.005, 0.02, 0.25, 0.25])  # [left, bottom, width, height] in figure fraction
    ax_logo.imshow(logo)
    ax_logo.axis("off")  # hide borders/ticks
except FileNotFoundError:
    print("Logo image not found (splash.png). Skipping image.")

# ==== Sliders ====
ax_slider_fftamp = plt.axes([0.4, 0.29, 0.3, 0.03])
slider_fftamp = Slider(ax_slider_fftamp, "FFT Amp (log10)", -3, 1, valinit=np.log10(amplitude_scale), valstep=0.001)

ax_slider_gain = plt.axes([0.4, 0.24, 0.3, 0.03])
slider_gain = Slider(ax_slider_gain, "Input Sensitivity", 0.01, 6.0, valinit=input_sensitivity, valstep=0.00001)

ax_slider_time = plt.axes([0.4, 0.19, 0.3, 0.03])
slider_time = Slider(ax_slider_time, "FFT Window (s)", 0.000001, min(0.3, BUFFER_SECONDS), valinit=time_window_s, valstep=0.000001)

ax_slider_timescale = plt.axes([0.4, 0.14, 0.3, 0.03])
slider_timescale = Slider(ax_slider_timescale, "Display Time (s)", 0.001, BUFFER_SECONDS, valinit=time_display_s, valstep=0.001)

# ==== Auto time-scale button ====
ax_button_autoscale = plt.axes([0.56, 0.05, 0.14, 0.04])
button_autoscale = Button(ax_button_autoscale, 'Auto Time Scale: ON')

# ==== Metrics ====
thd_text = ax_freq.text(0.02, 0.9, "", transform=ax_freq.transAxes, fontsize=10, color='green')
duty_text = ax_time.text(0.02, 0.9, "", transform=ax_time.transAxes, fontsize=10, color='purple')
freq_text = ax_time.text(0.84, 0.9, "Frequency: 1000Hz", transform=ax_time.transAxes, fontsize=10, color='green')

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
    print(f"FFT window set to: {time_window_s:.4f} s")

def update_time_display(val):
    global time_display_s
    time_display_s = float(val)
    print(f"Display time window set to: {time_display_s:.4f} s")

slider_fftamp.on_changed(update_amplitude)
slider_gain.on_changed(update_gain)
slider_time.on_changed(update_time_window)
slider_timescale.on_changed(update_time_display)


# ==== Auto-scale toggle ====
def toggle_auto_time_scale(event):
    global auto_time_scale
    auto_time_scale = not auto_time_scale
    button_autoscale.label.set_text(f"Auto Time Scale: {'ON' if auto_time_scale else 'OFF'}")
    slider_timescale.ax.set_facecolor('#dddddd' if auto_time_scale else '#ffffff')
    print(f"Auto Time Scale mode {'enabled' if auto_time_scale else 'disabled'}.")

button_autoscale.on_clicked(toggle_auto_time_scale)


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
    harmonics = fft_mag[2 * fund_idx:] if 2 * fund_idx < len(fft_mag) else np.array([])
    thd = np.sqrt(np.sum(harmonics ** 2)) / fund_mag * 100 if fund_mag > 0 else 0.0
    return thd

def calculate_duty_cycle(signal):
    sig = signal - np.mean(signal)
    peak = np.max(np.abs(sig)) + 1e-12
    if peak == 0:
        return None
    sig = sig / peak
    high = np.sum(sig > 0)
    duty_cycle = (high / len(sig)) * 100
    return duty_cycle

def measure_rise_fall_and_duty(sig, rate, fund_freq):
    """
    Returns (mean_rise_s, mean_fall_s, skew_percent, mean_duty_percent)
    or None if not enough cycles / signal too small.
    """
    sig = np.asarray(sig)
    if sig.size < 20:
        return None

    # basic smoothing to reduce jitter (window size scaled to freq)
    if fund_freq and fund_freq > 0:
        expected_period = max(3, int(rate / fund_freq))
    else:
        expected_period = max(3, sig.size // 6)

    smoothN = min(11, max(3, expected_period // 20))
    if smoothN % 2 == 0:
        smoothN += 1
    sig_s = np.convolve(sig, np.ones(smoothN) / smoothN, mode='same')

    amp = np.max(sig_s) - np.min(sig_s)
    if amp < 1e-4:   # too small to measure
        return None

    # peak detection params
    min_distance = max(3, int(0.4 * expected_period))  # at least ~40% of period between peaks
    prom = max(0.15 * amp, 1e-4)

    maxima, _ = find_peaks(sig_s, distance=min_distance, prominence=prom)
    minima, _ = find_peaks(-sig_s, distance=min_distance, prominence=prom)

    if minima.size < 2 or maxima.size < 1:
        return None

    maxima.sort()
    minima.sort()

    rise_samples = []
    fall_samples = []
    duties = []

    # iterate minima -> next maxima -> next minima
    # (treat each minima as cycle start)
    for mi in minima:
        # find next max after this min
        max_cands = maxima[maxima > mi]
        if max_cands.size == 0:
            continue
        max_idx = max_cands[0]

        # find next min after that max
        next_min_cands = minima[minima > max_idx]
        if next_min_cands.size == 0:
            continue
        next_min_idx = next_min_cands[0]

        rise_samples.append(max_idx - mi)
        fall_samples.append(next_min_idx - max_idx)

        seg = sig_s[mi:next_min_idx]
        if seg.size > 4:
            # duty relative to median (robust to DC offset)
            duties.append(np.sum(seg > np.median(seg)) / seg.size * 100.0)

    if len(rise_samples) == 0:
        return None

    mean_rise_s = float(np.mean(rise_samples)) / rate
    mean_fall_s = float(np.mean(fall_samples)) / rate
    skew_percent = 100.0 * (np.mean(rise_samples) / (np.mean(rise_samples) + np.mean(fall_samples)) - 0.5)
    mean_duty = float(np.mean(duties)) if len(duties) > 0 else None

    return mean_rise_s, mean_fall_s, skew_percent, mean_duty


# ==== Animation Update (with signal classification) ====
# ==== Improved Update Function (Hybrid FFT + Time-Domain Detection) ====
def update(frame):
    global ring_buffer, write_ptr, smoothed_fund_freq
    try:
        with lock:
            buf_copy = ring_buffer.copy()
            ptr = write_ptr

        # === FFT window ===
        window_samples = int(np.clip(time_window_s * RATE, 1, BUFFER_LEN))
        if ptr >= window_samples:
            window = buf_copy[ptr - window_samples:ptr]
        else:
            window = np.concatenate((buf_copy[BUFFER_LEN - (window_samples - ptr):], buf_copy[:ptr]))

        y = window * input_sensitivity
        nfft = int(2 ** np.ceil(np.log2(window_samples)))
        win = np.hanning(window_samples)
        fft_vals = np.fft.rfft(y * win, n=nfft)
        mag = np.abs(fft_vals) / (window_samples if window_samples > 0 else 1)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / RATE)
        mag_db = 20 * np.log10(np.maximum(mag * amplitude_scale, 1e-10))
        line_freq.set_data(freqs, mag_db)

        # === Find fundamental ===
        mag_copy = np.copy(mag)
        if len(mag_copy) > 1:
            mag_copy[0] = 0
            fund_idx = np.argmax(mag_copy)
            fund_freq = freqs[fund_idx]
            smoothed_fund_freq = (1 - SMOOTH_ALPHA) * smoothed_fund_freq + SMOOTH_ALPHA * fund_freq
            freq_text.set_text(f"Frequency: {smoothed_fund_freq:.1f}Hz")
            thd_val = calculate_thd_from_fft(mag_copy, fund_idx)
            max_db = np.max(mag_db)
        else:
            fund_freq = 0.0
            thd_val = 0.0
            max_db = -120

        
            #odd_ratio = (h3 + h5) / 2
            #even_ratio = (h2 + h4) / 2

        thd_text.set_text(f"THD: {thd_val:.2f} %")

        # === Tone present? ===
        signal_present = (max_db > -70) and (fund_freq > 10)

        # === Display window selection ===
        if auto_time_scale and signal_present and smoothed_fund_freq > 1.0:
            auto_time_display_s = AUTO_PERIODS / smoothed_fund_freq
            display_samples = int(np.clip(auto_time_display_s * RATE, 1, BUFFER_LEN))
        else:
            display_samples = int(np.clip(time_display_s * RATE, 1, BUFFER_LEN))

        if ptr >= display_samples:
            display_window = buf_copy[ptr - display_samples:ptr]
        else:
            display_window = np.concatenate((buf_copy[BUFFER_LEN - (display_samples - ptr):], buf_copy[:ptr]))

        x_display = np.linspace(-display_samples / RATE, 0.0, display_samples)
        # === Convert to volts ===
        y_display = display_window * input_sensitivity
        y_volts = y_display * AMP_TO_VOLT_RATIO

        # === Auto y-axis scaling (±5× current peak) ===
        peak_v = np.max(np.abs(y_volts)) + 1e-12
        view_scale = 1.5 * peak_v  # 1×5 the peak amplitude
        ax_time.set_ylim(-view_scale, view_scale)

        # === Plot ===
        line_time.set_data(x_display, y_volts)
        ax_time.set_xlim(x_display[0], x_display[-1])
        ax_time.set_ylabel("Voltage [V]")
        ax_time.set_title(f"Live Input")


        # === Classification: FFT for sine, time-domain for square/triangle/saw ===
        signal_type = "No Signal"
        #duty_text.set_text("")

        if signal_present:
            # if THD very low => sine
            if thd_val < 5.0:
                signal_type = "Sine"
                duty_text.set_text("Sine Wave")
            else:
                # measure rise/fall using peaks
                sig = y_display - np.mean(y_display)
                sig /= (np.max(np.abs(sig)) + 1e-12)
                # Rough duty cycle (fraction above mean)
                duty = np.sum(sig > 0) / len(sig) * 100

                meas = measure_rise_fall_and_duty(y_display, RATE, fund_freq)
                if meas is not None:
                    mean_rise_s, mean_fall_s, skew_pct, mean_duty = meas
                    # convert to samples for ratio decisions
                    mean_rise_samples = mean_rise_s * RATE
                    mean_fall_samples = mean_fall_s * RATE
                    total = mean_rise_samples + mean_fall_samples if (mean_rise_samples + mean_fall_samples) > 0 else 1.0
                    frac_rise = mean_rise_samples / total

                    print(f"Rise: {mean_rise_s:+.7f} | Fall: {mean_fall_s:+.7f} | Skew: {skew_pct:+.7f} | Duty: {duty:+.3f} | Frac_rise: {frac_rise:+.3f}")


                    # classify by fraction of period occupied by rise/fall
                    if duty > 49 and duty < 51: # and abs(skew_pct) > 2:
                        # === Extract harmonics ===
                        if fund_freq > 0:
                            harmonics = []
                            for n in range(2, 6):
                                idx = np.argmin(np.abs(freqs - n * fund_freq))
                                harmonics.append(mag[idx] if mag[fund_idx] > 0 else 0 )# / mag[fund_idx]
                            # Harmonic ratios
                            h2, h3, h4, h5 = harmonics

                            print(f"{h2 / mag[fund_idx]} | {mag[fund_idx] / h2}")

                        if mag[fund_idx] / h2 < 2200 and mag[fund_idx] / h2 > 1000: # abs(skew_pct) < 0.6:
                            signal_type = "Square"
                            duty_text.set_text(f"Square Wave - Duty: {duty:.1f}%")
                        elif abs(skew_pct) > 20.0:
                            signal_type = "Sawtooth"
                            duty_text.set_text(f"Sawtooth (skew) - Skew: {skew_pct:+.1f}%")
                        else:
                            signal_type = "Triangle"
                            duty_text.set_text(f"Triangle Wave - Skew: {skew_pct:+.1f}%")

                    elif duty > 0.1 or duty < 0.9:
                        signal_type = "Square"
                        duty_text.set_text(f"Square Wave - Duty: {duty:.1f}%")
                else:
                    signal_type = "Noise"
                    duty_text.set_text("Noise / insufficient cycles")
        else:
            duty_text.set_text("No Signal")

        # color feedback
        color_map = {
            "Sine": "blue",
            "Square": "red",
            "Triangle": "green",
            "Sawtooth": "orange",
            "No Signal": "gray",
            "Noise": "gray"
        }
        line_time.set_color(color_map.get(signal_type, "black"))

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
