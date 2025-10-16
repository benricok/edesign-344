import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import threading
import tkinter as tk
from tkinter import ttk
from scipy.signal import find_peaks

# ==== Configuration ====
CHUNK = 1024
RATE = 48000
CHANNELS = 1
AMP_TO_VOLT_RATIO = 10.0
BUFFER_SECONDS = 1.0
BUFFER_LEN = int(RATE * BUFFER_SECONDS)
AUTO_PERIODS = 20
SMOOTH_ALPHA = 0.2

# ==== State ====
ring_buffer = np.zeros(BUFFER_LEN, dtype=np.float32)
write_ptr = 0
running = False
selected_device_index = None
amplitude_scale = 10 ** -0.2
input_sensitivity = 2.0
time_window_s = 0.15
time_display_s = 0.15
auto_time_scale = True
smoothed_fund_freq = 0.0
lock = threading.Lock()

# ==== Devices ====
def list_input_devices():
    devices = sd.query_devices()
    return [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

devices = list_input_devices()
device_names = [name for _, name in devices]
device_indices = [i for i, _ in devices]

# ==== Audio callback ====
def audio_callback(indata, frames, time, status):
    global ring_buffer, write_ptr
    if not running:
        return
    if status:
        print(status)
    data = indata[:, 0]
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

# ==== Tkinter GUI ====
root = tk.Tk()
root.title("Audio Visualizer")
root.geometry("1600x900")

# ---- Controls Frame ----
ctrl_frame = tk.Frame(root)
ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

tk.Label(ctrl_frame, text="Input Device:").grid(row=0, column=0, sticky="w")
device_var = tk.StringVar(value=device_names[0] if device_names else "")
device_combo = ttk.Combobox(ctrl_frame, textvariable=device_var, values=device_names, width=50, state="readonly")
device_combo.grid(row=0, column=1, padx=5)
def on_device_select(event):
    global selected_device_index
    idx = device_names.index(device_var.get())
    selected_device_index = device_indices[idx]
device_combo.bind("<<ComboboxSelected>>", on_device_select)

start_button = tk.Button(ctrl_frame, text="Start", width=12)
start_button.grid(row=0, column=2, padx=5)
auto_button = tk.Button(ctrl_frame, text="Auto Time Scale: ON", width=18)
auto_button.grid(row=0, column=3, padx=5)

# ---- Sliders ----
def create_slider(parent, label, row, col, from_, to_, resolution, var, command):
    tk.Label(parent, text=label).grid(row=row, column=col, sticky="w")
    s = tk.Scale(parent, variable=var, from_=from_, to=to_, resolution=resolution,
                 orient=tk.HORIZONTAL, length=200, command=command)
    s.grid(row=row, column=col+1)
    return s

amp_var = tk.DoubleVar(value=np.log10(amplitude_scale))
gain_var = tk.DoubleVar(value=input_sensitivity)
fftwin_var = tk.DoubleVar(value=time_window_s)
disp_var = tk.DoubleVar(value=time_display_s)

def update_amp(val): global amplitude_scale; amplitude_scale = 10 ** float(val)
def update_gain(val): global input_sensitivity; input_sensitivity = float(val)
def update_fftwin(val): global time_window_s; time_window_s = float(val)
def update_disp(val): global time_display_s; time_display_s = float(val)

create_slider(ctrl_frame, "FFT Amp (log10)", 1, 0, -3, 1, 0.001, amp_var, update_amp)
create_slider(ctrl_frame, "FFT Window (s)", 1, 2, 0.0001, 0.3, 0.0001, fftwin_var, update_fftwin)

create_slider(ctrl_frame, "Input Sensitivity", 2, 0, 0.01, 6.0, 0.001, gain_var, update_gain)
create_slider(ctrl_frame, "Display Time (s)", 2, 2, 0.001, BUFFER_SECONDS, 0.001, disp_var, update_disp)

# ---- Matplotlib Plot ----
fig, (ax_time, ax_freq) = plt.subplots(2,1, figsize=(10,6))
plt.subplots_adjust(hspace=0.3)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

line_time, = ax_time.plot([], [])
ax_time.set_ylim([-1,1])
ax_time.set_title("Time Domain")
ax_time.set_xlabel("Time [s]")
ax_time.set_ylabel("Voltage [V]")

line_freq, = ax_freq.plot([], [])
ax_freq.set_xlim([0,RATE/2])
ax_freq.set_ylim([-120,0])
ax_freq.set_title("Frequency Domain")
ax_freq.set_xlabel("Freq [Hz]")
ax_freq.set_ylabel("Magnitude [dB]")

thd_text = ax_freq.text(0.02, 0.9, "", transform=ax_freq.transAxes, fontsize=10, color='green')
duty_text = ax_time.text(0.02, 0.9, "", transform=ax_time.transAxes, fontsize=10, color='purple')
freq_text = ax_time.text(0.84, 0.9, "Frequency: 0Hz", transform=ax_time.transAxes, fontsize=10, color='green')

# ==== Control Logic ====
stream = None
def toggle_run():
    global running, stream
    if not running:
        if selected_device_index is None:
            print("Select device first.")
            return
        running = True
        stream = start_audio_stream()
        stream.start()
        start_button.config(text="Stop")
    else:
        running = False
        if stream:
            stream.stop()
            stream.close()
        start_button.config(text="Start")
start_button.config(command=toggle_run)

def toggle_auto():
    global auto_time_scale
    auto_time_scale = not auto_time_scale
    auto_button.config(text=f"Auto Time Scale: {'ON' if auto_time_scale else 'OFF'}")
auto_button.config(command=toggle_auto)

# ==== Metrics ====
def calculate_thd_from_fft(fft_mag, fund_idx):
    if 2*fund_idx >= len(fft_mag): return 0.0
    harmonics = fft_mag[2*fund_idx:]
    thd = np.sqrt(np.sum(harmonics**2)) / fft_mag[fund_idx] * 100
    return float(thd)

def measure_rise_fall_and_duty(sig, rate, fund_freq):
    sig = np.asarray(sig)
    if sig.size < 20: return None
    expected_period = max(3,int(rate/fund_freq)) if fund_freq>0 else max(3,sig.size//6)
    smoothN = min(11, max(3, expected_period//20))
    if smoothN%2==0: smoothN+=1
    sig_s = np.convolve(sig,np.ones(smoothN)/smoothN, mode='same')
    amp = np.max(sig_s)-np.min(sig_s)
    if amp<1e-4: return None
    min_distance = max(3,int(0.4*expected_period))
    prom = max(0.15*amp,1e-4)
    maxima,_=find_peaks(sig_s,distance=min_distance,prominence=prom)
    minima,_=find_peaks(-sig_s,distance=min_distance,prominence=prom)
    if minima.size<2 or maxima.size<1: return None
    rise_samples=[]; fall_samples=[]; duties=[]
    for mi in minima:
        max_cands=maxima[maxima>mi]
        if max_cands.size==0: continue
        max_idx=max_cands[0]
        next_min_cands=minima[minima>max_idx]
        if next_min_cands.size==0: continue
        next_min_idx=next_min_cands[0]
        rise_samples.append(max_idx-mi)
        fall_samples.append(next_min_idx-max_idx)
        seg=sig_s[mi:next_min_idx]
        if seg.size>4: duties.append(np.sum(seg>np.median(seg))/seg.size*100.0)
    if len(rise_samples)==0: return None
    mean_rise_s=float(np.mean(rise_samples))/rate
    mean_fall_s=float(np.mean(fall_samples))/rate
    skew_percent=100.0*(np.mean(rise_samples)/(np.mean(rise_samples)+np.mean(fall_samples))-0.5)
    sig = sig - np.mean(sig)
    sig /= (np.max(np.abs(sig))+1e-12)
    mean_duty = np.sum(sig>0)/len(sig)*100
    return mean_rise_s, mean_fall_s, skew_percent, mean_duty

# ==== Animation Update ====
def update(frame):
    global ring_buffer, write_ptr, smoothed_fund_freq
    try:
        with lock:
            buf_copy = ring_buffer.copy()
            ptr = write_ptr
        window_samples=int(np.clip(time_window_s*RATE,1,BUFFER_LEN))
        if ptr>=window_samples:
            window=buf_copy[ptr-window_samples:ptr]
        else:
            window=np.concatenate((buf_copy[BUFFER_LEN-(window_samples-ptr):], buf_copy[:ptr]))
        y = window*input_sensitivity
        nfft=int(2**np.ceil(np.log2(window_samples)))
        win=np.hanning(window_samples)
        fft_vals=np.fft.rfft(y*win,n=nfft)
        mag=np.abs(fft_vals)/(window_samples if window_samples>0 else 1)
        freqs=np.fft.rfftfreq(nfft,d=1.0/RATE)
        mag_db=20*np.log10(np.maximum(mag*amplitude_scale,1e-10))
        line_freq.set_data(freqs, mag_db)
        mag_copy=np.copy(mag)
        if len(mag_copy)>1:
            mag_copy[0]=0
            fund_idx=np.argmax(mag_copy)
            fund_freq=freqs[fund_idx]
            smoothed_fund_freq=(1-SMOOTH_ALPHA)*smoothed_fund_freq+SMOOTH_ALPHA*fund_freq
            freq_text.set_text(f"Frequency: {smoothed_fund_freq:.1f}Hz")
            thd_val=calculate_thd_from_fft(mag_copy,fund_idx)
            max_db=np.max(mag_db)
        else:
            fund_freq=0.0; thd_val=0.0; max_db=-120
        thd_text.set_text(f"THD: {thd_val:.2f}%")
        signal_present=(max_db>-70) and (fund_freq>10)
        if auto_time_scale and signal_present and smoothed_fund_freq>1.0:
            auto_time_display_s=AUTO_PERIODS/smoothed_fund_freq
            display_samples=int(np.clip(auto_time_display_s*RATE,1,BUFFER_LEN))
        else:
            display_samples=int(np.clip(time_display_s*RATE,1,BUFFER_LEN))
        if ptr>=display_samples:
            display_window=buf_copy[ptr-display_samples:ptr]
        else:
            display_window=np.concatenate((buf_copy[BUFFER_LEN-(display_samples-ptr):], buf_copy[:ptr]))
        x_display=np.linspace(-display_samples/RATE,0.0,display_samples)
        y_display=display_window*input_sensitivity
        y_volts=y_display*AMP_TO_VOLT_RATIO
        peak_v=np.max(np.abs(y_volts))+1e-12
        view_scale=1.5*peak_v
        if abs(view_scale-ax_time.get_ylim()[1])>0.1*view_scale:
            ax_time.set_ylim(-view_scale,view_scale)
        line_time.set_data(x_display,y_volts)
        ax_time.set_xlim(x_display[0],x_display[-1])
        ax_time.set_ylabel("Voltage [V]")
        ax_time.set_title(f"Live Input")
        signal_type="No Signal"
        if signal_present:
            if thd_val<7.0:
                signal_type="Sine"
                duty_text.set_text("Sine Wave")
            else:
                meas=measure_rise_fall_and_duty(y_display,RATE,fund_freq)
                if meas is not None:
                    mean_rise_s, mean_fall_s, skew_pct, mean_duty=meas
                    if mean_duty>49 and mean_duty<51:
                        signal_type="Triangle"
                        duty_text.set_text(f"Triangle Wave - Skew: {skew_pct:+.1f}%")
                    elif mean_duty<49:
                        signal_type="Sawtooth"
                        duty_text.set_text(f"Sawtooth Wave - Skew: {skew_pct:+.1f}%")
                    else:
                        signal_type="Square"
                        duty_text.set_text(f"Square Wave - Duty: {mean_duty:.1f}%")
                else:
                    signal_type="Noise"
                    duty_text.set_text("Noise / insufficient cycles")
        else:
            duty_text.set_text("No Signal")
        color_map={"Sine":"blue","Square":"red","Triangle":"green","Sawtooth":"orange","No Signal":"gray","Noise":"gray"}
        line_time.set_color(color_map.get(signal_type,"black"))
        return line_time,line_freq,thd_text,duty_text,freq_text
    except Exception as e:
        print("Update error:", e)
        return line_time,line_freq,thd_text,duty_text,freq_text

ani=FuncAnimation(fig,update,interval=30,blit=False)
root.mainloop()
