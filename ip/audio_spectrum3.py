
# import pyaudio
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, FFMpegWriter
# from scipy.fftpack import fft
# from scipy.signal import butter, lfilter
# import time

# # Constants
# CHUNK = 1024 * 8             # Samples per frame
# FORMAT = pyaudio.paInt16     # Audio format (bytes per sample)
# CHANNELS = 1                 # Single channel for microphone
# RATE = 44100                 # Samples per second

# # Create matplotlib figure and axes
# fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))

# # PyAudio class instance
# p = pyaudio.PyAudio()

# # Stream object to get data from the microphone
# stream = p.open(
#     format=FORMAT,
#     channels=CHANNELS,
#     rate=RATE,
#     input=True,
#     output=True,
#     frames_per_buffer=CHUNK
# )

# # Variable for plotting
# x = np.arange(0, 2 * CHUNK, 2)       # Samples (waveform)
# xf = np.linspace(0, RATE, CHUNK)     # Frequencies (spectrum)

# # Create a line object with random data
# line, = ax1.plot(x, np.random.rand(CHUNK), '-', lw=2)

# # Create semilogx line for the spectrum
# line_fft, = ax2.semilogx(xf, np.random.rand(CHUNK), '-', lw=2)

# # Signal range is -32k to 32k, limiting amplitude to +/- 4k
# AMPLITUDE_LIMIT = 4096

# # For measuring frame rate
# frame_count = [0]
# start_time = time.time()

# def butter_lowpass_filter(data, cutoff, fs, order=5):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     filtered_data = lfilter(b, a, data)
#     return filtered_data

# # Format waveform axes
# ax1.set_title('AUDIO WAVEFORM')
# ax1.set_xlabel('Samples')
# ax1.set_ylabel('Volume')
# ax1.set_ylim(-AMPLITUDE_LIMIT, AMPLITUDE_LIMIT)
# ax1.set_xlim(0, 2 * CHUNK)
# plt.setp(ax1, xticks=[0, CHUNK, 2 * CHUNK], yticks=[-AMPLITUDE_LIMIT, 0, AMPLITUDE_LIMIT])

# # Format spectrum axes
# ax2.set_xlim(20, RATE / 2)

# print('Stream started')

# # Initialization function for the animation
# def init():
#     line.set_ydata(np.ma.array(x, mask=True))
#     line_fft.set_ydata(np.ma.array(xf, mask=True))
#     return line, line_fft

# # Update function for the animation
# def update(frame):
#     # Binary data
#     data = stream.read(CHUNK)
#     data_np = np.frombuffer(data, dtype='h')
#     filtered_data = butter_lowpass_filter(data_np, cutoff=1000, fs=RATE)

#     line.set_ydata(filtered_data)

#     # Compute FFT and update line
#     yf = fft(data_np)
#     line_fft.set_ydata(np.abs(yf[0:CHUNK]) / (512 * CHUNK))

#     frame_count[0] += 1

#     return line, line_fft

# # Create animation with a specific frame rate
# ani = FuncAnimation(fig, update, frames=None, init_func=init, blit=True, interval=1000/RATE)

# # Display Tkinter window
# plt.show(block=False)

# # Pause to keep the Tkinter window open for 5 seconds
# plt.pause(5)

# # Save the animation as an MP4 file with a specific frame rate
# writer = FFMpegWriter(fps=25, metadata=dict(artist='Me'), bitrate=1800)
# ani.save('animation_output2.mp4', writer=writer, savefig_kwargs={'facecolor': 'black'})

# # Wait for the user to close the Tkinter window
# p.terminate()


import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.fftpack import fft
from scipy.signal import butter, lfilter
import time
import tkinter as tk

# Constants
CHUNK = 1024 * 8             # Samples per frame
FORMAT = pyaudio.paInt16     # Audio format (bytes per sample)
CHANNELS = 1                 # Single channel for microphone
RATE = 44100                 # Samples per second
FPS = 25                     # Frames per second for recording

# Create matplotlib figure and axes
fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))

# PyAudio class instance
p = pyaudio.PyAudio()

# Stream object to get data from the microphone
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# Variable for plotting
x = np.arange(0, 2 * CHUNK, 2)       # Samples (waveform)
xf = np.linspace(0, RATE, CHUNK)     # Frequencies (spectrum)

# Create a line object with random data
line, = ax1.plot(x, np.random.rand(CHUNK), '-', lw=2)

# Create semilogx line for the spectrum
line_fft, = ax2.semilogx(xf, np.random.rand(CHUNK), '-', lw=2)

# Signal range is -32k to 32k, limiting amplitude to +/- 4k
AMPLITUDE_LIMIT = 4096

# For measuring frame rate
frame_count = [0]
start_time = time.time()

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

# Format waveform axes
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('Samples')
ax1.set_ylabel('Volume')
ax1.set_ylim(-AMPLITUDE_LIMIT, AMPLITUDE_LIMIT)
ax1.set_xlim(0, 2 * CHUNK)
plt.setp(ax1, xticks=[0, CHUNK, 2 * CHUNK], yticks=[-AMPLITUDE_LIMIT, 0, AMPLITUDE_LIMIT])

# Format spectrum axes
ax2.set_xlim(20, RATE / 2)

print('Stream started')

# Initialization function for the animation
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    line_fft.set_ydata(np.ma.array(xf, mask=True))
    return line, line_fft

# Update function for the animation
def update(frame):
    # Binary data
    data = stream.read(CHUNK)
    data_np = np.frombuffer(data, dtype='h')
    filtered_data = butter_lowpass_filter(data_np, cutoff=1000, fs=RATE)

    line.set_ydata(filtered_data)

    # Compute FFT and update line
    yf = fft(data_np)
    line_fft.set_ydata(np.abs(yf[0:CHUNK]) / (512 * CHUNK))

    frame_count[0] += 1

    return line, line_fft

# Create animation with a specific frame rate
ani = FuncAnimation(fig, update, frames=None, init_func=init, blit=True, interval=1000/FPS)

# Display Tkinter window
plt.show(block=False)

# Schedule the Tkinter window to close after 5 seconds
root = tk.Tk()
root.withdraw()
root.after(5000, root.destroy)  # 5000 milliseconds = 5 seconds

# Save the animation as an MP4 file with a specific frame rate
writer = FFMpegWriter(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
ani.save('animation_output2.mp4', writer=writer, savefig_kwargs={'facecolor': 'black'})

# Wait for the user to close the Tkinter window
p.terminate()
