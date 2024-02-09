'''
import cv2
import numpy as np
import pyaudio
import wave
import threading
import time
from moviepy.editor import *
import pyautogui

# Screen recording parameters
width = 1920
height = 1080
fps = 30
record_seconds = 10

# Microphone recording parameters
chunk = 1024
sample_format = pyaudio.paInt16
channels = 1
fs = 44100
seconds = 10
filename = "audio.wav"

# Function to list available audio input devices
def list_audio_devices():
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        print((i, dev['name'], dev['maxInputChannels']))
    audio.terminate()

# Function to find the index of the microphone corresponding to the given name
def find_microphone_index(name):
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        if name in dev['name']:
            return i
    return None

# List available audio input devices
list_audio_devices()

# Specify the name of your microphone
microphone_name = 'Microphone (Realtek(R) Audio)'

# Find the index of the microphone corresponding to the headphones
mic_index = find_microphone_index(microphone_name)

if mic_index is not None:
    print("Microphone found:", mic_index)
else:
    print("Microphone not found.")

# Start recording audio using the selected input device
audio = pyaudio.PyAudio()
stream = audio.open(format=sample_format, channels=channels,
                    rate=fs, frames_per_buffer=chunk, input=True,
                    input_device_index=mic_index)
frames = []

# Flag to indicate when to stop recording audio
stop_audio = False

# Start recording video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

def record_audio():
    print("Recording audio...")
    global stop_audio
    while not stop_audio:
        data = stream.read(chunk)
        frames.append(data)

def record_video():
    print("Recording video...")
    for _ in range(record_seconds * fps):
        img = pyautogui.screenshot()
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
        time.sleep(1/fps)
    # Set the stop_audio flag to True when done recording video
    global stop_audio
    stop_audio = True

# Start audio recording in a separate thread
audio_thread = threading.Thread(target=record_audio)
audio_thread.start()

# Start video recording
record_video()

# Stop audio recording
stream.stop_stream()
stream.close()
audio.terminate()

# Save audio to file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(audio.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

# Close the video file
out.release()

# Combine video and audio
video = VideoFileClip("output.avi")
audio = AudioFileClip("audio.wav")
final_clip = video.set_audio(audio)
final_clip.write_videofile("final_output.mp4", codec='libx264')

# Clean up
cv2.destroyAllWindows()
'''



#ok
'''
import cv2
import numpy as np
import pyaudio
import wave
import threading
import time
from moviepy.editor import *
import pyautogui

# Screen recording parameters
width = 1920
height = 1080
fps = 30
record_seconds = 10

# Microphone recording parameters
chunk = 1024
sample_format = pyaudio.paInt16
channels = 1
fs = 44100
seconds = 10
filename = "audio.wav"

# Function to list available audio input devices
def list_audio_devices():
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        print((i, dev['name'], dev['maxInputChannels']))
    audio.terminate()

# Function to find the index of the microphone corresponding to the given name
def find_microphone_index(name):
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        if name in dev['name']:
            return i
    return None

# List available audio input devices
list_audio_devices()

# Specify the name of your microphone
microphone_name = 'Microphone (Realtek(R) Audio)'

# Find the index of the microphone corresponding to the headphones
mic_index = find_microphone_index(microphone_name)

if mic_index is not None:
    print("Microphone found:", mic_index)
else:
    print("Microphone not found.")

# Start recording audio using the selected input device
audio = pyaudio.PyAudio()
stream = audio.open(format=sample_format, channels=channels,
                    rate=fs, frames_per_buffer=chunk, input=True,
                    input_device_index=mic_index)
frames = []

# Flag to indicate when to stop recording audio
stop_audio = False

# Start recording video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

def record_audio():
    print("Recording audio...")
    global stop_audio
    while not stop_audio:
        data = stream.read(chunk)
        frames.append(data)

def record_video():
    print("Recording video...")
    for _ in range(record_seconds * fps):
        img = pyautogui.screenshot()
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
        time.sleep(1/fps)
    # Set the stop_audio flag to True when done recording video
    global stop_audio
    stop_audio = True

# Start audio recording in a separate thread
audio_thread = threading.Thread(target=record_audio)
audio_thread.start()

# Start video recording
record_video()

# Stop audio recording
stream.stop_stream()
stream.close()
audio.terminate()

# Save audio to file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(audio.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

# Close the video file
out.release()

# Combine video and audio
video = VideoFileClip("output.avi")
audio = AudioFileClip("audio.wav")
final_clip = video.set_audio(audio)
final_clip.write_videofile("final_output.mp4", codec='libx264')

# Clean up
cv2.destroyAllWindows()
'''

import cv2
import numpy as np
import pyaudio
import wave
import threading
import time
from moviepy.editor import *
import pyautogui

# Screen recording parameters
width = 1920
height = 1080
fps = 30
record_seconds = 10

# Microphone recording parameters
chunk = 1024
sample_format = pyaudio.paInt16
channels = 1
fs = 44100
filename = "audio.wav"

# Function to list available audio input devices
def list_audio_devices():
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        print((i, dev['name'], dev['maxInputChannels']))
    audio.terminate()

# Function to find the index of the microphone corresponding to the given name
def find_microphone_index(name):
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        if name in dev['name']:
            return i
    return None

# List available audio input devices
list_audio_devices()

# Specify the name of your microphone
microphone_name = 'Microphone (Realtek(R) Audio)'

# Find the index of the microphone corresponding to the headphones
mic_index = find_microphone_index(microphone_name)

if mic_index is not None:
    print("Microphone found:", mic_index)
else:
    print("Microphone not found.")

# Start recording audio using the selected input device
audio = pyaudio.PyAudio()
stream = audio.open(format=sample_format, channels=channels,
                    rate=fs, frames_per_buffer=chunk, input=True,
                    input_device_index=mic_index)
frames = []

# Flag to indicate when to stop recording audio
stop_audio = False

# Start recording video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

def record_audio():
    print("Recording audio...")
    global stop_audio
    while not stop_audio:
        data = stream.read(chunk)
        frames.append(data)

def record_video():
    print("Recording video...")
    frame_duration = 1 / fps  # Duration of each frame in seconds
    start_time = time.time()  # Record the start time
    for _ in range(record_seconds * fps):
        img = pyautogui.screenshot()
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
        
        # Calculate the time to sleep to maintain the desired frame rate
        elapsed_time = time.time() - start_time
        time_to_sleep = max(0, frame_duration - elapsed_time)
        time.sleep(time_to_sleep)
        
        # Update the start time for the next frame
        start_time = time.time()
    # Set the stop_audio flag to True when done recording video
    global stop_audio
    stop_audio = True


# Start audio recording in a separate thread
audio_thread = threading.Thread(target=record_audio)
audio_thread.start()

# Start video recording
record_video()

# Stop audio recording
stream.stop_stream()
stream.close()
audio.terminate()

# Save audio to file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(audio.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

# Close the video file
out.release()

# Combine video and audio
video = VideoFileClip("output.avi")
audio = AudioFileClip("audio.wav")

# Calculate the difference in duration between audio and video
audio_duration = audio.duration
video_duration = record_seconds
duration_diff = abs(audio_duration - video_duration)

# Trim the longer of the two to match the duration
if audio_duration > video_duration:
    audio = audio.subclip(0, video_duration)
else:
    video = video.subclip(0, audio_duration)

# Set audio to video
final_clip = video.set_audio(audio)
final_clip.write_videofile("final_output.mp4", codec='libx264')

# Clean up
cv2.destroyAllWindows()
