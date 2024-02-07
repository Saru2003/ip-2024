import cv2
import numpy as np
import pyautogui
import time

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1920, 1080))  # Adjust resolution if needed

# Record for 10 seconds
start_time = time.time()
while (time.time() - start_time) < 10:
    # Capture the screen
    img = pyautogui.screenshot()
    frame = np.array(img)

    # Convert RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Write the frame to the video file
    out.write(frame)

# Release the VideoWriter and close any open windows
out.release()
cv2.destroyAllWindows()
