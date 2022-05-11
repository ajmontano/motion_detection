import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2

# Define the dimensions, in pixels, of a full screen
width = 1920
height = 1080

# Define the focus level of the camera
focus = 0

# Define the wait to display the next frame
frame_wait_ms = 1000

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Change the focus
cap.set(cv2.CAP_PROP_FOCUS, focus)

root = tk.Tk()

# Setup the image frame
image_frame = tk.Frame(root, width=width, height=height)
image_frame.grid(row=0, column=0, padx=10, pady=10)

label_main = tk.Label(root)
label_main.grid(row=0, column=0)


def show_frame():
    # Capture frame-by-frame
    ret_val, frame = cap.read()

    # Convert the frame into an ImageTk
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    image_tk = ImageTk.PhotoImage(image=img)

    # Display the window
    label_main.imgtk = image_tk
    label_main.configure(image=image_tk)
    label_main.after(frame_wait_ms, show_frame)


# Run the display loop
show_frame()
root.mainloop()

# When finished, release the capture
cap.release()
