import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import argparse


def parse_commandline():
    parser = argparse.ArgumentParser(description='Motion detection app')
    parser.add_argument('--wait-ms', type=int, default=10, help='The wait time to display frames in milliseconds')
    parser.add_argument('--image-width', type=int, default=1920, help='The width of the capture in pixels')
    parser.add_argument('--image-height', type=int, default=1080, help='The height of the capture in pixels')
    parser.add_argument('--focus', type=int, default=0, help='The focus value of the capture')

    return parser.parse_args()


def setup_capture(args):
    # Define the dimensions, in pixels, of a full screen
    width = args.image_width
    height = args.image_height

    # Define the focus level of the camera
    focus = args.focus

    # Define the wait to display the next frame
    frame_wait_ms = args.wait_ms

    # Create the capture object
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Change the focus
    cap.set(cv2.CAP_PROP_FOCUS, focus)

    # Create a background subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    return cap, background_subtractor


def show_frame(cap, background_subtractor):
    # Display the window
    while True:
        # Capture frame-by-frame
        ret_val, frame = cap.read()

        # Create the foreground mask
        foreground_mask = background_subtractor.apply(frame)

        # TODO: Figure out how to control the frames display ("framerate")
        cv2.imshow('Foreground Masked View', foreground_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When finished, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_commandline()
    cap, background_subtractor = setup_capture(args)

    # Run the display loop
    show_frame(cap, background_subtractor)


if __name__ == '__main__':
    main()
