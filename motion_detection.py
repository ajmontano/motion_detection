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
    previous_frame = None
    while True:
        # Capture frame-by-frame
        ret_val, frame = cap.read()

        if previous_frame is None:
            previous_frame = frame
            continue

        # # Get the greyscale version of the image and blur it
        # prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

        # Create the foreground masks
        foreground_mask_current = background_subtractor.apply(frame)
        foreground_mask_previous = background_subtractor.apply(previous_frame)

        # Subtract the previous masked frame and the current then set the previous frame
        pixel_difference_frame = cv2.absdiff(src1=foreground_mask_previous, src2=foreground_mask_current)
        previous_frame = frame

        # Dilate and filter the result based on a threshold value
        kernel = np.ones((5, 5))
        pixel_difference_frame = cv2.dilate(pixel_difference_frame, kernel, 1)
        threshold_frame = cv2.threshold(src=pixel_difference_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

        # TODO: Figure out how to control the frames display ("framerate")
        cv2.imshow('Foreground Masked & Diff View', threshold_frame)
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
