import numpy as np
import time
import cv2
import argparse


def parse_commandline():
    parser = argparse.ArgumentParser(description='Motion detection app')
    parser.add_argument('--wait-ms', type=int, default=0, help='The wait time to display frames in milliseconds')
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
    # TODO: Look into if this can keep track of static objects as well
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    return cap, background_subtractor


def show_frame(cap, background_subtractor, wait_ms):
    # Display the window
    while True:
        # Capture frame-by-frame
        ret_val, frame = cap.read()

        # Create the foreground mask
        foreground_mask = background_subtractor.apply(frame)

        # Blur the images to reduce noise
        foreground_mask = cv2.GaussianBlur(src=foreground_mask, ksize=(5, 5), sigmaX=0)

        # Dilate and filter the result based on a threshold value
        kernel = np.ones((5, 5))
        foreground_mask= cv2.dilate(foreground_mask, kernel, 1)
        threshold_frame = cv2.threshold(src=foreground_mask, thresh=200, maxval=255, type=cv2.THRESH_BINARY)[1]

        # Find the contours of the image
        contours, _ = cv2.findContours(image=threshold_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)

        cv2.imshow('Foreground Masked & Diff View', threshold_frame)
        cv2.imshow('Contour Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Wait to display the next frame
        # TODO: Figure out how to best control the framerate
        time.sleep(wait_ms / 1000)

    # When finished, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_commandline()
    cap, background_subtractor = setup_capture(args)

    # Run the display loop
    show_frame(cap, background_subtractor, args.wait_ms)


if __name__ == '__main__':
    main()
