import cv2
import os
import numpy as np

def count_birds(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Grayscale', grayscale)
    cv2.waitKey(0)

    # Apply a Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

    cv2.imshow('Blurred', blurred)
    cv2.waitKey(0)

    # Apply global thresholding
    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow('Thresholded', thresholded)
    cv2.waitKey(0)

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create()

    # Detect blobs.
    keypoints = detector.detect(thresholded)

    # The number of birds is the number of blobs
    num_birds = len(keypoints)

    return num_birds

# Iterate over all images in the directory
image_dir = './birds/bird_miniatures'  # Replace with your directory
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    num_birds = count_birds(image_path)
    print(f'{image_name}: {num_birds} birds')

cv2.destroyAllWindows()