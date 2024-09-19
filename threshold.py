import cv2
import numpy as np

"""
image_comparison.py

This script compares two grayscale images by computing the absolute difference between them. It resizes images to match dimensions, applies a threshold to highlight significant differences, and calculates the average intensity of the thresholded difference.

Functions:
- load_and_resize_images: Loads and resizes images to match dimensions.
- compute_difference: Computes the absolute difference between two images.
- apply_threshold: Applies a threshold to the difference image.
- calculate_threshold_result: Computes the average intensity of the thresholded difference.
- resize_for_display: Resizes images for display purposes.
- show_images: Displays the difference and thresholded difference images.
- save_thresholded_image: Saves the thresholded difference image to disk.
- run_threshold: Orchestrates the image comparison process and returns the threshold result.

Dependencies:
- OpenCV
- NumPy
"""

"""
Load and resize images to ensure they have the same dimensions.
"""
def load_and_resize_images(image1_path, image2_path):

    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    return image1, image2

"""
Compute the absolute difference between two images.
"""
def compute_difference(image1, image2):
    
    return cv2.absdiff(image1, image2)

"""
Apply a threshold to highlight significant differences.
"""
def apply_threshold(diff, threshold_value):
    
    _, thresholded_diff = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_diff


"""
Calculate the average intensity of the thresholded difference.
"""
def calculate_threshold_result(thresholded_diff):
    return np.mean(thresholded_diff)


"""
Resize the image for display.
"""
def resize_for_display(image, display_scale):
    return cv2.resize(image, (int(image.shape[1] * display_scale), int(image.shape[0] * display_scale)))

"""
Display the images.
"""
def show_images(diff, thresholded_diff, display_scale):
    
    resized_diff = resize_for_display(diff, display_scale)
    resized_thresholded_diff = resize_for_display(thresholded_diff, display_scale)

    cv2.imshow('Resized Difference Image', resized_diff)
    cv2.imshow('Resized Thresholded Difference', resized_thresholded_diff)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
Save the thresholded difference image.
"""
def save_thresholded_image(thresholded_diff, save_path='thresholded_difference.jpg'):
    
    cv2.imwrite(save_path, thresholded_diff)


"""
Perform all steps of the comparison and return the threshold result.
"""
def run_threshold(image1_path, image2_path, threshold_value=30, display_scale=0.5):
   
    image1, image2 = load_and_resize_images(image1_path, image2_path)
    diff = compute_difference(image1, image2)
    thresholded_diff = apply_threshold(diff, threshold_value)
    result = calculate_threshold_result(thresholded_diff)
    save_thresholded_image(thresholded_diff)

    # Open a popup images that will show threshold of the two images 
    # ===== Uncomment only for debugging =====

    # show_images(diff, thresholded_diff, display_scale)
    
    # ========================================


    return result