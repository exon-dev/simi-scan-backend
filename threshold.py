import cv2
import numpy as np

def load_and_resize_images(image1_path, image2_path):
    """
    Load and resize images to ensure they have the same dimensions.
    """
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    return image1, image2

def compute_difference(image1, image2):
    """
    Compute the absolute difference between two images.
    """
    return cv2.absdiff(image1, image2)

def apply_threshold(diff, threshold_value):
    """
    Apply a threshold to highlight significant differences.
    """
    _, thresholded_diff = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_diff

def calculate_threshold_result(thresholded_diff):
    """
    Calculate the average intensity of the thresholded difference.
    """
    return np.mean(thresholded_diff)

def resize_for_display(image, display_scale):
    """
    Resize the image for display.
    """
    return cv2.resize(image, (int(image.shape[1] * display_scale), int(image.shape[0] * display_scale)))

def show_images(diff, thresholded_diff, display_scale):
    """
    Display the images.
    """
    resized_diff = resize_for_display(diff, display_scale)
    resized_thresholded_diff = resize_for_display(thresholded_diff, display_scale)

    cv2.imshow('Resized Difference Image', resized_diff)
    cv2.imshow('Resized Thresholded Difference', resized_thresholded_diff)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_thresholded_image(thresholded_diff, save_path='thresholded_difference.jpg'):
    """
    Save the thresholded difference image.
    """
    cv2.imwrite(save_path, thresholded_diff)

def run_threshold(image1_path, image2_path, threshold_value=30, display_scale=0.5):
    """
    Perform all steps of the comparison and return the threshold result.
    """
    image1, image2 = load_and_resize_images(image1_path, image2_path)
    diff = compute_difference(image1, image2)
    thresholded_diff = apply_threshold(diff, threshold_value)
    result = calculate_threshold_result(thresholded_diff)
    show_images(diff, thresholded_diff, display_scale)
    save_thresholded_image(thresholded_diff)
    return result