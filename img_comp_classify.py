from torchvision import models, transforms
from PIL import Image
from scipy.spatial.distance import cosine

import json
import joblib
import os
import numpy as np
import cv2
import torch
import torch.nn as nn

"""
This script compares images and classifies them as 'Real' or 'Forged' using MobileNetV3 for feature extraction and an SVM classifier. It also calculates similarity indices, threshold results, and confidence levels.

Dependencies:
- OpenCV
- PIL
- PyTorch
- torchvision
- joblib
- scipy

Usage:
- Call `compare_images(img1_path, img2_path)` to get similarity index, threshold result, confidence level, and classification result.
"""

# Define the device setup (Check if the device has a grapics card).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Set the directory
HOME = os.getcwd()
print(f"Working directory: {HOME}")

# Define the path to save/load the PyTorch MobileNetV3 model and SVM classifier
model_save_path = os.path.join(HOME, 'model_package/mobilenet_v3_small_feature_extractor.pth')
svm_model_path = os.path.join(HOME, 'model_package/svm_classifier.pkl')
model_accuracy_data = os.path.join(HOME, 'model_package/data.json')

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])


"""
Initialize MobileNetV3Small as a feature extractor by removing the classifier layer,
load a saved model state if available, or save the model state if not.
"""
def load_mobileNet_v3_small_feature_extractor():
    base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    base_model.classifier = nn.Identity()  # Remove the classifier layer

    if os.path.exists(model_save_path,):
        base_model.load_state_dict(torch.load(model_save_path), strict=False)
        print("Loaded model from saved state.")
    else:
          print("Custom pre-trained model not found.")

    base_model.eval()  # Set to evaluation mode
    return base_model


# Instantiate the MobileNetV3Small feature extractor model, loading or saving its state as necessary
mobilenet_v3_small_feature_extractor = load_mobileNet_v3_small_feature_extractor()


"""
Load the SVM classifier from a specified file path.
Raises FileNotFoundError if the model file does not exist.
"""
def load_svm_classifier():
    if os.path.exists(svm_model_path):
        svm_model = joblib.load(svm_model_path)
    else:
        raise FileNotFoundError(f"SVM model not found at {svm_model_path}")
    return svm_model

svm_classifier = load_svm_classifier()


"""
Preprocess an image by reading, converting color space, and applying transformations.
"""
def preprocess_image(img_path):
    # Read the image using OpenCV
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Convert the NumPy array (OpenCV image) to a PIL image
    img = Image.fromarray(img)

    # Apply the transformations (Resize, ToTensor, Normalize)
    img = transform(img)
    
    return img.unsqueeze(0)  # Add batch dimension


"""
Extract features from an image using the MobileNetV3 model.
"""
def extract_features(img_path):
    img = preprocess_image(img_path)
    with torch.no_grad():
        features = mobilenet_v3_small_feature_extractor(img).squeeze().numpy()  # Extract features and flatten
    return features



"""
Read accuracy from a JSON file and return it as a percentage.
"""
def confidence_level(file_path):
    # Open the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)
    
    # Access the 'accuracy' value from the JSON data
    accuracy = data['accuracy']
    
    return round(accuracy * 100, 2) 


"""
Preprocess the image, extract features using MobileNetV3, and classify it as
'Real' or 'Forged' using the SVM classifier.
"""
def predict_image(image_path):
    
    # Preprocess the image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found or unable to read the image at path: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.equalizeHist(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Convert to PIL Image for the transforms
    image = Image.fromarray(image)

    # Apply data transforms and prepare tensor
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Extract features
    with torch.no_grad():
        features = mobilenet_v3_small_feature_extractor(image).cpu().numpy()

    # Predict using the SVM classifier
    prediction = svm_classifier.predict(features)

    # Map prediction to class name
    return 'Real' if prediction[0] == 0 else 'Forged'



"""
Compares two images by extracting features, calculating similarity, and evaluating differences. 
Returns the similarity index, threshold result, confidence level, and classification result.
"""
def compare_images(img1_path, img2_path):
    features1 = extract_features(img1_path)
    features2 = extract_features(img2_path)

    # Calculate similarity score
    similarity_score = cosine(features1, features2)
    similarity_index = (1 - similarity_score) * 100

    confidence_result = confidence_level(model_accuracy_data)
    final_result = predict_image(img1_path)

    return similarity_index, confidence_result, final_result

# Uncomment only for debugging (This way you can test the script by running this file)
# ===== Debugging script =====
# similarity_index, threshold_result, confidence_result, final_result = compare_images("101output.png", "101output.png")
# print("Similarity Index:  ", similarity_index)
# print("Confidence Level: ", confidence_result)
# print("Final Result: ", final_result)
# ============================
