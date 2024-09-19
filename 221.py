import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
from scipy.spatial.distance import cosine
from threshold import run_threshold
import json

# Set the directory
HOME = os.getcwd()
print(f"Working directory: {HOME}")

# Define the path to save/load the PyTorch MobileNetV3 model and SVM classifier
model_save_path = os.path.join(HOME, 'model_package/mobilenet_v3_small_feature_extractor.pth')
svm_model_path = os.path.join(HOME, 'model_package/svm_classifier.pkl')
model_accuracy_data = os.path.join(HOME, 'model_package/result.json')

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load the MobileNetV3Small model for feature extraction
def create_mobilenet_v3_small_feature_extractor():
    base_model = models.mobilenet_v3_small(pretrained=True)
    base_model.classifier = nn.Identity()  # Remove the classifier layer

    if os.path.exists(model_save_path):
        base_model.load_state_dict(torch.load(model_save_path))
        print("Loaded model from saved state.")
    else:
        torch.save(base_model.state_dict(), model_save_path)
        print("Saved model state dict.")

    base_model.eval()  # Set to evaluation mode
    return base_model

mobilenet_v3_small_feature_extractor = create_mobilenet_v3_small_feature_extractor()

# Load the SVM classifier
def load_svm_classifier():
    if os.path.exists(svm_model_path):
        svm_model = joblib.load(svm_model_path)
    else:
        raise FileNotFoundError(f"SVM model not found at {svm_model_path}")
    return svm_model

svm_classifier = load_svm_classifier()

def preprocess_image(img_path):
    # Read the image using OpenCV
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Convert the NumPy array (OpenCV image) to a PIL image
    img = Image.fromarray(img)

    # Apply the transformations (Resize, ToTensor, Normalize)
    img = transform(img)
    
    return img.unsqueeze(0)  # Add batch dimension

def extract_features(img_path):
    img = preprocess_image(img_path)
    with torch.no_grad():
        features = mobilenet_v3_small_feature_extractor(img).squeeze().numpy()  # Extract features and flatten
    return features

def calculate_confidence_value(percentage_similarity, threshold):
    if threshold >= 100:
        raise ValueError("Threshold must be less than 100.")
    
    confidence_value = max(0, (percentage_similarity - threshold) / (100 - threshold)) * 100
    return confidence_value

def compare_images(img1_path, img2_path):
    features1 = extract_features(img1_path)
    features2 = extract_features(img2_path)

    # Calculate similarity score
    similarity_score = cosine(features1, features2)
    similarity_index = (1 - similarity_score) * 100

    threshold_result = run_threshold(img1_path, img2_path)
    # confidence_result = calculate_confidence_value(similarity_index, threshold_result)
    confidence_result = confidence_level(model_accuracy_data)

    return similarity_index, threshold_result, confidence_result

def confidence_level(file_path):
    # Open the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)
    
    # Access the 'accuracy' value from the JSON data
    accuracy = data['accuracy']
    
    return round(accuracy * 100, 2) 




# Example usage
# similarity_index, threshold_result, confidence_result = compare_images("101output.png", "101output.png")
# print("Similarity Index:  ", similarity_index)
# print("Threshold: ", threshold_result)
# print("Confidence Level: ", confidence_result)
