import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
import joblib
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set the directory
HOME = os.getcwd()
print(f"Working directory: {HOME}")

# Paths
feature_extractor_path = os.path.join(HOME, 'mobilenet_v3_small_feature_extractor.pth')
svm_classifier_path = os.path.join(HOME, 'svm_classifier.pkl')

# Ensure the paths are correct
print("Feature Extractor Path:", feature_extractor_path)
print("SVM Classifier Path:", svm_classifier_path)

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the MobileNetV3 Small model (feature extractor)
model = models.mobilenet_v3_small(weights=None)
model.classifier = nn.Identity()  # Remove the classifier layer to get features
model.load_state_dict(torch.load(feature_extractor_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Load the pre-trained SVM classifier
svm_classifier = joblib.load(svm_classifier_path)

# Prediction function
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
    image = data_transforms(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Extract features
    with torch.no_grad():
        features = model(image).cpu().numpy()

    # Predict using the SVM classifier
    prediction = svm_classifier.predict(features)
    return 'Real' if prediction[0] == 0 else 'Forged'

# Test accuracy function
def test_accuracy(test_images, test_labels):
    predictions = []
    for image_path in test_images:
        predictions.append(predict_image(image_path))

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Accuracy: {accuracy:.4f}')

    return accuracy

# Example image for testing
test_image_path = os.path.join(HOME, "101output.png")

# Predict the class of the test image
prediction = predict_image(test_image_path)
print(f'The predicted label for the test image is: {prediction}')

# Optionally show the image
img = Image.open(test_image_path)
plt.imshow(img)
plt.show()

# Example of running the accuracy test (if you have a test set of images and labels)
# test_images = ["/path/to/image1.png", "/path/to/image2.png", ...]
# test_labels = ['Real', 'Forged', ...]  # Corresponding labels for test images
# test_accuracy(test_images, test_labels)
