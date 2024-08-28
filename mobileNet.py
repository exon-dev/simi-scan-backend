import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial.distance import cosine

# Set the directory
HOME = os.getcwd()
print(f"Working directory: {HOME}")

# Define the path to save the model
model_save_path = os.path.join(HOME, 'mobilenet_v3_small_feature_extractor.keras')
# model_save_path = os.path.join(HOME, 'mobilenet_v3_small_feature_extractor.h5')

# Create MobileNetV3Small model for feature extraction
def create_mobilenet_v3_small_feature_extractor():
    if os.path.exists(model_save_path):
        model = load_model(model_save_path)
    else:
        base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # No extra arguments passed
        model = Model(inputs=base_model.input, outputs=x)
        model.save(model_save_path)  # Save with HDF5 format (.h5)
    return model

mobilenet_v3_small_feature_extractor = create_mobilenet_v3_small_feature_extractor()

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = img_to_array(img)
    img = preprocess_input(img)  # Preprocess for MobileNetV3
    return np.expand_dims(img, axis=0)

def extract_features(img_path):
    img = preprocess_image(img_path)
    features = mobilenet_v3_small_feature_extractor.predict(img)
    return features.flatten()

def compare_images(img1_path, img2_path):
    features1 = extract_features(img1_path)
    features2 = extract_features(img2_path)
    
    # Calculate similarity score
    similarity_score = cosine(features1, features2)
    
    # Convert similarity score to percentage
    percentage_similarity = (1 - similarity_score) * 100
    
    return f"{percentage_similarity:.2f}"
