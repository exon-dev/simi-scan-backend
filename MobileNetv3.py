import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D
from scipy.spatial.distance import cosine
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

model_save_path = 'mobilenet_v3_small_feature_extractor.keras'
svm_model_path = 'svm_model.joblib'

def create_mobilenet_v3_small_feature_extractor():
    if os.path.exists(model_save_path):
        model = load_model(model_save_path)
    else:
        base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        model = Model(inputs=base_model.input, outputs=x)
        model.save(model_save_path)
    return model

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # Resize to 128x128
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def extract_features(img_path):
    img = preprocess_image(img_path)
    features = mobilenet_v3_small_feature_extractor.predict(img)
    return features.flatten()

def calculate_similarity(img1_features, img2_features):
    similarity_score = cosine(img1_features, img2_features)
    similarity_index = (1 - similarity_score) * 100
    return similarity_index

def load_svm_model():
    if os.path.exists(svm_model_path):
        svm_model = joblib.load(svm_model_path)
    else:
        raise FileNotFoundError("SVM model not found. Train the SVM model first.")
    return svm_model

def load_dataset(genuine_dir, forged_dir, validation_dir=None):
    features = []
    labels = []
    
    # Load genuine signatures from training set
    for filename in os.listdir(genuine_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(genuine_dir, filename)
            features.append(np.load(image_path))
            labels.append(0)  # Label 0 for genuine

    # Load forged signatures from training set
    for filename in os.listdir(forged_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(forged_dir, filename)
            features.append(np.load(image_path))
            labels.append(1)  # Label 1 for forged

    if validation_dir:
        # Load validation set genuine signatures
        val_genuine_dir = os.path.join(validation_dir, 'genuine')
        for filename in os.listdir(val_genuine_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(val_genuine_dir, filename)
                features.append(np.load(image_path))
                labels.append(0)  # Label 0 for genuine

        # Load validation set forged signatures
        val_forged_dir = os.path.join(validation_dir, 'forged')
        for filename in os.listdir(val_forged_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(val_forged_dir, filename)
                features.append(np.load(image_path))
                labels.append(1)  # Label 1 for forged

    return np.array(features), np.array(labels)

def train_svm_model(genuine_dir, forged_dir, validation_dir=None):
    features, labels = load_dataset(genuine_dir, forged_dir, validation_dir)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train SVM model
    svm_model = svm.SVC(kernel='linear', C=1)
    svm_model.fit(X_train, y_train)
    
    # Validate the model
    y_val_pred = svm_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation Accuracy: {val_accuracy:.2f}')
    
    # Predict on the test set
    y_test_pred = svm_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.2f}')
    
    # Save the trained SVM model
    joblib.dump(svm_model, svm_model_path)
    
    return svm_model

def classify_image(image_features, svm_model):
    prediction = svm_model.predict([image_features])
    return prediction[0]

def compare_images(img1_path, img2_path):
    img1_features = extract_features(img1_path)
    img2_features = extract_features(img2_path)
    
    similarity_index = calculate_similarity(img1_features, img2_features)
    
    try:
        svm_model = load_svm_model()
    except FileNotFoundError:
        print("SVM model not found. Training the model now...")
        # Update the paths as needed
        genuine_dir = 'path/to/genuine'  # Replace with your genuine dataset directory
        forged_dir = 'path/to/forged'    # Replace with your forged dataset directory
        validation_dir = 'path/to/validation'  # Replace with your validation dataset directory, if any
        train_svm_model(genuine_dir, forged_dir, validation_dir)
        # Load the model again after training
        svm_model = load_svm_model()
    
    classified_result = classify_image(img1_features, svm_model)
    threshold_result = np.mean(img1_features)

    return similarity_index, threshold_result, img1_features, classified_result
