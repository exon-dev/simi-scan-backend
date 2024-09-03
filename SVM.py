import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import os
import joblib

# Load pre-trained MobileNetV3 model without top layers
base_model = tf.keras.applications.MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)

# Function to preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to extract features using MobileNetV3
def extract_features(image_path):
    img = preprocess_image(image_path)
    features = feature_extractor.predict(img)
    features = features.flatten()
    return features

# Function to load dataset
def load_dataset(genuine_dir, forged_dir, validation_dir=None):
    features = []
    labels = []
    
    # Load genuine signatures from training set
    for filename in os.listdir(genuine_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(genuine_dir, filename)
            features.append(extract_features(image_path))
            labels.append(0)  # Label 0 for genuine

    # Load forged signatures from training set
    for filename in os.listdir(forged_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(forged_dir, filename)
            features.append(extract_features(image_path))
            labels.append(1)  # Label 1 for forged

    if validation_dir:
        # Load validation set genuine signatures
        val_genuine_dir = os.path.join(validation_dir, 'genuine')
        for filename in os.listdir(val_genuine_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(val_genuine_dir, filename)
                features.append(extract_features(image_path))
                labels.append(0)  # Label 0 for genuine

        # Load validation set forged signatures
        val_forged_dir = os.path.join(validation_dir, 'forged')
        for filename in os.listdir(val_forged_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(val_forged_dir, filename)
                features.append(extract_features(image_path))
                labels.append(1)  # Label 1 for forged

    return np.array(features), np.array(labels)

# Load dataset
genuine_dir = 'dataset/train/genuine'
forged_dir = 'dataset/train/forged'
validation_dir = 'dataset/validation'  # Path to the validation directory
features, labels = load_dataset(genuine_dir, forged_dir, validation_dir)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model_path = 'svm_model.pkl'

if os.path.exists(model_path):
    # Load the trained model
    svm_model = joblib.load(model_path)
    print('Loaded saved model.')
else:
    # Train SVM model
    svm_model = svm.SVC(kernel='linear', C=1)
    svm_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(svm_model, model_path)
    print('Trained and saved model.')

# Validate the model
y_val_pred = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Predict on the test set
y_test_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Predict on new images (example)
new_image_path = 'realSample.png'
new_features = extract_features(new_image_path)
prediction = svm_model.predict([new_features])
print('Prediction (0: genuine, 1: forged):', prediction[0])
# a 