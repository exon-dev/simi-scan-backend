from flask import Flask, request, jsonify
# from mobileNet import compare_images
from image_comparator import compare_images
from werkzeug.utils import secure_filename
from datetime import datetime
import pytz

import os

app = Flask(__name__)

# Upload folder configuration
app.config['UPLOAD_FOLDER'] = 'uploads'  
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16 MB

@app.route('/')
def main():
    return "<h1>SimiScan Flask Server is Running!</h1>"

@app.route('/scan', methods=["POST"])
def compare_signature():

    # Check if the POST request has files
    if 'scanned_image' not in request.files or 'original_image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    scanned_image = request.files['scanned_image']
    original_image = request.files['original_image']

    # Check if images are provided
    if scanned_image.filename == '' or original_image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Save the files
    scanned_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(scanned_image.filename))
    original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(original_image.filename))
    
    scanned_image.save(scanned_image_path)
    original_image.save(original_image_path)

    # Get the current date and time
    tz = pytz.timezone('Asia/Manila')
    utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
    local_time = utc_now.astimezone(tz)
    
    # result = compare_images(scanned_image_path, original_image_path)
    similarity_index, threshold_result, confidence_result = compare_images(scanned_image_path, original_image_path)


    return jsonify({
        'Scanned Image' : scanned_image_path,
        'Original Image' : original_image_path,
        'Similarity Index': round(similarity_index, 2),
        'Threshold': round(threshold_result, 2),
        'Confidence': round(confidence_result, 2),
        'Date' : local_time
    })

if __name__ == '__main__':
    app.run(debug=True)
