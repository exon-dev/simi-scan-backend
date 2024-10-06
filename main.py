from flask import Flask, request, jsonify
from img_comp_classify import compare_images
from werkzeug.utils import secure_filename
from datetime import datetime

import pytz
import os


app = Flask(__name__)

# Upload folder configuration
app.config['UPLOAD_FOLDER'] = 'uploads'  
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

@app.route('/')
def main():
    return "<h1>SimiScan Flask Server is Running!</h1>"

@app.route('/scan', methods=["POST"])
def compare_signature():
    # Ensure that both files are provided in the form data
    if 'original_signature' not in request.files or 'scanned_signature' not in request.files:
        return jsonify({'Error': 'Files not provided'}), 400

    # Retrieve the files from the request
    original_file = request.files['original_signature']
    scanned_file = request.files['scanned_signature']

    # Define file paths
    original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('original_signature.png'))
    scanned_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('scanned_signature.png'))

    # Save the uploaded files
    original_file.save(original_image_path)
    scanned_file.save(scanned_image_path)

    # Get the current date and time
    tz = pytz.timezone('Asia/Manila')
    utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
    local_time = utc_now.astimezone(tz)
    
    similarity_index, confidence_result, final_result = compare_images(scanned_image_path, original_image_path)

    return jsonify({
        # ====Use only for debugging puposes ===
        # 'Scanned Image': scanned_image_path,
        # 'Original Image': original_image_path,
        'Similarity Index': round(similarity_index, 2),
        'Confidence': round(confidence_result, 2),
        'Date': local_time,
        'Prediction': final_result
    })


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)