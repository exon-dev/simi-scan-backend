from flask import Flask, request, jsonify
# from mobileNet import compare_images
from image_comparator import compare_images
from werkzeug.utils import secure_filename
from datetime import datetime
import pytz
import base64
import os

from middleware import is_authenticated

app = Flask(__name__)

# Upload folder configuration
app.config['UPLOAD_FOLDER'] = 'uploads'  
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16 MB

@app.route('/')
def main():
    return "<h1>SimiScan Flask Server is Running!</h1>"

"""
    Sample API Request:
    
    POST /scan
    
    header: {
        "Authorization": "Bearer <token>"
    }
    body: {
        "original_signature": "base64_encoded_image",
        "scanned_signature": "base64_encoded_image",
    }
"""

@app.route('/scan', methods=["POST"])
def compare_signature():
    req = request.get_json()

    is_valid, error_response = is_authenticated()
    if not is_valid:
        return error_response
       
    original_signature = req.get('original_signature')
    scanned_signature = req.get('scanned_signature')

    if not original_signature or not scanned_signature:
        return jsonify({'Error': 'Images not provided'}), 400

    # decode base64 image to original image data
    original_img_data = base64.b64decode(original_signature)
    scanned_img_data = base64.b64decode(scanned_signature)

    original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('original_signature.png'))
    scanned_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('scanned_signature.png'))

    with open(original_image_path, 'wb') as f:
        f.write(original_img_data)

    # Get the current date and time
    tz = pytz.timezone('Asia/Manila')
    utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
    local_time = utc_now.astimezone(tz)
    
    # result = compare_images(scanned_image_path, original_image_path)
    similarity_index, threshold_result, confidence_result = compare_images(scanned_image_path, original_image_path)
    # similarity_index, threshold_result, confidence_result = compare_images(scanned_image_path, original_image_path)


    return jsonify({
        'Scanned Image' : scanned_image_path,
        'Original Image' : original_image_path,
        'Similarity Index': round(similarity_index, 2),
        'Threshold': round(threshold_result, 2),
        'Confidence': round(confidence_result, 2),
        'Date' : local_time
    })

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
