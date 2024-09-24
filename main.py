from flask import Flask, request, jsonify
from flask_cors import CORS
from image_comparator import compare_images
from werkzeug.utils import secure_filename
from datetime import datetime
import pytz
import base64
import os
import logging

app = Flask(__name__)

CORS(app)

# Upload folder configuration
app.config['UPLOAD_FOLDER'] = 'uploads'  
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16 MB

logging.basicConfig(level=logging.INFO)

@app.before_request
def log_request_info():
    logging.info(f"Received {request.method} request for {request.url}")
    logging.info(f"Request headers: {request.headers}")
    logging.info(f"Request body: {request.get_data()}")

@app.after_request
def log_response_info(response):
    logging.info(f"Response status: {response.status}")
    return response

@app.route('/')
def main():
    return "<h1>SimiScan Flask Server is Running!</h1>"

"""
    POST /scan
    
    header: {
        "Authorization": "Bearer <token>"
    }
    body: {
        "original_signature": "base64_encoded_image",
        "scanned_signature": "base64_encoded_image",
    }
"""

logging.basicConfig(level=logging.INFO)

@app.route('/scan', methods=["POST"])
def compare_signature():
    req = request.get_json()

    original_signature = req.get('original_signature')
    scanned_signature = req.get('scanned_signature')

    if not original_signature or not scanned_signature:
        return jsonify({'Error': 'Both original and scanned signatures must be provided'}), 400

    try:
        original_img_data = base64.b64decode(original_signature)
        scanned_img_data = base64.b64decode(scanned_signature)

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f'original_signature_{timestamp}.png'))
        scanned_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f'scanned_signature_{timestamp}.png'))

        with open(original_image_path, 'wb') as f:
            f.write(original_img_data)

        with open(scanned_image_path, 'wb') as f:
            f.write(scanned_img_data)

        tz = pytz.timezone('Asia/Manila')
        utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
        local_time = utc_now.astimezone(tz)

        #confidence_result
        similarity_index, threshold_result = compare_images(scanned_image_path, original_image_path)

        return jsonify({
            'similarity_idx': round(similarity_index, 2),
            'threshold_val': round(threshold_result, 2),
            # 'Confidence': round(confidence_result, 2),
            'date': local_time.strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        logging.error(f'Error occurred: {str(e)}')
        return jsonify({'Error': f'An error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    # Ensure the upload directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    # change to your ip address
    # make sure na ang ip address kay gikan sa Wireless LAN adapter Wi-Fi: ipv4 address
    app.run(host='192.168.1.14', port=5000, debug=False)
