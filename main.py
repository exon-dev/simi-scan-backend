from flask import Flask, request, jsonify
from img_comp_classify import compare_images
from werkzeug.utils import secure_filename
from datetime import datetime

import pytz
import os
import base64
import logging


app = Flask(__name__)

# Upload folder configuration
app.config['UPLOAD_FOLDER'] = 'uploads'  
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

@app.route('/')
def main():
    return "<h1>SimiScan Flask Server is Running!</h1>"

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
        
        similarity_index, confidence_result, final_result = compare_images(scanned_image_path, original_image_path)
        
        return jsonify({
        # ====Use only for debugging puposes ===
        # 'Scanned Image': scanned_image_path,
        # 'Original Image': original_image_path,
            'similarity_idx': round(similarity_index, 2),
            'confidence': round(confidence_result, 2),
            'date': local_time,
            'pred': final_result
        })

    except Exception as e:
        logging.error(f'Error occurred: {str(e)}')
        return jsonify({'Error': f'An error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='192.168.1.8', port='5000', debug=True)