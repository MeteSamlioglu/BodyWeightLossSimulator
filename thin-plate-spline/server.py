from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/BodyWeightLoss", methods=['GET', 'POST'])
def body_weight_loss():
    if request.method == 'POST':
        # Handle the JSON data
        body_parts_data = request.form['data']
        body_parts_data = json.loads(body_parts_data)
        
        # Handle the image file
        if 'image' not in request.files:
            return jsonify({"error": "No image part"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Read the image using OpenCV
        image = cv2.imread(filepath)

        if image is None:
            return jsonify({"error": "Failed to read image"}), 400

        # Display the image using OpenCV
        # cv2.imshow("Received Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Extract the necessary data from the received JSON
        extracted_data = {
            'score': body_parts_data['score'],
            'nose': body_parts_data['keypoints'].get('nose'),
            'leftEye': body_parts_data['keypoints'].get('leftEye'),
            'rightEye': body_parts_data['keypoints'].get('rightEye'),
            'leftEar': body_parts_data['keypoints'].get('leftEar'),
            'rightEar': body_parts_data['keypoints'].get('rightEar'),
            'leftShoulder': body_parts_data['keypoints'].get('leftShoulder'),
            'rightShoulder': body_parts_data['keypoints'].get('rightShoulder'),
            'leftElbow': body_parts_data['keypoints'].get('leftElbow'),
            'rightElbow': body_parts_data['keypoints'].get('rightElbow'),
            'leftWrist': body_parts_data['keypoints'].get('leftWrist'),
            'rightWrist': body_parts_data['keypoints'].get('rightWrist'),
            'leftHip': body_parts_data['keypoints'].get('leftHip'),
            'rightHip': body_parts_data['keypoints'].get('rightHip'),
            'leftKnee': body_parts_data['keypoints'].get('leftKnee'),
            'rightKnee': body_parts_data['keypoints'].get('rightKnee'),
            'leftAnkle': body_parts_data['keypoints'].get('leftAnkle'),
            'rightAnkle': body_parts_data['keypoints'].get('rightAnkle')
        }
        
        print("Received data:", extracted_data)
        print("Image saved to:", filepath)
        return jsonify({"message": "Data and image received successfully"}), 200
    else:
        return {"members": ["Member1", "Member2", "Member3"]}

if __name__ == "__main__":
    app.run(debug=True)
