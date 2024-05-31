from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import json
from torsoFront import torsoFront
from Arm import Arm
from upperLeg import upperLeg
from PIL import Image
from Body import Body
import os


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
  # Extract the necessary data from the received JSON
        body_parts = {}
        body_part_scores = {}

        for part, data in body_parts_data['keypoints'].items():
            body_parts[part] = data['position']
            body_part_scores[part] = {'score': data['score']}
        
        # print("Received data:", body_parts_data)
        #print("Body parts:", body_parts)
        print("Body part scores:", body_part_scores)
        #body.showWarpingPoints('torso')
        #body.showWarpingPoints('face')

        #body.warp('torso')
        #body.warp('rightArm')
        #body.warp('leftArm')

        #body.warp('rightLeg')
        #body.warp('leftLeg')
        #body.warp('leftLowerLeg')
        #body.warp('hip')
        #body.warp('rightLowerLeg')
        #body.showWarpingPoints('face')
        #body.warp('face')
        #body.showWarpingPoints('torso')
        #body.showWarpingPoints('arms')
        
        body = Body(body_parts, image, True)
        body.warp('face')
        body.save(cropImage=False)
        edited_filepath = os.path.join(UPLOAD_FOLDER, "edited.png")
        
        return send_file(edited_filepath, mimetype='image/png')
    else:
        return jsonify({"members": ["Member1", "Member2", "Member3"]})

if __name__ == "__main__":
    app.run(debug=True)