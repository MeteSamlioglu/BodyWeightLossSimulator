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
from core import pointMath

SCORE_TRESHOLD = 0.35
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
        print(f'file path {filepath}')
        if image is None:
            return jsonify({"error": "Failed to read image"}), 400

        # Extract the necessary data from the received JSON
        body_parts = {}
        body_part_scores = {}

        for part, data in body_parts_data['keypoints'].items():
            body_parts[part] = data['position']
            body_part_scores[part] = {'score': data['score']}
        
        detected_body_part_set = pointMath.generate_boolean_scores(body_part_scores, SCORE_TRESHOLD)
        
        percentages = {
            'face': 0.01, 
            'torso': 0.02, 
            'upperLegs': 0.01, 
            'hips': 0.02,
            'Arms': 0.005, 
            'lowerLeg': 0.01
        }
        body = Body(body_parts, detected_body_part_set, percentages, image, True)
        body.warpAllDetectedParts()
        body.save(cropImage=False)
        edited_filepath = os.path.join(UPLOAD_FOLDER, "resized_image.png")
        
        return send_file(edited_filepath, mimetype='image/png')
    else:
        return jsonify({"members": ["Member1", "Member2", "Member3"]})

@app.route("/UploadImage", methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    month = request.form.get('month')
    if not month:
        return jsonify({"error": "No month information"}), 400
    
    body_parts_data = request.form.get('data')
    if not body_parts_data:
        return jsonify({"error": "No body parts data"}), 400

    body_parts_data = json.loads(body_parts_data)

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"{month}_{filename}")
    file.save(filepath)
    
    filepath = "uploads/resized_image.png"
    
    image = cv2.imread(filepath)

    if image is None:
        return jsonify({"error": "Failed to read image"}), 400
    
    body_parts = {}    
    body_part_scores = {}

    for part, data in body_parts_data['keypoints'].items():
        body_parts[part] = data['position']
        body_part_scores[part] = {'score': data['score']}
    
    detected_body_part_set = pointMath.generate_boolean_scores(body_part_scores, SCORE_TRESHOLD)
    
    print(f'body parts {body_parts}')
    
    percentages = {
        'face': 0.01, 
        'torso': 0.02, 
        'upperLegs': 0.01, 
        'hips': 0.02,
        'Arms': 0.005, 
        'lowerLeg': 0.01
    }
    x  = 1
    if(month == "1th"):
        x = 1
    if(month == "3th"):
        x = 2
    if(month == "6th"):
        x = 3
    if(month == "12th"):
        x = 4    
    if(month == "15th"):
        x = 5   
    if(month == "18th"):
        x = 6  
    if(month == "24th"):
        x = 7  
    updated_percentages = {key: value * x for key, value in percentages.items()}

    body2 = Body(body_parts, detected_body_part_set, updated_percentages, image, True)
    body2.warpAllDetectedParts()
    body2.save(cropImage=False)
    edited_filepath = os.path.join(UPLOAD_FOLDER, "edited.png")    

    return send_file(edited_filepath, mimetype='image/png')


@app.route("/AdvancedWeightLoss", methods=['POST'])
def advanced_weight_loss():
    if 'image' not in request.files:
        print("No image part in request files")
        return jsonify({"error": "No image part"}), 400

    file = request.files['image']
    if file.filename == '':
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400

    face = request.form.get('face')
    torso = request.form.get('torso')
    upperLegs = request.form.get('upperLegs')
    hips = request.form.get('hips')
    arms = request.form.get('arms')
    lowerLegs = request.form.get('lowerLegs')

    print("Received values:", face, torso, upperLegs, hips, arms, lowerLegs)

    if not all([face, torso, upperLegs, hips, arms, lowerLegs]):
        print("Missing percentage values")
        return jsonify({"error": "Missing percentage values"}), 400

    body_parts_data = request.form.get('data')
    if not body_parts_data:
        print("No body parts data")
        return jsonify({"error": "No body parts data"}), 400

    body_parts_data = json.loads(body_parts_data)

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    print(f'File saved to {filepath}')

    if image is None:
        print("Failed to read image")
        return jsonify({"error": "Failed to read image"}), 400

    body_parts = {}
    body_part_scores = {}

    for part, data in body_parts_data['keypoints'].items():
        body_parts[part] = data['position']
        body_part_scores[part] = {'score': data['score']}
    
    detected_body_part_set = pointMath.generate_boolean_scores(body_part_scores, SCORE_TRESHOLD)

    percentages = {
        'face': float(face), 
        'torso': float(torso), 
        'upperLegs': float(upperLegs), 
        'hips': float(hips),
        'Arms': float(arms), 
        'lowerLeg': float(lowerLegs)
    }
    print(f'face {float(face)} torso {float(torso)} upperLegs {float(upperLegs)} hips {float(hips)}  Arms {float(arms)} lowerLeg {float(lowerLegs)}')
    
    body = Body(body_parts, detected_body_part_set, percentages, image, True)
    body.warpAllDetectedParts()
    body.save(cropImage=False)
    edited_filepath = os.path.join(UPLOAD_FOLDER, "edited.png")

    return send_file(edited_filepath, mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)
