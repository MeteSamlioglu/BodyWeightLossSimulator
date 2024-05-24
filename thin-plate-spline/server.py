from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/BodyWeightLoss", methods=['GET', 'POST'])
def body_weight_loss():
    if request.method == 'POST':
        body_parts_data = request.get_json()
        
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
        return jsonify({"message": "Data received successfully"}), 200
    else:
        return {"members": ["Member1", "Member2", "Member3"]}

if __name__ == "__main__":
    app.run(debug=True)
