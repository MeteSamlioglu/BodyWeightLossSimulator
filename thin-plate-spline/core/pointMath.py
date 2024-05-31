import math

def custom_round(value):
    if value - int(value) >= 0.5:
        return math.ceil(value)
    else:
        return math.floor(value)
    

def generate_boolean_scores(body_part_scores, threshold):
    boolean_scores = {}
    
    face_parts = ['leftEar', 'rightEar', 'nose', 'leftEye', 'rightEye']
    torso_parts = ['leftHip', 'rightHip', 'leftShoulder', 'rightShoulder']
    
    left_arm_parts = ['leftShoulder', 'leftElbow', 'leftWrist']
    right_arm_parts = ['rightShoulder', 'rightElbow', 'rightWrist']
    
    upper_left_leg_parts = ['leftKnee', 'leftHip', 'rightHip']
    upper_right_leg_parts = ['rightKnee', 'leftHip', 'rightHip']

    hips_parts = ['leftHip', 'rightHip']
    
    lower_left_leg_parts = ['leftKnee', 'leftAnkle']
    lower_right_leg_parts = ['rightKnee', 'rightAnkle']
    
    boolean_scores['face'] = all(body_part_scores[part]['score'] > threshold for part in face_parts)
    boolean_scores['torso'] = all(body_part_scores[part]['score'] > threshold for part in torso_parts)
    boolean_scores['leftArm'] = all(body_part_scores[part]['score'] > threshold for part in left_arm_parts)
    boolean_scores['rightArm'] = all(body_part_scores[part]['score'] > threshold for part in right_arm_parts)
    boolean_scores['leftUpperLeg'] = all(body_part_scores[part]['score'] > threshold for part in upper_left_leg_parts)
    boolean_scores['rightUpperLeg'] = all(body_part_scores[part]['score'] > threshold for part in upper_right_leg_parts)
    boolean_scores['hips'] = all(body_part_scores[part]['score'] > threshold for part in hips_parts)
    boolean_scores['leftLowerLeg'] = all(body_part_scores[part]['score'] > threshold for part in lower_left_leg_parts)
    boolean_scores['rightLowerLeg'] = all(body_part_scores[part]['score'] > threshold for part in lower_right_leg_parts)

    # Optionally, you can also generate boolean scores for individual parts
    # for part, data in body_part_scores.items():
    #     boolean_scores[part] = data['score'] >= threshold
    
    return boolean_scores