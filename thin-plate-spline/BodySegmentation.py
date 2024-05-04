import numpy as np
import cv2
import os
from core import Point


 
# body_parts = {
#     'nose': {'x': 140.37497220298118, 'y': 55.910094292609244},
#     'leftEye': {'x': 147.79016284794773, 'y': 48.112344106038414},
#     'rightEye': {'x' : 127.25704479956813, 'y': 47.50153932990609 },
#     'leftEar': {'x' : 165.67790807679643, 'y': 57.405897827703},
#     'rightEar': {'x' : 114.28676995565726, 'y': 57.06981493396201},
#     'leftShoulder': {'x' : 192.1682789529017, 'y': 120.77464233999288},
#     'rightShoulder': {'x' : 94.3108498920766, 'y': 121.4811002570631},
#     'leftElbow': {'x' : 201.93303669145868, 'y': 196.94535177443927},
#     'rightElbow': {'x' : 83.03741319050161, 'y': 191.2138579721416},
#     'leftWrist': {'x' : 205.09550254289496, 'y': 267.17328204078115},
#     'rightWrist': {'x' : 77.92971506044847, 'y': 254.22849078492806},
#     'leftHip': {'x' : 167.87436809096226, 'y': 261.15734365134887},
#     'rightHip': {'x' : 120.21855589400892, 'y': 249.63132546672892},
#     'leftKnee': {'x' : 170.73546759287518, 'y': 381.0585201633719},
#     'rightKnee': {'x' : 105.185312951258, 'y': 376.1818199122782},
#     'leftAnkle': {'x' : 168.36852695960408, 'y': 491.7963871999538},
#     'rightAnkle': {'x' : 101.11777239714483, 'y': 488.409336907523}
# }
# body_parts = {
#     'nose': {'x': 420.58755502194236, 'y': 178.5611058174169},
#     'leftEye': {'x': 443.9297769482637, 'y': 155.00383722986706},
#     'rightEye': {'x' : 400.32382274830314, 'y': 154.5108393834273},
#     'leftEar': {'x' : 484.14493676641524, 'y': 169.86089124694442},
#     'rightEar': {'x' : 373.1273169814852, 'y': 166.2998638175393},
#     'leftShoulder': {'x' : 546.6067086777566, 'y': 303.03412830215905},
#     'rightShoulder': {'x' : 304.9749470682122, 'y': 299.44493342859323},
#     'leftElbow': {'x' : 630.3275056446947, 'y': 474.6997010503284},
#     'rightElbow': {'x' : 226.439762177148, 'y': 456.13560834652543},
#     'leftWrist': {'x' : 549.1664952681191, 'y': 619.3380513168905},
#     'rightWrist': {'x' : 277.1653803677812, 'y': 623.2991806244515},
#     'leftHip': {'x' : 525.0838155702411, 'y': 651.4093179970561},
#     'rightHip': {'x' : 339.0650141365809, 'y': 658.5777996854737},
#     'leftKnee': {'x' : 548.7685618642847, 'y': 871.7984340864112},
#     'rightKnee': {'x' : 330.1582183749791, 'y': 893.0001537067097},
#     'leftAnkle': {'x' : 577.1193602663265, 'y': 1150.7460334557638},
#     'rightAnkle': {'x' : 309.0422612558061, 'y': 1159.517769210982}
# }


def show_detected_points(body_parts):
    for part, coords in body_parts.items():
        x = int(coords['x'])
        y = int(coords['y'])
        cv2.circle(im, (x, y), 6, (255, 0, 0), -1)  # Blue color with filled circle
    
    cv2.imshow('Image with Keypoints', im)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

im = cv2.imread('sample5.jpg')

if im is None:
    print("Error loading image. Check the file path.")
    exit(1)

# show_detected_points(body_parts)

epsilon_waist = 1
epsilon_bust = 1
epsilon_hip = 1
epsilon_shoulder = 0.1

x1, y1 = int(body_parts['leftShoulder']['x']), int(body_parts['leftShoulder']['y'])
x2, y2 = int(body_parts['rightShoulder']['x']), int(body_parts['rightShoulder']['y'])

if body_parts['leftHip']['y'] < body_parts['rightHip']['y']:
    right_aligned = False
    left_aligned = True
else:
    left_aligned = False
    right_aligned = True
    
left_shoulder_x =  int(body_parts['leftShoulder']['x']) # Shoulder's x coordinate
left_shoulder_y =  int(body_parts['leftShoulder']['y']) # Shoulder's y coordinate
left_hip_x = int(body_parts['leftHip']['x'])            # Hip's x coordinate
diff_left = (left_shoulder_x - left_hip_x) /3
left_hip_x = int(left_hip_x + 2*diff_left/3)            #Fixed Hip point
left_hip_y = int(body_parts['leftHip']['y'])            # Hip's y coordinate
                                       
right_shoulder_x = int(body_parts['rightShoulder']['x']) # Shoulder's x coordinate
right_shoulder_y = int(body_parts['rightShoulder']['y']) # Shoulder's y coordinate
right_hip_x = int(body_parts['rightHip']['x'])           # Hip's x coordinate
diff_right = (right_hip_x - right_shoulder_x) / 3
right_hip_x = int(right_shoulder_x + diff_right)
right_hip_y = int(body_parts['rightHip']['y'])           # Hip's x coordinate

if(left_aligned):
    #For Left
    left_bust_x = int(left_shoulder_x - (left_shoulder_x - left_hip_x) / 3)
    left_bust_y = int(left_shoulder_y - (left_shoulder_y - left_hip_y) / 3)
    left_waist_x = int(left_shoulder_x - 2 * (left_shoulder_x - left_hip_x) / 3)
    left_waist_y = int(left_shoulder_y - 2 * (left_shoulder_y - left_hip_y) / 3)

    right_hip_y = left_hip_y
    right_bust_x = int(right_shoulder_x - (right_shoulder_x - right_hip_x) / 3)
    right_bust_y = left_bust_y
    right_waist_x = int(right_shoulder_x - 2 * (right_shoulder_x - right_hip_x) / 3)
    right_waist_y = left_waist_y
else:
    #For Right
    right_bust_x = int(right_shoulder_x - (right_shoulder_x - right_hip_x) / 3)
    right_bust_y = int(right_shoulder_y - (right_shoulder_y - right_hip_y) / 3)
    right_waist_x = int(right_shoulder_x - 2 * (right_shoulder_x - right_hip_x) / 3)
    right_waist_y = int(right_shoulder_y - 2 * (right_shoulder_y - right_hip_y) / 3)

    left_hip_y = right_hip_y
    left_bust_x = int(left_shoulder_x - (left_shoulder_x - left_hip_x) / 3)
    left_bust_y = right_bust_y
    left_waist_x = int(left_shoulder_x - 2 * (left_shoulder_x - left_hip_x) / 3)
    left_waist_y = right_waist_y

bust = Point(right_bust_x, right_bust_y, left_bust_x, left_bust_y, epsilon_bust)
hip  = Point(right_hip_x, right_hip_y, left_hip_x, left_hip_y, epsilon_hip)
shoulder = Point(right_shoulder_x, right_shoulder_y, left_shoulder_x, left_shoulder_y, epsilon_shoulder)
waist1 = Point(right_waist_x, right_waist_y, left_waist_x, left_waist_y, epsilon_waist)

print(bust)
print(hip)
print(shoulder)
print(waist1)

# Draw small red circles at the division points
cv2.circle(im, (right_shoulder_x, right_shoulder_y), 4, (0, 0, 255), -1)                # Right-Shoulder-Point 
cv2.circle(im, (right_bust_x, right_bust_y), 4, (0, 0, 255), -1)                        # Right-Bust-Point 
cv2.circle(im, (right_waist_x, right_waist_y), 4, (0, 0, 255 ), -1)                     # Right-Waist-Point 
cv2.circle(im, (right_hip_x, right_hip_y), 4, (0, 0, 255 ), -1)                         # Right-Hip-Point 

cv2.circle(im, (left_shoulder_x, left_shoulder_y), 4, (0, 0, 255), -1)                # Left-Shoulder-Point 
cv2.circle(im, (left_bust_x, left_bust_y), 4, (0, 0, 255), -1)                        # Left-Bust-Point 
cv2.circle(im, (left_waist_x, left_waist_y), 4, (0, 0, 255 ), -1)                     # Left-Waist-Point 
cv2.circle(im, (left_hip_x, left_hip_y), 4, (0, 0, 255 ), -1)                         # Left-Hip-Point 

cv2.line(im, (right_shoulder_x, right_shoulder_y), (right_hip_x, right_hip_y), (0, 255, 0), 2)          # Green line
cv2.line(im, (left_shoulder_x, left_shoulder_y), (left_hip_x, left_hip_y), (0, 255, 0), 2)              # Green line


cv2.imshow('line',im)
cv2.waitKey(0)