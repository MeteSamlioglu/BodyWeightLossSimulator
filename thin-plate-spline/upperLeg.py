import numpy as np
import cv2
import os
from core import Point
from core import tps

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

body_parts = {
    'nose': {'x': 420.58755502194236, 'y': 178.5611058174169},
    'leftEye': {'x': 443.9297769482637, 'y': 155.00383722986706},
    'rightEye': {'x' : 400.32382274830314, 'y': 154.5108393834273},
    'leftEar': {'x' : 484.14493676641524, 'y': 169.86089124694442},
    'rightEar': {'x' : 373.1273169814852, 'y': 166.2998638175393},
    'leftShoulder': {'x' : 546.6067086777566, 'y': 303.03412830215905},
    'rightShoulder': {'x' : 304.9749470682122, 'y': 299.44493342859323},
    'leftElbow': {'x' : 630.3275056446947, 'y': 474.6997010503284},
    'rightElbow': {'x' : 226.439762177148, 'y': 456.13560834652543},
    'leftWrist': {'x' : 549.1664952681191, 'y': 619.3380513168905},
    'rightWrist': {'x' : 277.1653803677812, 'y': 623.2991806244515},
    'leftHip': {'x' : 525.0838155702411, 'y': 651.4093179970561},
    'rightHip': {'x' : 339.0650141365809, 'y': 658.5777996854737},
    'leftKnee': {'x' : 548.7685618642847, 'y': 871.7984340864112},
    'rightKnee': {'x' : 330.1582183749791, 'y': 893.0001537067097},
    'leftAnkle': {'x' : 577.1193602663265, 'y': 1150.7460334557638},
    'rightAnkle': {'x' : 309.0422612558061, 'y': 1159.517769210982}
}

global im
im = cv2.imread('samples//sample5.jpg')


if im is None:
    print("Error loading image. Check the file path.")
    exit(1)

def show_detected_points(body_parts):
    for part, coords in body_parts.items():
        x = int(coords['x'])
        y = int(coords['y'])
        cv2.circle(im, (x, y), 6, (255, 0, 0), -1)  # Blue color with filled circle
    
    #cv2.imshow('Image with Keypoints', im)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

show_detected_points(body_parts)
leg_json = body_parts['leftKnee']

height, width = im.shape[:2]

class upperLeg:
    
    def __init__(self, epsilon_,  body_parts):
        
        global im
        
        self.im_ = im
        
        self.epsilon_leg = epsilon_
    
        self.rightKnee_x =  int(body_parts['rightKnee']['x']) # Leg Knee's x coordinate
        self.rightKnee_y = int(body_parts['rightKnee']['y'])  # Leg Knee's y coordinate
        
        self.leftKnee_x =  int(body_parts['leftKnee']['x']) # Leg Knee's x coordinate
        self.leftKnee_y =  int(body_parts['leftKnee']['y'])  # Leg Knee's y coordinate
        
        self.hipLeft_x = int(body_parts['leftHip']['x'])    # Left Hip's x coordinate
        self.hipLeft_y = int(body_parts['leftHip']['y'])    # Left Hip's y coordinate
        
        self.hipRight_x = int(body_parts['rightHip']['x'])    # Left Hip's x coordinate
        self.hipRight_y = int(body_parts['rightHip']['y'])    # Left Hip's y coordinate
        
        self.midX, self.midY = self.findHipsMiddlePoints()
        
        self.r1, self.r2, self.r3 = self.getRightLegPoints()
        self.l1, self.l2, self.l3 = self.getLeftLegPoints()
                
        #self.showRightLegPoints()
        #self.showLeftLegPoints()
  
    
    def findHipsMiddlePoints(self):
        
        midX = int((self.hipLeft_x + self.hipRight_x) / 2)
        
        midY = int((self.hipLeft_y + self.hipRight_y) / 2)

        return midX, midY
        
    def getRightLegPoints(self):
             
        rightLeg_midX = int((self.hipRight_x + self.rightKnee_x) / 2) 
        rightLeg_midY = int((self.hipRight_y + self.rightKnee_y) / 2)
        
        rLs1x_ = int((rightLeg_midX + self.midX) / 2)
        rLs1y_ = int((rightLeg_midY + self.midY) / 2)

        rLs2x_ = int((self.midX + rightLeg_midX) / 2)
        rLs2y_ = rightLeg_midY
                
        rLs3x_ = rLs2x_
        rLs3y_ = rLs2y_ + (rLs2y_ - rLs1y_)
                
        rLs1x = rightLeg_midX - (rLs1x_- rightLeg_midX)
        rLs1y = rLs1y_
        
        rLs2x = rightLeg_midX - (rLs2x_- rightLeg_midX)
        rLs2y = rLs2y_
        
        rLs3x = rightLeg_midX - (rLs3x_- rightLeg_midX)
        rLs3y = rLs3y_
        
        p1 = Point(rLs1x, rLs1y, rLs1x_, rLs1y_, self.epsilon_leg)
        p2 = Point(rLs2x, rLs2y, rLs2x_, rLs2y_, self.epsilon_leg)
        p3 = Point(rLs3x, rLs3y, rLs3x_, rLs3y_, self.epsilon_leg)
        
        return p1, p2, p3 
    
    def getLeftLegPoints(self):
        
        leftLeg_midX = int((self.hipLeft_x + self.leftKnee_x) / 2) 
        leftLeg_midY = int((self.hipRight_y + self.leftKnee_y) / 2)
        
        lLs1x = int((leftLeg_midX + self.midX) / 2)
        lLs1y = int((leftLeg_midY + self.midY) / 2)
        
        lLs2x = int((self.midX + leftLeg_midX) / 2)
        lLs2y = leftLeg_midY
        
        lLs3x = lLs2x
        lLs3y = lLs2y + (lLs2y - lLs1y)
        
        lLs1x_ = leftLeg_midX + (leftLeg_midX - lLs1x)
        lLs1y_ = lLs1y
        
        lLs2x_ = leftLeg_midX + (leftLeg_midX - lLs2x)
        lLs2y_ = leftLeg_midY
        
        lLs3x_ = leftLeg_midX + (leftLeg_midX - lLs3x)
        lLs3y_ = lLs3y
        
        p1 = Point(lLs1x, lLs1y, lLs1x_, lLs1y_, self.epsilon_leg)
        p2 = Point(lLs2x, lLs2y, lLs2x_, lLs2y_, self.epsilon_leg)
        p3 = Point(lLs3x, lLs3y, lLs3x_, lLs3y_, self.epsilon_leg)
        
        return p1, p2, p3
        
    def performWarpingRightLeg(self):
        s1x, s1y, s1x_, s1y_ = self.r1.getSourcePoints()
        d1x, d1y, d1x_, d1y_ = self.r1.getDestinationPoints()
        
        s2x, s2y, s2x_, s2y_ = self.r2.getSourcePoints()
        d2x, d2y, d2x_, d2y_ = self.r2.getDestinationPoints()
        
        s3x, s3y, s3x_, s3y_ = self.r3.getSourcePoints()
        d3x, d3y, d3x_, d3y_ = self.r3.getDestinationPoints()
        
        source_points = np.array([
        [0, 0], [width, 0], [0, height], [width, width], 
        [s1x, s1y], [s1x_, s1y_],
        [s2x, s2y], [s2x_, s2y_],
        [s3x, s3y], [s3x_, s3y_]
        ])
        
        destination_points = np.array([
            [0, 0], [width, 0], [0, height], [width, width], 
            [d1x, d1y], [d1x_, d1y_],
            [d2x, d2y], [d2x_, d2y_],
            [d3x, d3y], [d3x_, d3y_]
        ])
        
        new_im = tps.warpPoints(self.im_, source_points, destination_points)
        
        self.im_ = new_im                

    
    def performWarpingLefttLeg(self):
        
        l1x, l1y, l1x_, l1y_ = self.l1.getSourcePoints()
        d1x, d1y, d1x_, d1y_ = self.l1.getDestinationPoints()
        
        l2x, l2y, l2x_, l2y_ = self.l2.getSourcePoints()
        d2x, d2y, d2x_, d2y_ = self.l2.getDestinationPoints()
        
        l3x, l3y, l3x_, l3y_ = self.l3.getSourcePoints()
        d3x, d3y, d3x_, d3y_ = self.l3.getDestinationPoints()
        
        source_points = np.array([
        [0, 0], [width, 0], [0, height], [width, width], 
        [l1x, l1y], [l1x_, l1y_],
        [l2x, l2y], [l2x_, l2y_],
        [l3x, l3y], [l3x_, l3y_]
        ])
        
        destination_points = np.array([
            [0, 0], [width, 0], [0, height], [width, width], 
            [d1x, d1y], [d1x_, d1y_],
            [d2x, d2y], [d2x_, d2y_],
            [d3x, d3y], [d3x_, d3y_]
        ])
        
        new_im = tps.warpPoints(self.im_, source_points, destination_points)
        
        self.im_ = new_im                
    

    def showRightLegPoints(self):
        
        copy_im = self.im_.copy()
        
        self.r1.drawPoint(copy_im)
        self.r2.drawPoint(copy_im)
        self.r3.drawPoint(copy_im)
        
        cv2.imshow("Right Leg Points",copy_im)        
        cv2.waitKey(0)
        
    def showLeftLegPoints(self):
        
        copy_im = self.im_.copy()
        
        self.l1.drawPoint(copy_im)
        self.l2.drawPoint(copy_im)
        self.l3.drawPoint(copy_im)
        
        cv2.imshow("Left Leg Points",copy_im)        
        cv2.waitKey(0)
    
        
if __name__ == "__main__":
    
    leftLeg = upperLeg(5, body_parts)
    
    leftLeg.showRightLegPoints()
    
    leftLeg.performWarpingLefttLeg()

    leftLeg.showRightLegPoints()



    




