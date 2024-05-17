import numpy as np
import cv2
import os
from torsoFront import torsoFront

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

# show_detected_points(body_parts)
#height, width = im.shape[:2]


class Arm:

    def __init__(self, im_, body_parts_, epsilon_):
        
        self.epsilon_arm = epsilon_
        self.body_parts = body_parts_
        self.im_ = im_
        self.height, self.width = im_.shape[:2]

        self.rightShoulder_x =  int( self.body_parts['rightShoulder']['x']) #  rightShoulder's x coordinate
        self.rightShoulder_y = int( self.body_parts['rightShoulder']['y'])  # rightShoulder's y coordinate
        
        self.rightElbow_x =  int( self.body_parts['rightElbow']['x']) #  rightElbow's x coordinate
        self.rightElbow_y = int( self.body_parts['rightElbow']['y'])  # rightElbow's y coordinate
        
        self.leftShoulder_x =  int( self.body_parts['leftShoulder']['x']) #  leftShoulder's x coordinate
        self.leftShoulder_y = int( self.body_parts['leftShoulder']['y'])  # leftShoulder's y coordinate
      
           
        self.leftElbow_x =  int( self.body_parts['leftElbow']['x']) #  leftElbow's x coordinate
        self.leftElbow_y = int( self.body_parts['leftElbow']['y'])  # leftElbow's y coordinate  
    
        self.setMiddlePointRightArm()
        self.setMiddlePointLeftArm()
        
                
        self.setRightArmPoints()
        self.setLeftArmPoints()
        
        #self.showAllArmPoints()
        #self.performWarpingLeftArm()
        #self.performWarpingRightArm()

    def getImage(self):
    
        return self.im_
    
    def setRightArmPoints(self):
                
        rRs2x_ = int(self.midRightX)
        rRs2y_ = int(self.midRightY)
        
        rRs1x_ = int((self.rightShoulder_x + rRs2x_) / 2)
        rRs1y_ = int((self.rightShoulder_y + rRs2y_) / 2)

        rRs3x_ = int((self.rightElbow_x + self.midRightX) / 2)        
        rRs3y_ = int((self.rightElbow_y + self.midRightY) / 2)   
                
        diff_x = (rRs1x_ - rRs3x_)

        rRs1x = rRs1x_ - diff_x
        rRs2x = rRs2x_ - diff_x
        rRs3x = rRs3x_ - diff_x
        
        rRs1y = rRs1y_
        rRs2y = rRs2y_
        rRs3y = rRs3y_
        
        self.r1 = Point(rRs1x, rRs1y, rRs1x_, rRs1y_, self.epsilon_arm)
        self.r2 = Point(rRs2x, rRs2y, rRs2x_, rRs2y_, self.epsilon_arm)
        self.r3 = Point(rRs3x, rRs3y, rRs3x_, rRs3y_, self.epsilon_arm)

    def showRightArmPoints(self):
        copy_im = self.im_.copy()
        
        self.r1.drawPoint(copy_im)
        self.r2.drawPoint(copy_im)
        self.r3.drawPoint(copy_im)
        
        return copy_im
    
    def showLeftArmPoints(self):
        copy_im = self.im_.copy()
        
        self.l1.drawPoint(copy_im)
        self.l2.drawPoint(copy_im)
        self.l3.drawPoint(copy_im)
        
        return copy_im
    
    def showAllArmPoints(self):
        
        copy_im = self.im_.copy()
        
        self.r1.drawPoint(copy_im)
        self.r2.drawPoint(copy_im)
        self.r3.drawPoint(copy_im)

        self.l1.drawPoint(copy_im)
        self.l2.drawPoint(copy_im)
        self.l3.drawPoint(copy_im)
        
        return copy_im
    
    def setLeftArmPoints(self):
            
        lRs2x = int(self.midLeftX)
        lRs2y = int(self.midLeftY)
           
        lRs1x = int((self.leftShoulder_x + lRs2x) / 2)
        lRs1y = int((self.leftShoulder_y + lRs2y) / 2)
        
        lRs3x = int((self.leftElbow_x + self.midLeftX) / 2)
        lRs3y = int((self.leftElbow_y + self.midLeftY) / 2)   
        
        diff_x = (lRs3x - lRs1x)
        
        lRs1x_ = lRs1x + diff_x
        lRs2x_ = lRs2x + diff_x
        lRs3x_ = lRs3x + diff_x
    
        lRs1y_ = lRs1y
        lRs2y_ = lRs2y
        lRs3y_ = lRs3y
        
        self.l1 = Point(lRs1x, lRs1y, lRs1x_, lRs1y_, self.epsilon_arm)
        self.l2 = Point(lRs2x, lRs2y, lRs2x_, lRs2y_, self.epsilon_arm)
        self.l3 = Point(lRs3x, lRs3y, lRs3x_, lRs3y_, self.epsilon_arm)
        
    def performWarpingLeftArm(self, im):
        
        """
            Performs TPS on left arm
        """
        self.im_ = im                

        l1x, l1y, l1x_, l1y_ = self.l1.getSourcePoints()
        d1x, d1y, d1x_, d1y_ = self.l1.getDestinationPoints()
        
        l2x, l2y, l2x_, l2y_ = self.l2.getSourcePoints()
        d2x, d2y, d2x_, d2y_ = self.l2.getDestinationPoints()
        
        l3x, l3y, l3x_, l3y_ = self.l3.getSourcePoints()
        d3x, d3y, d3x_, d3y_ = self.l3.getDestinationPoints()
        
        source_points = np.array([
        [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
        [l1x, l1y], [l1x_, l1y_],
        [l2x, l2y], [l2x_, l2y_],
        [l3x, l3y], [l3x_, l3y_]
        ])
        
        destination_points = np.array([
            [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
            [d1x, d1y], [d1x_, d1y_],
            [d2x, d2y], [d2x_, d2y_],
            [d3x, d3y], [d3x_, d3y_]
        ])
        
        new_im = tps.warpPoints(self.im_, source_points, destination_points)  
        self.im_ = new_im                

        return self.im_
    
    def performWarpingRightArm(self, im):
        """
            
            Performs TPS on right arm
        
        """        
        self.im_ = im                

        s1x, s1y, s1x_, s1y_ = self.r1.getSourcePoints()
        d1x, d1y, d1x_, d1y_ = self.r1.getDestinationPoints()
        
        s2x, s2y, s2x_, s2y_ = self.r2.getSourcePoints()
        d2x, d2y, d2x_, d2y_ = self.r2.getDestinationPoints()
        
        s3x, s3y, s3x_, s3y_ = self.r3.getSourcePoints()
        d3x, d3y, d3x_, d3y_ = self.r3.getDestinationPoints()
        
        source_points = np.array([
        [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
        [s1x, s1y], [s1x_, s1y_],
        [s2x, s2y], [s2x_, s2y_],
        [s3x, s3y], [s3x_, s3y_]
        ])
        
        destination_points = np.array([
            [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
            [d1x, d1y], [d1x_, d1y_],
            [d2x, d2y], [d2x_, d2y_],
            [d3x, d3y], [d3x_, d3y_]
        ])
        
        new_im = tps.warpPoints(self.im_, source_points, destination_points)
        
        self.im_ = new_im
                        
        return self.im_

    def setMiddlePointRightArm(self):
        
        """
            Sets middle point of right arm
        """
        self.midRightX = int((self.rightShoulder_x + self.rightElbow_x) / 2)
        
        self.midRightY = int((self.rightShoulder_y + self.rightElbow_y) / 2)
        
        

    def setMiddlePointLeftArm(self):
        
        """
            Sets middle point of left and right hips.
        """
        self.midLeftX = int((self.leftShoulder_x + self.leftElbow_x) / 2)
        
        self.midLeftY = int((self.rightShoulder_y + self.leftElbow_y) / 2)



    
if __name__ == "__main__":
    
    # Arms = Arm(im,body_parts, 5)
    # #Arms.showAllArmPoints()
    
    # Arms.performWarpingLeftArm()
    # Arms.performWarpingRightArm()
    # im_ = Arms.getImage()

    torso = torsoFront(im, body_parts)
    im = torso.performHorizontalWarping('belly', im)
    
    im = torso.performHorizontalWarping('waist', im)

    cv2.imwrite('edited.png',im)

    # torso.showAllSourcePoints()
    
    # torso.performHorizontalWarping('belly')
    # torso.performHorizontalWarping('waist')


    # im_ = torso.getImage()
    
    # cv2.imshow("pls",im)
    # cv2.waitKey(0)

    # cv2.imshow("pls",im_)
    # cv2.waitKey(0)