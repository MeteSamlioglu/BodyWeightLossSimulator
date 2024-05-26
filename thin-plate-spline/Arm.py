import numpy as np
import cv2
import os
from torsoFront import torsoFront

from core import Point
from core import tps
from core import pointMath

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

MIN_ARM_DIFF = 6

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
        
        self.rightWrist_x =  int( self.body_parts['rightWrist']['x'])
        self.rightWrist_y =  int( self.body_parts['rightWrist']['y'])
        
        self.leftWrist_x =  int( self.body_parts['leftWrist']['x'])
        self.leftWrist_y =  int( self.body_parts['leftWrist']['y'])
    
        self.setMiddlePointRightArm()
        self.setMiddlePointLeftArm()
        
                
        self.setRightArmPoints()
        
        self.setLeftArmPoints()
        
        #self.showAllArmPoints()
        #self.performWarpingLeftArm()
        #self.performWarpingRightArm()

    def getImage(self):
    
        return self.im_
    
    def getDifference(self):
        """
            Find the optimal difference to set distance between left and right source point 
        """
        lRs2x = int(self.midLeftX)
        lRs1x = int((self.leftShoulder_x + lRs2x) / 2)
        lRs3x = int((self.leftElbow_x + self.midLeftX) / 2)
        rRs2x_ = int(self.midRightX)
        rRs1x_ = int((self.rightShoulder_x + rRs2x_) / 2)
        rRs3x_ = int((self.rightElbow_x + self.midRightX) / 2)        

        diffLeft = abs(lRs3x - lRs1x)
        diffRight = abs(rRs1x_ - rRs3x_)

        if(diffLeft < MIN_ARM_DIFF and diffRight < MIN_ARM_DIFF):
            return MIN_ARM_DIFF
        if(diffLeft <= diffRight):
            return diffRight
        else:
            return diffLeft
    
    def setRightArmPoints(self):
                
        rRs2x_ = int(self.midRightX)
        rRs2y_ = int(self.midRightY)
        
        rRs1x_ = int((self.rightShoulder_x + rRs2x_) / 2)
        rRs1y_ = int((self.rightShoulder_y + rRs2y_) / 2)

        rRs3x_ = int((self.rightElbow_x + self.midRightX) / 2)        
        rRs3y_ = int((self.rightElbow_y + self.midRightY) / 2)   
                
        diff_x = self.getDifference()
        print(f'Right Arm diff {diff_x}')

        rRs1x = rRs1x_ - diff_x
        rRs2x = rRs2x_ - diff_x
        rRs3x = rRs3x_ - diff_x
        
        rRs1y = rRs1y_
        rRs2y = rRs2y_
        rRs3y = rRs3y_
        
        # Lower Arm
 
        self.midLowerArm_x = int((self.rightWrist_x + self.rightElbow_x) / 2)
        self.midLowerArm_y = int((self.rightWrist_y + self.rightElbow_y) / 2)
        
        rRs7x = self.rightElbow_x - diff_x
        rRs7y = self.rightElbow_y
        
        rRs7x_ = self.rightElbow_x
        rRs7y_ = self.rightElbow_y
        
        rRs4x_ =  int((self.midLowerArm_x + self.rightElbow_x)/ 2)
        rRs4y_ =  int((self.midLowerArm_y + self.rightElbow_y)/ 2)
        rRs4x =  int((self.midLowerArm_x + self.rightElbow_x)/ 2) - diff_x
        rRs4y =  int((self.midLowerArm_y + self.rightElbow_y)/ 2)
        
        rRs5x_ =  self.midLowerArm_x
        rRs5y_ = self.midLowerArm_y
        rRs5x =  self.midLowerArm_x - diff_x
        rRs5y = self.midLowerArm_y
        
        midRs6x_ =  int((self.midLowerArm_x + self.rightWrist_x)/ 2)
        midRs6y_ =  int((self.midLowerArm_y + self.rightWrist_y)/ 2)
        
        rRs6x_ =  midRs6x_ 
        rRs6y_ = midRs6y_
        rRs6x =  midRs6x_  - diff_x
        rRs6y = midRs6y_

        self.r1 = Point(rRs1x, rRs1y, rRs1x_, rRs1y_, self.epsilon_arm)
        self.r2 = Point(rRs2x, rRs2y, rRs2x_, rRs2y_, self.epsilon_arm)
        self.r3 = Point(rRs3x, rRs3y, rRs3x_, rRs3y_, self.epsilon_arm)
        
        self.r4 = Point(rRs4x, rRs4y , rRs4x_, rRs4y_, self.epsilon_arm)
        self.r5 = Point(rRs5x, rRs5y , rRs5x_, rRs5y_, self.epsilon_arm)
        self.r6 = Point(rRs6x, rRs6y, rRs6x_, rRs6y_, self.epsilon_arm)
        self.r7 = Point(rRs7x, rRs7y, rRs7x_, rRs7y_, self.epsilon_arm)

        
        #self.r7 = Point(rRs6x, rRs6y, rRs6x_, rRs6y_, self.epsilon_arm)


    def showRightArmPoints(self):
        copy_im = self.im_.copy()
        print('Activated')

        
        self.r1.drawPoint(copy_im)
        self.r2.drawPoint(copy_im)
        self.r3.drawPoint(copy_im)
        
        self.r4.drawPoint(copy_im)
        self.r5.drawPoint(copy_im)
        self.r6.drawPoint(copy_im)
        self.r7.drawPoint(copy_im)

        return copy_im
    
    def showLeftArmPoints(self):
        copy_im = self.im_.copy()
        
        self.l1.drawPoint(copy_im)
        self.l2.drawPoint(copy_im)
        self.l3.drawPoint(copy_im)
        
        self.l4.drawPoint(copy_im)
        self.l5.drawPoint(copy_im)
        self.l6.drawPoint(copy_im)
        
        return copy_im
    
    def showAllArmPoints(self):
        
        copy_im = self.im_.copy()
        
        self.r1.drawPoint(copy_im)
        self.r2.drawPoint(copy_im)
        self.r3.drawPoint(copy_im)
        self.r4.drawPoint(copy_im)
        self.r5.drawPoint(copy_im)
        self.r6.drawPoint(copy_im)
        self.r7.drawPoint(copy_im)

        self.l1.drawPoint(copy_im)
        self.l2.drawPoint(copy_im)
        self.l3.drawPoint(copy_im)
        self.l4.drawPoint(copy_im)
        self.l5.drawPoint(copy_im)
        self.l6.drawPoint(copy_im)
        self.l7.drawPoint(copy_im)

        return copy_im
    
    def setLeftArmPoints(self):
            
        lRs2x = int(self.midLeftX)
        lRs2y = int(self.midLeftY)
           
        lRs1x = int((self.leftShoulder_x + lRs2x) / 2)
        lRs1y = int((self.leftShoulder_y + lRs2y) / 2)
        
        lRs3x = int((self.leftElbow_x + self.midLeftX) / 2)
        lRs3y = int((self.leftElbow_y + self.midLeftY) / 2)   
        
        diff_x = self.getDifference() 
        print(f'Left Arm diff {diff_x}')
        
        lRs1x_ = lRs1x + diff_x
        lRs2x_ = lRs2x + diff_x
        lRs3x_ = lRs3x + diff_x
    
        lRs1y_ = lRs1y
        lRs2y_ = lRs2y
        lRs3y_ = lRs3y
        
        self.l1 = Point(lRs1x, lRs1y, lRs1x_, lRs1y_, self.epsilon_arm)
        self.l2 = Point(lRs2x, lRs2y, lRs2x_, lRs2y_, self.epsilon_arm)
        self.l3 = Point(lRs3x, lRs3y, lRs3x_, lRs3y_, self.epsilon_arm)
        
        self.midLowerArm_x = int((self.leftWrist_x + self.leftElbow_x) / 2)
        self.midLowerArm_y = int((self.leftWrist_y + self.leftElbow_y) / 2)
             
        # Lower Arm
        
        lRs4x =  int((self.midLowerArm_x + self.leftElbow_x)/ 2)
        lRs4y =  int((self.midLowerArm_y + self.leftElbow_y)/ 2)
        lRs4x_ =  int((self.midLowerArm_x + self.leftElbow_x)/ 2) + diff_x
        lRs4y_ =  int((self.midLowerArm_y + self.leftElbow_y)/ 2)
        
        lRs7x = self.leftElbow_x
        lRs7x_ = self.leftElbow_x + diff_x
        lRs7y = self.leftElbow_y
        lRs7y_ = self.leftElbow_y

        lRs5x =  self.midLowerArm_x
        lRs5y = self.midLowerArm_y
        lRs5x_ =  self.midLowerArm_x + diff_x
        lRs5y_ = self.midLowerArm_y
        
        midRs6x_ =  int((self.midLowerArm_x + self.leftWrist_x)/ 2)
        midRs6y_ =  int((self.midLowerArm_y + self.leftWrist_y)/ 2)
        
        lRs6x =  midRs6x_ 
        lRs6y = midRs6y_
        lRs6x_ =  midRs6x_  + diff_x
        lRs6y_ = midRs6y_

        self.l1 = Point(lRs1x, lRs1y, lRs1x_, lRs1y_, self.epsilon_arm)
        self.l2 = Point(lRs2x, lRs2y, lRs2x_, lRs2y_, self.epsilon_arm)
        self.l3 = Point(lRs3x, lRs3y, lRs3x_, lRs3y_, self.epsilon_arm)
        
        self.l4 = Point(lRs4x, lRs4y , lRs4x_, lRs4y_, self.epsilon_arm)
        self.l5 = Point(lRs5x, lRs5y , lRs5x_, lRs5y_, self.epsilon_arm)
        self.l6 = Point(lRs6x, lRs6y, lRs6x_, lRs6y_, self.epsilon_arm)
        self.l7 = Point(lRs7x, lRs7y, lRs7x_, lRs7y_, self.epsilon_arm)

        
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
        
        l4x, l4y, l4x_, l4y_ = self.l4.getSourcePoints()
        d4x, d4y, d4x_, d4y_ = self.l4.getDestinationPoints()
        
        l5x, l5y, l5x_, l5y_ = self.l5.getSourcePoints()
        d5x, d5y, d5x_, d5y_ = self.l5.getDestinationPoints()
        
        l6x, l6y, l6x_, l6y_ = self.l6.getSourcePoints()
        d6x, d6y, d6x_, d6y_ = self.l6.getDestinationPoints() 
        
        l7x, l7y, l7x_, l7y_ = self.l7.getSourcePoints()
        d7x, d7y, d7x_, d7y_ = self.l7.getDestinationPoints() 
        
        source_points = np.array([
        [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
        #edited
        [0, l1y], [0, l2y], [0, l3y], [0, l4y], [0, l5y], [0, l6y], [0, l7y],
        [self.width, l1y_], [self.width, l2y_], [self.width, l3y_], [self.width, l4y_], [self.width, l5y_], [self.width, l6y_], [self.width, l7y_],
        #---------------
        
        [l1x, l1y], [l1x_, l1y_],
        [l2x, l2y], [l2x_, l2y_],
        [l3x, l3y], [l3x_, l3y_],
        
        [l4x, l4y], [l4x_, l4y_],
        [l5x, l5y], [l5x_, l5y_],
        [l6x, l6y], [l6x_, l6y_],
        [l7x, l7y], [l7x_, l7y_],
        
        ])
        
        destination_points = np.array([
            [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
            
            #edited
            [0, l1y], [0, l2y], [0, l3y],  [0, l4y],  [0, l5y],  [0, l6y], [0, l7y],
            [self.width, l1y_], [self.width, l2y_], [self.width, l3y_], [self.width, l4y_], [self.width, l5y_], [self.width, l6y_], [self.width, l7y_], 
            #---------------
            #---------------
            
            [d1x, d1y], [d1x_, d1y_],
            [d2x, d2y], [d2x_, d2y_],
            [d3x, d3y], [d3x_, d3y_],
            
            [d4x, d4y], [d4x_, d4y_],
            [d5x, d5y], [d5x_, d5y_],
            [d6x, d6y], [d6x_, d6y_],
            [d7x, d7y], [d7x_, d7y_]
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
        
        s4x, s4y, s4x_, s4y_ = self.r4.getSourcePoints()
        d4x, d4y, d4x_, d4y_ = self.r4.getDestinationPoints()
        
        s5x, s5y, s5x_, s5y_ = self.r5.getSourcePoints()
        d5x, d5y, d5x_, d5y_ = self.r5.getDestinationPoints()
        
        s6x, s6y, s6x_, s6y_ = self.r6.getSourcePoints()
        d6x, d6y, d6x_, d6y_ = self.r6.getDestinationPoints() 
        
        s7x, s7y, s7x_, s7y_ = self.r7.getSourcePoints()
        d7x, d7y, d7x_, d7y_ = self.r7.getDestinationPoints() 

        source_points = np.array([
        [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
        #edited
        [0, s1y], [0, s2y], [0, s3y], [0, s4y], [0, s5y], [0, s6y], [0, s7y],
        [self.width, s1y_], [self.width, s2y_], [self.width, s3y_], [self.width, s4y_], [self.width, s5y_], [self.width, s6y_], [self.width, s7y_],
        #---------------
        [s1x, s1y], [s1x_, s1y_],
        [s2x, s2y], [s2x_, s2y_],
        [s3x, s3y], [s3x_, s3y_],
        
        [s4x, s4y], [s4x_, s4y_],
        [s5x, s5y], [s5x_, s5y_],
        [s6x, s6y], [s6x_, s6y_],
        [s7x, s7y], [s7x_, s7y_]

        ])
        
        destination_points = np.array([
        [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
        #edited
        [0, s1y], [0, s2y], [0, s3y], [0, s4y], [0, s5y], [0, s6y], [0, s7y],
        [self.width, s1y_], [self.width, s2y_], [self.width, s3y_], [self.width, s4y_], [self.width, s5y_], [self.width, s6y_], [self.width, s7y_], 
        #--------------------------------
        [d1x, d1y], [d1x_, d1y_],
        [d2x, d2y], [d2x_, d2y_],
        [d3x, d3y], [d3x_, d3y_],
        
        [d4x, d4y], [d4x_, d4y_],
        [d5x, d5y], [d5x_, d5y_],
        [d6x, d6y], [d6x_, d6y_],
        [d7x, d7y], [d7x_, d7y_]

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

    
    def getPixelDistance(self, part):
        
        if(part == 'rightArm'):
            
            r1_right_source_x, _, r1_left_source_x, _  = self.r1.getSourcePoints()
            r2_right_source_x, _, r2_left_source_x, _ =  self.r2.getSourcePoints()
            r3_right_source_x, _, r3_left_source_x, _ =  self.r3.getSourcePoints()
            
            r4_right_source_x, _, r4_left_source_x, _  = self.r4.getSourcePoints()
            r5_right_source_x, _, r5_left_source_x, _ =  self.r5.getSourcePoints()
            r6_right_source_x, _, r6_left_source_x, _ =  self.r6.getSourcePoints()
            r7_right_source_x, _, r7_left_source_x, _ =  self.r7.getSourcePoints()

            d1 = r1_left_source_x - r1_right_source_x
            d2 = r2_left_source_x - r2_right_source_x
            d3 = r3_left_source_x - r3_right_source_x   
            
            d4 = r4_left_source_x - r4_right_source_x
            d5 = r5_left_source_x - r5_right_source_x
            d6 = r6_left_source_x - r6_right_source_x   
            d7 = r7_left_source_x - r7_right_source_x   

            # print(f'rightArm d1 {d1} d2 {d2} d3 {d3}')
            return d1, d2, d3, d4, d5, d6, d7
        
        elif (part == 'leftArm'):
            l1_right_source_x, _, l1_left_source_x, _  = self.l1.getSourcePoints()
            l2_right_source_x, _, l2_left_source_x, _ =  self.l2.getSourcePoints()
            l3_right_source_x, _, l3_left_source_x, _ =  self.l3.getSourcePoints()
            
            l4_right_source_x, _, l4_left_source_x, _  = self.l4.getSourcePoints()
            l5_right_source_x, _, l5_left_source_x, _ =  self.l5.getSourcePoints()
            l6_right_source_x, _, l6_left_source_x, _ =  self.l6.getSourcePoints()
            l7_right_source_x, _, l7_left_source_x, _ =  self.l7.getSourcePoints()

            

            d1 = l1_left_source_x - l1_right_source_x
            d2 = l2_left_source_x - l2_right_source_x
            d3 = l3_left_source_x - l3_right_source_x   
            
            d4 = l4_left_source_x - l4_right_source_x
            d5 = l5_left_source_x - l5_right_source_x
            d6 = l6_left_source_x - l6_right_source_x   
            d7 = l7_left_source_x - l7_right_source_x   

            # print(f'leftArm d1 {d1} d2 {d2} d3 {d3}')

            return d1, d2, d3, d4, d5, d6, d7
        else:
            return None
    
    
    def setByPercentage(self, part, percentage):
        
        if(part == 'leftArm' or part == 'rightArm'):
            
            # print(f'{part}')

            d1, d2, d3, d4, d5, d6, d7 = self.getPixelDistance(part)
            
            per_part_d1 = pointMath.custom_round((d1 * percentage) / 2)
            per_part_d2 = pointMath.custom_round((d2 * percentage) / 2)
            per_part_d3 = pointMath.custom_round((d3 * percentage) / 2)
            per_part_d4 = pointMath.custom_round((d4 * percentage) / 2)
            per_part_d5 = pointMath.custom_round((d5 * percentage) / 2)
            per_part_d6 = pointMath.custom_round((d6 * percentage) / 2)
            per_part_d7 = pointMath.custom_round((d7 * percentage) / 2)



            if(part == 'leftArm'):
                # print(f'Left epsilons {per_part_d1} {per_part_d2} {per_part_d3}')
                self.l1.setEpsilonX(per_part_d1)
                self.l2.setEpsilonX(per_part_d2)
                self.l3.setEpsilonX(per_part_d3)
                self.l4.setEpsilonX(per_part_d4)
                self.l5.setEpsilonX(per_part_d5)
                self.l6.setEpsilonX(per_part_d6)
                self.l7.setEpsilonX(per_part_d6)

                  
                self.l1.updateDestinationPoints()
                self.l2.updateDestinationPoints()
                self.l3.updateDestinationPoints()
                self.l4.updateDestinationPoints()
                self.l5.updateDestinationPoints()
                self.l6.updateDestinationPoints()
                self.l7.updateDestinationPoints()


            if(part == 'rightArm'):                
                #print(f'Right epsilons {per_part_d1} {per_part_d2} {per_part_d3}')
                self.r1.setEpsilonX(per_part_d1)
                self.r2.setEpsilonX(per_part_d2)
                self.r3.setEpsilonX(per_part_d3)
                self.r4.setEpsilonX(per_part_d4)
                self.r5.setEpsilonX(per_part_d5)
                self.r6.setEpsilonX(per_part_d6)
                self.r7.setEpsilonX(per_part_d6)

                
                self.r1.updateDestinationPoints()
                self.r2.updateDestinationPoints()
                self.r3.updateDestinationPoints()
                self.r4.updateDestinationPoints()
                self.r5.updateDestinationPoints()
                self.r6.updateDestinationPoints()
                self.r7.updateDestinationPoints()

            return True

        else: 
            return False


    
# if __name__ == "__main__":
    
    # Arms = Arm(im,body_parts, 5)
    # #Arms.showAllArmPoints()
    
    # Arms.performWarpingLeftArm()
    # Arms.performWarpingRightArm()
    # im_ = Arms.getImage()

    # torso = torsoFront(im, body_parts)
    # im = torso.performHorizontalWarping('belly', im)
    
    # im = torso.performHorizontalWarping('waist', im)

    # cv2.imwrite('edited.png',im)

    # torso.showAllSourcePoints()
    
    # torso.performHorizontalWarping('belly')
    # torso.performHorizontalWarping('waist')


    # im_ = torso.getImage()
    
    # cv2.imshow("pls",im)
    # cv2.waitKey(0)

    # cv2.imshow("pls",im_)
    # cv2.waitKey(0)