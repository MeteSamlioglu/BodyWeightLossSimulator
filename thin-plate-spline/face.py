import cv2
# import mediapipe as mp
import numpy as np

import os
import logging
import warnings
from scipy.interpolate import Rbf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

from core import Point
from core import tps
from core import pointMath


# Define the landmarks for the left cheek, right cheek, chin tip, and nose tip
LEFT_CHEEK = 93
RIGHT_CHEEK = 323
CHIN_TIP = 152
NOSE_TIP = 1
MIN_DIFF = 6
class Face:
    def __init__(self, im_, body_parts_, epsilon_cheeks = 0, epsilon_neck = 0, isDetected_ = True):
        
        self.epsilon_cheeks = epsilon_cheeks
        
        self.epsilon_neck = epsilon_neck 
        
        self.isDetected = isDetected_
        
        self.im_ = im_
        self.height, self.width = im_.shape[:2]
        self.body_parts = body_parts_
        
        self.nose = (int(self.body_parts['nose']['x']), int(self.body_parts['nose']['y']))
        self.leftEar = (int(self.body_parts['leftEar']['x']), int(self.body_parts['leftEar']['y']))
        self.rightEar = (int(self.body_parts['rightEar']['x']), int(self.body_parts['rightEar']['y']))
        self.leftEye = (int(self.body_parts['leftEye']['x']), int(self.body_parts['leftEye']['y']))
        self.rightEye = (int(self.body_parts['rightEye']['x']), int(self.body_parts['rightEye']['y']))

        self.setFacePoints()
    
    def isFaceDetected(self):
        return self.isDetected
    
    def setFacePoints(self):

        rightCheekMid_x = int((self.rightEar[0] + self.nose[0])/2)
        rightCheekMid_y = int(self.nose[1])
        
        leftCheekMid_x =  int((self.leftEar[0] + self.nose[0])/2)
        leftCheekMid_y =  int(self.nose[1])

        rightCheek_x = int(rightCheekMid_x - (rightCheekMid_x - self.rightEar[0])/3)
        rightCheek_y = int(rightCheekMid_y)
        
        rightCheek_x_ = int(rightCheekMid_x + (self.nose[0] - rightCheekMid_x)/3)
        rightCheek_y_ = int(rightCheekMid_y)
        
        leftCheek_x =  int(leftCheekMid_x - (leftCheekMid_x - self.nose[0])/3)
        leftCheek_y = int(leftCheekMid_y)
        
        leftCheek_x_ =  int(leftCheekMid_x + (self.leftEar[0] - leftCheekMid_x)/3)
        leftCheek_y_ = int(leftCheekMid_y)
        #----------------------------------------------------------------------
        #Neck Points
        if(self.body_parts['rightShoulder']['y'] > self.body_parts['leftShoulder']['y']):
            midNeck_y = int(self.body_parts['rightShoulder']['y'])
        else:
            midNeck_y =int (self.body_parts['leftShoulder']['y'])
        
        midNeck_x = self.nose[0]
        midNeck_y1 = int(midNeck_y - (midNeck_y - self.nose[1])/3)
        midNeck_y2 = int((midNeck_y + self.nose[1])/2)
        midNeck_y = int((midNeck_y1 + midNeck_y2)/2)

        
        if(self.leftEye[1]  > self.rightEye[1]):
            offSet = abs(self.nose[1] - self.rightEye[1])
        else:
            offSet = abs(self.nose[1] - self.leftEye[1])
        
        offSet = int(offSet)
                
        self.neck = Point(rightCheek_x_, midNeck_y , leftCheek_x , midNeck_y, self.epsilon_neck)
        self.RightCheek = Point(rightCheek_x, rightCheek_y + offSet , rightCheek_x_, rightCheek_y_ + offSet, self.epsilon_cheeks)
        self.LeftCheek = Point(leftCheek_x, leftCheek_y + offSet , leftCheek_x_, leftCheek_y_ + offSet, self.epsilon_cheeks)

        # print(f'rightCheek_y {rightCheek_y} rightCheek_y_ {rightCheek_y_}')
        # print(f'leftCheek_y {leftCheek_y} leftCheek_y_ {leftCheek_y_}')
        # print(f'nose[1] {self.nose[1]}')

        
    def getDiff(self):
        
        diffLeft = abs((self.nose_tip_coords[0] - self.left_cheek_coords[0])/2)
        diffRight = abs((self.right_cheek_coords[0] - self.nose_tip_coords[0])/2)
        
        diffLeft = int(diffLeft)
        diffRight = int(diffRight)
        if( diffLeft < MIN_DIFF and diffRight < MIN_DIFF):
            return MIN_DIFF
        if(diffLeft <= diffRight):
            return diffRight
        else:
            return diffLeft
        
    def performWarpingFace(self, im):    
        """ 
            Performs TPS on hip leg
        """
        self.im_ = im
        
        s1x, s1y, s1x_, s1y_ = self.RightCheek.getSourcePoints()
        s2x, s2y, s2x_, s2y_ = self.LeftCheek.getSourcePoints()
        
        d1x, d1y, d1x_, d1y_ = self.RightCheek.getDestinationPoints()
        d2x, d2y, d2x_, d2y_ = self.LeftCheek.getDestinationPoints()
        
        n1x, n1y, n1x_, n1y_ = self.neck.getSourcePoints()
        nd1x, nd1y, nd1x_, nd1y_ = self.neck.getDestinationPoints()
 
        source_points = np.array([
            [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
            [0, n1y], [0, s2y],
            [self.width, nd1y_], [self.width, d2y_], #[self.width, n1y],
            
            [s1x, s1y], #[s1x_ , s1y_], 
            [s2x_, s2y_], #[s2x_ , s2y_], 
            [n1x, n1y], [n1x_, n1y_],
        ])
        
        destination_points = np.array([
            [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
            [0, n1y], [0, s2y], 
            [self.width, nd1y_], [self.width, d2y_], #[self.width, n1y],
            # [self.width, s2y_], [self.width, n1y_],
            
            [d1x, d1y], #[d1x_ , d1y_],  
            [d2x_, d2y_], #[d2x_ , d2y_],  
            [nd1x, nd1y], [nd1x_, nd1y_],
        ])
        
        new_im = tps.warpPoints(im, source_points, destination_points)

        # self.cheeks.updateDestinationPoints()

        return new_im
  
    def showAllPoints(self):
        
        copy_im = self.im_.copy()
        
        self.neck.drawPoint(copy_im)
        # self.cheeks.drawPoint(copy_im)
        self.LeftCheek.drawPoint(copy_im)
        self.RightCheek.drawPoint(copy_im)

        return copy_im
    
    def setImage(self, im):
        """     
            Sets image 
        """
        self.im_ = im 
    
    def getImage(self):
        """
            Returns:
                Returns the image
        """
        return self.im_
    
    def getPixelDistance(self, part):
        
        if(part == 'neck'):
            nRx, _, nLx, _  = self.neck.getSourcePoints()
            d1 = abs(nLx - nRx)
            return d1, d1
        
        elif (part == 'cheeks'):
            cRsx, _, cRsx_, _  = self.RightCheek.getSourcePoints()
            # cRdx, _, _, _  = self.RightCheek.getDestinationPoints()

            cLx, _, cLx_, _ = self.LeftCheek.getSourcePoints() 
            # _, _, cLdx, _ = self.LeftCheek.getDestinationPoints() 

            d1 = abs(cRsx_ - cRsx)
            d2 = abs(cLx_ - cLx)
            return d1, d2
        
        else:
            return None
        
    def setByPercentage(self, part, percentage):
        
        if(part == 'neck' or part == 'cheeks'):
            d1, d2= self.getPixelDistance(part)            
            per_part_d1 = pointMath.custom_round((d1 * percentage) / 2) #Righ
            per_part_d2 = pointMath.custom_round((d2 * percentage) / 2) #Left

            # print(f'd1 {d1} percentage {percentage} new epsilon {per_part_d1}')

            if(part == 'neck'):
                d1, _ = self.getPixelDistance(part)            
                per_part_d1 = pointMath.custom_round((d1 * percentage) / 2) 
                # print(f'Epsilon  neck {per_part_d1} {per_part_d2}')

                self.neck.setEpsilonX(per_part_d1)          
                self.neck.updateDestinationPoints()
                
            if(part == 'cheeks'):
                d1, d2= self.getPixelDistance(part)            
                # print(f' d1 {d1}  d2 {d2}')
                per_part_d1 = pointMath.custom_round((d1 * percentage)) #Right
                per_part_d2 = pointMath.custom_round((d2 * percentage)) #Left
                # print(f'Epsilon  Cheek {per_part_d1} {per_part_d2}')
                
                self.RightCheek.setEpsilonX(per_part_d1)
                self.LeftCheek.setEpsilonX(per_part_d2)
                
                self.RightCheek.updateDestinationPoints()
                self.LeftCheek.updateDestinationPoints()

            return True
        
        else: 
            return False

# if __name__ == "__main__":
    
#     image_path = 'samples/4.png'  # Replace with the path to your image
#     im = cv2.imread(image_path)
    
#     face = Face(im, 10 , 10)

#     face.setByPercentage("neck", 0.14)
#     face.setByPercentage("cheeks", 0.14)
    
#     im = face.showAllPoint()
#     cv2.imshow('w', im)
#     cv2.waitKey(0)
    
#     im_ = face.performWarpingFace(im)
#     cv2.imshow('w', im_)

#     cv2.waitKey(0)