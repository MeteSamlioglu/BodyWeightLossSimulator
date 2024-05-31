import numpy as np
import cv2
import math

from core import Point
from core import tps
from core import pointMath

MIN_LEG_DIFF = 6

class lowerLeg:
    
    def __init__(self,im, body_parts, epsilon_, isDetected_ = True):
        
        
        self.im_ = im
        self.height, self.width = im.shape[:2]
        self.epsilon_lowerLeg = epsilon_
        self.isDetected = isDetected_
        self.rightKnee_x =  int(body_parts['rightKnee']['x']) 
        self.rightKnee_y = int(body_parts['rightKnee']['y']) 
        
        self.leftKnee_x =  int(body_parts['leftKnee']['x'])
        self.leftKnee_y =  int(body_parts['leftKnee']['y']) 
        
        self.leftAnkle_x = int(body_parts['leftAnkle']['x'])   
        self.leftAnkle_y = int(body_parts['leftAnkle']['y'])   
        
        self.rightAnkle_x = int(body_parts['rightAnkle']['x'])    
        self.rightAnkle_y = int(body_parts['rightAnkle']['y'])    
        
        
        self.setRightLegPoints()
        self.setLeftLegPoints()
                
    def isLowerLegDetected(self):
        return self.isDetected
        
    def setRightLegPoints(self):
        
        """
            Setting Right leg's initial source and destination points
        """
        self.rightLeg_midX = int((self.rightAnkle_x + self.rightKnee_x) / 2) 
        self.rightLeg_midY = int((self.rightAnkle_y + self.rightKnee_y) / 2)
        diff = self.getDifference()
                
        mid1x = int((self.rightKnee_x + self.rightLeg_midX) / 2)
        mid1y = int((self.rightKnee_y + self.rightLeg_midY) / 2)
        
        mid2x = int((self.rightAnkle_x + self.rightLeg_midX) / 2)
        mid2y = int((self.rightAnkle_y + self.rightLeg_midY) / 2)
        
        rRs1x = mid1x - diff
        rRs1y = mid1y
        rRs1x_ = mid1x + diff
        rRs1y_ = mid1y
        
        rRs2x = self.rightLeg_midX - diff
        rRs2y = self.rightLeg_midY
        rRs2x_ = self.rightLeg_midX + diff 
        rRs2y_ = self.rightLeg_midY
        
        rRs3x = mid2x - diff
        rRs3y = mid2y
        rRs3x_ = mid2x + diff
        rRs3y_ = mid2y        
        
        self.r1 = Point(rRs1x, rRs1y, rRs1x_, rRs1y_, self.epsilon_lowerLeg)
        self.r2 = Point(rRs2x, rRs2y, rRs2x_, rRs2y_, self.epsilon_lowerLeg)
        self.r3 = Point(rRs3x, rRs3y, rRs3x_, rRs3y_, self.epsilon_lowerLeg)

    def getDifference(self):
        """
            Find the optimal difference to set distance between left and right source point 
        """
        self.leftLeg_midX = int((self.leftAnkle_x + self.leftKnee_x) / 2) 
        self.leftLeg_midY = int((self.leftAnkle_y + self.leftKnee_y) / 2)
        self.rightLeg_midX = int((self.rightAnkle_x + self.rightKnee_x) / 2) 
        self.rightLeg_midY = int((self.rightAnkle_y + self.rightKnee_y) / 2)
        diffLeft = abs(self.leftAnkle_x - self.leftKnee_x)
        diffRight = abs(self.rightAnkle_x - self.rightKnee_x)

        if(diffLeft < MIN_LEG_DIFF and diffRight < MIN_LEG_DIFF):
            return MIN_LEG_DIFF
        if(diffLeft <= diffRight):
            return diffRight
        else:
            return diffLeft
        
    def setLeftLegPoints(self):
        
        """
            Setting Left leg's initial source and destination points
        """
        self.leftLeg_midX = int((self.leftAnkle_x + self.leftKnee_x) / 2) 
        self.leftLeg_midY = int((self.leftAnkle_y + self.leftKnee_y) / 2)
        
        diff = self.getDifference()
        

        mid1x = int((self.leftKnee_x + self.leftLeg_midX) / 2)
        mid1y = int((self.leftKnee_y + self.leftLeg_midY) / 2)
        
        mid2x = int((self.leftAnkle_x + self.leftLeg_midX) / 2)
        mid2y = int((self.leftAnkle_y + self.leftLeg_midY) / 2)
        
        lRs1x = mid1x - diff
        lRs1y = mid1y
        lRs1x_ = mid1x + diff
        lRs1y_ = mid1y
        
        lRs2x = self.leftLeg_midX - diff
        lRs2y =  self.leftLeg_midY
        lRs2x_ = self.leftLeg_midX + diff 
        lRs2y_ = self.leftLeg_midY
        
        lRs3x = mid2x - diff
        lRs3y = mid2y
        lRs3x_ = mid2x + diff
        lRs3y_ = mid2y        
        
        self.l1 = Point(lRs1x, lRs1y, lRs1x_, lRs1y_, self.epsilon_lowerLeg)
        self.l2 = Point(lRs2x, lRs2y, lRs2x_, lRs2y_, self.epsilon_lowerLeg)
        self.l3 = Point(lRs3x, lRs3y, lRs3x_, lRs3y_, self.epsilon_lowerLeg)
        
    def performWarpingRightLeg(self, im):
        """
            
            Performs TPS on right leg
        
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
        
        #edited
        [0, s1y], [0, s2y], [0, s3y],
        [self.width, s1y_], [self.width, s2y_], [self.width, s3y_],
        #---------------
        
        [s1x, s1y], [s1x_, s1y_],
        [s2x, s2y], [s2x_, s2y_],
        [s3x, s3y], [s3x_, s3y_]
        ])
        
        destination_points = np.array([
        [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
        
        #edited
        [0, s1y], [0, s2y], [0, s3y],
        [self.width, s1y_], [self.width, s2y_], [self.width, s3y_],
        #---------------
        
        [d1x, d1y], [d1x_, d1y_],
        [d2x, d2y], [d2x_, d2y_],
        [d3x, d3y], [d3x_, d3y_]
        ])
        
        new_im = tps.warpPoints(self.im_, source_points, destination_points)
        
        self.im_ = new_im                
        
        self.r1.updateSourcePoints()
        self.r2.updateSourcePoints()
        self.r3.updateSourcePoints()
        
        self.r1.updateDestinationPoints()
        self.r2.updateDestinationPoints()
        self.r3.updateDestinationPoints()

        
        return self.im_

    
    def performWarpingLefttLeg(self, im):
        
        """
            
            Performs TPS on left leg
        
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
        
        #edited
        [0, l1y], [0, l2y], [0, l3y],
        [self.width, l1y_], [self.width, l2y_], [self.width, l3y_],
        #---------------
        
        [l1x, l1y], [l1x_, l1y_],
        [l2x, l2y], [l2x_, l2y_],
        [l3x, l3y], [l3x_, l3y_]
        ])
        
        destination_points = np.array([
        [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
        
        #edited
        [0, l1y], [0, l2y], [0, l3y],
        [self.width, l1y_], [self.width, l2y_], [self.width, l3y_],
        #---------------
        
        [d1x, d1y], [d1x_, d1y_],
        [d2x, d2y], [d2x_, d2y_],
        [d3x, d3y], [d3x_, d3y_]
        ])
        
        new_im = tps.warpPoints(self.im_, source_points, destination_points)
        
        self.im_ = new_im                
        
        self.l1.updateSourcePoints()
        self.l2.updateSourcePoints()
        self.l3.updateSourcePoints()
        
        self.l1.updateDestinationPoints()
        self.l2.updateDestinationPoints()
        self.l3.updateDestinationPoints()

        return self.im_

    
    
    def showRightLegPoints(self):
        """

            Displays all the source and destination points on right leg.
       
        """
        copy_im = self.im_.copy()
        self.r1.drawPoint(copy_im)
        self.r2.drawPoint(copy_im)
        self.r3.drawPoint(copy_im)
        
        return copy_im
    
    def showLeftLegPoints(self):
        """
        
            Displays all the source and destination points on left leg.
        
        """
        copy_im = self.im_.copy()
        self.l1.drawPoint(copy_im)
        self.l2.drawPoint(copy_im)
        self.l3.drawPoint(copy_im)
        
        return copy_im

    def showAllPoints(self):
        """
            Returns:
                Returns all lower leg points.
        """
        copy_im = self.im_.copy()
        

        self.r1.drawPoint(copy_im)
        self.r2.drawPoint(copy_im)
        self.r3.drawPoint(copy_im)
        self.l1.drawPoint(copy_im)
        self.l2.drawPoint(copy_im)
        self.l3.drawPoint(copy_im)
        
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
        
        if(part == 'leftLeg'):
            l1_right_source_x, _, l1_left_source_x, _  = self.l1.getSourcePoints()
            l2_right_source_x, _, l2_left_source_x, _ =  self.l2.getSourcePoints()
            l3_right_source_x, _, l3_left_source_x, _ =  self.l3.getSourcePoints()

            d1 = l1_left_source_x - l1_right_source_x
            d2 = l2_left_source_x - l2_right_source_x
            d3 = l3_left_source_x - l3_right_source_x   

            return d1, d2, d3
        
        elif (part == 'rightLeg'):
            r1_right_source_x, _, r1_left_source_x, _  = self.r1.getSourcePoints()
            r2_right_source_x, _, r2_left_source_x, _ =  self.r2.getSourcePoints()
            r3_right_source_x, _, r3_left_source_x, _ =  self.r3.getSourcePoints()

            d1 = r1_left_source_x - r1_right_source_x
            d2 = r2_left_source_x - r2_right_source_x
            d3 = r3_left_source_x - r3_right_source_x   
            
            return d1, d2, d3
        else:
            return None
    
    
    def setByPercentage(self, part, percentage):
        
        if(part == 'leftLeg' or part == 'rightLeg'):
            d1, d2, d3 = self.getPixelDistance(part)
            
            per_part_d1 = pointMath.custom_round((d1 * percentage) / 2)
            per_part_d2 = pointMath.custom_round((d2 * percentage) / 2)
            per_part_d3 = pointMath.custom_round((d3 * percentage) / 2)
            

            if(part == 'leftLeg'):
                self.l1.setEpsilonX(per_part_d1)
                self.l2.setEpsilonX(per_part_d2)
                self.l3.setEpsilonX(per_part_d3)
                
                self.l1.updateDestinationPoints()
                self.l2.updateDestinationPoints()
                self.l3.updateDestinationPoints()
                
 
            if(part == 'rightLeg'):
                self.r1.setEpsilonX(per_part_d1)
                self.r2.setEpsilonX(per_part_d2)
                self.r3.setEpsilonX(per_part_d3)
                self.r1.updateDestinationPoints()
                self.r2.updateDestinationPoints()
                self.r3.updateDestinationPoints()
            return True
        else: 
            return False
        