
import cv2

class Point:
    def __init__(self, rightX = 0, rightY = 0, leftX = 0, leftY = 0, epsilonX = 0, epsilonY = 0):
        
        self.epsilon_x = epsilonX 
        self.epsilon_y = epsilonY
        
        self.right_source_x = rightX
        self.right_source_y = rightY
        
        self.right_destination_x = rightX + epsilonX
        self.right_destination_y = rightY + epsilonY
        
        self.left_source_x = leftX
        self.left_source_y = leftY
        
        self.left_destination_x = leftX + epsilonX
        self.left_destination_y = leftY + epsilonY
        
    
    def setEpsilonX(self, epsilon_):
        self.epsilon_x = epsilon_
    
    def setEpsilonY(self, epsilon_):
        self.epsilon_y = epsilon_
    
    def setDestinationAxisX(self):
        self.right_destination_x += self.right_source_x + self.epsilon_x
        self.left_destination_x += self.left_source_x - self.epsilon_x
    
    def setDestinationAxisY(self):
        self.right_destination_y += self.right_source_y - self.epsilon_y
        self.left_destination_y += self.left_source_y - self.epsilon_y
    
    def __repr__(self):
        return f"Point(right x={self.right_source_x}, right y={self.right_source_y}, left x={self.left_source_x}, left y={self.left_source_y} )"
    
    # def showAllPoints(self):
    #     cv2.circle(im, (self.right_shoulder_x, self.right_shoulder_y), 4, (0, 0, 255), -1) 