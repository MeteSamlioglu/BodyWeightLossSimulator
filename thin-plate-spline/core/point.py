
import cv2

class Point:
    def __init__(self, rightX = 0, rightY = 0, leftX = 0, leftY = 0, epsilon = 0):
        
        self.epsilon_x = epsilon 
        self.epsilon_y = epsilon
        
        self.right_source_x = rightX
        self.right_source_y = rightY
        
        self.right_destination_x = self.right_source_x + self.epsilon_x
        self.right_destination_y = self.right_source_y
        
        #print(f'source : {self.right_source_x} destination: {self.right_destination_x}')
        
        self.left_source_x = leftX
        self.left_source_y = leftY
        
        self.left_destination_x = leftX -  epsilon
        self.left_destination_y = leftY 

    def drawPoint(self, im):
        cv2.circle(im, ( self.right_source_x,   self.right_source_y ), 4, (0, 0, 255), -1) 
        cv2.circle(im, ( self.right_destination_x, self.right_destination_y ), 4, (255, 0, 0), -1) 

        cv2.circle(im, ( self.left_source_x,   self.left_source_y ), 4, (0, 0, 255), -1) 
        cv2.circle(im, ( self.left_destination_x,   self.left_destination_y ), 4, (255, 0, 0), -1) 

        
    def setEpsilonX(self, epsilon_):
        """
            Returns:
                Returns the epsilon on X axi
        """
        if(epsilon_ == 0):
            epsilon_+= 1
            
        self.epsilon_x = epsilon_
    
    def setEpsilonY(self, epsilon_):
        """
            Returns:
                Returns the epsilon on Y axis
        """
        self.epsilon_y = epsilon_
    
    # def setDestinationAxisX(self):
    #     self.right_destination_x = self.right_source_x + self.epsilon_x
    #     self.left_destination_x = self.left_source_x - self.epsilon_x
    
    def getSourcePoints(self):
        """
            Returns:
                Returns the source points
        """
        return self.right_source_x, self.right_source_y, self.left_source_x, self.left_source_y
   
    def updateSourcePoints(self):
        """
            Updates source points after TPS 
        """
        self.right_source_x = self.right_source_x + self.epsilon_x
        self.right_source_y = self.right_source_y
        
        self.left_source_x =   self.left_source_x - self.epsilon_x
        self.left_source_y =  self.left_source_y
    
    def updateDestinationPoints(self):
        """
            Updates destination points after TPS 
        """
        # print(f'right source x {self.right_source_x} left source x {self.left_source_x} epsilon {self.epsilon_x}')
        
        #self.left_destination_x =  self.left_source_x - self.epsilon_x
        
        left = self.left_source_x - self.epsilon_x
        self.left_destination_x = left
        self.right_destination_x = self.right_source_x + self.epsilon_x
        
                
        # print(f'right destination x {self.right_destination_x} left destination x {self.left_destination_x} epsilon {self.epsilon_x} left {left}')


    def getDestinationPoints(self):
        """
            Returns:
                Returns the destination points 
        """
        return self.right_destination_x, self.right_destination_y, self.left_destination_x,self.left_destination_y
    
    def setVerticalDestinationPoints(self, vertical_epsilon):
        """
        Setting the vertical destination points
        
        Args:
            vertical_epsilon (int): Epsilon for destination point
        """
        self.right_destination_y = self.right_source_y + vertical_epsilon
        
        self.left_destination_y  = self.left_destination_y + vertical_epsilon
        
    def setHorizontalDesinationPoints(self, vertical_epsilon):
        """
        Setting the horizontal destination points
        
        Args:
            vertical_epsilon (int): Epsilon for destination point
        """
        self.right_destination_x = self.right_source_x + vertical_epsilon
        
        self.left_destination_x = self.left_source_x + vertical_epsilon
    
    def getEpsilonY(self):
        return self.epsilon_y
    
    # def setDestinationAxisY(self):
    #     self.right_destination_y += self.right_source_y - self.epsilon_y
    #     self.left_destination_y += self.left_source_y - self.epsilon_y
    
    def __repr__(self):
        return f"Point(right x={self.right_source_x}, right y={self.right_source_y}, left x={self.left_source_x}, left y={self.left_source_y} )"
    
    # def showAllPoints(self):
    #     cv2.circle(im, (self.right_shoulder_x, self.right_shoulder_y), 4, (0, 0, 255), -1) 