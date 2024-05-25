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
        cv2.circle(im, (x, y), 8, (255, 0, 0), -1)  # Blue color with filled circle
    
    cv2.imshow('Image with Keypoints', im)
    cv2.waitKey(0)
    cv2.imwrite('s.png',im)
    #cv2.destroyAllWindows()

# Get the height and width of the image
#height, width = im.shape[:2]

class torsoFront:

    def __init__(self,im_, body_parts_, epsilon_waist_ = 0, epsilon_belly_ = 0, epsilon_bust_ = 0, epsilon_hip_ = 0):
        
        self.im_ = im_
        self.height, self.width = self.im_.shape[:2]

        self.epsilon_waist = epsilon_waist_
        self.epsilon_belly = epsilon_belly_
        self.epsilon_bust = epsilon_bust_
        self.epsilon_hip = epsilon_hip_
        self.epsilon_shoulder = 5
        
        self.body_parts = body_parts_
        
        self.setShoulders()
        self.setHips()
        
        if self.body_parts['leftHip']['y'] < self.body_parts['rightHip']['y']:
            right_aligned = False
            left_aligned = True
        else:
            left_aligned = False
            right_aligned = True
        
        self.setBustAndWaist(left_aligned, right_aligned)
        self.setBelly()
        self.setAllParts()        
        # Initialization of parts 
        self.torsoFront_parts = [{'shoulder':self.shoulder}, {'bust':self.bust}, {'waist':self.waist}, {'belly':self.belly}, {'hip':self.hip}]
        # self.showAllPoints(self)
        
    def setShoulders(self):
        #Left Shoulder
        self.left_shoulder_x =  int(self.body_parts['leftShoulder']['x']) # Shoulder's x coordinate
        self.left_shoulder_y =  int(self.body_parts['leftShoulder']['y']) # Shoulder's y coordinate
        #Right Shoulder
        self.right_shoulder_x = int(self.body_parts['rightShoulder']['x']) # Shoulder's x coordinate
        self.right_shoulder_y = int(self.body_parts['rightShoulder']['y']) # Shoulder's y coordinate
        
    def setHips(self):
        #Left Hip
        self.left_hip_x = int(self.body_parts['leftHip']['x'])             # Hip's x coordinate
        diff_left = (self.left_shoulder_x - self.left_hip_x) /3
        self.left_hip_x = int(self.left_hip_x + 2*diff_left/3)       #Fixed Hip point
        self.left_hip_y = int(self.body_parts['leftHip']['y'])            # Hip's y coordinate
        
        #Right Hip
        self.right_hip_x = int(self.body_parts['rightHip']['x'])           # Hip's x coordinate
        diff_right = (self.right_hip_x - self.right_shoulder_x) / 3
        self.right_hip_x = int(self.right_shoulder_x + diff_right)
        self.right_hip_y = int(self.body_parts['rightHip']['y'])           # Hip's x coordinate
        
        
    def setBustAndWaist(self, left_aligned, right_aligned):
        
        if(left_aligned):
            #For Left
            self.left_bust_x = int(self.left_shoulder_x - (self.left_shoulder_x - self.left_hip_x) / 3)
            self.left_bust_y = int(self.left_shoulder_y - (self.left_shoulder_y - self.left_hip_y) / 3)
            self.left_waist_x = int(self.left_shoulder_x - 2 * (self.left_shoulder_x - self.left_hip_x) / 3)
            self.left_waist_y = int(self.left_shoulder_y - 2 * (self.left_shoulder_y - self.left_hip_y) / 3)

            self.right_hip_y = self.left_hip_y
            self.right_bust_x = int(self.right_shoulder_x - (self.right_shoulder_x - self.right_hip_x) / 3)
            self.right_bust_y = self.left_bust_y
            self.right_waist_x = int(self.right_shoulder_x - 2 * (self.right_shoulder_x - self.right_hip_x) / 3)
            self.right_waist_y = self.left_waist_y
        
        elif(right_aligned):
            #For Right
            self.right_bust_x = int(self.right_shoulder_x - (self.right_shoulder_x - self.right_hip_x) / 3)
            self.right_bust_y = int(self.right_shoulder_y - (self.right_shoulder_y - self.right_hip_y) / 3)
            self.right_waist_x = int(self.right_shoulder_x - 2 * (self.right_shoulder_x - self.right_hip_x) / 3)
            self.right_waist_y = int(self.right_shoulder_y - 2 * (self.right_shoulder_y - self.right_hip_y) / 3)

            self.left_hip_y = self.right_hip_y
            self.left_bust_x = int(self.left_shoulder_x - (self.left_shoulder_x - self.left_hip_x) / 3)
            self.left_bust_y = self.right_bust_y
            self.left_waist_x = int(self.left_shoulder_x - 2 * (self.left_shoulder_x - self.left_hip_x) / 3)
            self.left_waist_y = self.right_waist_y
    
    def getBustCoordinates(self):
        """ 

        Returns:
            Returns right and left bust coordinates.
        
        """
        return self.right_bust_x, self.right_bust_y, self.left_bust_x, self.left_bust_y        
    
    def setBelly(self):
        
        self.right_belly_x = int((self.right_waist_x + self.right_hip_x)/2)
        self.right_belly_y = int((self.right_waist_y + self.left_hip_y)/2)
        self.left_belly_x = int((self.left_waist_x + self.left_hip_x)/2)
        self.left_belly_y = int((self.left_waist_y + self.left_hip_y)/2)
        
    def setAllParts(self):
        """
        Setting all the coordinates of each body part on torseFront
        
        """
        self.bust = Point(self.right_bust_x, self.right_bust_y, self.left_bust_x, self.left_bust_y, self.epsilon_bust)
        self.hip  = Point(self.right_hip_x, self.right_hip_y, self.left_hip_x, self.left_hip_y, self.epsilon_hip)
        self.shoulder = Point(self.right_shoulder_x, self.right_shoulder_y, self.left_shoulder_x, self.left_shoulder_y, self.epsilon_shoulder)
        self.waist = Point(self.right_waist_x, self.right_waist_y, self.left_waist_x, self.left_waist_y, self.epsilon_waist)
        self.belly = Point(self.right_belly_x, self.right_belly_y, self.left_belly_x, self.left_belly_y, self.epsilon_belly)
        
    def get_point_by_part_name(self, part_name,):
        """
        Fetches the coordinates of a body part by its name from a list of dictionaries.
        
        :param part_name: The name of the body part to search for.
        :param parts_list: The list containing dictionaries of body parts.
        :return: The coordinates of the part if found, or None if not found.
        """
        for part in self.torsoFront_parts:
            if part_name in part:
                return part[part_name]
        return None
    
    def add_part(self, part_name, points):
        """
        Adds a new part to the torsoFront_parts list.
        :param part_name: The name of the new body part (string).
        :param coordinates: The coordinates of the new part, as a tuple (x, y).
        """
        self.torsoFront_parts.append({part_name: points})

    def showAllSourcePoints(self):
        """
        Shows all source points on the image
      
        """
        # print(self.bust)
        # print(self.hip)
        # print(self.shoulder)
        # print(self.waist)
        # print(self.belly)
        cv2.circle(self.im_, (self.right_shoulder_x, self.right_shoulder_y), 4, (0, 0, 255), -1)           # Right-Shoulder-Point 
        cv2.circle(self.im_, (self.right_bust_x, self.right_bust_y), 4, (0, 0, 255), -1)                   # Right-Bust-Point 
        cv2.circle(self.im_, (self.right_waist_x, self.right_waist_y), 4, (0, 0, 255 ), -1)                # Right-Waist-Point 
        cv2.circle(self.im_, (self.right_hip_x, self.right_hip_y), 4, (0, 0, 255 ), -1)                    # Right-Hip-Point 
        cv2.circle(self.im_, (self.right_belly_x, self.right_belly_y), 4, (0, 0, 255 ), -1)                # Right-belly-Point 

        cv2.circle(self.im_, (self.left_shoulder_x, self.left_shoulder_y), 4, (0, 0, 255), -1)             # Left-Shoulder-Point 
        cv2.circle(self.im_, (self.left_bust_x, self.left_bust_y), 4, (0, 0, 255), -1)                     # Left-Bust-Point 
        cv2.circle(self.im_, (self.left_waist_x, self.left_waist_y), 4, (0, 0, 255 ), -1)                  # Left-Waist-Point 
        cv2.circle(self.im_, (self.left_hip_x, self.left_hip_y), 4, (0, 0, 255 ), -1)                      # Left-Hip-Point 
        cv2.circle(self.im_, (self.left_belly_x, self.left_belly_y), 4, (0, 0, 255 ), -1)                  # Left-Belly-Point 

        #cv2.line(self.im_, (self.right_shoulder_x, self.right_shoulder_y), (self.right_hip_x, self.right_hip_y), (0, 255, 0), 2)          # Green line
        #cv2.line(self.im_, (self.left_shoulder_x, self.left_shoulder_y), (self.left_hip_x, self.left_hip_y), (0, 255, 0), 2)              # Green line
        cv2.imshow("Source Points",self.im_)
        cv2.waitKey(0)

    def showWarpingPoints(self):
        copy_im = self.im_.copy()
        for part in self.torsoFront_parts:
            for key, value in part.items():
                right_source_x, right_source_y, left_source_x, left_source_y = value.getSourcePoints()
                right_destination_x, right_destination_y, left_destination_x, left_destination_y = value.getDestinationPoints()
                
                # Convert source and destination points to integers
                right_source_x, right_source_y = int(right_source_x), int(right_source_y)
                left_source_x, left_source_y = int(left_source_x), int(left_source_y)
                right_destination_x, right_destination_y = int(right_destination_x), int(right_destination_y)
                left_destination_x, left_destination_y = int(left_destination_x), int(left_destination_y)
                
                # Draw circles on the image
                cv2.circle(copy_im, (right_source_x, right_source_y), 4, (0, 0, 255), -1)           # Show Source Right
                cv2.circle(copy_im, (left_source_x, left_source_y), 4, (0, 0, 255), -1)             # Show Source Left
                cv2.circle(copy_im, (right_destination_x, right_destination_y), 4, (255, 0, 0), -1)  # Show Destination Right
                cv2.circle(copy_im, (left_destination_x, left_destination_y), 4, (255, 0, 0), -1)    # Show Destination Left

        return copy_im
    
    def getBodyPartCoordinates(self, body_part):
        
        flag = False
        for part in self.torsoFront_parts:
            for key, value in part.items():
                if(key == body_part):
                    coordinates = value
                    flag = True
                    break
        if flag:
            return coordinates
        else:
            return None         
        
    def performHorizontalWarping(self, body_part, im, step = 0):
        
        points = self.getBodyPartCoordinates(body_part)
        
        if points == None:
            print(f'Body Part {body_part} is not found!')
            return     #Exception
        
        right_source_x, right_source_y, left_source_x, left_source_y = points.getSourcePoints()
        
        right_destination_x, right_destination_y, left_destination_x, left_destination_y = points.getDestinationPoints()
        
        right_source_x, right_source_y = int(right_source_x), int(right_source_y)
        left_source_x, left_source_y = int(left_source_x), int(left_source_y)
        right_destination_x, right_destination_y = int(right_destination_x), int(right_destination_y)
        left_destination_x, left_destination_y = int(left_destination_x), int(left_destination_y)
        
        source_points = np.array([
            [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
            [right_source_x, right_source_y], [left_source_x, left_source_y]
        ])
        destination_points = np.array([
            [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
            [right_destination_x, right_destination_y], [left_destination_x, left_destination_y]
        ])
        
        # print(f'right source {right_source_x} right destination {right_destination_x}')
        copy_im = im.copy()
        
        new_im = tps.warpPoints(copy_im, source_points, destination_points)
        
        return new_im
    
        # points.updateSourcePoints()
        # points.updateDestinationPoints()
       
        # print(f'right source {right_source_x} right destination {right_destination_x}')
        
        # cv2.imshow('Warped', new_im)
        # cv2.waitKey(0)
        
        
    def getPixelDistance(self, part):
        if(part == 'belly'):
            belly_points = self.getBodyPartCoordinates('belly')
            belly_right_source_x, belly_right_source_y, belly_left_source_x, belly_left_source_y = belly_points.getSourcePoints()
            distance_pixel = belly_left_source_x - belly_right_source_x
            return distance_pixel
        elif (part == 'waist'):
            waist_points = self.getBodyPartCoordinates('waist')
            waist_right_source_x, waist_right_source_y, waist_left_source_x, waist_left_source_y = waist_points.getSourcePoints()
            distance_pixel = waist_left_source_x - waist_right_source_x
            return distance_pixel

        elif (part == 'bust'):
            bust_points = self.getBodyPartCoordinates('bust')
            bust_right_source_x, bust_right_source_y, bust_left_source_x, bust_left_source_y = bust_points.getSourcePoints()
            distance_pixel = bust_left_source_x - bust_right_source_x
            return distance_pixel
        
        elif (part == 'hip'):
            hip_points = self.getBodyPartCoordinates('hip')
            hip_right_source_x, hip_right_source_y, hip_left_source_x, hip_left_source_y = hip_points.getSourcePoints()
            distance_pixel = hip_left_source_x - hip_right_source_x
            return distance_pixel
        else:
            return None
        
    def setByPercentage(self, part, percentage):
        if(part == 'belly' or part == 'waist' or part == 'bust' or part == 'hip'):
            distance = self.getPixelDistance(part)
            per_part = int((distance * percentage)/2)
            
            #print(f'distance {distance} per_part {per_part}')
            if(part == 'bust'):
                self.bust.setEpsilonX(per_part)
                self.bust.updateDestinationPoints()

            elif(part == 'waist'):
                self.waist.setEpsilonX(per_part)
                self.waist.updateDestinationPoints()

            elif(part == 'hip'):
                self.hip.setEpsilonX(per_part)  
                self.hip.updateDestinationPoints()

            elif(part == 'belly'):
                self.belly.setEpsilonX(per_part)  
                self.belly.updateDestinationPoints()

        else: 
            return False
    
    def performVerticalWarping(self, body_part, step = 0):
        
        if (body_part != 'belly' and body_part != 'bust'):
            return #Exception
        
        belly_points = self.getBodyPartCoordinates('belly')
        waist_points = self.getBodyPartCoordinates('waist')
        
        #Get left and righy belly points
        belly_epsilon_y = belly_points.getEpsilonY()
        belly_points.setVerticalDestinationPoints(-(belly_epsilon_y))
        waist_points.setVerticalDestinationPoints(belly_epsilon_y)
        
        belly_right_source_x, belly_right_source_y, belly_left_source_x, belly_left_source_y = belly_points.getSourcePoints()
        belly_right_destination_x, belly_right_destination_y, belly_left_destination_x, belly_left_destination_y = belly_points.getDestinationPoints()
        
        belly_right_source_x, belly_right_source_y = int(belly_right_source_x), int(belly_right_source_y)
        belly_left_source_x, belly_left_source_y = int(belly_left_source_x), int(belly_left_source_y)
        
        belly_right_destination_x, belly_right_destination_y = int(belly_right_source_x), int(belly_right_destination_y)
        belly_left_destination_x, belly_left_destination_y = int(belly_left_source_x), int(belly_left_destination_y)
        
        #Get left and right waist belly points
        waist_right_source_x, waist_right_source_y, waist_left_source_x, waist_left_source_y = waist_points.getSourcePoints()
        waist_right_destination_x, waist_right_destination_y, waist_left_destination_x, waist_left_destination_y = waist_points.getDestinationPoints()
        
        waist_right_source_x, waist_right_source_y = int(waist_right_source_x), int(waist_right_source_y)
        waist_left_source_x, waist_left_source_y = int(waist_left_source_x), int(waist_left_source_y)
        
        waist_right_destination_x, waist_right_destination_y = int(waist_right_source_x), int(waist_right_destination_y)
        waist_left_destination_x, waist_left_destination_y = int(waist_left_source_x), int(waist_left_destination_y)
        
        cv2.circle(self.im_, (belly_right_source_x, belly_right_source_y), 4, (0, 0, 255), -1)           # Show Source Right
        cv2.circle(self.im_, (belly_left_source_x, belly_left_source_y), 4, (0, 0, 255), -1)             # Show Source Left
        cv2.circle(self.im_, (belly_right_destination_x, belly_right_destination_y), 4, (255, 0, 0), -1)  # Show Destination Right
        cv2.circle(self.im_, (belly_left_destination_x, belly_left_destination_y), 4, (255, 0, 0), -1)    # Show Destination Left
        
        cv2.circle(self.im_, (waist_right_source_x, waist_right_source_y), 4, (0, 0, 255), -1)           # Show Source Right
        cv2.circle(self.im_, (waist_left_source_x, waist_left_source_y), 4, (0, 0, 255), -1)             # Show Source Left
        cv2.circle(self.im_, (waist_right_destination_x, waist_right_destination_y), 4, (255, 0, 0), -1)  # Show Destination Right
        cv2.circle(self.im_, (waist_left_destination_x, waist_left_destination_y), 4, (255, 0, 0), -1)    # Show Destination Left
        
        # cv2.imshow('Warped', self.im_)
        # cv2.waitKey(0)
 
        source_points = np.array([
            [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
            [belly_right_source_x, belly_right_source_y], [belly_left_source_x, belly_left_source_y],
            [waist_right_source_x, waist_right_source_y], [waist_left_source_x, waist_left_source_y],
        ])
        destination_points = np.array([
            [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
            [belly_right_destination_x, belly_right_destination_y], [belly_left_destination_x, belly_left_destination_y],
            [waist_right_destination_x, waist_right_destination_y], [waist_left_destination_x, waist_left_destination_y]
        ])
        
        new_im = tps.warpPoints(self.im_, source_points, destination_points)
        
        self.im_ = new_im
        
        # cv2.imshow('Warped', self.im_)
        # cv2.waitKey(0)
    
    def getImage(self):
        return self.im_
   
        
# if __name__ == "__main__":
#     body_parts
    
#     torso = torsoFront(im, body_parts)
#     torso.showAllSourcePoints()
    
    #show_detected_points(body_parts)
    #torso.showWarpingPoints()
    
    #torso.performVerticalWarping('belly')
    #torso.performVerticalWarping('belly')

    #torso.performHorizontalWarping('belly')
    #torso.showWarpingPoints()

    #torso.performHorizontalWarping('belly', 5)
    # torso.performHorizontalWarping('belly', 5)
    # torso.performHorizontalWarping('belly', 5)


    