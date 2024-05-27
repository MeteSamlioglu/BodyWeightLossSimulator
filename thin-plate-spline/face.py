import cv2
import mediapipe as mp
import numpy as np

import os
import logging
import warnings
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
    def __init__(self, im_, epsilon_cheeks = 10, epsilon_neck = 2):
        
        self.epsilon_cheeks = epsilon_cheeks
        
        self.epsilon_neck = epsilon_neck 
        
        self.im_ = im_
        self.height, self.width = im_.shape[:2]

        self.setFacePoints()
    
    
    def setFacePoints(self):
        
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        
        image = self.im_.copy()
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # Initialize Face Mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
            # Process the image
            results = face_mesh.process(rgb_image)

            # If face landmarks are detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the face landmarks
                    mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                            landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

                    # Get coordinates of the specific landmarks
                    h, w, _ = image.shape
                    left_cheek = face_landmarks.landmark[LEFT_CHEEK]
                    right_cheek = face_landmarks.landmark[RIGHT_CHEEK]
                    chin_tip = face_landmarks.landmark[CHIN_TIP]
                    nose_tip = face_landmarks.landmark[NOSE_TIP]

                    self.left_cheek_coords = (int(left_cheek.x * w), int(left_cheek.y * h))
                    self.right_cheek_coords = (int(right_cheek.x * w), int(right_cheek.y * h))
                    self.chin_tip_coords = (int(chin_tip.x * w), int(chin_tip.y * h))
                    self.nose_tip_coords = (int(nose_tip.x * w), int(nose_tip.y * h))


            rightMid_x =  int((self.right_cheek_coords[0] + self.nose_tip_coords[0])/2)
            rightMid_y =  int((self.right_cheek_coords[1] +  self.nose_tip_coords[1])/2)
            
            leftMid_x =  int((self.left_cheek_coords[0] + self.nose_tip_coords[0])/2)
            leftMid_y =  int((self.left_cheek_coords[1] + self.nose_tip_coords[1])/2)

            self.cheeks = Point(leftMid_x, leftMid_y, rightMid_x, rightMid_y, self.epsilon_cheeks)
            
            diff = self.getDiff()            
            self.neck = Point(int(self.chin_tip_coords[0] - diff), self.chin_tip_coords[1], int(self.chin_tip_coords[0] + diff), self.chin_tip_coords[1],  self.epsilon_neck)

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
        
        s1x, s1y, s1x_, s1y_ = self.cheeks.getSourcePoints()
        d1x, d1y, d1x_, d1y_ = self.cheeks.getDestinationPoints()
  
        n1x, n1y, n1x_, n1y_ = self.neck.getSourcePoints()
        nd1x, nd1y, nd1x_, nd1y_ = self.neck.getDestinationPoints()
        
        source_points = np.array([
            [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
            [0, n1y],[n1y_ , self.width, ],
            
            [n1x, n1y], [n1x_, n1y_],
            [s1x, s1y], [s1x_ , s1y_], 
        
        ])
        
        destination_points = np.array([
            [0, 0], [self.width, 0], [0, self.height], [self.width, self.width], 
            [0, n1y],[n1y_ , self.width, ],

            [nd1x, nd1y], [nd1x_, nd1y_],
            [d1x, d1y], [d1x_ , d1y_],  
        ])
        
        new_im = tps.warpPoints(im, source_points, destination_points)

        self.cheeks.updateDestinationPoints()

        return new_im

    def showAllPoint(self):
        
        copy_im = self.im_.copy()
        
        self.neck.drawPoint(copy_im)
        self.cheeks.drawPoint(copy_im)
        
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
            return d1
        
        elif (part == 'cheeks'):
            cLx, _, cRx, _  = self.cheeks.getSourcePoints()
            d1 = abs(cRx - cLx)
            return d1
        
        else:
            return None
        
    def setByPercentage(self, part, percentage):
        
        if(part == 'neck' or part == 'cheeks'):
            d1= self.getPixelDistance(part)            
            per_part_d1 = pointMath.custom_round((d1 * percentage) / 2)
            
            # print(f'd1 {d1} percentage {percentage} new epsilon {per_part_d1}')

            if(part == 'neck'):
                self.neck.setEpsilonX(per_part_d1)          
                self.neck.updateDestinationPoints()
                
            if(part == 'cheeks'):
                self.cheeks.setEpsilonX(per_part_d1)
                self.cheeks.updateDestinationPoints()
            return True
        
        else: 
            return False

if __name__ == "__main__":
    
    image_path = 'samples/before_.png'  # Replace with the path to your image
    im = cv2.imread(image_path)
    
    face = Face(im, 3 , 3)

    face.setByPercentage("neck", 0.01)
    face.setByPercentage("cheeks", 0.01)
    #im = face.showAllPoint()
    im_ = face.performWarpingFace(im)
    #cv2.imshow('w', im)
    cv2.imshow('w', im_)

    cv2.waitKey(0)