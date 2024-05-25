import numpy as np
import cv2
import os
from torsoFront import torsoFront
from Arm import Arm
from upperLeg import upperLeg
from PIL import Image

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

# im = cv2.imread('samples//sample5.jpg')
# if im is None:
#     print("Error loading image. Check the file path.")
#     exit(1)
CROP_AMOUNT = 50
class Body:

    def __init__(self, body_parts_, img, setByPercentage = False):
        self.initial_img = img.copy()
        self.curr_im  = img
        self.height, self.width = img.shape[:2]
        self.steps = []
        self.step_counter = 1
        self.steps.append(self.curr_im)
        
        self.body_parts = body_parts_ 
        
        self.torso = torsoFront(img, self.body_parts, 3, 3, 3, 3) # waist, belly, bust  hip
        self.leftArm = Arm(img, self.body_parts, 1)
        self.rightArm = Arm(img, self.body_parts, 1)
        self.leftLeg = upperLeg(img, self.body_parts, 5)
        self.rightLeg = upperLeg(img, self.body_parts, 5)
        self.hip = upperLeg(img, self.body_parts, 5)
        
        self.setMaxCrop()
       #BodyMassIndex'i girecez 22.5 > overweight aylar alt alta yazılacak  her resmin altın ay ve body mass  badfa
        if(setByPercentage):
            percentage_torso = 0.12
            self.torso.setByPercentage('belly', percentage_torso)
            self.torso.setByPercentage('waist', percentage_torso)
            self.torso.setByPercentage('hip', percentage_torso)
            self.torso.setByPercentage('bust', percentage_torso)
            #---------------------------------------------------------------
            percentage_legs = 0.15
            self.leftLeg.setByPercentage('leftLeg', percentage_legs)
            self.rightLeg.setByPercentage('rightLeg', percentage_legs)
            self.hip.setByPercentage('hip', percentage_legs)
            
            #---------------------------------------------------------------
            percentage_upperArms = 0.12
            self.leftArm.setByPercentage('leftArm', percentage_upperArms)
            self.rightArm.setByPercentage('rightArm', percentage_upperArms)
            
    def setMaxCrop(self):
        
        a1 = ("Shoulder", int(self.body_parts['leftShoulder']['x'] - self.body_parts['rightShoulder']['x']))
        a2   =  ("Elbow" , int(self.body_parts['leftElbow']['x'] - self.body_parts['rightElbow']['x']))
        a3   =  ("Wrist", int(self.body_parts['leftWrist']['x'] - self.body_parts['rightWrist']['x']))

        distances = [a1, a2, a3]
    
        max_distance_tuple = max(distances, key=lambda x: x[1])
        
        max_name, max_distance = max_distance_tuple
        self.cropRightMax = self.body_parts["right" + max_name]
        self.cropLeftMax = self.body_parts ["left" + max_name]


    def resetImage(self):
        """
            Returns:
                Returns the initial image
        """
        return self.initial_img 

    def getHight(self):
        """
            Returns:
        
                Returns the height of the image
                
        """
        return self.height

    def getWidth(self):
        """

        Returns:
            Returns the height of the image
        """
        return self.width
    
    def showDetectedPoints(self):
        copy_im = self.curr_im.copy()
        for part, coords in self.body_parts.items():
            x = int(coords['x'])
            y = int(coords['y'])
            cv2.circle(copy_im, (x, y), 8, (255, 0, 0), -1)  # Blue color with filled circle
        
        cv2.imshow('Image with Keypoints', copy_im)
        cv2.waitKey(0)
        
# body_parts['rightKnee']['x']
    
    def showWarpingPoints(self, part_body):
        
        if(part_body == "torso"):
            im = self.torso.showWarpingPoints()
            cv2.imshow('Torso Warping Points',im)
            cv2.waitKey(0)
        
        if(part_body == 'leftArm'):
            im = self.leftArm.showLeftArmPoints()
            cv2.imshow('Left Arm Warping Points',im)
            cv2.waitKey(0)
        
        if(part_body == 'rightArm'):
            im = self.leftArm.showRightArmPoints()
            cv2.imshow('Right Arm Warping Points',im)
            cv2.waitKey(0)
        
        if(part_body == 'arms'):
            im = self.leftArm.showAllArmPoints()
            cv2.imshow('Arms All Warping Points',im)
            cv2.waitKey(0)
         
        if(part_body == 'leftLeg'):
            im = self.leftLeg.showLeftLegPoints()
            cv2.imshow('Left Leg Warping Points',im)
            cv2.waitKey(0)
        
        if(part_body == 'rightLeg'):
            im = self.rightLeg.showRightLegPoints()
            cv2.imshow('Right Leg Warping Points',im)
            cv2.waitKey(0)
        
        if(part_body == 'upperLegs'):
            im = self.leftLeg.showAllPoints()
            cv2.imshow('Leg Warping Points',im)
            cv2.waitKey(0)
        
    
    def warp(self, body_part):
    
        im = self.steps[self.step_counter - 1]
        
        # cv2.imshow('sent', im)
        # cv2.waitKey(0)
        # cv2.imshow('def', self.leftArm.showLeftArmPoints())
        # cv2.waitKey(0)
        if(body_part == "belly"):
            self.curr_im = self.torso.performHorizontalWarping('belly', im)

        if(body_part == "waist"):
            self.curr_im = self.torso.performHorizontalWarping('waist', im)
            
        if(body_part == "bust"):
            self.curr_im = self.torso.performHorizontalWarping('bust', im)
        
        if(body_part == "hip"):
            self.curr_im = self.torso.performHorizontalWarping('hip', im)    
    
        if(body_part == 'leftArm'):
            self.curr_im = self.leftArm.performWarpingLeftArm(im)
        
        if(body_part == 'rightArm'):
            self.curr_im = self.rightArm.performWarpingRightArm(im)
         
        if(body_part == 'leftLeg'):
            self.curr_im = self.leftLeg.performWarpingLefttLeg(im)
                    
        if(body_part == 'rightLeg'):
            self.curr_im = self.rightLeg.performWarpingRightLeg(im)
        
        
        im = self.curr_im.copy()
        self.steps.append(im)
        self.step_counter+=1
        
    def show(self, step = 0):
        if(step == 0):
            im = self.steps[self.step_counter]
            cv2.imshow("Body", im)
            cv2.waitKey(0)
        else:
            counter = self.step_counter - step
            if(counter > 0):
                im = self.steps[counter]
                cv2.imshow("Body", im)
                cv2.waitKey(0)
    def save(self, step = 0, cropImage = False):
        if(step == 0):
            im = self.steps[self.step_counter - 1]
            if(cropImage):
                self.crop(im)
            else:
                cv2.imwrite('uploads/edited.png',im)

    def crop(self,im_):
        crop_amount = 50  # Adjust as needed
        im = im_
        
        image_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # print(f'selfRight max {self.cropRightMax} selfLeftMax {self.cropLeftMax}')
        # # Calculate the new bounding box
        crop_amount = CROP_AMOUNT  

        left = crop_amount
        top = 0
        right = pil_image.width - crop_amount
        bottom = pil_image.height

        cropped_image = pil_image.crop((left, top, right, bottom))

        cropped_image_path = "uploads/edited.png"
        cropped_image.save(cropped_image_path, format='PNG')
        
        #cropped_image.show()
        
       
# if __name__ == "__main__":
#     body = Body(body_parts, im)
    
#     body.showDetectedPoints()
#     # body.showWarpingPoints('leftLeg')
#     # body.showWarpingPoints('rightLeg')
#     # body.showWarpingPoints('torso')
#     body.warp('belly')
#     #body.warp('belly')
#     body.warp('waist')
#     #body.warp('waist')
#     body.warp('hip')
#     body.warp('leftLeg')
#     body.warp('rightLeg')
#     # body.show()
#     # body.warp('rightArm')s
#     # body.warp('leftArm')
#     # body.warp('leftLeg')
#     # body.warp('rightLeg')
#     # body.showWarpingPoints('legs')
#     # body.showWarpingPoints('leftLeg')
#     # body.showWarpingPoints('rightLeg')
#     body.save(cropImage=True)


    

