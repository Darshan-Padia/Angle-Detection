import mediapipe as mp
import cv2
import numpy as np
from scipy.stats import linregress
# import uuid
# import os
# import face_recognition
# from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# from matplotlib import pyplot as plt

joint_list = [[7,6,5]]

def draw_finger_angles(image, results, joint_list):
    
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        for joint in joint_list:
            # a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            # b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            # radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            # angle = np.abs(radians*180.0/np.pi)
            # angle = 180-angle
            
            # if angle < 0.0:
            #     angle = -1*angle
                
            # cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
            #            cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 2, cv2.LINE_AA)


            # markAttendanceangle(angle)

            upper_part_x =  [results.multi_hand_landmarks[0].landmark[joint_list[0]].x, results.multi_hand_landmarks[0].landmark[joint_list[1]].x]
            upper_part_y =  [results.multi_hand_landmarks[0].landmark[joint_list[0]].y, results.multi_hand_landmarks[0].landmark[joint_list[1]].y]
            
            lower_part_x =  [results.multi_hand_landmarks[0].landmark[joint_list[1]].x, results.multi_hand_landmarks[0].landmark[joint_list[2]].x]
            lower_part_y =  [results.multi_hand_landmarks[0].landmark[joint_list[1]].y, results.multi_hand_landmarks[0].landmark[joint_list[2]].y]




            lower_slope = (np.arctan(linregress(lower_part_x, lower_part_y)[0]))*(180/np.pi)
            upper_slope = (np.arctan(linregress(upper_part_x,upper_part_y)[0]))*(180/np.pi)

            raad = 180 - upper_slope - lower_slope

            raad = -1*raad
    
            # if angle < 0.0:
            #     angle = -1*angle
            if raad < 0.0:
                raad = -1*raad  
            cv2.putText(image, str(round(raad, 2)), tuple(np.multiply(c, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 2, cv2.LINE_AA)

            return image

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
#         print(results)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                
            draw_finger_angles(image, results, joint_list)
           
            
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()