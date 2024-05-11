import mediapipe as mp
import cv2
import numpy as np
import scipy as sp
from scipy.stats import linregress


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles



joint_list = [7,6,5]

def draw_finger_angles(image, output):

    a = [ output.multi_hand_landmarks[0].landmark[joint_list[0]].x,  output.multi_hand_landmarks[0].landmark[joint_list[0]].y] # First coord
    b = [ output.multi_hand_landmarks[0].landmark[joint_list[1]].x,  output.multi_hand_landmarks[0].landmark[joint_list[1]].y] # Second coord
    c = [ output.multi_hand_landmarks[0].landmark[joint_list[2]].x,  output.multi_hand_landmarks[0].landmark[joint_list[2]].y] # Third coord
            
    upper_part_x =  [output.multi_hand_landmarks[0].landmark[joint_list[0]].x, output.multi_hand_landmarks[0].landmark[joint_list[1]].x]
    upper_part_y =  [output.multi_hand_landmarks[0].landmark[joint_list[0]].y, output.multi_hand_landmarks[0].landmark[joint_list[1]].y]
    
    lower_part_x =  [output.multi_hand_landmarks[0].landmark[joint_list[1]].x, output.multi_hand_landmarks[0].landmark[joint_list[2]].x]
    lower_part_y =  [output.multi_hand_landmarks[0].landmark[joint_list[1]].y, output.multi_hand_landmarks[0].landmark[joint_list[2]].y]




    lower_slope = (np.arctan(linregress(lower_part_x, lower_part_y)[0]))*(180/np.pi)
    upper_slope = (np.arctan(linregress(upper_part_x,upper_part_y)[0]))*(180/np.pi)

    raad = 180 - upper_slope - lower_slope
    # print(f"l {lower_slope} ")
    
    rad_final = lower_slope - upper_slope
    print(rad_final)
    radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    angle = 180-angle
    # angle = 180-angle
    rad_final = 180-rad_final
    raad = -1*raad
            
    if rad_final < 0.0:
        rad_final = -1*rad_final
    if angle < 0.0:
        angle = -1*angle
    if raad < 0.0:
        raad = -1*raad
        
        
    cv2.putText(image, str(round(rad_final, 2)), tuple(np.multiply([ output.multi_hand_landmarks[0].landmark[joint_list[1]].x,  output.multi_hand_landmarks[0].landmark[joint_list[1]].y], [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(c, [640, 480]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 2, cv2.LINE_AA)
    return image

cap = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=0,min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        image = cv2.flip(image, 1)        
        
        results = hands.process(image)
# Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
        # if results.multi_hand_world_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
            # for num, hand in enumerate(results.multi_hand_world_landmarks):
            
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        # mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        # mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                
            draw_finger_angles(image, results)
           
            
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()