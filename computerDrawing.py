import cv2    
import mediapipe as mp
import socket

width, height = 1280, 720

cap = cv2.VideoCapture(0)       
if not cap.isOpened():
    print("Camera can't open")
    exit()
mpHands = mp.solutions.hands        
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

chunk_size = 1024

while True:
    success, img = cap.read()      
    if not success:
        print("Failed frame read")
        continue
    if img is None:
        print("Empty frame is received")
        continue
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if imgRGB is None:
        print("Failed to convert color space of input")
        continue
    results = hands.process(imgRGB)        
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:         
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            data = []
            lmList = results.multi_hand_landmarks[0].landmark
            print(lmList)
            for lm in lmList:
                data.extend([lm.x, lm.y, lm.z])
            data_str = str(data)
            for i in range(0, len(data_str), chunk_size):
                chunk = data_str[i:i+chunk_size]
                sock.sendto(str.encode(chunk), serverAddressPort)

    
    cv2.imshow("Image", img)        
    cv2.waitKey(1)


