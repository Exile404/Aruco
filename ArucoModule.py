import cv2
import cv2.aruco as aruco
import numpy
import os

def findArucoMarkers(img,markerSize=5,totalMarkers=250,draw=True):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict=aruco.Dictionary_get(key)
    arucoParam= aruco.DetectorParameters_create()
    bbox,ids,rejected=aruco.detectMarkers(imgGray,arucoDict,parameters=arucoParam)
    print(ids)
    if draw:
        aruco.drawDetectedMarkers(img,bbox)





def main():
    cap=cv2.VideoCapture(0)

    while True:
        success,img=cap.read()
        findArucoMarkers(img)
        # Loop Through all markerts and augment each one
        cv2.imshow("Image",img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()