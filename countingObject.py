# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 22:34:54 2018

@author: jorda
"""

import numpy as np
import cv2
import os, os.path
import imutils

def loadImage(dataBase):
    #dataBase_path = "";
    if dataBase == "Urban":
        dataBase_path = "M:\\Periodo 2\\Estudo Orientado\\DB\\Urban1\\Urban1\\";
    else:
        print("Data base does not found")
        return
    print(dataBase_path);

def backgroundSubtraction(video_path):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    #fgbg = cv2.createBackgroundSubtractorMOG2()
    while(1):
        text = "Unoccupied"
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)

        # filters
        filter = cv2.GaussianBlur(fgmask, (11, 11), 0)
        #filter = cv2.bilateralFilter(fgmask, 9, 75, 75)
        #filter = cv2.medianBlur(fgmask,3)

        # transformations
        kernelDilate = np.ones((7,7),np.uint8)
        kernelErode = np.ones((3, 3), np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        transformacion = cv2.dilate(filter,kernelDilate,iterations = 1)
        transformacion = cv2.erode(transformacion,kernelErode,iterations = 1)
        #transformacion = cv2.morphologyEx(filter,cv2.MORPH_OPEN,kernel)
        transformacion = cv2.morphologyEx(transformacion,cv2.MORPH_CLOSE,kernel)

        # countors
        contours = cv2.findContours(transformacion.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]
        # loop over the contours
        for contour in contours:
    		# ignore the small contours
            if cv2.contourArea(contour) < 1000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"

        grey_3_channel = cv2.cvtColor(transformacion, cv2.COLOR_GRAY2BGR)

        print("original: ", frame.dtype, frame.shape, frame.ndim, frame.size)
        print("processed: ", grey_3_channel.dtype, grey_3_channel.shape, grey_3_channel.ndim, grey_3_channel.size)

        # print original and
        numpy_horizontal = np.hstack((frame, grey_3_channel))
        cv2.imshow('frame',numpy_horizontal)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    backgroundSubtraction("M:\\Periodo 2\\Estudo Orientado\\DB\\rheinhafen\\rheinhafen.mpg");
    #print(image.dtype)
    #print(image.shape)
    #print(image.ndim)
    #print(image.size)


if __name__ == "__main__":
    main()
