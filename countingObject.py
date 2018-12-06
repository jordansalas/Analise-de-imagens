# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 22:34:54 2018

@author: jorda
"""

import numpy as np
import cv2
import os, os.path
import imutils
import argparse

def filter(frame):
    frame = cv2.GaussianBlur(frame, (9, 9), 0)
    #frame = cv2.bilateralFilter(frame, 9, 75, 75)
    #frame = cv2.medianBlur(frame,3)
    return frame

def transformation(frame):
    kernelDilate = np.ones((7,7),np.uint8)
    kernelErode = np.ones((3, 3), np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    frame = cv2.dilate(frame,kernelDilate,iterations = 1)
    frame = cv2.erode(frame,kernelErode,iterations = 1)
    #frame = cv2.morphologyEx(frame,cv2.MORPH_OPEN,kernel)
    frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel)
    return frame

def detection(frameOriginal, frameProcessed):
    font = cv2.FONT_HERSHEY_SIMPLEX
    contours = cv2.findContours(frameProcessed.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    # loop over the contours
    for contour in contours:
        # ignore the small contours
        if cv2.contourArea(contour) < 1500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        distanceX = abs(x-w)
        distanceY = abs(y-h)
        if distanceX > distanceY:
            proportion = distanceX/(distanceY+1)
            proportion = round(proportion, 2)
            if proportion < 3:
                cv2.putText(frameOriginal,'Proportion: ' + str(proportion),(x,y), font, 0.5, (255,255,100), 1, cv2.LINE_AA)
                cv2.rectangle(frameOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            proportion = distanceY/(distanceX+1)
            proportion = round(proportion, 2)
            if proportion < 5:
                cv2.putText(frameOriginal,'Proportion: ' + str(proportion),(x,y), font, 0.5, (255,255,100), 1, cv2.LINE_AA)
                cv2.rectangle(frameOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if proportion > 5:
                cv2.putText(frameOriginal,'Proportion: ' + str(proportion),(x,y), font, 0.5, (255,255,100), 1, cv2.LINE_AA)
                cv2.rectangle(frameOriginal, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frameOriginal

def backgroundSubtraction(video_path):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    #fgbg = cv2.createBackgroundSubtractorMOG2()
    while(1):
        text = "Unoccupied"
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)

        # filters
        img_filter = filter(fgmask)

        # transformations
        img_transformed = transformation(img_filter)

        # countors
        img_detection = detection(frame, img_transformed)

        # print original and processed frames
        grey_3_channel = cv2.cvtColor(img_transformed, cv2.COLOR_GRAY2BGR)
        numpy_horizontal = np.hstack((img_detection, grey_3_channel))
        cv2.imshow('frame',numpy_horizontal)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def converImageToVideo(path):
    ap = argparse.ArgumentParser()
    ap.add_argument("-ext", "--extension", required=False, default='jpg', help="extension name. default is 'jpg'.")
    ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
    args = vars(ap.parse_args())

    # Arguments
    dir_path = '.'
    ext = args['extension']
    output = args['output']

    images = []
    for file in os.listdir(path):
        if file.endswith(ext):
            images.append(file)

    imagePath = os.path.join(path, images[0])
    frame = cv2.imread(imagePath)
    print("qwe: ", frame.shape)
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for image in images:
        imagePath = os.path.join(path, image)
        frame = cv2.imread(imagePath)

        out.write(frame)



def main():
    backgroundSubtraction("M:\\Periodo 2\\Estudo Orientado\\DB\\rheinhafen\\rheinhafen.mpg")
    #backgroundSubtraction("M:\\Periodo 2\\Estudo Orientado\\DB\\Urban1\\outout.mp4")
    #converImageToVideo("M:\\Periodo 2\\Estudo Orientado\\DB\\Urban1\\M:\Periodo 2\\Estudo Orientado\\DB\\Urban1\\test")
    #print(image.dtype)
    #print(image.shape)
    #print(image.ndim)
    #print(image.size)


if __name__ == "__main__":
    main()
