# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 18:34:55 2018

@author: jorda
"""

import numpy as np
import cv2
import os, os.path

def loadImage(dataBase):
    if dataBase == "GRAM1":
        imgPath1 = "M:\\Periodo 2\\Estudo Orientado\\DB\\M-30\\M-30\\image000001.jpg"
        imgPath2 = "M:\\Periodo 2\\Estudo Orientado\\DB\\M-30\\M-30\\image000004.jpg"
    elif dataBase == "GRAM2":
        imgPath1 = "M:\\Periodo 2\\Estudo Orientado\\DB\\Urban1\\Urban1\\image000001.jpg"
        imgPath2 = "M:\\Periodo 2\\Estudo Orientado\\DB\\Urban1\\Urban1\\image000004.jpg"
    elif dataBase == "PETS1":
        imgPath1 = "M:\\Periodo 2\\Estudo Orientado\\DB\\S0_CC\\Crowd_PETS09\\S0\\City_Center\\Time_12-34\\View_002\\frame_0001.jpg"
        imgPath2 = "M:\\Periodo 2\\Estudo Orientado\\DB\\S0_CC\\Crowd_PETS09\\S0\\City_Center\\Time_12-34\\View_002\\frame_0004.jpg"
    elif dataBase == "PETS2":
        imgPath1 = "M:\\Periodo 2\\Estudo Orientado\\DB\\S0_CC\\Crowd_PETS09\\S0\\City_Center\\Time_12-34\\View_007\\frame_0001.jpg"
        imgPath2 = "M:\\Periodo 2\\Estudo Orientado\\DB\\S0_CC\\Crowd_PETS09\\S0\\City_Center\\Time_12-34\\View_007\\frame_0002.jpg"
    elif dataBase == "PETS3":
        imgPath1 = "M:\\Periodo 2\\Estudo Orientado\\DB\\S0_CC\\Crowd_PETS09\\S0\\City_Center\\Time_14-55\\View_003\\frame_0001.jpg"
        imgPath2 = "M:\\Periodo 2\\Estudo Orientado\\DB\\S0_CC\\Crowd_PETS09\\S0\\City_Center\\Time_14-55\\View_003\\frame_0004.jpg"
    img1 = cv2.imread(imgPath1)
    img2 = cv2.imread(imgPath2)

    return img1, img2

def showImage(img_list):
    numpy_horizontal = np.hstack(img_list)
    cv2.imshow('output', numpy_horizontal)
    cv2.waitKey(0)
    cv2.destroyWindow('output')

def scaleImage(img_list, scale):
    img_listResult = []
    for img_original in img_list:
        img_scaled = cv2.resize(img_original, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        img_listResult.append(img_scaled)
    return img_listResult


def basicMethod(img_grey1, img_grey2):
    img_listResult = []
    # applying blur
    #img_blur1 = cv2.GaussianBlur(img_grey1, (21, 21), 0)
    #img_blur2 = cv2.GaussianBlur(img_grey2, (21, 21), 0)

    # absolute subtraction
    #img_subtraction = cv2.absdiff(img_blur1, img_blur2)
    img_subtraction = cv2.absdiff(img_grey1, img_grey2)
    # threshold
    img_threshold = cv2.threshold(img_subtraction, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(img_threshold, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    print(cnts)

    img_listResult.append(img_grey1)
    #img_listResult.append()
    #img_listResult.append(img_blur1)
    img_listResult.append(img_threshold)
    #scaling imagens
    img_listResult = scaleImage(img_listResult, 0.5)
    # print original image and result
    showImage(img_listResult)

def runningAverage(img_grey1, img_grey2):
    img_listResult =[]
    moving_average = np.float32(img_grey1)
    # running average
    cv2.accumulateWeighted(img_grey2, moving_average, 0.5, None)
    # convert to 8-bit image
    img_8bit= cv2.convertScaleAbs(moving_average)
    # absolute subtraction
    img_subtraction = cv2.absdiff(img_grey1, img_8bit)
    # threshold
    img_threshold = cv2.threshold(img_subtraction, 25, 255, cv2.THRESH_BINARY)[1]

    img_listResult.append(img_grey1)
    img_listResult.append(img_8bit)
    img_listResult.append(img_threshold)
    img_listResult = scaleImage(img_listResult, 0.5)
    # print original image and result
    showImage(img_listResult)

def runningGaussianAverage():
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/M-30/M-30/"
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/Urban1/Urban1/"
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_12-34/View_002/"
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_12-34/View_007/"
    img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_14-55/View_003/"

    #img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_12-34/View_002/"
    i = 0
    k = 2.5
    all_frames = 15
    mean_frame = all_frames - 1 # (4)
    img_list = []
    img_listResult = []

    #read images
    for file in os.listdir(img_path):
        if i < all_frames:
            img_original = cv2.imread(os.path.join(img_path, file), 0)
            img_list.append(img_original)
            i += 1
        else:
            break
    img_evaluate = img_list[mean_frame]
    w, h = img_evaluate.shape[:2]

    #mean from N images
    sum_img = np.zeros((w, h), dtype=np.int32)
    for img in img_list[:mean_frame]:
        sum_img += img
    img_mean = np.divide(sum_img, mean_frame)
    img_mean = img_mean.astype(np.uint8)

    #standar deviation from N images
    sum_subtraction = np.zeros((w, h), dtype=np.int)
    for img in img_list[:mean_frame]:
        sum_subtraction += ((img - img_mean)**2)
    img_deviation = np.divide(sum_subtraction, (mean_frame - 1))
    img_deviation = img_deviation.astype(np.uint8)

    #evaluation foreground
    img_foreground = np.zeros((w, h), dtype=np.uint8)
    img_difference = cv2.absdiff(img_list[mean_frame], img_mean)
    for row in range(w):
        for column in range(h):
            if img_difference[row][column] > (k*np.sqrt(img_deviation[row][column])):
                img_foreground[row][column] = 255

    #resize images
    img_listResult.append(img_evaluate)
    #img_listResult.append(img_mean)
    #img_listResult.append(img_deviation)
    #img_listResult.append(img_difference)
    img_listResult.append(img_foreground)
    img_listResult = scaleImage(img_listResult, 0.5)
    # print original image and result
    showImage(img_listResult)

def mixtureGaussian():
    img_path = "M:/Periodo 2/Estudo Orientado/DB/M-30/M-30/"
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/Urban1/Urban1/"
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_12-34/View_002/"
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_12-34/View_007/"
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_14-55/View_003/"

    #img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_12-34/View_002/"
    i = 0
    all_frames = 20
    img_list = []
    img_result = []
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    img_listResult = []

    #read images and get the mask
    for file in os.listdir(img_path):
        if i < all_frames:
            img_original = cv2.imread(os.path.join(img_path, file), 0)
            img_list.append(img_original)
            img_result = fgbg.apply(img_original)
            i += 1
        else:
            break

    #resize image
    img_listResult.append(img_list[all_frames-1])
    img_listResult.append(img_result)
    img_listResult = scaleImage(img_listResult, 0.5)
    # print original image and result
    showImage(img_listResult)

def backgroundSubtractorMOG2():
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/M-30/M-30/"
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/Urban1/Urban1/"
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_12-34/View_002/"
    img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_12-34/View_007/"
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_14-55/View_003/"

    #img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_12-34/View_002/"
    i = 0
    all_frames = 30
    img_list = []
    img_result = []
    fgbg = cv2.createBackgroundSubtractorMOG2()
    img_listResult = []

    #read images and get the mask
    for file in os.listdir(img_path):
        if i < all_frames:
            img_original = cv2.imread(os.path.join(img_path, file), 0)
            img_list.append(img_original)
            img_result = fgbg.apply(img_original)
            i += 1
        else:
            break

    #resize image
    #img_originalResize = cv2.resize(img_list[all_frames-1], None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    #img_resultResize = cv2.resize(img_result, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    img_listResult.append(img_list[all_frames-1])
    img_listResult.append(img_result)
    img_listResult = scaleImage(img_listResult, 0.5)
    showImage(img_listResult)

def backgroundSubtractorGMG():
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/M-30/M-30/"
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/Urban1/Urban1/"
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_12-34/View_002/"
    #img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_12-34/View_007/"
    img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_14-55/View_003/"

    #img_path = "M:/Periodo 2/Estudo Orientado/DB/S0_CC/Crowd_PETS09/S0/City_Center/Time_12-34/View_002/"
    i = 0
    all_frames = 140
    img_result = []
    img_listResult = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    img_result = np.zeros((576, 768), dtype=np.int32)
    #read images and get the mask
    for file in os.listdir(img_path):
        if i < all_frames:
            img_original = cv2.imread(os.path.join(img_path, file), 0)
            img_result = fgbg.apply(img_original)
            img_result = cv2.morphologyEx(img_result, cv2.MORPH_OPEN, kernel)
            i += 1
        else:
            break

    #resize image
    #img_originalResize = cv2.resize(img_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    #img_resultResize = cv2.resize(img_result, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    img_listResult.append(img_original)
    img_listResult.append(img_result)
    img_listResult = scaleImage(img_listResult, 0.5)
    showImage(img_listResult)

def main():
    # GRAM1, GRAM2, PETS1, PETS2, PETS3
    #img1, img2 = loadImage("GRAM2")
    img1, img2 = loadImage("PETS2")
    img_grey1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img_grey2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    basicMethod(img_grey1, img_grey2)
    #runningAverage(img_grey1, img_grey2)
    #runningGaussianAverage()
    #mixtureGaussian()
    #backgroundSubtractorMOG2()
    #backgroundSubtractorGMG()

    #image = res
    #print(image.dtype)
    #print(image.shape)
    #print(image.ndim)
    #print(image.size)


if __name__ == "__main__":
    main()
