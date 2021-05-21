import numpy as np
import images as images
import torch
from torchvision.models import resnet18
from torchvision import transforms
import pandas as pd
import streamlit as st
import base64
from io import BytesIO
from random import randint
import SessionState
import cv2
import numpy as np

def get_contour_areas(contours):

    all_areas= []

    for cnt in contours:
        area= cv2.contourArea(cnt)
        all_areas.append(area)

    return all_areas

def main():
    #1000, 600, 6300, 4100
    im = cv2.imread('fail.png')
    #crop_img = img[y:y + h, x:x + w]
    #crop to remove axis
    crop_img = im[600:4100, 1000:6300]
    imgray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    max = 2
    for i in range(2,20):
        if cv2.contourArea(sorted_contours[i]) > 200000:
            max = i
        print(cv2.contourArea(sorted_contours[i]))
    print(len(contours))
    print("MAX:", str(max))
    imz = cv2.drawContours(crop_img, sorted_contours[2:max], -1, (0, 255, 0), 2)
    cv2.imwrite("elipse.png", imz)

   # cv2.namedWindow('fail.png', cv2.WINDOW_NORMAL)
    #imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(imz, (800, 600))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(1)


if __name__ == '__main__':
    main()