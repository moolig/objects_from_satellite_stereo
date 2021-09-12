
import numpy as np
import cv2 as cv


def corner_detection(X, Y):
    pass



im = cv.imread(r'/home/shmuelgr/Downloads/chshImg.jpg')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

ksize = 2
sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
ret = cv.getGaussianKernel(ksize, sigma, cv.CV_64FC1)

print(ret)
print(sigma)
