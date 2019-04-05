"""
@author:  wangquaxiu
@time:  2018/9/4 16:02
"""
import cv2
import numpy as np

img = cv2.imread("G:/photos/test.jpg")
cv2.namedWindow("lenatest")
cv2.imshow("lena", img)
cv2.waitKey()
cv2.destroyAllWindows()