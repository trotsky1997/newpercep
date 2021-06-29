'''
this scripts is aimed for a high-performance video human detection proccessing tool.

'''

import os
import sys
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
sys.path.append(root_dir)
from apis import misc
from model.yolov5.utils.datasets import LoadImages
cpus = os.cpu_count()


yolo = misc.get_yolov5()





tracker = misc.get_sort()
deepmar = misc.get_deepmar()

img = cv2.imread("party.jpg")#h,w,c(gbr)

res,ori_sz,new_sz = yolo.preprocess(img)
print(ori_sz,new_sz)
res = yolo.inference(res)
res = yolo.postprocess(res,new_sz,ori_sz)
print(res.int())
xywhc = tracker.preprocess(res)
xyxyi = tracker.inference(xywhc,img)
print(xyxyi)