# !/usr/bin/python3
# coding: utf-8
# @author: Deng Junwei
# @date: 2019/3/21
# @institute:SJTU

import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import math



def savepic(output, filename):
    output = output.cpu().detach().numpy()
    img = _pic_coloring(output)
    cv2.imwrite(filename, img) # pay attention to the /255
    print(img.shape)

def _pic_coloring(output):
    img = np.zeros((output.shape[1], output.shape[2], 3))
    Unlabelled  =   np.array([205, 250, 255]) #black, background
    A           =   np.array([80,    0,      255 ]) #red, inflammatory cells 
    B           =   np.array([139,   64,    39 ]) #Light blue, nuclei
    C           =   np.array([235, 206, 135]) #green, cytoplasm
    maxindex = np.argmax(output,axis=0)
    img[maxindex==0]=Unlabelled
    img[maxindex==1]=A
    img[maxindex==2]=B
    img[maxindex==3]=C
    return img