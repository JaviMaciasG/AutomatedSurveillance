from ctypes import *

import random
from typing import List, Any, Tuple

import numpy as np
import cv2

import os

import re

import time


def erasethis(folder):

    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def getfiles(path,format):
    if format == 0 or format == None:
        ending = ".jpg"
    else:
        ending = ".png"

    files = []

    Directory = path
    for filename in os.listdir(Directory):  # type: str
        if filename.endswith(ending):
            files.append(Directory+"/"+filename)
            #print(os.path.join(Directory, filename))
            continue
        else:
            continue

    return files


def saveimage(img, framepassed, path2, format):
    if format == 0 or None:
        ending = ".jpg"
    else:
        ending = ".png"

    if path2 == None:
        path = '/mypath/git/darknet-master/results'
    else:
        path = path2
    name = str(framepassed) + ending
    cv2.imwrite(os.path.join(path, name), img)


