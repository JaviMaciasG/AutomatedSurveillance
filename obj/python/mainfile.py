from ctypes import *

import random

from typing import List, Any, Tuple

from pathlib import Path

import numpy as np

import cv2

import os

import re

import time

import psutil

import objectclass

from Anomaly.code.myflowr import Stampade

#import kalman

from PIL import Image

import argparse

import Anomaly

from kalman3 import KalmanFilter



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

""""
class cd:
    Context manager for changing the current working directory
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
"""

def saveimage(img, framepassed, path2, format):
    if format == 0 or None:
        ending = ".jpg"
    else:
        ending = ".png"

    if path2 == None:
        path = '/mypath/darknet-master/results'
    else:
        path = path2
    name = str(framepassed) + ending
    cv2.imwrite(os.path.join(path, name), img)

def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1

def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL(b"/mypath/darknet-master/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])  # type: List[Tuple[Any, Any]]
    return res


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def detect(net, meta, image,classespath,thresh=.5, hier_thresh=.5, nms=.45):
    img = image.encode("UTF-8")
    im = load_image(img, 0, 0)
    with open(classespath, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                label = str(classes[i])
                b = dets[j].bbox
                res.append((label, dets[j].prob[i], (round(b.x), round(b.y), round(b.w), round(b.h))))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


def printeliminate(img, personremove, elimlist):
    global frameprint
    global maintaintext

    # print("frameprint is ", frameprint)

    elimlistrange = len(elimlist)

    # print("elimlistrange is ", elimlistrange)

    if personremove == 1:
        for x in range(elimlistrange):
            frameprint = 5
            maintaintext = elimlist[x]

    if frameprint > 0:
        for x in range(elimlistrange):
            cv2.putText(img, "borrando persona " + str(elimlist[x]), (10 + (x * 10), 1000), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 2)
            # print("printing remove on picture : borrando persona ", elimlist[x])

        frameprint = frameprint - 1

    if frameprint == 0 and elimlistrange > 0:
        for x in range(elimlistrange):
            del elimlist[0]

    return img


def desres(res, num):
    commas = []
    news = str(res[num])
    newslen = len(news)
    foundat = 0
    # print(news)
    while foundat < (newslen - 8):
        foundat = news.find(",", foundat + 1)
        # print(foundat)
        commas.append(foundat)
    # print(commas)

    upper1 = news.find("'")
    upper2 = news.find("'", upper1 + 1)
    firstp = news.find("(", 5)
    lastp = news.find(")")

    name = news[upper1 + 1:commas[0] - 1]
    #print("name",name)
    probs = news[commas[0] + 2:commas[1]]
    #print("probs",probs)
    x = news[firstp + 1:commas[2]]
    y = news[commas[2] + 2:commas[3]]
    w = news[commas[3] + 2:commas[4]]
    h = news[commas[4] + 2:lastp]
    #print("x",x,"y",y,"w","h",h)

    return name, float(probs), int(x), int(y), int(w), int(h)


def kalmanupdate(x,y,w,h,img,kalman,speed,label,linked,id,frameN):

    print("-",frameN)
    x = np.float32(x)
    y = np.float32(y)
    w = np.float32(w)
    h = np.float32(h)

    cox = np.float32(x + (w-x)/2)

    coy = np.float32(y + (h-y)/2)

    def correction(xpos, ypos, cox, coy, speed, label, linked, id,img):
        current_measurement = np.array([[np.float32(cox)], [np.float32(coy)]])
        velocity = np.array([[np.float32(speed[0])], [np.float32(speed[1])]])

        cmx, cmy = int(current_measurement[0]), int(current_measurement[1])


        vmx, vmy = int(speed[0]), int(speed[1])

        previouspred = kalman.getprediction()

        prevcx,prevcy,prevvx,prevvy = previouspred[0],previouspred[1],previouspred[2],previouspred[3]

        update = np.array([[cmx],[cmy],[vmx],[vmy]])
        print("kalman update for ",label,id," is ",update)
        kalman.correct(update)

        current_prediction = kalman.predict()

        cpx,cpy,cvx,cvy = current_prediction[0], current_prediction[1], current_prediction[2], current_prediction[3]

        print(" -- Real prediction ",label," --",id)

        print(" bag ",id," update position : ", cmx," ",cmy, " ,and velocity ", vmx," ", vmy)
        print(" predicted position for this bag ",cpx," ",cpy,"and velocity ", cvx, " ", cvy)

        print(" previous error real : ", cmx-prevcx," ", cmy-prevcy," ",vmx-prevvx," ",vmy-prevvy)


        x2 = int(cpx - (cmx-xpos))
        y2 = int(cpy - (cmy-ypos))
        w2 = (cpx + (cmx-xpos))
        h2 = (cpy + (cmy-ypos))

        '''
        print(" x2 is ", x2)
        print(" y2 is ", y2)
        print("w2 is ", w2)
        print("h2 is ", h2)
        '''
        if ShowPrediction :
            cv2.rectangle(img, (x2, y2), (w2, h2), (255, 255, 255), 2)
            cv2.putText(img, "            " + str(id), (x2 - 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # return current_prediction
        prediction = np.array([[cpx], [cpy], [vmx], [vmy]])
        return prediction

    prediction = correction(x, y, cox, coy, speed, label, linked, id,img)

    return prediction

def Drawmissing(x,y,w,h,img,id):
    if ShowPrediction:

        cv2.rectangle(img, (x, y), (w, h), (100, 0, 0), 2)
def DrawKalman3(x,y,w,h,img,kalman,speed,label,linked,id,frameN):
    print(frameN)

    x = np.float32(x)
    y = np.float32(y)
    w = np.float32(w)
    h = np.float32(h)
    '''
    print("Drawing Kalman bag id ",id)
    print(x)
    print(y)
    print(w)
    print(h)
    '''


    cox=np.float32(x + (w-x)/2)

    coy = np.float32(y + (h-y)/2)



    def correction(xpos,ypos,cox,coy,speed,label,linked,id,img):

        current_measurement = np.array([[np.float32(cox)], [np.float32(coy)]])

        previouspred = kalman.getprediction()

        prevcx,prevcy,prevvx,prevvy = previouspred[0],previouspred[1],previouspred[2],previouspred[3]

        current_prediction = kalman.predict()

        velocity = np.array([[np.float32(speed[0])], [np.float32(speed[1])]])

        cmx, cmy = int(current_measurement[0]), int(current_measurement[1])

        cpx, cpy, cvx, cvy = current_prediction[0], current_prediction[1], current_prediction[2], current_prediction[3]



        vmx, vmy = int(speed[0]), int(speed[1])

        prediction = np.array([cpx, cpy])


        update2 = np.array([[cmx],[cmy],[vmx],[vmy]])



        x2 = int(cpx - (cmx-xpos))
        y2 = int(cpy - (cmy-ypos))
        w2 = (cpx + (cmx-xpos))
        h2 = (cpy + (cmy-ypos))


        w2 = int(w2)
        h2 = int(h2)

        #print("w2 is ", w2)
        #print("h2 is ", h2)


        print("-- Repetitive prediction ",label," --", id)

        print(" bag ", id, " update position : ", cpx, " ", cpy, " ,and velocity ", vmx, " ", vmy)
        #print(" predicted position for this bag ", fpx, " ", fpy, " ,and velocity ", fvx, " ", fvy)

        print(" previous error real : ", cpx-prevcx," ", cpy-prevcy," ",vmx-prevvx," ",vmy-prevvy)


        if ShowPrediction:
            cv2.rectangle(img, (x2, y2), (w2, h2), (0, 0, 0), 2)
            #cv2.putText(img, label, (x2 - 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(img, "             " + str(id), (x2 - 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 250), 2)



        #return current_prediction

        return prediction


    prediction = correction(x,y,cox,coy,speed,label,linked,id,img)

    return prediction

def createkalman(xo,yo):

        AT = 1

        #stateMatrix = np.zeros((4, 1), np.float32)  # [x, y, delta_x, delta_y]
        stateMatrix = np.array([[xo],[yo],[0],[0]],np.float32)
        estimateCovariance = np.eye(stateMatrix.shape[0])
        transitionMatrix = np.array([[1, 0, AT, 0], [0, 1, 0, AT], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        measurementStateMatrix = np.zeros((4, 1), np.float32)
        observationMatrix = np.array([[1,0,0,0],[0,1,0,0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        #measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1
        measurementNoiseCov = np.array([[10,0,0,0],[0,10,0,0], [0, 0, 5, 0], [0, 0, 0, 5]], np.float32)
        kalmancreated = KalmanFilter(X=stateMatrix,
                                     P=estimateCovariance,
                                     A=transitionMatrix,
                                     Q=processNoiseCov,
                                     Z=measurementStateMatrix,
                                     H=observationMatrix,
                                     R=measurementNoiseCov)
        return kalmancreated


def createkalman2():
    stateMatrix = np.zeros((4, 1), np.float32)  # [x, y, delta_x, delta_y]
    estimateCovariance = np.eye(stateMatrix.shape[0])
    transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.001
    measurementStateMatrix = np.zeros((4, 1), np.float32)
    observationMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    # measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1
    measurementNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]], np.float32)
    kalmancreated = KalmanFilter(X=stateMatrix,
                                 P=estimateCovariance,
                                 A=transitionMatrix,
                                 Q=processNoiseCov,
                                 Z=measurementStateMatrix,
                                 H=observationMatrix,
                                 R=measurementNoiseCov)
    return kalmancreated


class associatedkalman:

    def __init__(self,id,name,kalmanfilter):
        self.id = id
        self.name = name
        self.kalmanfilter = kalmanfilter

def getfilter(id,name,associateds):

        Len = len(associateds)
        exists = 0
        for x in range(Len):

            if associateds[x].name == name:

                if associateds[x].id == id:
                    exists = 1
                    return exists,associateds[x]

        if exists == 0 :

           return exists,None


def array_to_image(arr):
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr / 255.0).flatten()
    data = c_array(c_float, arr)
    im = IMAGE(w, h, c, data)
    return im

#Detection function used
def detection(net, meta, image, classespath,thresh=.5, hier_thresh=.5, nms=.45):
    # IF IT IS A VIDEO
    global frameprint
    global elimlist

    start = time.time()


    #isanimage = image.find("png") + image.find("jpg")
    #isavideo = image.find("avi")
    isanimage = 5
    isavideo = 0

    # read class names from text file
    classes = None
    with open(classespath, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    if (isanimage > 0):

        if camera == 0:
            print("image detected")

            newim = image.encode("UTF-8")
        # print(newim)
            im = load_image(newim, 0, 0)
            img = cv2.imread(image)

        else:
            img = image
            im = array_to_image(image)
            rgbgr_image(im)

        # print(im.w)
        # print(im.h)
        num = c_int(0)
        pnum = pointer(num)
        # slow process
        predict_image(net, im)

        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]






        if (nms): do_nms_obj(dets, num, meta.classes, nms);

        res = []

        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    label = str(classes[i])
                    person = "person"
                    backpack = "backpack"
                    bag = "bag"
                    suitcase = "suitcase"
                    handbag = "handbag"

                    if ((bag in label or suitcase in label or handbag  in label or backpack in label or person in label) ):

                        b = dets[j].bbox

                        x = round(b.x - b.w / 2)
                        y = round(b.y - b.h / 2)
                        w = round(x + b.w)
                        h = round(y + b.h)

                        res.append((label, dets[j].prob[i], (x, y, w, h)))
        res = sorted(res, key=lambda x: -x[1])

        if camera == 0: free_image(im)
        free_detections(dets, num)

        end = time.time()

        dif = end-start
        fps = 1/dif
        print("fps :",fps)

        return res

def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)

    return image

def heatmapping(res,image,framespassed,savedic):

    global PathLocation

    global totalfifties
    fiftiesrep = 0
    fiftieson = 0
    cycle = 0
    maxheatmapframes = 50
    tracehistory = 49
    # variasmochilas014
    #path = '/mypath/Videos/extract/linked'

    reslong = len(res)

    basepath = PathLocation + "/darknet-master/Heatmap/base"

    #Determine heatmapframe

    heatmapframes = framespassed

    # Cycles are defined to merge binary images later on

    if framespassed > tracehistory:
        fiftieson =1
        cycle = 1

    iteration = 0

    # We do cycles from 50 to 50 images to update the trace history.
    if cycle == 1:

            while ( (tracehistory*iteration) < heatmapframes):
                iteration = iteration + 1
                print ("iteration",iteration)

    if cycle == 1:

        iteration = iteration -1

    heatmapframes = heatmapframes - (tracehistory*iteration)

    print(heatmapframes)

    # read the image

    img = cv2.imread(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    accum_image_ellipse = np.zeros((height, width), np.uint8)
    accum_image = np.zeros((height, width), np.uint8)
    accum_image2 = np.zeros((height, width), np.uint8)

    # YOLO detection to find where the people are at.

    for b in range(reslong):

        label, prob, x, y, w, h = desres(res, b)
        print("label", label, " prob", prob, " x", x, " y", y, "w ", w, " h", h)
        id = 0

        if label == "person":

            # The elipse is drawn on the people position

            cv2.ellipse(accum_image_ellipse, (round(x + (w - x) / 2), round(h)), (round((w - x) / 2), round((w - x) / 4)), 0, -180, 180,(255, 255, 255), -1)



    thresh = 1
    maxValue = 1
    ret, th1 = cv2.threshold(accum_image_ellipse, thresh, maxValue, cv2.THRESH_BINARY)
    saveimage(th1, "test3", savedic,0)


    # heatmapframes represent the value of files we have in the temporary folder

    heatmapframes


    # The current image is saved and then, we check all the images in the temp. folder to merge them together
    # and hence, draw the traces.
    saveimage(th1,heatmapframes, PathLocation + "/darknet-master/Heatmap/temp",0)


    Directory2 = PathLocation + "/darknet-master/Heatmap/temp"

    files2 = getfiles(Directory2,0)  # jpg files

    files2.sort(key=lambda var: [x if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    x = 0

    if (fiftieson == 1):

        Directory3 = PathLocation + "/darknet-master/Heatmap/temp/fifties"
        files3 = getfiles(Directory3, 0)  # jpg files

        print("files in fifties",image)
        for it in range(iteration):

            if it == 0:
                start =1
            if iteration > 6:
                start = (iteration -6) + it
            else:
                start = it+1
            fif = PathLocation + "/darknet-master/Heatmap/temp/fifties/" + str(start) + ".jpg"
            print("adding ",fif)
            fif.encode('utf-8')

            fifties = cv2.imread(fif)
            cv2.waitKey(1)
            newfiftiesgray = cv2.cvtColor(fifties, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            accum_image = cv2.add(accum_image, newfiftiesgray)

            if start == iteration:
                break

    # we get the acummulated image of the elipses

    for stri2 in files2:

        newth = cv2.imread(stri2)
        newthgray = cv2.cvtColor(newth, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        saveimage(newthgray, "befacu", PathLocation + "/darknet-master/Heatmap", 0)

        accum_image2 = cv2.add(accum_image2, newthgray)



        if x < heatmapframes:
            x = x+1
        else:

            break



    if heatmapframes == tracehistory:
        totalfifties = totalfifties + 1
        print(" totalfifties to save ", totalfifties)
        saveimage(accum_image2, str(totalfifties), PathLocation + "/darknet-master/Heatmap/temp/fifties", 0)

    accum_image = cv2.add(accum_image2, accum_image)

    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    cv2.imwrite('diff-color.jpg', color_image)

    # for testing purposes, show the accumulated image
    saveimage(accum_image,"acc",PathLocation + "/darknet-master/Heatmap",0)

    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    # for testing purposes, show the colorMap image
    cv2.imwrite('diff-color.jpg', color_image)



     # overlay the color mapped image to the first frame ( base picture)

    exists = 0
    for the_file in os.listdir(basepath):
        file_path = os.path.join(basepath, the_file)
        try:
            if os.path.isfile(file_path):
                exists = 1

            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    print("exists is ",exists)
    if exists:
        basepathimg = basepath + "/base.png"
        baseimg = cv2.imread(basepathimg)
        result_overlay = cv2.addWeighted(baseimg, 0.7, color_image, 0.8, 0)
    else:
        # Keep presets

        result_overlay = cv2.addWeighted(img, 0.7, color_image, 0.8, 0)
    #cv2.imshow("overlay", result_overlay)
    #cv2.waitKey(1)

    saveimage(result_overlay,framespassed,savedic,0)

    framespassed += 1

    return result_overlay,framespassed

def aspectratio(res,image):

    global maxratio,minratio,maxarea,minarea
    global max
    cycle = 0
    path = '/mypath/Videos/extract/linked'

    reslong = len(res)

    img = cv2.imread(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    print(height, width)
    mean = []
    areapeoplemean =[]

    for b in range(reslong):

        label, prob, x, y, w, h = desres(res, b)
        print("label", label, " prob", prob, " x", x, " y", y, "w ", w, " h", h)
        id = 0
        hpeople = (h-y)
        wpeople = (w-x)
        areapeople = hpeople*wpeople
        aspect = (h-y)/(w-x)
        print("aspect video ratio ", height/width)
        print("aspect ratio ",b," is h/w =",aspect)
        areapeoplemean.append(areapeople)
    if mean:

        ##aspects
        meanofmean = np.mean(mean)
        print("mean is ",meanofmean)
        if meanofmean> maxratio:
         maxratio = meanofmean
        print("max ratio ",maxratio)
        if meanofmean < minratio:
         minratio = meanofmean
        print("min ratio ", minratio)
        ##areas
        meanofareas = np.mean(areapeoplemean)
        print("mean is ",meanofareas)
        if meanofareas> maxarea:
            maxarea = meanofareas
        print("max area ",maxarea)
        if meanofareas < minarea:
            minarea = meanofareas
        print("min area ", minarea)
        print("total people ",reslong)

def IdentificationProcess(r,stri, people, bags, framespassed, KalmanArray,kalmanbags,kalmanpeople,predictionbags,showprediction):

    img, elimpeople, elimbags = identification(r, stri, people, bags, framespassed, KalmanArray,kalmanbags,kalmanpeople,predictionbags,showprediction)
    lasremvalue = elimpeople[0]
    elimlist = elimpeople[1]
    img = printeliminate(img, lasremvalue, elimlist)
    if camera >= 0:
        saveimage(img, framespassed, '/mypath/git/darknet-master/results', 1)
    framespassed += 1

    return framespassed,img

def identification(res, image, people, bags, framespassed,KalmanArray,kalmanbags,kalmanpeople,predictionbags,showprediction):

    global frameprint
    global elimlist
    global ShowPrediction
    global KalmanPerson

    lasremvalue = 0
    lasremvalue2 = 0
    KalmanPerson = kalmanpeople
    KalmanBag = 0
    PersonOK = 1
    ShowPrediction = showprediction
    ShowOwner = 1

    start = time.time()

    LimitPerson = 100
    LimitPersonKalmanUpdate = 30
    LimitBag = 300
    # YOLO returns an array with strings of the label and position of the bounding boxes for every object detected.
    # The length of the array is obtained and for every item in the array,an iteration is done to detect if the
    # object is a person or a bag.

    reslong = len(res)
    print(reslong)
    if camera==0:
        img = cv2.imread(image)
        img2 = cv2.imread(image)
    else:
        img = image
        img2 = img

    for b in range(reslong):

        # desres() is the function that obtains label and coordinates of the objects.

        label, prob, x, y, w, h = desres(res, b)
        # print("label",label," prob",prob," x",x," y",y,"w ",w," h",h)
        id = 0
        linkedstat = 0

        # If a person label is found, the identification process begins.
        # In this conditional, with the new bounding box found, the program will determine if it is a new person
        # that has entered in the scope of the camera or if it is a person that has moved in its sight.

        if label == "person":

            # GetHistValue returns the histogram value of the cropped image of a person in this case.
            # It is not used currently given the bad results obtained.

            histogramvalue = []
            histogramvalue = GetHistValue(img2, y, h, x, w)

            print('')
            print(" ## determine new or update person ##")
            print('')

            # DetermineNewOrUpdate() is a function that returns an id of the object with the bounding box found
            # in the new frame. This function will return the id of the object that in the previous frame had
            # the closest distance and area compared to the current frame or the id of a new object if there is
            # any similarities.

            id = objectclass.person.DetermineNewOrUpdate(x, y, w, h, people, framespassed, img, histogramvalue,
                                                         LimitPersonKalmanUpdate)

            # Lastupdate () checks if the object was undetected for a certain number of frames. If that is the case,
            # the object will be removed from the array in the following if conditional bellow.

            printremove, personremove, eliminate = objectclass.person.LastUpdate(framespassed, people, LimitPerson,
                                                                                 label)

            # This line is used to display a message on the screen later for an extended period of time.
            lasremvalue = lasremvalue or personremove

            # Checkposition() returns the position of an object in the array when the id is introduced and therefore
            # if the object still exists.
            objectposition = objectclass.CheckPosition(people, id)

            # The variable speed will be used for the kalman filter later.

            speed = people[objectposition].speed
            print("person ", id, " velocity is :", speed)

            # The variable item status will be obtained later from this object.
            itemstatus = people[objectposition]


            # If conditional to remove person from people array.
            if personremove == 1:
                frameprint = 5
                print(printremove)
                print("\n")
                cv2.putText(img, "eliminando people", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # print("eliminando people ", people[eliminate].id)

                # Introducing deleted people in the array

                elimlist.append(eliminate)
                del people[eliminate]

        if label == "backpack" or label == "handbag" or label == "suitcase":

            histogramvalue = GetHistValue(img2, y, h, x, w)

            # Label is simplified and the labels of backpack, handbag and suitcase are condensed in the bag label.

            label = "bag"

            print()
            print(" ## determine new or update bag ##")
            print()

            # DetermineNewOrUpdate() is a function that returns an id of the object with the bounding box found
            # in the new frame. This function will return the id of the object that in the previous frame had
            # the closest distance and area compared to the current frame or the id of a new object if there is
            # any similarities.

            id = objectclass.backpack.DetermineNewOrUpdate(x, y, w, h, bags, framespassed, people, img, histogramvalue)

            # Lastupdate () checks if the object was undetected for a certain number of frames. If that is the case,
            # the object will be removed from the array in the following if conditional bellow.

            printremove2, bagsremove, eliminate2 = objectclass.backpack.LastUpdate(framespassed, bags, LimitBag, label)
            lasremvalue2 = lasremvalue2 or bagsremove

            # Checkposition() returns the position of an object in the array when the id is introduced and therefore
            # if the object still exists.

            objectposition = objectclass.CheckPosition(bags, id)
            linkedstat = bags[objectposition].linked

            # The variable speed will be used for the kalman filter later.

            speed = bags[objectposition].speed
            print("bag", bags[objectposition].id, " speed is ", speed)

            # do the kalman update for updated bag

            if bags[objectposition].linked == 1:
                personposition = objectclass.CheckPosition(people, bags[objectposition].personlinked)
                if personposition != None:
                    speed = people[personposition].speed
                    print("bag ", bags[objectposition].id, " speed is ", bags[objectposition].speed,
                          ", linked to person ", people[personposition].id, " ,speed is :", speed)

                    if ShowPrediction:
                        #ShowOwner
                        bagx = bags[objectposition].xcentroid
                        bagy = bags[objectposition].ycentroid
                        personx = people[personposition].xcentroid
                        persony = people[personposition].ycentroid
                        cv2.line(img, (bagx, bagy), (personx, persony), (255, 0, 0), 5)

            # If conditional to remove bag from people array.

            itemstatus = bags[objectposition]
            if bagsremove == 1:
                print(printremove2)
                print("\n")
                cv2.putText(img, "eliminando backpack", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)
                # print("eliminando people ", bags[eliminate].id)
                elimlist2.append(eliminate2)
                del bags[eliminate2]

        # The object  is drawed in the image.

        if label == "person" or label == "bag":
            status = objectclass.GetStatus(itemstatus)
            cv2.rectangle(img, (x, y), (w, h), (status), 2)
            cv2.rectangle(img, (x - 1, y - 20), (x + 80, y), (status), -1)
            # + " " + str(id) IF WE WANT TO ADD ID
            # if label == "person":
            cv2.putText(img, (label + str(id)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            '''+ str(id)'''

            # check the kalman prediction

            exists, newkalman = getfilter(id, label, KalmanArray)

            if exists == 0:
                print("creating new Kalman for ", label, id)
                # newkalmanfilter = createkalman((x + (w-x)/2),(y + (h-y)/2))
                newkalmanfilter = createkalman2()
                newkalman = associatedkalman(id, label, newkalmanfilter)
                KalmanArray.append(newkalman)


            # Dawing kalman prediction and doing update stage

            if (label == "bag") and (kalmanbags == 1):
                objectposition = objectclass.CheckPosition(bags, id)
                AT = framespassed - bags[objectposition].prevlastframe
                print(" ## kalman update for bag##")
                kalmanupdate(x, y, w, h, img, newkalman.kalmanfilter, speed, label, linkedstat, id, framespassed)

            if (label == "person") and (kalmanpeople == 1):
                personposition = objectclass.CheckPosition(people, id)
                AT = framespassed - people[personposition].prevlastframe
                print(" ## kalman update for person ##")
                kalmanupdate(x, y, w, h, img, newkalman.kalmanfilter, speed, label, linkedstat, id, framespassed)



    bagslen = len(bags)
    peoplelen = len(people)

    print('')
    print(" ## linking backpack ##")
    print('')
    objectclass.LinkedBackpack(people, bags, KalmanArray)


    # Process to update the fake position for the missing backpacks. Done with kalman or quick prediction method
    if (objectclass.changed2 == 1):

        if KalmanPerson == 1:
            for i in range(peoplelen):

                updated = people[i].updated
                speed = people[i].speed

                if (updated == 0) and (framespassed - people[i].lastframe < 30) and (speed[0] < 30 and speed[1] < 30):

                    fake = people[i].GetFake()

                    speed = people[i].speed

                    xcentroid = int(fake[0] + speed[0])
                    ycentroid = int(fake[1] + speed[1])
                    print()
                    xmissing = round(xcentroid - people[i].rwidth / 2)
                    wmissing = round(xcentroid + people[i].rwidth / 2)
                    ymissing = round(ycentroid - people[i].rheight / 2)
                    hmissing = round(ycentroid + people[i].rheight / 2)
                    exists, newkalman = getfilter(people[i].id, "person", KalmanArray)

                    print("Kalman of unupdated person ", people[i].id, ", exists is ", exists, ", not found for",
                          framespassed - people[i].lastframe)
                    print("centroid is x:", fake[0], " y:", fake[1])

                    AT = framespassed - people[i].prevlastframe
                    # KalmanUp =DrawKalman3(x, y, w, h, img, newkalman.kalmanfilter, speed, "bag", linkedstat,(bags[i].id),AT,framespassed)

                    if  (exists == 1):
                        label = "people"
                        KalmanUp = DrawKalman3(xmissing, ymissing, wmissing, hmissing, img, newkalman.kalmanfilter,
                                               speed,
                                               label, 1, people[i].id, framespassed)

                        people[i].GetFakeCentroid(int(KalmanUp[0]), int(KalmanUp[1]))

        for i in range(bagslen):
            status = objectclass.GetStatus(bags[i])
            # missingcolor= (0, 240, 240) #yellow
            # suspiciouspersoncolor = (180,180,0) #ligth cyan
            owner = bags[i].personlinked

            if (owner != "none"):

                personlinked = objectclass.CheckPosition(people, owner)
                if personlinked!= None:

                    if (status == (0, 240, 240)) and (people[personlinked].updated == 1):

                        fake = bags[i].GetFake()

                        xcentroid = int(fake[0] + speed[0])
                        ycentroid = int(fake[1] + speed[1])
                        xmissing = round(xcentroid - bags[i].rwidth / 2)
                        wmissing = round(xcentroid + bags[i].rwidth / 2)
                        ymissing = round(ycentroid - bags[i].rheight / 2)
                        hmissing = round(ycentroid + bags[i].rheight / 2)
                        exists, newkalman = getfilter(bags[i].id, "bag", KalmanArray)
                        personposition = objectclass.CheckPosition(people, bags[i].personlinked)
                        print(" Kalman of bag ", bags[i].id, " with people", people[personposition].id, " speed.")
                        speed = people[personposition].speed

                        AT = framespassed - bags[i].prevlastframe
                        # KalmanUp =DrawKalman3(x, y, w, h, img, newkalman.kalmanfilter, speed, "bag", linkedstat,(bags[i].id),AT,framespassed)
                        print("Kalman of unupdated bag ", bags[i].id, ", exists is ", exists)
                        if kalmanbags == 1:
                            speed = bags[i].speed
                            label = "bag"

                            KalmanUp = kalmanupdate(xmissing, ymissing, wmissing, hmissing, img, newkalman.kalmanfilter,
                                                    speed,
                                                    label, linkedstat, bags[i].id, framespassed)

                            bags[i].GetFakeCentroid(int(KalmanUp[0]), int(KalmanUp[1]))
                            bags[i].speed = np.array([[KalmanUp[2]], [KalmanUp[3]]])

                        if predictionbags == 1:

                            if (bags[i].lastframe < (framespassed - 1)) and (bags[i].lastframe > (framespassed - 5)):
                                if x > people[personlinked].posx:
                                    bags[i].extradistance = round(bags[i].rwidth / 3)
                                else:
                                    bags[i].extradistancextra = - round(bags[i].rwidth / 3)

                            xcentroid = int(fake[0] + speed[0])
                            ycentroid = int(fake[1] + speed[1])
                            xmissing = round(xcentroid - bags[i].rwidth / 2)
                            wmissing = round(xcentroid + bags[i].rwidth / 2)
                            ymissing = round(ycentroid - bags[i].rheight / 2)
                            hmissing = round(ycentroid + bags[i].rheight / 2)

                            bags[i].GetFakeCentroid(xcentroid, ycentroid)
                            Drawmissing(xmissing, ymissing, wmissing, hmissing, img, (bags[i].id))

                            # KalmanUp = kalmanupdate(xmissing, ymissing, wmissing, hmissing,img,newkalman.kalmanfilter,speed,label,linkedstat,bags[i].id,framespassed)

    objectclass.restartupdatedflag(0, people)
    objectclass.restartupdatedflag(0, bags)

    end = time.time()

    dif = end - start
    fps = 1 / dif
    print("identification fps :", fps)

    return img, (lasremvalue, elimlist), (lasremvalue2, elimlist2)

def GetHistValue(image,x1,x2,y1,y2):

    x1 = abs(x1)
    x2 = abs(x2)
    y1 = abs(y1)
    y2 = abs(y2)

    crop_img = image[x1:x2,y1:y2]

    hist = np.zeros(256)

    #cv2.imshow("CROP",crop_img)
    cv2.waitKey(1)
    print("shape is ", image.shape," crop x", x2-x1, " crop y", y2-y1)
    if image.shape[0] > 1 :

        hist = cv2.calcHist(crop_img,[0],None,[256],[0,256])
        '''
        HistogramDiv = np.zeros(13)
        histlen = len(hist)
        print("hist len is ", histlen," total hist is ", hist)
        
        for i in range(histlen):
            grid = 0
            gridvalue = 0
            while hist[i] < gridvalue:
                grid = grid + 1
                gridvalue = gridvalue + 20
    
            HistogramDiv[grid] =+ 1
        
        for i in range(len(HistogramDiv)):
            HistogramDiv[i] = round((HistogramDiv[i]/histlen)*100)
        #cv2.imshow(str(TotalValue),(crop_img))
        cv2.waitKey(10)
        print("histogram is ")
        print(HistogramDiv)
        '''
    return hist

def Main2():

    # For fixing and testing purposes, the real main is Main()
    global totalfifties
    global frameprint
    global elimlist
    global elimlist2
    global maxratio, minratio, maxarea, minarea

    global PathLocation

    PathLocation = "/mypath/"


    maxratio, maxarea = 0, 0
    minratio, minarea = 10000, 10000000000000

    elimlist2 = []
    elimlist = []
    frameprint = 0
    totalfifties = 0

    framespassed = 0

    # Create a list for people and bags detected

    people = []
    bags = []

    objectclass.CreateBunchOfPeople(1, people)
    objectclass.CreateBunchOfBags(1, bags)

    # Create Kalman Filter Object
    KalmanArray = []

    Kalmanzero = createkalman(0,0)
    KalmanArray.append(associatedkalman(0, "zero", Kalmanzero))
    KalmanArray.append(associatedkalman(0, "zero", Kalmanzero))
    print(KalmanArray)

    # saving locations

    heatfolder = PathLocation + "/darknet-master/resultsheat"
    erasethis(heatfolder)

    # yolo 3

    lostobjfolder = "/mypath/darknet-master/results"
    erasethis(lostobjfolder)

    net = load_net(b"/mypath/darknet-master/cfg/yolov3.cfg",
                   b"/mypath/darknet-master/cfg/yolov3.weights", 0)
    meta = load_meta(b"/mypath/darknet-master/cfg/coco.data")
    classespath = ("/mypath/darknet-master/data/coco.names")

    # --- check if we have to get the image from a folder or video --- #

    Directory = "/mypath/Videos/extract/myvideo"
                #minnesota/2_train1"
    #Directory = "/mypath/Videos/extract/minnesota/2_train4"
    files = getfiles(Directory, 1)  # png files

    files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    framespassed = 0

    for frame in files:

        print("frame n ", framespassed)

        # ============== DETECTION ================= #
        r = detection(net, meta, frame, classespath)

        # ============ IDENTIFICATION ============== #
        #framespassed = IdentificationProcess(r, frame, people, bags, framespassed, KalmanArray,kalmanbags,kalmanpeople,predictionbags,showprediction))

        # ============== HEATMAP =========================== #

        img, framespassed = heatmapping(r, frame, framespassed, heatfolder)

def Main():

     global totalfifties
     global frameprint
     global elimlist
     global elimlist2
     global maxratio, minratio, maxarea, minarea
     global camera

     camera = 0

     parser = argparse.ArgumentParser()
     parser.add_argument("-m", "--mode", choices=["lostbag", "heatmap", "stampede", "stampedeT"],
                         help=" Select the activity to perform. ")
     parser.add_argument("-i", "--input", help=" input of the folder with the images")
     parser.add_argument("-kb", "--kalmanbags", type=int, default=0, help=" Activate kalman filters for bags")
     parser.add_argument("-kp", "--kalmanpeople",type=int, default=1, help=" Activate kalman filters for people")
     parser.add_argument("-pb", "--predictionbags",type=int, default=0, help=" Activate quick position prediction for bags")
     parser.add_argument("-sp", "--showprediction",type=int, default=0, help=" Activate prediction's bounding boxes and grid displaying on screen.")
     args = parser.parse_args()

     kalmanbags = args.kalmanbags
     kalmanpeople = args.kalmanpeople
     predictionbags = args.predictionbags
     showprediction = args.showprediction

     folder = "/mypath/darknet-master/results"
     erasethis(folder)
     folder2 = "/mypath/darknet-master/Anomaly/code/results"
     erasethis(folder2)
     folder3 = "/mypath/darknet-master/resultsheat"
     erasethis(folder3)

     VideoInput = 0
     inputtext = args.input
     Inputvideo = inputtext.find(".avi") + inputtext.find(".MP4")
     Inputcamera = inputtext.find("era")

     if Inputvideo > 0:
         VideoInput = 1

     if Inputcamera > 0:
        VideoInput = 5
        camera = 1

     print("input is",args.input)
     print("inputcamera is",Inputcamera)
     print("VideoInput is",VideoInput)

     if args.mode == "lostbag" or args.mode == "heatmap" :

         maxratio, maxarea = 0, 0
         minratio, minarea = 10000, 10000000000000

         elimlist2 = []
         elimlist = []
         frameprint = 0
         totalfifties = 0

         framespassed = 0

         # Create a list for people and bags detected

         people = []
         bags = []

         objectclass.CreateBunchOfPeople(1, people)
         objectclass.CreateBunchOfBags(1, bags)

         # Create Kalman Filter Object
         KalmanArray = []

         Kalmanzero = createkalman(0,0)
         KalmanArray.append(associatedkalman(0, "zero", Kalmanzero))
         KalmanArray.append(associatedkalman(0, "zero", Kalmanzero))
         print(KalmanArray)

         # saving locations

         heatmapsavegba = '/darknet-master/Heatmap'
         heatmapsaveavenue = '/darknet-master/HeatmapAv'

         # yolo 3

         net = load_net(b"/mypath/nnsurveillance/cfg/yolov3.cfg",
                        b"/mypath/nnsurveillance/cfg/yolov3.weights", 0)
         meta = load_meta(b"/mypath/nnsurveillance/cfg/coco.data")
         classespath = ("/mypath/nnsurveillance/data/coco.names")

         # --- check if we have to get the image from a folder or video --- #

         if (VideoInput > 1):

             if VideoInput ==1:
                frameNo = 0
                capture = cv2.VideoCapture(args.input)
                transmission, frame = capture.read()

             if VideoInput > 2:
                capture = cv2.VideoCapture(2)

             while (True):


                 transmission, frame = capture.read()
                 # cv2.imshow('frame', frame)
                 #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                 # ============== DETECTION ================= #
                 r = detection(net, meta, frame, classespath)

                 if args.mode == "lostbag":
                     # ============== DETECTION ================= #
                     #r = detection(net, meta, frame, classespath)
                     # ============== IDENTIFICATION ================= #

                     framespassed,img = IdentificationProcess(r, frame, people, bags, framespassed, KalmanArray,kalmanbags,kalmanpeople,predictionbags,showprediction)


                     cv2.imshow('img', img)
                     if cv2.waitKey(1) & 0xFF == ord('q'):
                         break

                 if args.mode == "heatmap":
                     # ============== HEATMAP ===========================
                    print("HEATMAP")
                     #img, framespassed = heatmapping(r, frame, framespassed, heatmapsaveavenue)



                 # When everything done, release the capture
             capture.release()
             cv2.destroyAllWindows()


         if VideoInput == 0:

             Directory = args.input
             files = getfiles(Directory, 1)  # png files

             files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

             framespassed = 0

             for frame in files:

                 # ============== DETECTION ================= #
                 r = detection(net, meta, frame, classespath)

                 if args.mode == "lostbag":
                     # ============== DETECTION ================= #
                     r = detection(net, meta, frame, classespath)
                     # ============== IDENTIFICATION ================= #
                     framespassed,img = IdentificationProcess(r, frame, people, bags, framespassed, KalmanArray,kalmanbags,kalmanpeople,predictionbags,showprediction)

                 if args.mode == "heatmap":
                     # ============== HEATMAP =========================== #

                     img, framespassed = heatmapping(r, frame, framespassed, heatmapsaveavenue)

     if args.mode == "stampede" or args.mode == "stampedeT":
         Stampade(args.input, args.mode, VideoInput,showprediction)

if __name__ == "__main__":

    #Main should be the main function used to use the application through the console

    Main()

    # Main2 is for local bug fixing purposes
    #Main2()
