from ctypes import *

import random
from typing import List, Any, Tuple

import numpy as np
import cv2

import os

import re

import time

import psutil

import objectclass

#import kalman

from PIL import Image

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


# ndarray_image = lib.ndarray_to_image
# ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
# ndarray_image.restype = IMAGE


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

'''
def DrawKalman(x,y,w,h, img, filter2):

        filter = kalman.KalmanFilter(4,2,0)



        predictedCoords = np.zeros((2, 1), np.float32)



        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        filter = kalman.KalmanFilter()

        predictedCoords = filter.Estimate(x, y)

        predx = round(predictedCoords.item(0))
        predy = round(predictedCoords.item(1))

        wo = w - x
        ho = h - y
        w2 = round(wo + predx)
        h2 = round(ho + predy)


        print(" KF es x :", predx, ". y :", predy, ". w:", w2, ". h: ", h2, ".")

        # Draw Kalman Filter Predicted output
        # cv2.rectangle(img, (x, y), (w, h), color, 2)
        cv2.rectangle(img, (predx, predy), (w2, h2), (0, 255, 255), 2)
        # cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img, "prediction", (predx - 10, predy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 250), 2)

        return img, kalman
'''
from kalman3 import KalmanFilter



def DrawKalman3(x,y,w,h,img,kalman,speed,label,linked):

    x = np.float32(x)
    y = np.float32(y)
    w = np.float32(w)
    h = np.float32(h)


    cox=np.float32(x + (w-x)/2)

    coy = np.float32(y + (h-y)/2)



    def correction(xpos,ypos,cox,coy,speed,label,linked):

        current_measurement = np.array([[np.float32(cox)], [np.float32(coy)]])
        current_prediction = kalman.predict()
        velocity = np.array([[np.float32(speed[0])], [np.float32(speed[1])]])

        cmx, cmy = current_measurement[0], current_measurement[1]

        cpx, cpy = current_prediction[0], current_prediction[1]

        vmx, vmy = speed[0], speed[1]

        cpx = int(cpx)
        cpy = int(cpy)

        x2 = int(cpx - (cmx-xpos))
        y2 = int(cpy - (cmy-ypos))
        w2 = (cpx + (cmx-xpos))
        h2 = (cpy + (cmy-ypos))


        w2 = int(w2)
        h2 = int(h2)

        print(" x is ", xpos, "cox is ", cox)
        print(" y is ", ypos, "coy is ", coy)
        print("w2 is ", w2)
        print("h2 is ", h2)


        #frame = np.ones((800,800,3),np.uint8) * 255
        #cv2.putText(frame, "Measurement: ({:.1f}, {:.1f})".format(np.float(cmx), np.float(cmy)),
                    #(30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (50, 150, 0))
        cv2.putText(img, "prediction".format(np.float(cpx), np.float(cpy)),
                    (30, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255))

        #cv2.rectangle(img, (x2, y2), (w2, h2), (0, 0, 0), 2)
        # cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        #cv2.putText(img, "prediction", (x2 - 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 250), 2)
        update = np.array([cmx, cmy,vmx,vmy])


        kalman.correct(update)

        return current_prediction


    prediction = correction(x,y,cox,coy,speed,label,linked)

    return prediction

def createkalman():

        stateMatrix = np.zeros((4, 1), np.float32)  # [x, y, delta_x, delta_y]
        estimateCovariance = np.eye(stateMatrix.shape[0])
        transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.001
        measurementStateMatrix = np.zeros((4, 1), np.float32)
        observationMatrix = np.array([[1,0,0,0],[0,1,0,0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        #measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1
        measurementNoiseCov = np.eye(observationMatrix.shape[0])
        kalmancreated = KalmanFilter(X=stateMatrix,
                                     P=estimateCovariance,
                                     F=transitionMatrix,
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




#this is the good one
def detection(net, meta, image, classespath,thresh=.5, hier_thresh=.5, nms=.45):
    # IF IT IS A VIDEO
    global frameprint
    global elimlist

    start = time.time()


    isanimage = image.find("png") + image.find("jpg")
    isavideo = image.find("avi")

    # read class names from text file
    classes = None
    with open(classespath, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    if (isanimage > 0):

        print("image detected")

        newim = image.encode("UTF-8")
        # print(newim)
        im = load_image(newim, 0, 0)
        # print(im.w)
        # print(im.h)
        num = c_int(0)
        pnum = pointer(num)
        # slow process
        predict_image(net, im)

        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]

        img = cv2.imread(image)

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
        free_image(im)
        free_detections(dets, num)

        end = time.time()

        dif = end-start
        fps = 1/dif
        print("fps :",fps)

        return res

def heatmapping(res,image,framespassed,savedic):

    global totalfifties
    fiftiesrep = 0
    fiftieson = 0
    cycle = 0
    maxheatmapframes = 50
    # variasmochilas014
    path = '/mypath/Videos/extract/linked'

    reslong = len(res)

    #Determine heatmapframe


    heatmapframes = framespassed

    if framespassed > 49:
        fiftieson =1
        cycle = 1
    '''

    if heatmapframes >= maxheatmapframes and cycle == 0:
        cycle = 1

    iteration = 0
    if cycle == 1:
        while ( 300*iteration <= heatmapframes):
            iteration = iteration + 1

    heatmapframes = heatmapframes - (300*iteration)
    '''
    iteration = 0

    if cycle == 1:

            while ( (49*iteration) < heatmapframes):
                iteration = iteration + 1
                print ("iteration",iteration)

    if cycle == 1:

        iteration = iteration -1

    heatmapframes = heatmapframes - (49*iteration)

    print(heatmapframes)

    # read the image

    img = cv2.imread(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    accum_image_ellipse = np.zeros((height, width), np.uint8)
    accum_image = np.zeros((height, width), np.uint8)
    accum_image2 = np.zeros((height, width), np.uint8)

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
    #et2, th2 = cv2.threshold(accum_image, 40, 80, cv2.THRESH_BINARY)
    saveimage(th1, "test3", savedic,0)
    #saveimage(th2, "BIG", '/mypath/nnsurveillance/Heatmap', 0)

    # The image is saved in the temp folder

    heatrange = heatmapframes

    #files2 = []
    if (cycle == 0):
        #print(th1)
        # OPTION A, SAVING PICS IN FILES

        saveimage(th1,heatmapframes,'/mypath/nnsurveillance/Heatmap/temp',0)


    elif(cycle ==1):

        # OPTION A, SAVING PICS IN FILES

        saveimage(th1,heatmapframes,'/mypath/nnsurveillance/Heatmap/temp',0)

        # OPTION B, SAVING PICS IN ARRAY

        #thsaved[heatmapframes] = th1.copy()

        #heatrange = maxheatmapframes

    Directory2 = "/mypath/nnsurveillance/Heatmap/temp"
    files2 = getfiles(Directory2,0)  # jpg files

    files2.sort(key=lambda var: [x if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    x = 0

    if (fiftieson == 1):

        Directory3 = "/mypath/nnsurveillance/Heatmap/temp/fifties"
        files3 = getfiles(Directory3, 0)  # jpg files
        '''
        files3.sort(key=lambda var: [x if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        #fif = str(totalfifties) + ".jpg"

        for fif in files3:
            print("total fifties in",str(fif),fif)
            fifties = cv2.imread(fif)
            newfiftiesgray = cv2.cvtColor(fifties, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            #saveimage(newfiftiesgray, "befacu", '/mypath/nnsurveillance/Heatmap', 0)
            accum_image = cv2.add(accum_image, newfiftiesgray)
        '''

        print("files in fifties",stri)
        for it in range(iteration):

            if it == 0:
                start =1
            if iteration > 6:
                start = (iteration -6) + it
            else:
                start = it+1
            fif = '/mypath/nnsurveillance/Heatmap/temp/fifties/' + str(start) + '.jpg'
            print("adding ",fif)
            fif.encode('utf-8')

            fifties = cv2.imread(fif)
            cv2.waitKey(1)
            newfiftiesgray = cv2.cvtColor(fifties, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            # saveimage(newfiftiesgray, "befacu", '/mypath/nnsurveillance/Heatmap', 0)
            accum_image = cv2.add(accum_image, newfiftiesgray)

            if start == iteration:
                break

    # we get the acumm image of the elipses
    for stri2 in files2:



        newth = cv2.imread(stri2)
        newthgray = cv2.cvtColor(newth, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        saveimage(newthgray, "befacu", '/mypath/nnsurveillance/Heatmap', 0)

        accum_image2 = cv2.add(accum_image2, newthgray)



        if x < heatrange:
                    x = x+1
        else:
            break



    if heatmapframes == 49:

        #if (totalfifties) == 6:
        #    totalfifties = 0
        totalfifties = totalfifties + 1
        print(" totalfifties to save ", totalfifties)
        saveimage(accum_image2, str(totalfifties), "/mypath/nnsurveillance/Heatmap/temp/fifties", 0)

    accum_image = cv2.add(accum_image2, accum_image)

    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    cv2.imwrite('diff-color.jpg', color_image)
            # for testing purposes, show the accumulated image
    saveimage(accum_image,"acc","/mypath/nnsurveillance/Heatmap",0)

    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
        # for testing purposes, show the colorMap image
    cv2.imwrite('diff-color.jpg', color_image)



        # overlay the color mapped image to the first frame
    result_overlay = cv2.addWeighted(img, 0.7, color_image, 0.8, 0)
    #cv2.imshow("overlay", result_overlay)
    #cv2.waitKey(1)

    saveimage(result_overlay,framespassed,savedic,0)



    return result_overlay, framespassed

def aspectratio(res,image):

    global maxratio,minratio,maxarea,minarea
    global max
    cycle = 0
    path = '/home/sergio.lopez/Videos/extract/linked'

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
        mean.append(aspect)
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

def identification(res, image, people, bags, framespassed,KalmanArray):

    global frameprint
    global elimlist
    lasremvalue = 0
    lasremvalue2 = 0

    reslong = len(res)
    print(reslong)
    img = cv2.imread(image)
    img2 = cv2.imread(image)
    speed = []

    for b in range(reslong):

        label, prob, x, y, w, h = desres(res, b)
        print("label",label," prob",prob," x",x," y",y,"w ",w," h",h)
        id = 0
        linkedstat = 0

        if label == "person":

            histogramvalue = []
            histogramvalue= GetHistValue(img2,y,h,x,w)
            print('')
            print(" ## determine new or update person ##")
            print('')
            id = objectclass.person.DetermineNewOrUpdate(x, y, w, h, people, framespassed, img,histogramvalue)
            printremove, personremove, eliminate = objectclass.person.LastUpdate(framespassed, people)
            lasremvalue = lasremvalue or personremove
            objectposition = objectclass.CheckPosition(people, id)

            speed = people[objectposition].speed
            print("person ", id, " speed is :", speed)
            itemstatus = people[objectposition]


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


            histogramvalue= GetHistValue(img2,y,h,x,w)

            label = "bag"
            # or label =="handbag"
            print()
            print(" ## determine new or update bag ##")
            print()
            id = objectclass.backpack.DetermineNewOrUpdate(x, y, w, h, bags, framespassed, people, img,histogramvalue)
            printremove2, bagsremove, eliminate2 = objectclass.backpack.LastUpdate(framespassed, bags)
            lasremvalue2 = lasremvalue2 or bagsremove
            objectposition = objectclass.CheckPosition(bags,id)
            linkedstat = bags[objectposition].linked

            speed = bags[objectposition].speed
            print("bag", bags[objectposition].id, " speed is ", speed)

            if bags[objectposition].linked == 1:
                personposition = objectclass.CheckPosition(people, bags[objectposition].personlinked)
                if personposition != None:
                    speed = people[personposition].speed
                    print("bag ",bags[objectposition].id," speed is ",bags[objectposition].speed,", linked to person ", people[personposition].id, " ,speed is :", speed)





            itemstatus = bags[objectposition]
            if bagsremove == 1:
                print(printremove2)
                print("\n")
                cv2.putText(img, "eliminando backpack", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)
                # print("eliminando people ", bags[eliminate].id)
                elimlist2.append(eliminate2)
                del bags[eliminate2]



        # checking the linked backpacks
        # print("changed2 is ", objectclass.changed2,", j is",j," and i is",i)

        if label == "person" or label == "bag":
            status = objectclass.GetStatus(itemstatus)
            cv2.rectangle(img, (x, y), (w, h), (status), 2)
            cv2.rectangle(img, (x - 1, y - 20), (x + 80, y), (status), -1)
            # + " " + str(id) IF WE WANT TO ADD ID
            cv2.putText(img, (label+ str(id) ), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            '''+ str(id)'''

            #check the kalman prediction
            #print("getting filter")
            exists,newkalman = getfilter(id, label, KalmanArray)
            if exists == 0:
                #print("creating new Kalman")
                newkalmanfilter = createkalman()
                newkalman = associatedkalman(id,label,newkalmanfilter)
                KalmanArray.append(newkalman)

                #print("existing Kalman")
            # Dawing kalman prediction

            if label == "bag":
                DrawKalman3(x,y,w,h,img,newkalman.kalmanfilter,speed,label,linkedstat)
                #print(" x state is ", newkalman.kalmanfilter.getprediction())

        # histogramvalue = GetHistValue(img2, y, h, x, w)
    bagslen = len(bags)

    print('')
    print(" ## linking backpack ##")
    print('')
    objectclass.LinkedBackpack(people, bags, KalmanArray)
    # was before inside de loop (linkedbackback)
    if (objectclass.changed2 == 1):
        for i in range(bagslen):
            status = objectclass.GetStatus(bags[i])
            # missingcolor= (0, 240, 240) #yellow
            # suspiciouspersoncolor = (180,180,0) #ligth cyan
            owner = bags[i].personlinked

            if (owner != "none"):

                personlinked = objectclass.CheckPosition(people, owner)

                if (status == (0, 240, 240)) and (people[personlinked].updated == 1):

                    fake = bags[i].GetFake()
                    x = round(fake[0] - bags[i].rwidth / 2)
                    y = round(fake[1] - bags[i].rheight / 2)
                    w = round(fake[0] + bags[i].rwidth / 2)
                    h = round(fake[1] + bags[i].rheight / 2)
                    #cv2.rectangle(img, (x, y), (w, h), (status), 2)
                    #cv2.rectangle(img, (x - 1, y - 20), (x + 80, y), (status), -1)
                    # + " " + str(bags[i].id) IF WE WANT TO SHOW ID
                    #cv2.putText(img, ("bag"+ str(bags[i].id)+"prediction"), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    exists, newkalman = getfilter(bags[i].id, "bag", KalmanArray)
                    personposition = objectclass.CheckPosition(people, bags[i].personlinked)
                    speed = people[personposition].speed
                    #DrawKalman3(x, y, w, h, img, newkalman.kalmanfilter, speed, "bag", linkedstat)

    objectclass.restartupdatedflag(0, people)
    objectclass.restartupdatedflag(0, bags)

    return img, (lasremvalue, elimlist), (lasremvalue2, elimlist2)

'''

def detectndraw(net, meta, image, classespath, COLORS, people, bags, framespassed, thresh=.5, hier_thresh=.5, nms=.45):
    # IF IT IS A VIDEO
    global frameprint
    global elimlist
    fake = []
    status = (0, 0, 0)

    isanimage = image.find("png") + image.find("jpg")
    isavideo = image.find("avi")

    # read class names from text file
    classes = None
    with open(classespath, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    if (isanimage > 0):

        print("image detected")

        newim = image.encode("UTF-8")
        # print(newim)
        im = load_image(newim, 0, 0)
        # print(im.w)
        # print(im.h)
        num = c_int(0)
        pnum = pointer(num)
        # slow process
        predict_image(net, im)

        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]

        # generate different colors for different classes 
        # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        img = cv2.imread(image)

        # generate different colors for different classes
        # color = np.random.uniform(0, 255, 150)

        if (nms): do_nms_obj(dets, num, meta.classes, nms);

        res = []
        # SOME VARIABLES CREATED
        lasremvalue = 0
        lasremvalue2 = 0

        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    label = str(classes[i])
                    color = COLORS[i]
                    b = dets[j].bbox
                    x = round(b.x - b.w / 2)
                    y = round(b.y - b.h / 2)
                    w = round(x + b.w)
                    h = round(y + b.h)
                    id = 0
                    printremove = 0

                    if (label == "person"):

                        print(" entering in if condition of ",label)

                        id = objectclass.person.DetermineNewOrUpdate(x, y, w, h, people, framespassed, img)
                        printremove, personremove, eliminate = objectclass.person.LastUpdate(framespassed, people)
                        lasremvalue = lasremvalue or personremove

                        if personremove == 1:
                            frameprint = 5
                            print(printremove)
                            print("\n")
                            cv2.putText(img, "eliminando people", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 2)
                            # print("eliminando people ", people[eliminate].id)

                            # Introducing deleted people in the array

                            elimlist.append(eliminate)
                            del people[eliminate]

                        position = objectclass.CheckPosition(people, id)
                        itemstatus = people[position]

                    if (label == "backpack" or label == "handbag" or label == "suitcase"):

                        label = "bag"
                        print(" entering in if condition of ",label)
                        # or label =="handbag"
                        id = objectclass.backpack.DetermineNewOrUpdate(x, y, w, h, bags, framespassed, img)
                        printremove2, bagsremove, eliminate2 = objectclass.backpack.LastUpdate(framespassed, bags)
                        lasremvalue2 = lasremvalue2 or bagsremove

                        if bagsremove == 1:
                            print(printremove2)
                            print("\n")
                            cv2.putText(img, "eliminando backpack", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 2)
                            # print("eliminando people ", bags[eliminate].id)
                            elimlist2.append(eliminate2)
                            del bags[eliminate2]

                        position = objectclass.CheckPosition(bags, id)
                        itemstatus = bags[position]
                    # checking the linked backpacks
                    # print("changed2 is ", objectclass.changed2,", j is",j," and i is",i)

                    if (label == "person" or label == "bag"):
                        print(" entering in status if condition ")
                        res.append((str(meta.names[i]) + str(id), dets[j].prob[i], (x, y, w, h)))

                        status = objectclass.GetStatus(itemstatus)
                        print("status is ",status)
                        if status != (0, 240, 240):
                            cv2.rectangle(img, (x, y), (w, h), (status), 2)
                            cv2.rectangle(img, (x - 1, y - 20), (x + 80, y), (status), -1)
                        # + " " + str(id) IF WE WANT TO ADD ID
                            cv2.putText(img,str(id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        if status == (0, 0, 255):
                            cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

    cv2.waitKey(1000)
    bagslen = len(bags)
    objectclass.LinkedBackpack(people, bags)
    # was before inside de loop (linkedbackback)
    if (objectclass.changed2 == 1):
        for i in range(bagslen):
            status = objectclass.GetStatus(bags[i])
            if status == (0, 240, 240):
                fake = bags[i].GetFake()
                x = round(fake[0] )
                y = round(fake[1] )
                w = round(fake[0] + bags[i].rwidth)
                h = round(fake[1] + bags[i].rheight)
                cv2.rectangle(img, (x, y), (w, h), (status), 2)
                cv2.rectangle(img, (x - 1, y - 20), (x + 80, y), (status), -1)
                # + " " + str(bags[i].id) IF WE WANT TO SHOW ID
                cv2.putText(img, ("bag prediction"), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    objectclass.restartupdatedflag(0, people)
    objectclass.restartupdatedflag(0, bags)
    # for b in range(len(bags)):
    # print("backpack ",bags[b].id," updated status is ",bags[b].updated," after restart.")
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)

    return res, img, (lasremvalue, elimlist), (lasremvalue2, elimlist2)
'''

"""
def draw_bounding_box(img,res,meta):
    

        res = np.array(res)
        list = []
        list = [line.split(',') for line in res]
        #list = [x.strip() for x in res.split(',')]
          # generate different colors for different classes 
        color = np.random.uniform(0, 255, 3)
        
        j=0
        i=0
        
        while i < len(list):
        
            print("bucle ",i)
            name=list[i]
            print(name)
            prob=list[i+1]
            print(prob)
            x=list[i+2]
            print(x)
            y=list[i+3]
            print(y)
            w=list[i+4]
            print(w)
            h=list[i+4]
            print(h)
            
            if prob> 0.5:
        
                cv2.rectangle(img,(round(x),round(y)),(round(x+w),round(y+h)),color[j],2)
                    
                cv2.putText(img,name,(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[j], 2)
                
                i=i+5
                j+=1
            else:
                
                i=i+5
                j+=1
                    
"""
"""
def detect_np(net, meta, np_img, thresh=.5, hier_thresh=.5, nms=.45):

    im = ndarray_to_image(np_img)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
"""


def GetHistValue(image,x1,x2,y1,y2):

    x1 = abs(x1)
    x2 = abs(x2)
    y1 = abs(y1)
    y2 = abs(y2)

    crop_img = image[x1:x2,y1:y2]



    #cv2.imshow("CROP",crop_img)
    cv2.waitKey(1)
    print("shape is ", image.shape," crop x", x2-x1, " crop y", y2-y1)
    if image.shape[0]<1:
        hist = np.zeros(256)
    else:
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



if __name__ == "__main__":

    global totalfifties
    global frameprint
    global elimlist
    global elimlist2
    global maxratio,minratio,maxarea,minarea

    maxratio,maxarea = 0,0
    minratio,minarea = 10000,10000000000000
    min
    elimlist2 = []
    elimlist = []
    frameprint = 0
    totalfifties = 0

    # net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    # im = load_image("data/wolf.jpg", 0, 0)
    # meta = load_meta("cfg/imagenet1k.data")
    # r = classify(net, meta, im)
    # print r[:10]
    # imagepath = ( "/mypath/nnsurveillance/data/dog.jpg")
    heatmapsavegba = '/mypath/nnsurveillance/Heatmap'
    heatmapsaveavenue = '/mypath/nnsurveillance/HeatmapAv'

    # yolo 3

    net = load_net(b"/mypath/nnsurveillance/cfg/yolov3.cfg",
                   b"/mypath/nnsurveillance/cfg/yolov3.weights", 0)
    meta = load_meta(b"/mypath/nnsurveillance/cfg/coco.data")
    classespath = ("/mypath/nnsurveillance/data/coco.names")
    '''
    net = load_net(b"/mypath/nnsurveillance/cfg/yolov2.cfg",
                   b"/mypath/nnsurveillance/cfg/yolov2.weights", 0)
    meta = load_meta(b"/mypath/nnsurveillance/cfg/coco.data")
    classespath = ("/mypath/nnsurveillance/data/coco.names")
    '''
    #tiny yolo

    # Create a list for people and bags detected
    people = []
    bags = []
    KalmanArray = []

    objectclass.CreateBunchOfPeople(1, people)
    objectclass.CreateBunchOfBags(1, bags)

    Kalmanzero = createkalman()
    KalmanArray.append(associatedkalman(0,"zero",Kalmanzero))
    KalmanArray.append(associatedkalman(0, "zero", Kalmanzero))
    print(KalmanArray)

    # Create Kalman Filter Object
    #kfObj = kalman.KalmanFilter()


    #erasing temp

    folder = '/mypath/nnsurveillance/Heatmap/temp'
    folder2 = '/mypath/nnsurveillance/Heatmap/temp/fifties'
    folder3 = '/mypath/nnsurveillance/results'

    erasethis(folder)
    erasethis(folder2)
    erasethis(folder3)

    # read class names from text file
    classes = None
    with open(classespath, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # generate different colors for different classes
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    files = []

    # variasmochilas014
    # train2
    #linkedlastpics
    #waitgray
    #E1gather


    Directory = "/home/sergio.lopez/Videos/extract/0421-abnormal"
    #gold = "/home/sergio.lopez/Videos/extract/007-12-ABNORMAL-normal"
    #gold = "/home/sergio.lopez/Videos/extract/007-12-shorter"
    files = getfiles(Directory,1) #png files

    files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    framespassed = 0

    for stri in files:

        print("frame n ", framespassed)
        # im = cv2.imread(stri)
        # cv2.imshow("imagen",im)
        # newstr = stri.encode("UTF-8")
        # r =detect(net,meta,newstr)
        #r = detect(net,meta,stri,classespath)
        #print(r)
        r = detection(net, meta, stri, classespath)
        #print(r)
        print("done")

        #aspectratio(r,stri)
        # ============== IDENTIFICATION =================
        #'''
        img,elimpeople,elimbags = identification(r,stri,people,bags,framespassed,KalmanArray)
        #r, img, elimpeople, elimbags = detectndraw(net, meta, stri, classespath, COLORS, people, bags, framespassed)
        # Dawing kalman prediction
        #img,kfObj = DrawKalman(r,img,kfObj)
        lasremvalue = elimpeople[0]
        # print("lasremvalue es ",lasremvalue)
        elimlist = elimpeople[1]
        # print("elimlist value es ", elimlist)
        img = printeliminate(img, lasremvalue, elimlist)
        saveimage(img, framespassed, '/mypath/nnsurveillance/results', 1)

        #'''
        #============== HEATMAP ===========================
        '''

        img,framespassed = heatmapping(r,stri,framespassed,heatmapsaveavenue)

        #img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # print(img)
        # ya estoy haciendo save image dentro de heatmapping
        #saveimage(img,framespassed,'/mypath/nnsurveillance/resultsheat',1)
        #'''

        #imagenn = Image.fromarray(img)
        # imagenn , img = array_to_imag#!/usr/bin/python3e(img)
        # IMAGE IS DISPLAYED
        # imagenn.show()

        framespassed += 1




        '''
        imagenn.show()
        time.sleep(3)

        for proc in psutil.process_iter():
            if proc.name() == "display":
                proc.kill()
        # imagenn.transpose(2,0,1)
        # cv2.imread(imagenn)
        # cv2.imshow("image", img)
        # cv2.waitKey()

        # imagen = load_image(newstr, 0, 0)
        # print(imagen)

    # myimage,myarray= array_to_image("/mypath/nnsurveillance/data/dog.jpg")

   
    imagen = load_image(b"/mypath/nnsurveillance/data/dog.jpg",0,0)
    print("imagen")
    print(imagen)

    img = cv2.imread("/mypath/nnsurveillance/data/dog.jpg")
    #r = detect(net, meta, b"/mypath/nnsurveillance/data/dog.jpg")
    # r = detect_np(net, meta, img)
    print("img")
    print(img)

    im, image = array_to_image(img)
    rgbgr_image(im)
    #cvimg= cv2.imread('/mypath/nnsurveillance/data/dog.jpg')
    r = detect(net, meta,im)
    print("im")
    print(im)
    #img = load_image(b'/mypath/nnsurveillance/data/dog.jpg', 0, 0)

    #print(r)

    # cv2.waitKey()
    # cv2.destroyAllWindows()
    '''

