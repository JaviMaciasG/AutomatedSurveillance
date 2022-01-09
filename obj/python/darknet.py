from ctypes import *


import random
import numpy as np
import cv2

import os

import re

import time

import psutil

import objectclass

import kalman

from PIL import Image

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

def saveimage(img,framepassed,path2,format):
    if format == 0 or None :
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
    res = sorted(res, key=lambda x: -x[1])
    return res


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):

    im = load_image(image, 0, 0)
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

def printeliminate (img,personremove,elimlist):

    global frameprint
    global maintaintext

    #print("frameprint is ", frameprint)

    elimlistrange = len(elimlist)

    #print("elimlistrange is ", elimlistrange)

    if personremove == 1:
        for x in range(elimlistrange):
            frameprint = 5
            maintaintext = elimlist[x]

    if frameprint > 0:
        for x in range(elimlistrange):
            cv2.putText(img, "borrando persona " + str(elimlist[x]), (10 + (x*10), 1000), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0), 2)
            #print("printing remove on picture : borrando persona ", elimlist[x])

        frameprint = frameprint - 1


    if frameprint == 0 and elimlistrange > 0 :
        for x in range(elimlistrange):
            del elimlist[0]

    return img

def desres(res,num):

    commas = []
    news = str(res[num])
    newslen = len(news)
    foundat = 0
    #print(news)
    while foundat < (newslen - 8):
        foundat = news.find(",", foundat + 1)
        #print(foundat)
        commas.append(foundat)
    #print(commas)

    upper1 = news.find("'")
    upper2 = news.find("'", upper1+1)
    firstp = news.find("(", 5)
    lastp = news.find(")")

    name = news[upper1 + 1:commas[0]-1]
    probs = news[commas[0] + 2:commas[1]]
    x = news[firstp + 1:commas[2]]
    y = news[commas[2] + 2:commas[3]]
    w = news[commas[3] + 2:commas[4]]
    h = news[commas[4] + 2:lastp]

    return name, probs, x, y, w, h

def DrawKalman(res,img,kalman):

    reslong = len(res)

    for b in range(reslong):
        name,probs,x,y,w,h = desres(res,b)
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)



        predictedCoords = kalman.Estimate(x, y)

        predx = round(predictedCoords.item(0))
        predy = round(predictedCoords.item(1))

        wo = w-x
        ho = h-y
        w2 = round(wo + predx)
        h2 = round(ho + predy)

        print(name," es x: ",x,". y: ",y, ". w:",w,". h: ",h,".")
        print(" KF es x :",predx,". y :",predy,". w:",w2,". h: ",h2,".")

        # Draw Kalman Filter Predicted output
        #cv2.rectangle(img, (x, y), (w, h), color, 2)
        cv2.rectangle(img, (predx, predy), (w2, h2), (0, 255, 255), 2)
        #cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img, name, (predx-10, predy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(50, 200, 250),2)

    return img,kalman



def detectndraw(net, meta, image, classespath,COLORS, people,bags,framespassed,thresh=.5, hier_thresh=.5, nms=.45):
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
        #print(newim)
        im = load_image(newim, 0, 0)
        #print(im.w)
        #print(im.h)
        num = c_int(0)
        pnum = pointer(num)
        # slow process
        predict_image(net, im)

        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]

        # generate different colors for different classes 
        #COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        img = cv2.imread(image)

        # generate different colors for different classes
        #color = np.random.uniform(0, 255, 150)

        if (nms): do_nms_obj(dets, num, meta.classes, nms);

        res = []
        #SOME VARIABLES CREATED
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

                        id=objectclass.person.DetermineNewOrUpdate(x,y,w,h,people,framespassed,img)
                        printremove,personremove,eliminate= objectclass.person.LastUpdate(framespassed,people)
                        lasremvalue = lasremvalue or personremove
                        print("x", x, "y", y, "w", w, "h", h)
                        cv2.waitKey(3000)



                        if personremove == 1:
                            frameprint = 5
                            print(printremove)
                            print("\n")
                            cv2.putText(img, "eliminando people", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
                            #print("eliminando people ", people[eliminate].id)

                            #Introducing deleted people in the array

                            elimlist.append(eliminate)
                            del people[eliminate]


                        position = objectclass.CheckPosition(people, id)
                        itemstatus = people[position]

                    if (label == "backpack" or label == "handbag" or label == "suitcase"):
                        label = "bag"
                    # or label =="handbag"
                        id=objectclass.backpack.DetermineNewOrUpdate(x, y, w, h, bags, framespassed, img)
                        printremove2,bagsremove,eliminate2= objectclass.backpack.LastUpdate(framespassed, bags)
                        lasremvalue2 = lasremvalue2 or bagsremove

                        if bagsremove == 1:
                            print(printremove2)
                            print("\n")
                            cv2.putText(img, "eliminando backpack", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
                            #print("eliminando people ", bags[eliminate].id)
                            elimlist2.append(eliminate2)
                            del bags[eliminate2]

                        position = objectclass.CheckPosition(bags, id)
                        itemstatus = bags[position]
                    #checking the linked backpacks
                    #print("changed2 is ", objectclass.changed2,", j is",j," and i is",i)


                    if (label == "person" or label == "bag"):
                        res.append((str(meta.names[i]) + str(id), dets[j].prob[i], (x, y, w, h)))


                        status = objectclass.GetStatus(itemstatus)
                        cv2.rectangle(img, (x, y), (w, h), (status), 2)
                        cv2.rectangle(img, (x - 1, y - 20), (x + 80, y), (status), -1)
                        # + " " + str(id) IF WE WANT TO ADD ID
                        cv2.putText(img,(label ), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    bagslen = len(bags)
    objectclass.LinkedBackpack(people, bags)
    #was before inside de loop (linkedbackback)
    if (objectclass.changed2 == 1):
        for i in range(bagslen):
            status = objectclass.GetStatus(bags[i])
            if status == (0, 240, 240):
                fake = bags[i].GetFake()
                x = round(fake[0]-bags[i].rwidth/2)
                y = round(fake[1]-bags[i].rheight/2)
                w = round(fake[0]+bags[i].rwidth/2)
                h = round(fake[1]+bags[i].rheight/2)
                cv2.rectangle(img, (x, y), (w, h), (status), 2)
                cv2.rectangle(img, (x - 1, y - 20), (x + 80, y), (status), -1)
                #+ " " + str(bags[i].id) IF WE WANT TO SHOW ID
                cv2.putText(img, ("bag"), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    objectclass.restartupdatedflag(0, people)
    objectclass.restartupdatedflag(0, bags)
    #for b in range(len(bags)):
        #print("backpack ",bags[b].id," updated status is ",bags[b].updated," after restart.")
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)

    return res, img, (lasremvalue, elimlist),(lasremvalue2,elimlist2)


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

if __name__ == "__main__":

    global frameprint
    global elimlist
    global elimlist2

    elimlist2 = []
    elimlist = []
    frameprint = 0

    net = load_net(b"/mypath/darknet-master/cfg/yolov3.cfg",b"/mypath/darknet-master/cfg/yolov3.weights", 0)
    meta = load_meta(b"/mypath/darknet-master/cfg/coco.data")
    classespath = ("/mypath/darknet-master/data/coco.names")

    # Create a list for people and bags detected
    people = []
    bags = []

    objectclass.CreateBunchOfPeople(1,people)
    objectclass.CreateBunchOfBags(1, bags)

    # Create Kalman Filter Object
    kfObj = kalman.KalmanFilter()
    predictedCoords = np.zeros((2, 1), np.float32)




    # read class names from text file
    classes = None
    with open(classespath, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # generate different colors for different classes
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    files = []

    #videodirectoy
    Directory = "/mypath/Videos/extract/linkedlastpics"
    for filename in os.listdir(Directory):  # type: str
        if filename.endswith(".png"):
            files.append(Directory+"/"+filename)
            #print(os.path.join(Directory, filename))
            continue
        else:
            continue

    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])


    framespassed = 0

    for stri in files:
        print("frame n ", framespassed)
        #im = cv2.imread(stri)
        #cv2.imshow("imagen",im)
        #newstr = stri.encode("UTF-8")
        #r =detect(net,meta,newstr)
        r, img, elimpeople,elimbags = detectndraw(net,meta,stri,classespath,COLORS,people,bags,framespassed)
        print(r)
        #Dawing kalman prediction
        #img,kfObj = DrawKalman(r,img,kfObj)
        lasremvalue = elimpeople[0]
        #print("lasremvalue es ",lasremvalue)
        elimlist = elimpeople[1]
        #print("elimlist value es ", elimlist)
        img = printeliminate(img, lasremvalue, elimlist)

        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #print(img)

        saveimage(img, framespassed,'/mypath/nnsurveillance/results',1)

        imagenn = Image.fromarray(img2)
        #imagenn , img = array_to_imag#!/usr/bin/python3e(img)
        #IMAGE IS DISPLAYED
        #imagenn.show()

        framespassed += 1
        #imagenn.transpose(2,0,1)
        '''imagenn.show()
        time.sleep(0.7)


        for proc in psutil.process_iter():
            if proc.name() == "display":
                proc.kill()
        '''
        #cv2.imread(imagenn)
        #cv2.imshow("image", img)
        #cv2.waitKey()

        #imagen = load_image(newstr, 0, 0)
        #print(imagen)














    # myimage,myarray= array_to_image("/mypath/nnsurveillance/data/dog.jpg")

    '''
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
