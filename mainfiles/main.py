
import numpy as np
import cv2
import copy
import darknet
import darknetandheat

############
from ctypes import *


import os

import re

import time

import psutil


from PIL import Image


################

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


def main():

    files = []
    cycle = 0
    heatmapframes = 0
    maxheatmapframes = 300
    thsaved = []
#bag video for test
    path = '/home/sergio.lopez/Videos/extract/linked'

    files = getfiles(path,1)
    files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    first_iteration_indicator = 1
    framespassed = 0

    net = darknet.load_net(b"/mypath/nnsurveillance/cfg/yolov3.cfg",b"/mypath/nnsurveillance/cfg/yolov3.weights", 0)
    meta = darknet.load_meta(b"/mypath/nnsurveillance/cfg/coco.data")
    classespath = ("/mypath/nnsurveillance/data/coco.names")

    for stri in files:

        r = darknetandheat.detection(net,meta,stri,classespath)

        reslong = len(r)

        for b in range(reslong):

            label, prob, x, y, w, h = darknet.desres(r, b)

        if (first_iteration_indicator == 1):
            frame = cv2.imread(stri)
            first_frame = copy.deepcopy(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            #accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
            for b in range(maxheatmapframes):
                thsaved.append(np.zeros((height, width), np.uint8))
        else:
            #print(thsaved, "len ", len(thsaved))

            frame = cv2.imread(stri)  # read a frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale


            fgmask = fgbg.apply(gray)  # remove the background
            darknet.saveimage(fgmask, "test", '/mypath/nnsurveillance/Heatmap',0)

            # for testing purposes, show the result of the background subtraction
            kernel = np.ones((3, 3), np.uint8)
            kernel2 = np.ones((2, 2), np.uint8)
            opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
            darknet.saveimage(fgmask, "test2", '/mypath/nnsurveillance/Heatmap',0)
            #cv2.imshow('img', closing)
            #cv2.waitKey(1)

            # apply a binary threshold only keeping pixels above thresh and setting the result to maxValue.  If you want
            # motion to be picked up more, increase the value of maxValue.  To pick up the least amount of motion over time, set maxValue = 1
            thresh = 2
            maxValue = 2
            ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)
            darknet.saveimage(th1, "test3", '/mypath/nnsurveillance/Heatmap',0)
            # for testing purposes, show the threshold image
            #cv2.imwrite('temp/diff-th1.jpg', th1)
            #darknet.saveimage(th1,heatmapframes,'/mypath/nnsurveillance/Heatmap/temp')

            # add to the accumulated image
            # ACCUM IMAGE OF NUMBER OF FRAMES DESIRED

            heatrange = heatmapframes

            #files2 = []
            if (cycle == 0):
                #print(th1)
                # OPTION A, SAVING PICS IN FILES

                darknet.saveimage(th1,heatmapframes,'/mypath/nnsurveillance/Heatmap/temp',0)
                # OPTION B, SAVING PICS IN ARRAY
                # imagenn , img = array_to_imag#!/usr/bin/python3e(img)
                # IMAGE IS DISPLAYED

                #print("################")
                #thsaved.append(th1)
                #print(thsaved[heatmapframes])
                #print("################")
                #thsaved[heatmapframes] = th1.copy()
                #print(thsaved[heatmapframes])


            elif(cycle ==1):

                # OPTION A, SAVING PICS IN FILES

                darknet.saveimage(th1,heatmapframes,'/mypath/nnsurveillance/Heatmap/temp',0)

                # OPTION B, SAVING PICS IN ARRAY

                #thsaved[heatmapframes] = th1.copy()
                heatrange = maxheatmapframes

            #reset the picture
            accum_image = np.zeros((height, width), np.uint8)


            files2 = getfiles('/mypath/nnsurveillance/Heatmap/temp',0)
            (files2.sort(key=lambda var: [u if u.isdigit() else u for u in re.findall(r"[^0-9]|[0-9]+", var)]))
            x = 0
            for stri2 in files2:
                #print("heat range is ", heatrange)
                newth = cv2.imread(stri2)
                newthgray = cv2.cvtColor(newth, cv2.COLOR_BGR2GRAY)  # convert to grayscale
                darknet.saveimage(newthgray, "befacu", '/mypath/nnsurveillance/Heatmap',0)
                accum_image = cv2.add(accum_image, newthgray)

                # for testing purposes, show the colorMap image

                #imagenn = Image.fromarray(newth)
                #imagenn.show()
                #time.sleep(1)
                #th = cv2.imread(stri2)
                #print(sizeof(th))
                #print(sizeof(accum_image))



                if x < heatrange:
                    x = x+1
                else:
                    break
            color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
            cv2.imwrite('diff-color.jpg', color_image)
                # for testing purposes, show the accumulated image
            darknet.saveimage(accum_image,"acc","/mypath/nnsurveillance/Heatmap",0)

            # for testing purposes, control frame by frame
            # raw_input("press any key to continue")

            # for testing purposes, show the current frame
            # cv2.imshow('frame', gray)
            #cv2.imshow('image',accum_image)
            #cv2.waitKey()
            # apply a color map
            # COLORMAP_PINK also works well, COLORMAP_BONE is acceptable if the background is dark
            color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
            # for testing purposes, show the colorMap image
            cv2.imwrite('diff-color.jpg', color_image)

            # overlay the color mapped image to the first frame
            result_overlay = cv2.addWeighted(frame, 0.7, color_image, 0.8, 0)

            #cv2.imshow('result',result_overlay)
            #cv2.waitKey(1)
            # save the final overlay image
            #cv2.imwrite('diff-overlay.jpg', result_overlay)
            darknet.saveimage(result_overlay,framespassed,'/mypath/nnsurveillance/resultsheat',0)
            heatmapframes = heatmapframes + 1
            framespassed = framespassed + 1

            #img2 = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)


            #imagenn = Image.fromarray(img2)
            # imagenn , img = array_to_imag#!/usr/bin/python3e(img)
            # IMAGE IS DISPLAYED
            #imagenn.show()


            for proc in psutil.process_iter():
                if proc.name() == "display":
                    proc.kill()

            if heatmapframes == maxheatmapframes:
                heatmapframes = 0
                cycle = 1
            # cleanup
            cv2.destroyAllWindows()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(framespassed)


#########################################################################


#########################################################################

'''
        There are some important reasons this if statement exists:
            -in the first run there is no previous frame, so this accounts for that
            -the first frame is saved to be used for the overlay after the accumulation has occurred
            -the height and width of the video are used to create an empty image for accumulation (accum_image)
        '''


if __name__=='__main__':
    main()
