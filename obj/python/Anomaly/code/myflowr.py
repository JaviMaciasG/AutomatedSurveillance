'''
Created on 29 Dec 2018

@author: David
'''


from Anomaly.code.Editmyfiles import saveimage,getfiles

import cv2
import numpy as np

import re

from cv2 import FONT_HERSHEY_COMPLEX, waitKey


def SortArray(Array):
    Array2 = []
    Max = Array.max()
    if Max <= Array[0]:
        Array2




'''
def EucDist((x1, y1), (x2, y2)):
    dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    return dist
'''


def GetThreshold(FlowMeanArray, increment):
    ArrayLen = len(FlowMeanArray)
    Threshold = 0
    for i in range(ArrayLen):
        Threshold = Threshold + FlowMeanArray[i]
    Threshold = Threshold / ArrayLen
    Threshold = Threshold * (increment)
    return Threshold


# def TriggerAlarm():

def DrawNumberPos(frame2, BlockWidth, BlockHeigth, m, n, dist, heigth):
    size = (heigth / 240) * 0.25
    x1 = (n * BlockHeigth) + 5
    y1 = (m * BlockWidth) + dist

    # text = str((6*m+n)) y1 x1
    text = str(m + n * 6)

    # if n == 0:
    #   text = str(m)
    # if m == 0:
    #   text = str(n)
    cv2.putText(frame2, text, (x1, y1), FONT_HERSHEY_COMPLEX, size, (0, 0, 255), 1, cv2.LINE_AA)


def DisplayAlarm(frame2, dist, heigth):
    size = round((heigth / 240) * 0.75)
    cv2.putText(frame2, "A L A R M", (0, round(0 + dist)), FONT_HERSHEY_COMPLEX, size, (0, 0, 255), 1, cv2.LINE_AA)


def DrawFlowMag(frame2, BlockWidth, BlockHeigth, m, n, Maxflow, threshold, dist, heigth):
    size = (heigth / 240) * 0.25
    x1 = round((n * BlockHeigth) + 5)
    y1 = round((m * BlockWidth) + dist)

    text = str(round(Maxflow, 2))

    # if n == 0:
    #   text = str(m)
    # if m == 0:
    #   text = str(n)
    if Maxflow > threshold:
        cv2.putText(frame2, text, (x1, y1), FONT_HERSHEY_COMPLEX, size, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame2, text, (x1, y1), FONT_HERSHEY_COMPLEX, size, (0, 255, 0), 1, cv2.LINE_AA)


def DrawFlowMag2(frame2, BlockWidth, BlockHeigth, m, n, Maxflow, centers, dist, heigth):
    size = (heigth / 240) * 0.25
    x1 = round((n * BlockHeigth) + 5)
    y1 = round((m * BlockWidth) + dist)

    text = str(round(Maxflow, 2))

    # if n == 0:
    #   text = str(m)
    # if m == 0:
    #   text = str(n)
    if Maxflow > centers[2]:
        cv2.putText(frame2, text, (x1, y1), FONT_HERSHEY_COMPLEX, size, (0, 0, 255), 1, cv2.LINE_AA)

    elif Maxflow > centers[1]:
        cv2.putText(frame2, text, (x1, y1), FONT_HERSHEY_COMPLEX, size, (0, 255, 255), 1, cv2.LINE_AA)

    else:
        cv2.putText(frame2, text, (x1, y1), FONT_HERSHEY_COMPLEX, size, (0, 255, 0), 1, cv2.LINE_AA)


def ReArrangeArray(Array, pos, value):
    Array1 = pos  # 36
    Array2 = Array.shape[1]  # 3
    Array3 = Array.shape[2]  # 1
    Arraylen = (Array2)
    for i in range(Arraylen):
        if i < Arraylen - 1:
            Array[(pos - 1), i, (Array3 - 1)] = Array[(pos - 1), i + 1, (Array3 - 1)]
        if i == Arraylen - 1:
            Array[(pos - 1), i, (Array3 - 1)] = value
        # print("i",i," ",Array[(pos-1),i,(Array3-1)])


def TriggerThreshold(Array, pos, TrDuration):
    count = 0
    Array1 = pos  # 36
    Array2 = Array.shape[1]  # 3
    Array3 = Array.shape[2]  # 1
    Arraylen = (Array2)
    for i in range(0, Arraylen):
        if Array[(pos - 1), i, (Array3 - 1)] == 1:
            count = count + 1
            # print("i ",i," count",count)

    if count == TrDuration:
        Trigger = 1
    else:
        Trigger = 0

    return Trigger


def SetMax(Array1, Array2):
    leng = len(Array1)
    # print leng
    for i in range(leng):
        if Array1[i] < Array2[i]:
            Array1[i] = Array2[i]
        if Array1[i] > 10:
            Array1[i] = 0
    # print Array1


def SetMaxMatrix(Matrix):
    Max = 0

    for ind, val in np.ndenumerate(Matrix):
        if val > Max:
            Max = val

    # return Matrix.max()
    return Max


def drawOptFlowMap(flow, newframe, num, color):
    rows, cols = newframe.shape[0], newframe.shape[1]
    # print("rows ",rows," ,cols ",cols)
    # print("flow",flow.shape)
    flowrow, flowcol = flow.shape[0], flow.shape[1]
    # print("flowrows ",flowrow," ,flowcols ", flowcol)
    for row in range(0, rows, num):
        for col in range(0, cols, num):
            #      print("x ",x,", y ",y)
            fx = flow[row, col, 0]
            # print("fx is ",fx)
            fy = flow[row, col, 1]
            # print("fy is ",fy)
            cv2.line(newframe, (col, row), (int(round(col + fx)), int(round(row + fy))), color, 1)
            cv2.circle(newframe, (col, row), 1, color, -1)

    #       print("done")
    # cv2.circle(newframe,(320,240),15,(255,0,0),1)


def drawOptFlowBlock(flow, newframe, num, color):
    rows, cols = newframe.shape[0], newframe.shape[1]
    # print("rows ",rows," ,cols ",cols)
    # print("flow",flow.shape)
    flowrow, flowcol = flow.shape[0], flow.shape[1]
    # print("flowrows ",flowrow," ,flowcols ", flowcol)
    for x in range(0, rows, num):
        for y in range(0, cols, num):
            #      print("x ",x,", y ",y)
            fy = flow[x, y, 0]
            # print("fx is ",fx)
            fx = flow[x, y, 1]
            # print("fy is ",fy)
            cv2.line(newframe, (y, x), (int(round(y + fy + 15)), int(round(x + fx + 15))), color, 3)
            cv2.circle(newframe, (y, x), 1, color, -1)


def QuantifieArray(Array):
    lenofarray = len(Array)
    QuantifiedArray = []
    for i in range(lenofarray):
        quantified = 0
        while Array[i] >= quantified:
            quantified = quantified + 0.02
            if quantified > 20:
                break;
        quantified = round(quantified, 2)
        print(i, "i-", quantified, " of ", Array[i])
        QuantifiedArray.append(quantified)

    return QuantifiedArray


def StudyData(Array, centroids):
    # plt.hist(Array,100,[0,10]),plt.show()
    pa = np.asarray(Array, dtype=np.float32)

    print(pa.shape)

    leng = len(pa)

    np.reshape(pa, (leng))

    # print(pa.shape)
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans

    compactness, labels, centers = cv2.kmeans(pa, centroids, None, criteria, 10, flags)
    # print("compactness",compactness,"labels",labels,"centers",centers)
    '''   
    A = Array[labels==0]
    B = Array[labels==1]
    C = Array[labels==2]
    # Now plot 'A' in red, 'B' in blue, 'centers' in yellow

    plt.hist(A,100,[0,10],color = 'r')
    plt.hist(B,100,[0,10],color = 'b')
    plt.hist(B,100,[0,10],color = 'g')
    plt.hist(centers,32,[0,10],color = 'y')
    plt.show()
    '''
    output = np.zeros(4)
    for i in range(len(output)):
        if i < 3:
            output[i] = centers[i]
        else:
            output[i] = compactness

    return output


# This function takes a mean value from the first frames as a threshold

def mymainfilesongoing(stri,BlockWidth,BlockHeigth,BlockRowSize,BlockColumnSize,ImageHeigth,count,
                       ActualThreshold,ThresholdArray,NumpyTimeThresholdArray,MyMaxFlow,MaxFlow,
                       position,prvs, pyr_scale, levels, winsize, iterations, poly_n,
                       poly_sigma,flags,TrDuration):

    frame2 = cv2.imread(stri)

    if stri:
        print(count)

        # image resize
        # basewidth = 320
        # wpercent = (basewidth / float(frame2.shape[0]))
        # hsize = int((float(frame2.shape[1]) * float(wpercent)))
        # print("basewidth ",basewidth,", hsize ",hsize)
        frame2 = cv2.resize(frame2, dsize=(320, 240))
        ##############

        #### Obtaining the flow motion ######
        NewGray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, NewGray, None, pyr_scale, levels, winsize, iterations, poly_n,
                                            poly_sigma, flags)
        # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        prvs = NewGray

        #### Drawing the flow motion in the image #####
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # drawOptFlowMap(flow,frame2,10,(0,255,0))

        # cv2.imshow("output",frame2)
        ### ------------------METHOD 1 : WORKING WITH MAX AND MIN VALUES AS TRESHOLDS ----------------------####
        ### Dividing the frame in different segment to study the flow motion in every block ####

        MaxFlowBlockArray = []
        FlowBlockMean = 0
        FlowMeanArray = []
        Trigger = 0
        Alarm = 0
        ThresholdInfoDuration = 60

        for n in range(0, (BlockRowSize)):

            for m in range(0, (BlockColumnSize)):

                # For every block we have in the frame, a cluster is created with the dx and dy information.
                # For example, in this 240*320 frame, blocks will have aprox. 60x53 pixels.
                # The absolute value will be saved in a matrix and then, we will have a maximum threshold value for every mxn block.

                # With the map function, we can get a new matrix in which each element is the result of an operation

                FlowBlockMatrix = np.zeros((BlockWidth, BlockHeigth))
                # FlowBlockMatrix = np.zeros(50,50)
                Meaniterations = 0

                for ind, val in np.ndenumerate(FlowBlockMatrix):

                    #Checking display purposes
                    if ind[0] % 5 == 0:
                        cv2.circle(frame2,(ind[0]+(n*BlockWidth),ind[1]+(m*BlockHeigth)),1,(255,0,0),-1)
                        #print(ind[1]+(m*BlockHeigth))


                FlowBlockMean = FlowBlockMean / Meaniterations
                if m != 0 or n != 0 or m != 6 or n != 6:
                    if abs((FlowBlockMean)) < 500:
                        FlowMeanArray.append(FlowBlockMean)
                    # print ("mean ",FlowBlockMean, "in it", Meaniterations)
                MaxFlowBlock = SetMaxMatrix(FlowBlockMatrix)

                if MyMaxFlow[(m + 6 * n)] < MaxFlowBlock:
                    MyMaxFlow[(m + 6 * n)] = MaxFlowBlock
                    # print(" Block: ",(m+6*n)," flow : ",MaxFlowBlock)

                # ---- display the values in the frame---#
                if (ActualThreshold == 0):
                    InputThreshold = 100
                else:
                    InputThreshold = ActualThreshold
                    if (DisplayAlarmValue != 0):
                        DisplayAlarm(frame2, ((ImageHeigth / 240) * 10) + position, ImageHeigth)

                cv2.rectangle(frame2, (BlockWidth * n, BlockHeigth * m),
                              (BlockWidth * (n + 1), BlockHeigth * (m + 1)), (0, 255, 0), 1)
                DrawFlowMag(frame2, BlockHeigth, BlockWidth, m, n, FlowBlockMean, InputThreshold,
                            ((ImageHeigth / 240) * 10) + position, ImageHeigth)
                # DrawFlowMag(frame2,BlockHeigth,BlockWidth,m,n,MyMaxFlow[(m+(6*n))],InputThreshold,((ImageHeigth/240)*20)+position,ImageHeigth)
                DrawNumberPos(frame2, BlockHeigth, BlockWidth, m, n, position, ImageHeigth)
                Meaniterations = 0

                if (ActualThreshold != 0):

                    if ActualThreshold < FlowBlockMean:
                        print((m + 6 * n), " Value above threshold ", ActualThreshold, " < ", FlowBlockMean)
                        ReArrangeArray(NumpyTimeThresholdArray, (m + 6 * n), 1)
                    else:
                        ReArrangeArray(NumpyTimeThresholdArray, (m + 6 * n), 0)
                    Trigger = TriggerThreshold(NumpyTimeThresholdArray, (m + 6 * n), TrDuration)
                    print("Trigger is ", Trigger)
                    # print((m+6*n)," ",NumpyTimeThresholdArray[(m+6*n),...])
                    Alarm = Alarm + Trigger
                    print("Alarm is ", Alarm)

                    if (Alarm > 6) or (DisplayAlarmValue != 0):
                        DisplayAlarmValue = 1
                        print("alarm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if (len(ThresholdArray) <= ThresholdInfoDuration) and (ActualThreshold == 0):
            Threshold = GetThreshold(FlowMeanArray, 2.6)
            ThresholdArray.append(Threshold)
            print(ThresholdArray)

            for i in range(len(ThresholdArray)):
                print(i, ".", ThresholdArray[i])
        mynorm = np.linalg.norm(MyMaxFlow)

        if (len(ThresholdArray) > ThresholdInfoDuration) and (ActualThreshold == 0):
            ActualThreshold = SetMaxMatrix(ThresholdArray)

            # ---------------------------------------#
            # MaxFlowBlockArray has the maximum magnitude value in each block of the frame. Now we save the information in a new matrix and then compare if
            # in a new frame, we have a new maximum. Following this procedure, we will get the maximum threshold value.
        if count == 0:
            MaxFlow = MaxFlowBlockArray

        else:
            SetMax(MaxFlow, MaxFlowBlockArray)


            ###--------------------- END OF METHOD 1 -----------------------------------------------------------#####

            ### ------------------METHOD 2 : WORKING WITH KMEAN ----------------------####

            ### ------------------  END OFMETHOD 2 ----------------------####
    cv2.imshow("output", frame2)
    count = count + 1
    saveimage(frame2, count, "/mypath/darknet-master/Anomaly/code/results", 0)

    cv2.waitKey(1)

    # cv2.waitKey(1);


# np.save(r"mypath/Dataset\videos\scene1\maxvalues_myflow1.npy",MaxFlow)

# This function creates and save in a document the values of the clusters created from a training video

def mymainfilesgatherdata(stri,FlowMeanArray,BlockWidth,BlockHeigth,BlockRowSize,BlockColumnSize,ImageHeigth,count
                         ,ActualThreshold,DisplayAlarmValue,MyMaxFlow,MaxFlow,
                       position,prvs, pyr_scale, levels, winsize, iterations, poly_n,
                        poly_sigma,flags):




        frame2 = cv2.imread(stri)

        if stri:
            print(count)

            pyr_scale = 0.1
            levels = 3
            winsize = 5# image resize
            iterations = 5# basewidth = 320
            poly_n = 5# wpercent = (basewidth / float(frame2.shape[0]))
            poly_sigma = 1.1# hsize = int((float(frame2.shape[1]) * float(wpercent)))
            flags = 0# print("basewidth ",basewidth,", hsize ",hsize)

            frame2 = cv2.resize(frame2, dsize=(320, 240))
            ##############

            #### Obtaining the flow motion ######
            NewGray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs, NewGray, None, pyr_scale, levels, winsize, iterations, poly_n,
                                                poly_sigma, flags)
            # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            prvs = NewGray

            #### Drawing the flow motion in the image #####
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # drawOptFlowMap(flow,frame2,10,(0,255,0))

            # cv2.imshow("output",frame2)
            ### ------------------METHOD 1 : WORKING WITH MAX AND MIN VALUES AS TRESHOLDS ----------------------####
            ### Dividing the frame in different segment to study the flow motion in every block ####

            MaxFlowBlockArray = []
            FlowBlockMean = 0
            Trigger = 0
            Alarm = 0
            # ThresholdInfoDuration = 0

            for n in range(0, (BlockRowSize)):

                for m in range(0, (BlockColumnSize)):

                    # For every block we have in the frame, a cluster is created with the dx and dy information.
                    # For example, in this 240*320 frame, blocks will have aprox. 60x53 pixels.
                    # The absolute value will be saved in a matrix and then, we will have a maximum threshold value for every mxn block.

                    # With the map function, we can get a new matrix in which each element is the result of an operation

                    FlowBlockMatrix = np.zeros((BlockWidth, BlockHeigth))
                    # FlowBlockMatrix = np.zeros(50,50)
                    Meaniterations = 0

                    for ind, val in np.ndenumerate(FlowBlockMatrix):
                        '''
                        # HOW THE DRAWING WORKS   
                        print("#####")
                        #print("mag shape 0:",mag.shape[0]) 576
                        #print("mag shape 1:",mag.shape[1]) 768
                        # We obtain the abs value of each pixel
                        #FlowBlockMatrix[ind[0]][ind[1]] = mag[ind[0]][ind[1]], where mag[576][768]

                        #print((m+6*n))
                        print("####")
                        print(ind[0])# va hasta 128
                        print(n*BlockWidth)# n*128
                        print(ind[1])# va hasta 95
                        print(m*BlockHeigth)# m*96
                        print("####")
                        '''

                        # con m y n hasta 5, deberia ser mag[ind[1]+(m*blockheigth)][ind[0]+(n*blockwidth)]
                        FlowBlockMatrix[ind[0]][ind[1]] += mag[ind[1] + (m * BlockHeigth)][ind[0] + (n * BlockWidth)]
                        FlowBlockMean = (FlowBlockMatrix[ind[0]][ind[1]] + FlowBlockMean)
                        # print(FlowBlockMean)
                        Meaniterations = Meaniterations + 1

                        '''
                        #Checking display purposes
                        if ind[0] % 5 == 0:
                            cv2.circle(frame2,(ind[0]+(n*BlockWidth),ind[1]+(m*BlockHeigth)),1,(255,0,0),-1)
                            #print(ind[1]+(m*BlockHeigth))
                        '''

                    FlowBlockMean = FlowBlockMean / Meaniterations
                    print("FLowBlockMean ",FlowBlockMean)
                    print("Meaniterations ",Meaniterations)
                    if m != 0 or n != 0 or m != 6 or n != 6:
                        if (abs((FlowBlockMean)) < 100) and (count % 10 == 0) and (count != 0):
                            FlowMeanArray.append(FlowBlockMean)
                        print("mean array ", len(FlowMeanArray))
                    MaxFlowBlock = SetMaxMatrix(FlowBlockMatrix)

                    if MyMaxFlow[(m + 6 * n)] < MaxFlowBlock:
                        MyMaxFlow[(m + 6 * n)] = MaxFlowBlock
                        # print(" Block: ",(m+6*n)," flow : ",MaxFlowBlock)

                    # ---- display the values in the frame---#
                    if (ActualThreshold == 0):
                        InputThreshold = 100
                    else:
                        InputThreshold = ActualThreshold
                        if (DisplayAlarmValue != 0):
                            DisplayAlarm(frame2, ((ImageHeigth / 240) * 10) + position, ImageHeigth)

                        # if (ActualThreshold != 0):
                        '''
                        if ActualThreshold < FlowBlockMean:
                            print((m + 6 * n), " Value above threshold ", ActualThreshold, " < ", FlowBlockMean)
                            ReArrangeArray(NumpyTimeThresholdArray, (m + 6 * n), 1)
                        else:
                            ReArrangeArray(NumpyTimeThresholdArray, (m + 6 * n), 0)
                        Trigger = TriggerThreshold(NumpyTimeThresholdArray, (m + 6 * n),TrDuration)
                        print("Trigger is ", Trigger)
                        # print((m+6*n)," ",NumpyTimeThresholdArray[(m+6*n),...])
                        Alarm = Alarm + Trigger
                        print("Alarm is ", Alarm)

                        if (Alarm > 6) or (DisplayAlarmValue != 0):
                            DisplayAlarmValue = 1
                            print("alarm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        '''

                # for i in range(len(ThresholdArray)):
                #    print(i, ".", ThresholdArray[i])
            # mynorm = np.linalg.norm(MyMaxFlow)

            # ---------------------------------------#
            # MaxFlowBlockArray has the maximum magnitude value in each block of the frame. Now we save the information in a new matrix and then compare if
            # in a new frame, we have a new maximum. Following this procedure, we will get the maximum threshold value.
            if count == 0:
                MaxFlow = MaxFlowBlockArray
                # print("maxflow ",MaxFlow)
            else:
                SetMax(MaxFlow, MaxFlowBlockArray)
                # print ("maxflow",MaxFlow)

                ###--------------------- END OF METHOD 1 -----------------------------------------------------------#####

                ### ------------------METHOD 2 : WORKING WITH KMEAN ----------------------####

                ### ------------------  END OFMETHOD 2 ----------------------####
        # cv2.imshow("output", frame2)
        count = count + 1
        cv2.waitKey(1)

        if len(FlowMeanArray) != 0:
            Centers = StudyData(FlowMeanArray, 3)
            print(Centers)
            print(Meaniterations)


        if count > 300:

            print("len of array", len(FlowMeanArray))

            # now, the objective is to fit the value in standarized levels

            # QuantifiedArray = QuantifieArray(FlowMeanArray)

            Centers = StudyData(FlowMeanArray, 3)

            np.sort(Centers, axis=None)
            print("saving 3 centers and compactness")
            print(Centers)
            print(Centers[2])

            # cv2.waitKey(1);
            np.save(
                r"mypath/\Dataset\videos\scene1\maxvalues_myflow1.npy",
                Centers)

        return prvs,count,ActualThreshold,DisplayAlarmValue,MyMaxFlow,MaxFlow,FlowMeanArray


    # this

# This function load the values of the clusters created before in a document and analyze new videos with them

def mymainfilescheckdata(stri,DisplayAlarmValue,BlockWidth,BlockHeigth,BlockRowSize,BlockColumnSize,ImageHeigth,count
                         ,ThresholdArray,NumpyTimeThresholdArray,MyMaxFlow,MaxFlow,
                       position,prvs, pyr_scale, levels, winsize, iterations, poly_n,
                        poly_sigma,flags,TrDuration,showgrid):
        Centers = np.load(
            r"mypath/\Dataset\videos\scene1\maxvalues_myflow1.npy")
        print(Centers)
        Compactness = Centers[3]
        Multiplier = 1 + np.power((1 / Compactness), (1 / 5))
        Centers = np.multiply(Centers, Multiplier)
        Centers.sort(axis=0)
        print(Centers)
        print("multipliers", Multiplier)
        AlarmCells = 5

        frame2 = cv2.imread(stri)

        if stri:
            print(count)


            frame2 = cv2.resize(frame2, dsize=(320, 240))
            ##############

            #### Obtaining the flow motion ######

            NewGray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs, NewGray, None, pyr_scale, levels, winsize, iterations, poly_n,
                                                poly_sigma, flags)


            prvs = NewGray

            #### Drawing the flow motion in the image #####
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])


            ### ------------------METHOD 1 : WORKING WITH MAX AND MIN VALUES AS TRESHOLDS ----------------------####
            ### Dividing the frame in different segment to study the flow motion in every block ####

            MaxFlowBlockArray = []
            FlowBlockMean = 0
            FlowMeanArray = []
            Trigger = 0
            Alarm = 0
            ThresholdInfoDuration = 60

            for n in range(0, (BlockRowSize)):

                for m in range(0, (BlockColumnSize)):

                    # For every block we have in the frame, a cluster is created with the dx and dy information.
                    # For example, in this 240*320 frame, blocks will have aprox. 60x53 pixels.
                    # The absolute value will be saved in a matrix and then, we will have a maximum threshold value for every mxn block.

                    # With the map function, we can get a new matrix in which each element is the result of an operation

                    FlowBlockMatrix = np.zeros((BlockWidth, BlockHeigth))
                    # FlowBlockMatrix = np.zeros(50,50)
                    Meaniterations = 0

                    for ind, val in np.ndenumerate(FlowBlockMatrix):
                        '''
                        # HOW THE DRAWING WORKS   
                        print("#####")
                        #print("mag shape 0:",mag.shape[0]) 576
                        #print("mag shape 1:",mag.shape[1]) 768
                        # We obtain the abs value of each pixel
                        #FlowBlockMatrix[ind[0]][ind[1]] = mag[ind[0]][ind[1]], where mag[576][768]

                        #print((m+6*n))
                        print("####")
                        print(ind[0])# va hasta 128
                        print(n*BlockWidth)# n*128
                        print(ind[1])# va hasta 95
                        print(m*BlockHeigth)# m*96
                        print("####")
                        '''

                        # con m y n hasta 5, deberia ser mag[ind[1]+(m*blockheigth)][ind[0]+(n*blockwidth)]
                        FlowBlockMatrix[ind[0]][ind[1]] += mag[ind[1] + (m * BlockHeigth)][ind[0] + (n * BlockWidth)]
                        FlowBlockMean = (FlowBlockMatrix[ind[0]][ind[1]] + FlowBlockMean)
                        # print(FlowBlockMean)
                        Meaniterations = Meaniterations + 1



                    FlowBlockMean = FlowBlockMean / Meaniterations
                    if m != 0 or n != 0 or m != 6 or n != 6:
                        if abs((FlowBlockMean)) < 500:
                            FlowMeanArray.append(FlowBlockMean)
                        # print ("mean ",FlowBlockMean, "in it", Meaniterations)
                    MaxFlowBlock = SetMaxMatrix(FlowBlockMatrix)

                    if MyMaxFlow[(m + 6 * n)] < MaxFlowBlock:
                        MyMaxFlow[(m + 6 * n)] = MaxFlowBlock
                        # print(" Block: ",(m+6*n)," flow : ",MaxFlowBlock)

                    # ---- display the values in the frame---#

                    if (DisplayAlarmValue != 0):
                        DisplayAlarm(frame2, ((ImageHeigth / 240) * 10) + position, ImageHeigth)

                    if showgrid:
                        cv2.rectangle(frame2, (BlockWidth * n, BlockHeigth * m),
                                      (BlockWidth * (n + 1), BlockHeigth * (m + 1)), (0, 255, 0), 1)
                        DrawFlowMag2(frame2, BlockHeigth, BlockWidth, m, n, FlowBlockMean, Centers,
                                     ((ImageHeigth / 240) * 10) + position, ImageHeigth)
                        # DrawFlowMag(frame2,BlockHeigth,BlockWidth,m,n,MyMaxFlow[(m+(6*n))],InputThreshold,((ImageHeigth/240)*20)+position,ImageHeigth)
                        DrawNumberPos(frame2, BlockHeigth, BlockWidth, m, n, position, ImageHeigth)
                        Meaniterations = 0

                    if Centers[2] < FlowBlockMean:
                        # print((m + 6 * n), " Value above threshold ", Centers[2], " < ", FlowBlockMean)
                        ReArrangeArray(NumpyTimeThresholdArray, (m + 6 * n), 1)
                    else:
                        ReArrangeArray(NumpyTimeThresholdArray, (m + 6 * n), 0)
                    Trigger = TriggerThreshold(NumpyTimeThresholdArray, (m + 6 * n), TrDuration)
                    # print("Trigger is ", Trigger)
                    # print((m+6*n)," ",NumpyTimeThresholdArray[(m+6*n),...])
                    Alarm = Alarm + Trigger
                    # print("Alarm is ", Alarm)

                    if (Alarm > AlarmCells) or (DisplayAlarmValue != 0):
                        DisplayAlarmValue = 1
                        # print("alarm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            if (len(ThresholdArray) <= ThresholdInfoDuration):
                Threshold = GetThreshold(FlowMeanArray, 2.6)
                ThresholdArray.append(Threshold)
                # print(ThresholdArray)

                for i in range(len(ThresholdArray)):
                    # print(i, ".", ThresholdArray[i])
                    mynorm = np.linalg.norm(MyMaxFlow)

            if (len(ThresholdArray) > ThresholdInfoDuration):
                ActualThreshold = SetMaxMatrix(ThresholdArray)

                # ---------------------------------------#
                # MaxFlowBlockArray has the maximum magnitude value in each block of the frame. Now we save the information in a new matrix and then compare if
                # in a new frame, we have a new maximum. Following this procedure, we will get the maximum threshold value.
            if count == 0:
                MaxFlow = MaxFlowBlockArray
                # print("maxflow ",MaxFlow)
            else:
                SetMax(MaxFlow, MaxFlowBlockArray)
                # print ("maxflow",MaxFlow)

                ###--------------------- END OF METHOD 1 -----------------------------------------------------------#####

                ### ------------------METHOD 2 : WORKING WITH KMEAN ----------------------####

                ### ------------------  END OFMETHOD 2 ----------------------####
        cv2.imshow("output", frame2)
        count = count + 1
        saveimage(frame2, count, "/mypath/nnsurveillance/Anomaly/code/results", 0)
        cv2.waitKey(1)

        return prvs,count, DisplayAlarmValue, MyMaxFlow, MaxFlow, FlowMeanArray

        # cv2.waitKey(1);
    # np.save(r"mypath/\Dataset\videos\scene1\maxvalues_myflow1.npy",MaxFlow)


def Stampade(input, mode, VideoInput,showgrid):
    # --- general variables ---#

    framespassed = 0

    if (VideoInput >= 1):
        capture = cv2.VideoCapture(input)
        transmission, frame = capture.read()
        frame1 = cv2.imread(frame)

    else:
        print("aqui")
        Directory = input
        files = getfiles(Directory, 1)  # png files

        files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        frame1 = cv2.imread(files[1])


    # output resize
    # screen_res =1920,1080
    screen_res = 980, 520
    scale_width = screen_res[0] / 320
    scale_height = screen_res[1] / 240
    scale = min(scale_width, scale_height)
    window_width = int(320 * scale)
    window_height = int(240 * scale)

    #cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('output', window_width, window_height)

    position = 13

    global frameNo

    # parameters
    pyr_scale = 0.1
    levels = 3
    winsize = 5
    iterations = 5
    poly_n = 5
    poly_sigma = 1.1
    flags = 0

    frameNo = 0

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # --- image resize ---


    prvs = cv2.resize(prvs, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)
    ##############

    motionInfOfFrames = []
    count = 0

    # BLOCK INFO : Window divided in 6 rows and 4 columns , having then 24 blocks to analyse if there is a suspicious behavior going on

    BlockRowSize = 6
    BlockColumnSize = 6

    ImageHeigth = prvs.shape[0]

    ImageWidth = prvs.shape[1]

    BlockHeigth = int(ImageHeigth / BlockRowSize)
    print("Heigth", BlockHeigth)
    BlockWidth = int(ImageWidth / BlockColumnSize)
    print("Width", BlockWidth)

    MyMaxFlow = np.zeros(36)
    TrDuration = 3
    ThresholdArray = []
    # list()
    ActualThreshold = 0
    timeframe = 3
    NumpyTimeThresholdArray = np.zeros((BlockRowSize * BlockColumnSize, TrDuration, 1))
    DisplayAlarmValue = 0
    MaxFlow = 0
    FlowMeanArray = []

    print("here again")

    # ----------------------------------------------------------------------------------------------------------#

    print("done.")

    if (VideoInput > 1):


        while transmission:

            if mode == "stampede":
                # ============== VIDEO TO PROCESS ================= #
                '''
                mymainfilesongoing(frame, BlockWidth, BlockHeigth, BlockRowSize, BlockColumnSize, ImageHeigth, ImageWidth,
                                   count,
                                   ActualThreshold, ThresholdArray, NumpyTimeThresholdArray, MyMaxFlow, MaxFlow,
                                   position, prvs, pyr_scale, levels, winsize, iterations, poly_n,
                                   poly_sigma, flags, TrDuration,Maxflow)

                '''
                mymainfilescheckdata(frame,DisplayAlarmValue,BlockWidth,BlockHeigth,BlockRowSize,BlockColumnSize,ImageHeigth,count
                         ,ThresholdArray,NumpyTimeThresholdArray,MyMaxFlow,MaxFlow,
                       position,prvs, pyr_scale, levels, winsize, iterations, poly_n,
                        poly_sigma,flags,TrDuration,showgrid)

            if mode == "stampedeT":
                # ============== VIDEO TO TRAIN ===========================

                if count < 301:
                    prvs,count, ActualThreshold, DisplayAlarmValue, MyMaxFlow, MaxFlow,FlowMeanArray = mymainfilesgatherdata(frame,FlowMeanArray, BlockWidth, BlockHeigth, BlockRowSize, BlockColumnSize, ImageHeigth,
                                          count,ActualThreshold, DisplayAlarmValue, MyMaxFlow, MaxFlow,
                                          position, prvs, pyr_scale, levels, winsize, iterations, poly_n,
                                          poly_sigma, flags)

                else:
                    print("done")
                    break

    else:

        for frame in files:

            if mode == "stampede":
                # ============== VIDEO TO PROCESS ================= #
                '''
                mymainfilesongoing(frame, BlockWidth, BlockHeigth, BlockRowSize, BlockColumnSize, ImageHeigth, ImageWidth,
                                   count,
                                   ActualThreshold, ThresholdArray, NumpyTimeThresholdArray, MyMaxFlow, MaxFlow,
                                   position, prvs, pyr_scale, levels, winsize, iterations, poly_n,
                                   poly_sigma, flags, TrDuration,Maxflow)
                
                '''
                prvs,count, DisplayAlarmValue, MyMaxFlow, MaxFlow, FlowMeanArray = mymainfilescheckdata(frame,DisplayAlarmValue,BlockWidth,BlockHeigth,BlockRowSize,BlockColumnSize,ImageHeigth,count
                         ,ThresholdArray,NumpyTimeThresholdArray,MyMaxFlow,MaxFlow,
                       position,prvs, pyr_scale, levels, winsize, iterations, poly_n,
                        poly_sigma,flags,TrDuration,showgrid)

            if mode == "stampedeT":
                # ============== VIDEO TO TRAIN ===========================


                if count < 301:
                    flags = 0

                    prvs,count, ActualThreshold, DisplayAlarmValue, MyMaxFlow, MaxFlow, FlowMeanArray = mymainfilesgatherdata(frame,FlowMeanArray,BlockWidth, BlockHeigth, BlockRowSize, BlockColumnSize,
                                          ImageHeigth, count
                                          , ActualThreshold, DisplayAlarmValue, MyMaxFlow, MaxFlow,
                                          position, prvs, pyr_scale, levels, winsize, iterations, poly_n,
                                          poly_sigma, flags)


                else:
                    print("done")
                    break



if __name__ == "__main__":

        trainingset3 = [r"/mypath/nnsurveillance/Heatmap/vtest.avi"]
        trainingSet2 = [r"/mypath/nnsurveillance/Anomaly/Dataset/vids/videos/scene2/2_train1.avi"]

        Directory2 = "/home/sergio.lopez/Videos/extract/minnesota/scene3/3_test1"
        VideoInput = 0

        mode = "stampede"
        showgrid = 1


        Stampade(Directory2,mode,VideoInput,showgrid)





