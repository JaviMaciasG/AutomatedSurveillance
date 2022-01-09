import numpy as np
from statistics import mode
import cv2
import kalman3
from darknetandheat import getfilter,associatedkalman , createkalman
from kalman3 import KalmanFilter

import ast

#---------------status colors ---------------------#
# B G R
missingcolor = (0, 240, 240) #yellow
abandonedcolor = (0, 0, 255) #red
neutralpersoncolor = (240, 240, 0) #cyan
suspiciouspersoncolor = (180,180,0) #ligth cyan
neutralbagcolor = (240, 0, 240) #rose
suspiciousbagcolor = (180,0,180)#ligth rose
linkedpersoncolor = (0, 250, 0) #green
linkedbagcolor = (20, 250, 20) # ligth green
neutralcolor = (230, 230, 230) #white
lostcolor = (0, 0, 0) #black

#--------------------------------------------------#

totalpeople = 0
totalbackpack = 0
changed = 0
changed2 = 0
mean = 0

def MyMode(Thingy):

    LenArray = len(Thingy)
    MaxThingy = 0
    MaxThingyTimes = np.zeros(LenArray)
    for b in range(LenArray):
        for k in range(LenArray):
    #        print("Comparing ",Thingy[b]," with ",Thingy[k])
            if Thingy[b] == Thingy[k]:
                MaxThingyTimes[b] = 1 + MaxThingyTimes[b]
     #           print("yeah same, ", MaxThingyTimes[b])

    #print(" Maxthingytimes ", MaxThingyTimes)
    for x in range(LenArray):
        for y in range(LenArray):
            if MaxThingyTimes[x] == MaxThingyTimes[y]:
                if (MaxThingyTimes[x] != 0) and (MaxThingyTimes[x] > LenArray/2):
                    MaxThingy = x

    if MaxThingy != 0:
        return Thingy[MaxThingy]
    else:
        return None

def GetStatus(Thingy):

    return Thingy.status

def CheckPosition(Thingy,theid):

    LenArray = len(Thingy)

    for b in range(LenArray):
        if Thingy[b].id == int(theid):
            return b

def GetID(Thingy):

    return Thingy.id

def ReorgArray(Array):

    LenArray = len(Array)
    Array2 = Array.copy()

    b = 0

    for b in range(LenArray - 1):

        Array[b] = Array2[b+1]
        b = b + 1


    return Array

def ReorgAndAdd(Array,Value):


    Array = ReorgArray(Array)

    LenArray = len(Array)

    Array[LenArray-1] = Value



    return Array


def CreateBunchOfPeople(array, alist):
    for i in range(0, array):
        newperson = person(0, 0, 0, 0, 0, 0, 0, np.zeros(256))
        alist.append(newperson)


def CreateBunchOfBags(array, alist):
    for i in range(0, array):
        newbag = backpack(0, 0, 0, 0, 0, 0, 0,  np.zeros(256))
        alist.append(newbag)


def EuDist(extx, exty, newextx, newexty):
    return pow(pow(newextx - extx, 2) + pow(newexty - exty, 2), 0.5)


def AreaDiff(wNew, hNew, wOld, hOld, xNew, yNew, xOld, yOld, AbsPer):
    AreaNew = (wNew - xNew) * (hNew - yNew)
    AreaOld = (wOld - xOld) * (hOld - yOld)
    Greater = 0
    Area = AreaNew - AreaOld

    if (AreaNew - AreaOld) <= 0:
        Greater = 0

    elif (AreaNew - AreaOld) > 0:
        Greater = 1

    if AbsPer == 0:
        return abs(Area), Greater

    elif AbsPer == 1:
        if Greater == 1:
            Area = ((AreaNew - AreaOld) / AreaNew) * 100
            return Area, Greater
        elif Greater == 0:
            Area = ((AreaOld - AreaNew) / AreaOld) * 100
            return Area, Greater

def CompareHist(hist1,hist2):

    differencearray = []
    #print(" hist 1 is ",len(hist1))
    #print(" hist 2 is ",len(hist2))

    for i in range(len(hist1)):
        div = hist1[i]
        if hist1[i] == 0:
            div = 1
        difference = (((hist2[i] - hist1[i])/div)*100)

        differencearray.append(difference)
    totaldifference = np.mean(differencearray)

    return totaldifference

def CheckOwnerArea(People,Bags):
    if(People.posx < Bags.xcentroid) and (People.width > Bags.xcentroid) and (People.posy < Bags.ycentroid) and (People.height > Bags.xcentroid):
        return 1
    else:
        return 0
def CheckOwnerHist(mindistlist,People, maxdist,lastpersonlinked):

    listlen = len(mindistlist)
    possibledist = []
    possiblehist = []

    for i in range(listlen):

        #print(" owner ", i ," dist ", mindistlist[i]," and hist ", People[i].histogramvalue)
        if mindistlist[i] <= maxdist:
            #print(" possible owner ",i," dist", mindistlist[i])
            possibledist.append(int(i))

    if possibledist:
        for i in range(len(possibledist)):
            histiteration = CompareHist(People[possibledist[i]].histogramvalue,People[int(lastpersonlinked)].histogramvalue)
            #print(" possible owner ", possibledist[i], " hist ", histiteration)
            possiblehist.append(histiteration)

        #print("owner chosen ",possibledist[np.argmin(possiblehist)]," with hist ",possiblehist[np.argmin(possiblehist)])

        return 1,possibledist[np.argmin(possiblehist)]
    else:
        return 0,0

def checkabandon(BagLinkDistance, ThisBag, ThisPerson):

    if (BagLinkDistance >= mean*1.2) and (BagLinkDistance < 9999):

        print(" bag link distance ", BagLinkDistance, " >= ", mean*1.2)
        print("bag ", ThisBag.id, " away ", len(ThisBag.lostdistance), " from ",ThisPerson.id,"times.")
        ThisBag.lostdistance.append(BagLinkDistance)

        ThisBag.status = suspiciousbagcolor

        #if ThisBag.linked == 1:

        #print("bag away and linked.")
        timeslost = len(ThisBag.lostdistance)

        if timeslost < 6:

                ThisBag.lostdistance.append(BagLinkDistance)

        elif timeslost >= 6:

                #print(" timeslost is over 6 ")
                ThisBag.lostdistance = ReorgAndAdd(ThisBag.lostdistance, BagLinkDistance)
                count = 0

                for i in range(timeslost):

                    print("iteration ", i, " lost distance :", ThisBag.lostdistance[i],", Lost distance array len ", len(ThisBag.lostdistance) )
                    if (ThisBag.lostdistance[i + 1] > ThisBag.lostdistance[i]) or (
                            ThisBag.lostdistance[i] >= 0.6 * ThisPerson.rheight):
                        print(ThisBag.lostdistance[i], " >= ", ThisPerson.rwidth)
                        count = count + 1

                    print(" incremental times ", count, ", iterations ", i)

                    if count > 4:
                        ThisBag.abandoned = 1

                    if i == 5:
                        break

def LinkedBackpack(People, Bags, KalmanArray):

    LastPersonLinked = None
    LenPeople = len(People)
    LenBag = len(Bags)
    array = np.zeros(5)
    returnarray = []

    ## For loop to check all the bags in  the array

    for b in range(LenBag):

        minList = []

        #print("Bag ", Bags[b].id, "updated: ", Bags[b].updated, "linked : ", Bags[b].linked)

        ## If YOLO has found the bag, the classification for "linked", "lost" or "abandoned" begins with updated == 1.

        if Bags[b].updated == 1:

            for x in range(LenPeople):

                ## If the person has previously been updated, we have in consideration the distance.
                if People[x].updated == 1:

                    minlistval = EuDist(Bags[b].xcentroid, Bags[b].ycentroid, People[x].xcentroid, People[x].ycentroid)
                    #print(" Bag ", Bags[b].id, " distance with person ", People[x].id, " : ", minlistval)
                    minList.append(minlistval)


                ## If the person has not, it means it can be out of the scope, so we just put a inf. distance.
                elif People[x].updated == 0:

                    minlistval = 999999
                    #print(" Bag ", Bags[b].id," distance with person ",People[x].id, " : ", minlistval)
                    minList.append(minlistval)

            # We obtain the min distance and the position in the array to know which person is the one with the min distance.
            TheMinForThisBag = min(minList)
            personid = int(np.argmin(minList))


            LenBagLinked = len(Bags[b].near)

            #If the bag has been updated for at least 5 times, we try to link the bag to a person.

            if LenBagLinked < 5:
                Bags[b].near.append(People[personid].id)
            elif LenBagLinked >= 5:
                Bags[b].near = ReorgAndAdd(Bags[b].near,People[personid].id)

            print(" near ", Bags[b].near)
            print("bag ",Bags[b].id," length : ", LenBagLinked)

            if LenBagLinked >= 5:

            # The array is being modified to only check the last 5 frames were people has been closed to the bag

                for i in range(6):
                    if i == 0:
                        continue
                    else:
                        array[i-1] = Bags[b].near[-i]
                        #print("array ", array[i-1])
                #print("array ",array)
                if ((MyMode(array))) == None:
                    #print("NONE")
                    continue
                else:
                    mostr = int(MyMode(array))
                    #print("mostr ", mostr)

                # Just a check to know the previous person linked and the person closest to the bag
                # this is needed for the firts iteration. CheckOwner is going to have a error if we don't initialize the value
                LastPersonLinked = mostr

                print("lastpersonlinked is ",str(mostr))

                Ownerbyhist = 0

                if Bags[b].linked == 1:

                    LastPersonLinked = Bags[b].personlinked
                    LastPersonLinkedPosition = CheckPosition(People, LastPersonLinked)



                newpersontolink = str(mostr)
                personposition = CheckPosition(People,newpersontolink)

                if personposition != None:

                    print("newpersontolink is ", newpersontolink," with id ",People[personposition].id)

                #print("min distance with person is ",TheMinForThisBag ," and max distance is ",2.2*Bags[b].rwidth )


                # If the bag is too far away, we check if it is abandoned.

                if (TheMinForThisBag > 2.2*Bags[b].rwidth) and (Bags[b].linked == 0) and (personposition != None):

                    BagLinkDistance = EuDist(Bags[b].xcentroid, Bags[b].ycentroid, People[personid].xcentroid, People[personid].ycentroid)

                    if Bags[b].linked == 0:
                        Bags[b].personlinked = personposition
                    checkabandon(BagLinkDistance, Bags[b], People[personid])


                # If the bag is close enough to a person, the person is linked.

                if TheMinForThisBag < 2.2*Bags[b].rwidth and (personposition != None):

                    text = "Bag " + str(Bags[b].id) + " close to person " + str(People[personposition].id)
                    print(text)

                    if Bags[b].linked == 1:
                        Bags[b].status = linkedbagcolor
                        #People[personlinked].status = linkedpersoncolor
                        text = "Bag " + str(Bags[b].id) + " close to person " + str(Bags[b].personlinked)
                        Bags[b].lostdistance = []
                        print("bag clear with lostdistance  = ", len(Bags[b].lostdistance))
                        print(text)


                    if (personposition != None) and (Bags[b].linked == 0) and (Bags[b].abandoned == 0): # in case the person goes out the limits of the screen
                        Bags[b].personlinked = newpersontolink

                        Bags[b].linked = 1
                        Bags[b].status = linkedbagcolor
                        Bags[b].distancetoownerx = People[personposition].posx - Bags[b].posx
                        Bags[b].distancetoownery = People[personposition].posy - Bags[b].posy

                        text = "Bag " + str(Bags[b].id) + " linked to person " + str(Bags[b].personlinked)
                        print(text)


                        People[personposition].status = linkedpersoncolor




                if Bags[b].linked == 1 and (personposition != None) :

                    #personposition = CheckPosition(People, Bags[b].personlinked)
                    BagLinkDistance = EuDist(Bags[b].xcentroid, Bags[b].ycentroid, People[personposition].xcentroid, People[personposition].ycentroid)

                # The distance of the bag with the person is checked. In case the threshold is passed 7 times, the bag is going
                # to be considered lost.

                    checkabandon(BagLinkDistance, Bags[b], People[personposition])



        # We check posible states and print messages to know the actual status of the bag.


        elif (Bags[b].updated == 0) and (Bags[b].linked == 0):

            text = "Bag " + str(Bags[b].id) + " not found. "


        elif (Bags[b].updated == 0) and (Bags[b].linked == 1):

            personlinked = CheckPosition(People, Bags[b].personlinked)
            #print(" personlinked is in the pos ", personlinked, " and rwidth is ", People[0].rwidth)
            text = "Bag " + str(Bags[b].id) + " not found. Last time linked to person " + str(personlinked)



            if personlinked == None:
                #print("personlinked is none ",personlinked)
                Bags[b].status = lostcolor
                #continue

            #if People[personlinked].updated == 0:

             #   Bags[b].status = lostcolor


            elif People[personlinked]:

                #print("fake updating")
                #cv2.putText(img, text, (30, (500 + 16 * b)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 182, 193), 2)

                if Bags[b].leftsided == 0:
                    extra = (People[personlinked].rwidth)/6
                else:
                    extra = -(People[personlinked].rwidth)/6

                difference, greater = AreaDiff(People[personlinked].width, People[personlinked].height, People[personlinked].widthatlink, People[personlinked].heightatlink, People[personlinked].posx, People[personlinked].posy,
                                               People[personlinked].posxatlink,
                                               People[personlinked].posyatlink, 1)

                if greater == 1:

                 Bags[b].status = missingcolor


        if Bags[b].abandoned == 1:

            Bags[b].status = abandonedcolor
            personlinked = CheckPosition(People, Bags[b].personlinked)
            if Bags[b].linked ==1 and personlinked !=None :
                People[personlinked].status = abandonedcolor
                text = "Bag " + str(Bags[b].id) + " abandoned by person " + str(personlinked) + " bag data linked " + str( Bags[b].personlinked)
            else:

                text = "Bag " + str(Bags[b].id) + " abandoned "

            print(text)




def restartupdatedflag(updated, thingy):

    Lent = len(thingy)

    #print("restarting flags")

    for b in range(Lent):
        thingy[b].updated = updated


def LastUpdate(framespassed, thingy,framelimit,label):

        Lent = len(thingy)

        Array_length = int(Lent)

        #print(Array_length)

        for b in range(Array_length):
            #print(b)
            personremoved = 0
            printedremove = "NOTHING"
            #print("person ", people[b].id, " last frame is ", people[b].lastframe)
            position = 0
            #17 frames era buen valor
            if (framespassed - thingy[b].lastframe > framelimit):
                personremoved = 1
                printedremove ="object " + str(thingy[b].id) + " removed"
                #print("\n")
                #print(printedremove)
                #position = people[b].id
                position = b
                b = b - 1
                return (printedremove, personremoved, position)

class thing:


    def showvalue(self):
        print(self.posx,self.posy)


    def updatevalue(self,posx,posy,width,height,framepassed,histogramvalue):
        speedx = (posx +(width -posx)/2) - self.xcentroid
        speedy = (posy +(height -posy)/2) - self.ycentroid
        newspeed = np.array([[np.float32(speedx)], [np.float32(speedy)]])
        self.speed = newspeed.copy()
        self.posx = posx
        self.posy = posy
        self.width = width
        self.height = height
        self.prevlastframe = self.lastframe
        self.lastframe = framepassed
        self.updated = 1
        self.rwidth = width - posx
        self.rheight = height - posy
        self.xcentroid = round(self.posx + (self.rwidth)/2)
        self.ycentroid = round(self.posy + (self.rheight)/2)
        self.histogramvalue = histogramvalue

        #self.status = self.defaultstatus

    def GetFake(self):

        return self.fakexcentroid,self.fakeycentroid

    def GetFakeCentroid(self,fakex,fakey):

        self.fakexcentroid = round(fakex)
        self.fakeycentroid = round(fakey)



class person(thing):

    def __init__(self, posx , posy ,width, height,id,updated,framepassed,histogramvalue):
        global totalpeople
       # print("Im a person")
        self.posx = posx
        self.posy = posy
        self.width = width
        self.height = height
        self.posxatlink = posx
        self.posyatlink = posy
        self.widthatlink = width
        self.heightatlink = height
        self.rwidth = self.width - self.posx
        self.rheight = self.height - self.posy
        self.xcentroid = round(self.posx + (self.rwidth)/2)
        self.ycentroid = round(self.posy + (self.rheight)/2)
        self.fakexcentroid = round(self.posx + (self.rwidth) / 2)
        self.fakeycentroid = round(self.posy + (self.rheight) / 2)
        self.id = id
        self.updated = updated
        self.prevlastframe = framepassed
        self.lastframe = framepassed
        self.status = neutralpersoncolor
        self.defaultstatus = neutralpersoncolor
        self.histogramvalue = histogramvalue
        self.speed = [0,0]


        totalpeople += 1
       # print(totalpeople)
        if (totalpeople >98) :
            totalpeople = 0

    #def __del__(self):
    #        global totalpeople
    #        totalpeople -=1
    #        #print(totalpeople)


    def LastUpdate(framespassed,people,LimitPerson,label):

        Lent = len(people)

        Array_length = int(Lent)

        #print(Array_length)

        for b in range(Array_length):
            #print(b)
            personremoved = 0
            printedremove = "NOTHING"
            #print("person ", people[b].id, " last frame is ", people[b].lastframe)
            position = 0
            #17 frames era buen valor
            if (framespassed - people[b].lastframe > LimitPerson):
                personremoved = 1
                printedremove ="person " + str(people[b].id) + " removed"
                #print("\n")
                #print(printedremove)
                #position = people[b].id
                position = b
                b = b - 1
                return (printedremove, personremoved, position)


        return (printedremove,personremoved,position)




    def DetermineNewOrUpdate(posx,posy,width,height,people,framespassed,img,histogramvalue,LimitPersonKalmanUpdate):


        xcentroid = round(posx + (width - posx)/2)
        ycentroid = round(posy + (height - posy)/2)
        Isnew = 0
        mindist = 0
        minListdist = []
        minhist = 0
        minDiff = []
        minListhist = []
        global totalpeople
        global changed
        global mean
        global KalmanPerson
        KalmanPerson = 1
        Array_length = len(people)
        mean = 0

        #First update needs to go through this if statement

        if (changed != 1) and (EuDist(xcentroid, ycentroid, people[0].xcentroid, people[0].ycentroid) > 5):
            #print ( "update de la persona 0 por tener distancia ", (EuDist(posx, posy, people[0].posx, people[0].posy)))
            #print(people[0].id, " updated")
            people[0].updatevalue(posx, posy, width, height,framespassed,histogramvalue)
            changed = 1
            mean = people[0].rheight
            return (people[0].id)

        #For loop to find minimun distances and areas of existing objects and compare them to the bounding box found in the new frame.

        for x in range(Array_length):

            # minimun difference in distance


            mindistval = EuDist(xcentroid, ycentroid, people[x].xcentroid, people[x].ycentroid)

            print("comparing input x:", xcentroid , " y:", ycentroid," with people ", people[x].id, " position x: ", people[x].xcentroid," y :",people[x].ycentroid , " : ", mindistval )

            fakelistval = 9999


            # minimun difference in histogram

            minhist = CompareHist(histogramvalue,people[x].histogramvalue)
            #print(x, " hist is ", minhist)
            minListhist.append(minhist)

            # For debugging purposes
            if 1:
                fake = people[x].GetFake()
                fakelistval = EuDist(xcentroid, ycentroid, fake[0], fake[1])

                print("comparing input x:", xcentroid, " y:", ycentroid, " with FAKE people ", people[x].id, " position x: ",
                      people[x].fakexcentroid, " y :", people[x].fakeycentroid," : ", fakelistval )

            # minimun difference in area size


            difference, greater = AreaDiff(width, height, people[x].width, people[x].height,posx,posy,people[x].posx,people[x].posy,1)
            print("comparing areas.Input w :",(width - posx), ". h :", height - posy, ". people ",people[x].id," w:",people[x].rwidth,". h: ",people[x].rheight,". Difference:",difference)
            mean = (mean + (people[x].rheight))/2


            # YOLO sometimes had errors when it draws two bounding boxes for the same person. This should help to fix this error that NMS should have solved.
            if ((people[x].updated == 1)) or ((KalmanPerson == 1)and(framespassed - people[x].lastframe > LimitPersonKalmanUpdate)):


                #and(mindistval > mean/8)
                print("person",people[x].id," dist changed to 9999 for not being updated : 10 < ",framespassed - people[x].lastframe)
                print("person",people[x].id," dist changed to 9999 for being updated and with a distance> ",mindistval, " > ", mean/8)
                mindistval = 9999
                difference = 9999

            mindistval = min(mindistval, fakelistval)
            ("person", people[x].id," min dist is ",mindistval)

            minListdist.append(mindistval)
            minDiff.append(difference)

        themindist = min(minListdist)
        theminhist = min(minListhist)
        minHistid = np.argmin(minListhist)


        minDistid = np.argmin(minListdist)

        minAreaid = np.argmin(minDiff)

        personid = minAreaid
        onemayokey = 0
        secondmayokey = 0

        #print("likely hist, ", histogramvalue, ", real hist", people[minHistid].histogramvalue)

        print(" area id ",minAreaid,",minDistid",minDistid)

        if (minListdist[minAreaid] < mean/3.1):
            print( " case minlistdist[minAreaid = ",minAreaid ,"], with value ",minListdist[minAreaid]," lower than mean/3.5 ",mean/3.5 )
            Isnew = 0

            personid = minAreaid

            #print(" min 1 case. person ",personid)


            onemayokey = 1

        if (themindist < minListdist[minAreaid]) and (minDiff[minDistid]<50): # and (minDiff[minDist]) < 33:
            Isnew = 0
            print( " case minlistdist[minareaid], with value ",minListdist[minAreaid]," higher than mindist ",themindist )
            print(" mindist area difference is ",minDiff[minDistid])

            personid = minDistid

            secondmayokey = 1

        # This is not really used, given that only of the conditions works right now. We maintain this section for future updates.
        if (onemayokey ==1) and (secondmayokey) == 1:

            histcomparisonAreaid = CompareHist(people[minAreaid].histogramvalue,histogramvalue)
            histcomparisonDistid = CompareHist(people[minDistid].histogramvalue,histogramvalue)
            print(" Hist of person in the areid pos is ",histcomparisonAreaid)
            print(" Hist of person in the distid pos is ", histcomparisonDistid)

            if (histcomparisonAreaid < histcomparisonDistid):
                personid = minAreaid
            else:
                personid = minDistid



        # In case minimun threshold for update is not working, we will create a new object.
        if (onemayokey ==0) and (secondmayokey == 0):

            Isnew = 1

        if Isnew == 0:

            people[personid].updatevalue(posx, posy, width, height,framespassed,histogramvalue)
            print("\n")
            #print("check, histmin is person is ", minHistid)
            print("person ",people[personid].id, " updated")
            print("\n")


            return(people[personid].id)

        elif Isnew == 1:

            print("Denied, difference : ",themindist, "and", minListdist[minAreaid]," greater than ", mean/3)
            newperson = person(posx, posy, width, height, totalpeople, 1,framespassed,histogramvalue)
            people.append(newperson)
            print("\n")
            print("person ",newperson.id, "created")
            print("\n")
            return newperson.id



class backpack(thing):

    def __init__(self, posx , posy ,width, height,id,updated,framepassed,histogramvalue):
       # print("Im a backpack")
        global totalbackpack
        self.posx = posx
        self.posy = posy
        self.width = width
        self.height = height
        self.rwidth = width - posx
        self.rheight = height - posy
        self.xcentroid = round(self.posx + (self.rwidth) / 2)
        self.ycentroid = round(self.posy + (self.rheight) / 2)
        self.fakexcentroid = round(self.posx + (self.rwidth) / 2)
        self.fakeycentroid = round(self.posy + (self.rheight) / 2)

        self.id = id
        self.updated = updated
        self.prevlastframe = framepassed
        self.lastframe = framepassed
        self.near = []
        self.lostdistance = []
        self.linked = 0
        self.abandoned = 0
        self.personlinked = "none"
        self.status = neutralbagcolor
        self.defaultstatus = neutralbagcolor
        self.leftsided = 0
        self.distancetoownerx = 0
        self.distancetoownery = 0
        self.histogramvalue = histogramvalue
        self.extradistance = 0

        self.speed = np.array([[np.float32(0)], [np.float32(0)]])

        totalbackpack += 1


    #def __del__(self):
     #       global totalbackpack
      #      totalbackpack -=1
       #     print(totalbackpack)


    def DetermineNewOrUpdate(posx, posy, width, height, bags, framespassed, people, img, histogramvalue):
        xcentroid = round(posx + (width - posx) / 2)
        ycentroid = round(posy + (height - posy) / 2)
        # cv2.circle(img, (xcentroid, ycentroid), 5, (0, 0, 255), -1)
        Isnew = 0
        minlistval = 0
        fakelistval = 0
        minList = []
        minDiff = []
        global totalbackpack
        global changed2
        global mean
        Array_length = len(bags)

        if (changed2 != 1) and (EuDist(xcentroid, ycentroid, bags[0].xcentroid, bags[0].ycentroid) > 5):
            # print ( "update de la bag 0 por tener distancia ", (EuDist(xcentroid, ycentroid, bags[0].xcentroid, bags[0].ycentroid)))
            # print(bags[0].id, " updated")
            bags[0].updatevalue(posx, posy, width, height, framespassed, histogramvalue)
            changed2 = 1
            return (bags[0].id)

        AlternativeOwnerDist = 0
        AlternativeOwner = 9999
        breakfor = 0

        for x in range(Array_length):

            # minimun difference in distance
            if mean == 0:
                mean = img.shape[0]/10
            print("comparing input x:", xcentroid, " y:", ycentroid, " with bag ", bags[x].id, " position x: ", bags[x].xcentroid," y :",bags[x].ycentroid, " updated status :",bags[x].updated)
            # cv2.circle(img, (bags[x].xcentroid, bags[x].ycentroid), 5, (255, 0, 0), -1)

            minlistval = EuDist(xcentroid, ycentroid, bags[x].xcentroid, bags[x].ycentroid)
            #print(minlistval, " with bag", bags[x].id)
            fakelistval = 9999

            # minimun difference in area size

            # print (minlistval)
            difference, greater = AreaDiff(width, height, bags[x].width, bags[x].height, posx, posy, bags[x].posx,
                                           bags[x].posy, 1)
            # print("comparing areas.Input w :",(width - posx), ". h :", height - posy, ". people ",people[x].id," w:",people[x].rwidth,". h: ",people[x].rheight,". Difference:",difference)
            minDiff.append(difference)
            # print("\n")
            # cv2.circle(img, (people[x].xcentroid, people[x].ycentroid), 6, (255-(5*x), 0, 0), -1)

            minhist = CompareHist(histogramvalue, bags[x].histogramvalue)
            #print(x, " hist is ", minhist)

            if bags[x].lastframe < (framespassed - 2):
                print("comparing input x:", xcentroid, " y:", ycentroid, " with FAKE bag ", bags[x].id, " position x: ",
                      bags[x].fakexcentroid, " y :", bags[x].fakeycentroid)
                # cv2.circle(img, (bags[x].xcentroid, bags[x].ycentroid), 5, (255, 255, 0), -1)
                fakelistval = EuDist(xcentroid, ycentroid, bags[x].fakexcentroid, bags[x].fakeycentroid)

                minhist = CompareHist(histogramvalue, bags[x].histogramvalue)
                #print(x, " fake hist is ", minhist)

            if (bags[x].linked == 1) and bags[x].lastframe < (framespassed - 10):

                personlinked = CheckPosition(people, bags[x].personlinked)

                DistanceToOwner = 999

                if personlinked != None:

                    DistanceToOwner = EuDist(xcentroid, ycentroid, people[personlinked].xcentroid,
                                             people[personlinked].ycentroid)

                    #print(" distance to owner ", personlinked, ":", DistanceToOwner, " with rwidth",
                          #people[personlinked].rwidth)

                    if DistanceToOwner <= people[personlinked].rwidth:

                        if AlternativeOwnerDist == 0:
                            AlternativeOwnerDist = DistanceToOwner
                            AlternativeOwner = x

                        else:
                            if AlternativeOwnerDist > DistanceToOwner:
                                AlternativeOwner = x
                                AlternativeOwnerDist = DistanceToOwner

                # we try to check if there is a possibility of having a fake centroid that can potentially be a bag
                # detected by yolo

            print(fakelistval, " with fake bag", bags[x].id)
            minlistval = min(minlistval, fakelistval)
            if (bags[x].updated == 1) and (minlistval > mean/4):
                minlistval = 9999
            minList.append(minlistval)
            # print(minlistval)

        themin = min(minList)
        bagid = np.argmin(minList)

        difference, greater = AreaDiff(width, height, bags[bagid].width, bags[bagid].height, posx, posy,
                                       bags[bagid].posx,
                                       bags[bagid].posy, 1)

        '''


        if difference > 35:
            print(" difference with the min ", theminpos, " is ", difference," which is over 30")
            minList[theminpos]= 9999
            themin = min(minList)
            theminpos = np.argmin(themin)
            print(" new min is ",theminpos," with dist ",themin)


        RealOwnerClose = 0

        if bags[theminpos].linked == 1:

            personlinked = CheckPosition(people, bags[theminpos].personlinked)

            DistanceToOwner = EuDist(xcentroid, ycentroid, people[personlinked].xcentroid, people[personlinked].ycentroid)

            if DistanceToOwner<= people[personlinked].rwidth:

                print("for mindist, real owner is close")

                RealOwnerClose = 1

        if (RealOwnerClose == 0) and (theminpos!= AlternativeOwner) and (AlternativeOwnerDist > 0 ) and (AlternativeOwnerDist < mean/3) and(themin >1)  :

            print("dumped:  instead of" ,np.argmin(minList)," dist", themin," is ", AlternativeOwner," dist ", minList[AlternativeOwner])
            bagid = AlternativeOwner
            bags[bagid].framespassed = framespassed
            bags[bagid].updated = 1
            bags[bagid].GetFakeCentroid(xcentroid,ycentroid)
            return(bags[bagid].id)
        '''

        print("mean is ",mean/3)
        if (themin < mean / 3) and (bags[bagid].updated == 0):
            Isnew = 0
        elif themin >= mean / 3:
            Isnew = 1

        if Isnew == 0:

            bags[bagid].updatevalue(posx, posy, width, height, framespassed, histogramvalue)
            print("Bag ", bags[bagid].id, " updated")

            xcentroid = round(posx + (width- posx) /2)
            yycentroid = round(posy + (height- posy) /2)

            bags[bagid].GetFakeCentroid(xcentroid,ycentroid)

            return (bags[bagid].id)

        elif Isnew == 1:

            newbags = backpack(posx, posy, width, height, totalbackpack, 1, framespassed, histogramvalue)
            bags.append(newbags)
            print("Bag ", newbags.id, "created")
            return newbags.id


    def LastUpdate(framespassed,bags, limit, label):

        Lent = len(bags)

        Array_length = int(Lent)

        #print(Array_length)

        for b in range(Array_length):
            #print(b)
            personremoved = 0
            printedremove = "NOTHING"
            #print("bag ", bags[b].id, " last frame is ", bags[b].lastframe)
            position = 0
            #17 frames era buen valor
            if (framespassed - bags[b].lastframe > limit):
                personremoved = 1
                printedremove ="bag " + str(bags[b].id) + " removed"
                #print("\n")
                #print(printedremove)
                #position = people[b].id
                position = b
                b = b - 1
                return (printedremove, personremoved, position)


        return (printedremove,personremoved,position)
######################### OTHER FUNCTIONS ###############################


# Update de la función clase hecha, de esta no.

def DetermineNewOrUpdate(posx,posy,width,height,people,framepassed):
    Isnew = 0
    minlistval = 0
    minList = []
    global totalpeople
    global changed
    Array_length = len(people)
    

    if (changed != 1) and (EuDist(posx, posy, people[0].posx, people[0].posy) > 5):
        #print ( "update de la persona 0 por tener distancia ", (EuDist(posx, posy, people[0].posx, people[0].posy)))
        #print(people[0].id, " updated")
        people[0].updatevalue(posx, posy, width, height)
        changed = 1
        return (people[0].id)

    for x in range(Array_length):
        
        #COMPARING CENTROIDS TO DETERMINE IF IT'S THE SAME PERSON
        #print("comparing input :", posx , " with people ", people[x].id , " position : ", people[x].posx)
        minlistval = EuDist(posx, posy, people[x].posx, people[x].posy)
        minList.append(minlistval)
        #COMPARING ALSO AREAS. IF GREATER = 0, OLD > NEW. IF GREATER
        #print (minlistval)

    themin = min(minList)

    if themin < 11:
        Isnew = 0
    elif themin>=11:
        Isnew = 1


    if Isnew == 0:
        personid = np.argmin(minList)
        people[personid].updatevalue(posx, posy, width, height,framepassed)
        #print(people[personid].id, " updated")
        return (people[personid].id)

    elif Isnew == 1:
        newperson = person(posx, posy, width, height, totalpeople,1,framepassed)
        people.append(newperson)
        #print(newperson.id, "created")
        return (newperson.id)








