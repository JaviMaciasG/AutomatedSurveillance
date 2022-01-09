import numpy as np

totalpeople = 0
totalbackpack = 0


class thing:

    def showvalue(self):
        print(self.posx,self.posy)

    def updatevalue(self,posx,posy,width,height):
        self.posx = posx
        self.posy = posy
        self.width = width
        self.height = height

class person(thing):

    def __init__(self, posx , posy ,width, height,id ,exist):
        global totalpeople
       # print("Im a person")
        self.posx = posx
        self.posy = posy
        self.width = width
        self.height = height
        self.id = id
        self.exist = exist
        totalpeople += 1
       # print(totalpeople)
        if (totalpeople >98) :
            totalpeople = 0

    def __del__(self):
            global totalpeople
            totalpeople -=1
            #print(totalpeople)



class backpack(thing):
    def __init__(self, posx , posy ,width, height):
       # print("Im a backpack")
        global totalbackpack
        self.posx = posx
        self.posy = posy
        self.width = width
        self.height = height
        totalbackpack += 1


    def __del__(self):
            global totalbackpack
            totalbackpack -=1
            print(totalbackpack)

def CreateBunchOfPeople(array,alist):

    for i in range(0, array):
        newperson = person(0, 0, 0, 0, 0, 0)
        alist.append(newperson)


def EuDist(extx, exty, newextx, newexty):
    return pow(pow(newextx - extx, 2) + pow(newexty - exty, 2), 0.5)


def DetermineNewOrUpdate(object, posx, posy,width,height,people):
    Isnew = 0
    minList = []
    global totalpeople

    Array_length = len(object)
    for i in range(Array_length):
        minList.append(EuDist(posx, posy, object[i].posx, object[i].posy))
    if min(minList) < 1:
        Isnew = 0
    if Isnew == 0:
        people[np.argmin(minList)].updatevalue(posx, posy, width, height)
    elif Isnew == 1:
        newperson = person(posx, posy, width, height, totalpeople, 1)
        people.append(newperson)


''''
if __name__ == "__main__":

    people = []
    i=1
    while ( i < 25 ):
        newperson = person(0,0,0,0,0,0)
        people.append(newperson)
        i+=1


    Rober = person(5,6,1,1,30,1)
    people.append(Rober)
    Al = person(2,4,1,1,31,1)
    people.append(Al)


    apos1 = (2.1, 4.1)
    apos2 = (2.2, 4.2)

    DetermineNewOrUpdate(people,apos1[0],apos1[1])

    Array_length = len(people)
    i=0
    while (i <Array_length):
        print(people[i].id)
        people[i].showvalue()
        i+=1




    #Al.showvalue()

'''
