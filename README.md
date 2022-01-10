# Automated Surveillance

This project can be used to determine pattern recognition (abandoned object detection) and panic detector (abnormal crowd behavior). which is based on neural networks, deep learning and clustering algorithms mainly.. It is not the objective to be seen as a unique and single application but as a framework for automated surveilllance as it is described in the thesis that relies on. The thesis can be found [here](https://ebuah.uah.es/dspace/handle/10017/37872).

There is also special mention to [Darknet](https://github.com/pjreddie/darknet), an open source neural network that we use to perform detection in this application.

_________

## How to run the application

There are two main modules in this project:

   1. In one module, we have the main python wrap of YOLO and the application has been built up around it. Given that two out of three parts work with YOLO for the moment, the main file that leads this application can be found there too. This file counts with the functions that are use mainly by the Heatmap and Lost Bag Detector from this project.



https://user-images.githubusercontent.com/97040752/148806196-e3c6308e-3dd3-430f-a7c1-d30d2d6de1e9.MP4


![675](https://user-images.githubusercontent.com/97040752/148807654-21f424fd-6b1c-4272-91ef-2c27a2c07cb9.jpg)

   2. The second module takes care of abnormal crowd activity, and it contains the functions that calculate the optical flow density in the image and the procedures to train the software to enable the identification of abnormal crowd activities that provokes stampedes.



https://user-images.githubusercontent.com/97040752/148807520-2099140d-4728-4dd4-b5f2-f3c09bd95bb0.MP4


 
There are also other important modules that need a mention too because they contain important
functions to make this framework possible:

  - Kalman module contains the class based on Kalman filters that allow us to predict the position of
  the backpacks when they are not found by YOLO.
  
  -  The objects module has the classes of person and backpack, which are necessary to create the
  objects that contain the information of people and bags respectively in the videos analyzed by the
  Lost Object Detector module. This module also has functions related to update these classes and
  the function used to link backpacks to people.
  
  -  The configuration module has been created to ease the process of saving images and do other minor
  tasks that were repetitive in the process and were allocated in this file.
  
To run the application, we will need to open a console in the folder called obj/python, and it will be
necessary to have the inputs as it is shown below:

python3 mainfile.py -i /path/to/video –mode choose -kp -pb

Described with a double line, we can see the inputs we have available for our application. The possible
values that these input accept are:

  - Input(–i)
     We have to put a path that leads to the folder where we have the images of the desired video.

  - Mode(–m)
     Through mode input, we can choose the different modes available for our application. The modes
     available right now are:

           – lostobjectdetection, which activates the mode that detect people and bags and raises an alarm
           when the application finds an abandoned bag.

           – heatmap, which activates the mode draws the people’s trace. In case we have an image in the
           folder base, that image will be use to display the heatmap(this can be useful if we don’t want
           to see the traces over the people but in a empty image).

           – Stampede or StampedeT, which trains and calculates the thresholds for detecting abnormal
           crowd activity or raises an alarm when the algorithm detect that the optical flow is higher
           than the threshold calculated.
        
   -  Use kalman with people(–kp)
      Through using kalman with people, we enhance the reidentification process when YOLO does not
      detect people in the images by using kalman filters. We just need to put the –kp to activate it.
   
   -  Use kalman with bags(–kb)
      Through using kalman with bags, we enhance the reidentification process when YOLO does not
      detect people in the images by using kalman filters. We just need to put the –kp to activate it.
   
   -  Use quick position prediction with bags(–pb)
      Through using quick position prediction with bags with bags, we enhance the reidentification process
      when YOLO does not detect people in the images by roughly prediction the bag position with the
      gradient of movement of the person who is linked to the bag. We just need to put the –kp to
      activate it.
   
   -  show prediction(–sp)
      Given that sometimes it can be too much information on screen, we believe that there is no need
      to show the predictions in the images. This is actually an internal process that help us with the
      reidentification but it does not help to the person who is watching the footage. In case we want to
      watch the prediction, we should use –sp to activate it.
      To end, results will be allocated in the folder results. In the folder Heatmap, we will save temporary
      binary images in the folder temp.
      In case we put an image in the folder heatmap/base, that image will be used to draw the heatmap in
      it. If there is any image in the folder, the heatmap will be drawn always in the current image displayed.
      
      ## Additional notes
      
      It is needed to adjust the path that this repository has on your local files. As it is shown in the files, you should change "mypath" for the path that you decide to locate the repository. 
      Yolo weigths have not been added to this repository either, so you need to download those files from [Yolo`s website](https://pjreddie.com/darknet/yolo/) and place them where marked on the notes ask for them/ where the paths point to them in our application.
      
      ## Improvements to be done 
      
      - As it is seen, kalman filters have been good to make a quick prediction of the possible position of the backpacks when they are not found by YOLO. Nevertheless, not always responds good when we are continiously iterating from an unknown position. As mentioned in the documents, some inputs from the person attached to the backpack has been used, but the results can still be improved.
      
      - Paths to be global in the code.
