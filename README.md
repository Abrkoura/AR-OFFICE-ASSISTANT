# AR-OFFICE-ASSISTANT
Co-working spaces: plug in to your customized workspace,
an AI Powered AR Character in Unity with AR Foundation.

<B>The aim of this project is to:<B>

Aiming to develop an AR office assistant, the project is centered around exploring core notions about the future development of workspaces, what being a digital nomad means presently and how technology can better assist the users of co-working spaces.


<B>The objectives of this project is to:<B>

MEMORIZE user habits and work patterns

SYNC to calender and assist work-session

PROJECT customized workspace

![](images/3.png)

<B>WORKFLOW:<B>

![](images/FLOW.jpg)

<B>DATA COLLECTION & LOGGING SETUP<B>

1/Defining SET OF ACTIVITIES to work with

![](images/activities.jpg)

2/Selecting RANGE OF OBJECTS to detect

![](images/dataset.jpg)

3/Defining WORKSPACE BOUNDARIES

![](images/defining%20space.jpg)

4/Recording a series of WORKSPACE TASKS

![](images/raw1.gif) ![](images/raw4.gif)

5/Performing OBJECT DETECTION on dataset (Darknet Neural network, Yolo + OpenCV)

![](images/od1.gif) ![](images/od4.gif)

6/Define a workspace boundaries by marking the corners:
Size of the workspace: 800 x 650 mm 

Make a matrix of this bounding surface and define the position of each tracked object in this matrix over time (optical flow)

![](images/of2.gif) ![](images/of3.gif) 

then return for each activity and work patterns 

![](images/of1.gif) ![](images/5.png)

Log work patterns & log anchor points for the working space 

This is the data collection output to Unity. 


<B> AR UNITY APP <B>

Set up main APP framework.

![](images/AR.jpg)
![](images/Optical%20flow%20homography.jpg)
![](images/IBM%20cloud.jpg)
![](images/8.png)
![](images/8.gif)
![](images/unity1.gif)
