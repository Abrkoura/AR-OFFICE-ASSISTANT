# AR-OFFICE-ASSISTANT
Co-working spaces: plug in to your customized workspace,
an AI Powered AR Character in Unity with AR Foundation.

<B>The aim of this project is to:<B>

FACILITATE development of workspace habits and accessibility to preferred tools & objects.

TO have a quick workspace set-up, smooth transition between space set-ups and user-id based intuitive workspaces.

BY publish reminders, pop-up interactive tools and mindful space sharing suggestions based on calendar activities.


<B>The objectives of this project is to:<B>

MEMORIZE user habits and work patterns

SYNC to calender and assist work-session

PROJECT customized workspace


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

5/Performing OBJECT DETECTION on dataset

![](images/od1.gif) ![](images/od4.gif)

6/Define a workspace boundaries by marking the corners:
Size of the workspace: 800 x 650 mm 

Make a matrix of this bounding surface and define the position of each tracked object in this matrix over time (optical flow)

![](images/of2.gif) ![](images/of3.gif) 

then return for each activity and work patterns 

![](images/of1.gif) 

Log work patterns & log anchor points for the working space 

This is the data collection output to Unity. 


<B> AR UNITY APP <B>

Set up main APP framework .

Get the anchor points from the data collection step and perform either QR code recognition or image detection.

Setting 4 corners will be used as the anchor point to Anchor my AR projection. 

In AR, project the matrix from the DATA collection step (grid-ify our space): 

Get user calendar integration (build this in as the front-end interface-> in the AR app frontend builder, you will need to build a series of buttons and input fields so that the user can click and sync their calendar)

Implement object detection in real-time (tensorflow lite and or Wikitude): you need to look at the matrix and tell what objects are currently in the scene, and WHERE there is open space on the matrix 

Based on activity, you can recommend that the user moves the existing objects (on-screen pointers and then optical flow tracking of the object)

For the empty space on the matrix, you can project different third party apps: calculator, drawing tool etc.

Lastly, ImplementING finger tracking and response.
