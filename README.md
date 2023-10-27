# Measures Width Of Finger To Select Ring Size Using Image Processing And Hand Landmarks
The project is used to choose a ring that matches to finger size, using image processing and Hand Landmarks. 
First, the program will read the image of the hand and a coin besides the hand. The actual size of the coin diameter is known which will be used as a reference for measuring width of finger. then the program will look for the contour of the coin and fingers by determining the threshold using OpenCV.
After the contour of the coin and fingers are obtained, then the program will detect hand landmarks from the image.

The hand landmarks that will be measured are at the PIP joints on all fingers, namely at points 3, 6, 10, 14 and 18.
The PIP connection was chosen because this connection is the widest of the connections the other finger so that the ring 
can fit into the finger. To calculate width of finger at the PIP joint we have to create a line equation at the joint.
Because the PIP connection only has one point on each finger, to create the equation of the line
we have to find the gradient first, say m1. To find the gradient of m1, we can obtain the mutually perpendicular gradient formula, namely m1 x m2 = -1.
The gradient m2 can be obtained because we have 2 points on each finger, namely at the PIP joint and MCP joint
at points 2, 5, 9, 13 and 17.

by using the gradient formula from two coordinate points (x1, y1) and (x2, y2), namely m2 = (y2 - y1)/(x2-x1), and
with the perpendicular gradient formula m1 x m2 = -1 we can get the gradient value m1 so we can create
equation of the line on the width of the finger with the points of the PIP joint and the gradient m1 at each point.

to obtain the boundaries of two fingertip points using the line equation that we have obtained
then we can use the contours of the fingers that we already have.

The length of the fingers that we get is still in pixels and we have to convert it to millimeters.
Here we use the Euclidean distance formula using the previous coin reference we already know the actual size

# How to Run the program
1. open the cmd.exe and enter the path where Measures_Width_of_Finger.py is located.

2. Take the finger.png as the test image, input the command and Run:
```	
    python Measures_Width_of_Finger.py --image finger.png --width 25.4
```	

3. You could change the finger.png to your own test image name.
    -image : the name of the test image.
    -width : the width of the reference(coin). The unit of the results will be the same with the width.
    
4. Please make sure that reference(coin) is left on the hand. 

5. Please make sure that the contrast between background and foreground is obvious.
