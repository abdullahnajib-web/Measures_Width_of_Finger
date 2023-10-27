# python Measures_Width_of_Finger.py --image finger.png --width 25.4


import mediapipe as mp
import imutils
from imutils import perspective
from imutils import contours
import cv2
import numpy as np
import argparse
from scipy.spatial import distance as dist
import math


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def calSize(xA,yA,xB,yB,warnaLingkaran,warnaGaris):

	d = dist.euclidean((xA, yA), (xB, yB))
	global pixelsPerMetric
	if pixelsPerMetric is None:
		pixelsPerMetric = d / args["width"]
        
	dim = d / pixelsPerMetric

	cv2.circle(orig, (int(xA), int(yA)), 5, warnaLingkaran, -1)
	cv2.circle(orig, (int(xB), int(yB)), 5, warnaLingkaran, -1)
	cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),warnaGaris, 2)
	cv2.putText(orig, "{:.1f}".format(dim),
		(int(xA - 15), int(yA - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)


mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils


handsLM = mpHands.Hands(max_num_hands=1,min_detection_confidence=0.8,min_tracking_confidence=0.8)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image ")
args = vars(ap.parse_args())


pixelsPerMetric = None


image = cv2.imread(args["image"])
orig = image.copy()


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)



# find contours in thresholded image, then grab the largest
# one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

#grab_contours：return the actual contours array
cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts2, _) = contours.sort_contours(cnts)


cnts=()

for c in cnts2:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 100:
        continue
    cnts=cnts+(c,)

    
box = cv2.minAreaRect(cnts[0])
box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box
(tltrX, tltrY) = midpoint(tl, tr)
(blbrX, blbrY) = midpoint(bl, br)
(tlblX, tlblY) = midpoint(tl, bl)
(trbrX, trbrY) = midpoint(tr, br)
dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
if dA < dB:
    xA,yA,xB,yB =tltrX, tltrY,blbrX, blbrY
else:
    xA,yA,xB,yB =tlblX, tlblY,trbrX, trbrY


calSize(xA,yA,xB,yB,(255, 0, 0),(255, 0, 255))


#get the contour with the max area
c = max(cnts, key=cv2.contourArea)

# Get the palm defects
hull=cv2.convexHull(c,returnPoints=False)

#defects is a Three-dimensional array: N * 1 * 4
#defects[0]=N，defects[n][0]: start point index,end point index, far Point index, far distance
#start/end/far point index in contour
defects=cv2.convexityDefects(c,hull)

# convert the N*1*4 array to N*4
defectsSort = np.reshape(defects,(defects.shape[0],defects.shape[2]))
#sort the new N*4 by the distance from small to large
defectsSort = defectsSort[np.argsort(defectsSort[:,3]), :]
#get 6 largest distance elements in defects. Take them as the effect segment point of finger
defectsSort = defectsSort[(defects.shape[0] - 6):]

#get the finger roughly area
sPts=[]
ePts=[]
fPts=[]

for i in range(defectsSort.shape[0]):
    #start Point, endPoint, far Point, the depth of far point to convexity
    s,e,f,d=defectsSort[i]
    sPts.append(tuple(c[s][0]))
    ePts.append(tuple(c[e][0]))
    fPts.append(tuple(c[f][0]))

sPts = np.array(sPts)
ePts = np.array(ePts)
fPts = np.array(fPts)

# sort the sPts/ePts/fPts from left to right based on fPts x-coordinates
sPtsSort = sPts[np.argsort(fPts[:, 0]), :]
ePtsSort = ePts[np.argsort(fPts[:, 0]), :]
fPtsSort = fPts[np.argsort(fPts[:, 0]), :]
mPtsSort = np.floor(np.add(sPtsSort,ePtsSort)/2)

#get exact finger area
proimage = thresh.copy()
ROI = np.ones(thresh.shape, np.uint8)
#imgroi = np.ones(thresh.shape, np.uint8)
 
for index in range(len(fPtsSort) - 1):
    nIndex = index + 1   
    finger = [fPtsSort[index],mPtsSort[index],sPtsSort[index],ePtsSort[nIndex],mPtsSort[nIndex],fPtsSort[nIndex]]
    finger = np.array(finger,np.int32)    
    cv2.drawContours(ROI, [finger],-1,(255,255,255),-1)      
    imgroi= cv2.bitwise_and(ROI,proimage)
    #cv2.imshow('ROI',ROI)
    #cv2.imshow('imgroi_bt',imgroi)
    #cv2.waitKey(0)



imgroi = cv2.threshold(imgroi, 45, 255, cv2.THRESH_BINARY)[1]
imgroi = cv2.erode(imgroi, None, iterations=2)      


# roiCnts = cv2.findContours(imgroi, cv2.RETR_EXTERNAL,
	# cv2.CHAIN_APPROX_SIMPLE)
# roiCnts = imutils.grab_contours(roiCnts)
roiCnts,hierarchy = cv2.findContours(imgroi, cv2.RETR_EXTERNAL,
	 cv2.CHAIN_APPROX_SIMPLE)


imgH,imgW,imgC = image.shape


frame = image
frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

pr = handsLM.process(frame2)
if pr.multi_hand_landmarks:    
    for hand_landmarks in pr.multi_hand_landmarks:        
        hand = hand_landmarks
        #mpDraw.draw_landmarks(orig,hand,mpHands.HAND_CONNECTIONS)
        lmlist = []
        for id,landMark in enumerate(hand.landmark):
            xPos,yPos,zPos = int(landMark.x*imgW),int(landMark.y*imgH),int(landMark.z*imgW)
            lmlist.append([id,xPos,yPos])
        jariS=(
              [lmlist[3][1],lmlist[3][2],lmlist[2][1],lmlist[2][2]],
              [lmlist[6][1],lmlist[6][2],lmlist[5][1],lmlist[5][2]],
              [lmlist[10][1],lmlist[10][2],lmlist[9][1],lmlist[9][2]],
              [lmlist[14][1],lmlist[14][2],lmlist[13][1],lmlist[13][2]],
              [lmlist[18][1],lmlist[18][2],lmlist[17][1],lmlist[17][2]]
              )
        if len(lmlist)!=0:
            for cnt in roiCnts:
                for jari in jariS:
                    result = cv2.pointPolygonTest(cnt, (jari[0],jari[1]), False)
                    if result > 0:
                        m=(jari[1]-jari[3])/(jari[0]-jari[2])
                                                
                        
                        m=-1/m
                        a=jari[1]-m*jari[0]
                        
                        x2 = x1 = jari[0]
                        y1 = m*(x1)+a
                        y2 = m*(x2)+a        
                        result = 1.0
                        while result > 0:
                            result = cv2.pointPolygonTest(cnt, (x1,y1), False)
                            x1=x1+1
                            y1 = m*(x1)+a
                        x1 = x1 -1
                        y1 = m*(x1)+a
                        
                        result = 1.0
                        while result > 0:
                            result = cv2.pointPolygonTest(cnt, (x2,y2), False)
                            x2=x2-1
                            y2 = m*(x2)+a                        
                        x2=x2+1
                        y2 = m*(x2)+a                        
                                                
                        calSize(x1,y1,x2,y2,(255, 0, 0),(255, 0, 255))
                                                

cv2.imshow("tangan",orig)
cv2.imwrite("contour.png",imgroi)
cv2.imwrite("result.png",orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
