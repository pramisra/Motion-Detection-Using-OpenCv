import cv2

cap = cv2.VideoCapture('vtest.mp4')

_,intialFrame = cap.read()
_,nextFrame = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(intialFrame,nextFrame)
    grayScale = cv2.cvtColor(diff,cv2.COLOR_BGRA2GRAY)
    blur = cv2.GaussianBlur(grayScale,(5,5),0)
    _,threshold = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilate = cv2.dilate(threshold,None,iterations=3)
    contours,_= cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour)<900:
            continue
        cv2.rectangle(intialFrame,(x,y),(x+w,y+h),(0,200,0),2)
    cv2.imshow('Motion-Detection',intialFrame)
    intialFrame = nextFrame
    _,nextFrame = cap.read()
    if cv2.waitKey(30)==27:
        break

cv2.destroyAllWindows()
cap.release()