import cv2

#opening the machine learning cascadeclassifier
f_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#reading the image
img = cv2.imread("sachi.jpg",1)
g_img = cv2.imread("sachi.jpg", 0)
#rsized = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
#g_rsized = cv2.resize(g_img,(int(img.shape[1]/2),int(img.shape[0]/2)))

#face detection algorithm
face = f_cascade.detectMultiScale(g_img, scaleFactor = 1.3, minNeighbors = 5)

#making rectangle
for x,y,w,h in face:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    roi_gray = g_img[y:y+h,x:x+w]
    roi = img[y:y+h,x:x+w]
    #eye detection
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for x1,y1,w1,h1 in eyes:
        roi = cv2.rectangle(roi,(x1,y1),(x1+w1,y1+h1),(255,0,0),1)
    croped = img[y:y+h,x:x+w]
cv2.imshow("FaceDetection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
