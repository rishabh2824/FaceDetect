#import all the required libraries
import cv2

#Load the classifier file
detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Load and read your image
imp_img = cv2.VideoCapture("elon.jpg")
res, img = imp_img.read()

#Convert image to grey scale for use with haarcascade classifier
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect images of different sizes
faces = detect.detectMultiScale(gray, 1.3, 5)

#Draw a rectange to recognise the face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

#Display the image
cv2.imshow("Elon Image", img)
cv2.waitKey(0)
imp_img.release()
cv2.destroyAllWindows()
