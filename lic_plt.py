import cv2
import matplotlib.pyplot as plt

img = cv2.imread('car.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 200)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    
  
    if 2 < w/h < 6:  
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        break

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("License Plate Detection")
plt.axis('off')
plt.show()
