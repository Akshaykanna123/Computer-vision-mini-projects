import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

num_labels, labels = cv2.connectedComponents(thresh)

print("Number of objects:", num_labels - 1)

plt.subplot(1,2,1)
plt.title("Binary Image")
plt.imshow(thresh, cmap='gray')

plt.subplot(1,2,2)
plt.title("Labeled Image")
plt.imshow(labels, cmap='jet')

plt.show()
