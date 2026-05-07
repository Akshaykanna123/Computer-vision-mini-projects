import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('akshay.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

seed = (100,100)
threshold = 10

segmented = np.zeros_like(gray)   

stack = [seed]
segmented[seed] = 255

while stack:
    x, y = stack.pop()

    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy

        if 0 <= nx < gray.shape[0] and 0 <= ny < gray.shape[1]:
            if segmented[nx, ny] == 0:
                if abs(int(gray[nx, ny]) - int(gray[seed])) < threshold:  # ✅ use gray
                    segmented[nx, ny] = 255
                    stack.append((nx, ny))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(gray, cmap='gray')

plt.subplot(1,2,2)
plt.title("Region Growing")
plt.imshow(segmented, cmap='gray')

plt.show()
