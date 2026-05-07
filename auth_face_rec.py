import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('auth.jpg', 0)
img2 = cv2.imread('test.jpg', 0)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

if len(matches) > 20:
    print("Authorized Person")
else:
    print("Unauthorized Person")

img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None)

plt.imshow(img_match)
plt.title("Feature Matching")
plt.axis('off')
plt.show()
