import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

img = cv2.imread('person.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

res = pose.process(img_rgb)

for p in res.pose_landmarks.landmark:
    h, w, _ = img.shape
    cx, cy = int(p.x*w), int(p.y*h)
    cv2.circle(img, (cx,cy), 3, (0,255,0), -1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Pose")
plt.axis('off')
plt.show()



