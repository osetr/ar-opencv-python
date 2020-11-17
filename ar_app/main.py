from classifier import Classifier
import cv2
import numpy as np

# available descriptor for Classifier
from descriptors import ORB_Descriptor, SIFT_Descriptor

# available classification models for Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier


model = Classifier(descriptor=SIFT_Descriptor())
model.fit("with_object", "without_object")
# model.process_video("video_to_check.MOV", (640, 480), fps=30)

cap = cv2.VideoCapture(2)
cover_img = cv2.imread("img.jpg")
myVid = cv2.VideoCapture("video.mp4")

video_frame = cv2.imread("our_team_photo.jpg")
height, width, _ = cover_img.shape
video_frame = cv2.resize(video_frame, (width, height))

method = model.descriptor
method.compute(cover_img)
kp1, des1 = (method.points, method.descriptors)

bf = cv2.BFMatcher()

# fourcc = cv2.VideoWriter_fourcc(*"FMP4")
# out = cv2.VideoWriter(
#     "results/augmenting_reality.avi",
#     fourcc,
#     20,
#     (640, 480)
# )

while True:
    sucess, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    method.compute(imgAug)
    kp2, des2 = (method.points, method.descriptors)

    try:
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if model.predict(imgWebcam):
            srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
            pts = np.float32(
                [[0, 0], [0, height], [width, height], [width, 0]]
            ).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            imgWarp = cv2.warpPerspective(
                video_frame, matrix, (imgWebcam.shape[1], imgWebcam.shape[0])
            )

            maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
            cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
            maskInv = cv2.bitwise_not(maskNew)
            imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
            imgAug = cv2.bitwise_or(imgWarp, imgAug)
    except:
        # print("Bad frame")
        pass

    # out.write(imgAug)
    cv2.imshow("AugmentedReality", imgAug)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
