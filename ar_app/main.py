from classifier import Classifier
import cv2


model = Classifier()
model.fit("with_object", "without_object")
model.process_video("video_to_check.MOV", (640, 480), fps=30)
