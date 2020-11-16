import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from descriptors import ORB_Descriptor, SIFT_Descriptor, BRIEF_Descriptor
from sklearn.tree import DecisionTreeClassifier
import os


class Classifier:
    def __init__(self, descriptor=ORB_Descriptor, nfeatures=500, classifier = DecisionTreeClassifier()):
        self.descriptor = descriptor()
        self.nfeatures = nfeatures
        self.classifier = classifier
        self.model = None

    def fit(self, path_to_dir_with_obj, path_to_dir_without_obj):
        pure_data = []
        y = []

        for ind, dir in enumerate([path_to_dir_without_obj, path_to_dir_with_obj]):
            list_of_files = os.listdir(dir)
            for file in list_of_files:
                img = cv2.imread(dir + "/" + file)
                self.descriptor.compute(img)
                descriptors = self.descriptor.descriptors

                try:
                    matches = np.zeros((self.nfeatures, self.descriptor.desc_size))
                    for i in range(min(len(descriptors), len(matches))):
                        matches[i, :] = descriptors[i, :]
                    pure_data.append(matches.ravel() / 256)
                    y.append(ind)
                except:
                    print(f"Bad frame!")
        pure_data = np.array(pure_data)
        y = np.array(y)

        X_train, X_test, Y_train, Y_test = train_test_split(
            pure_data, y, random_state=0, test_size=0.5
        )

        model = self.classifier
        model.fit(X_train, Y_train)
        self.model = model
        return model

    def predict(self, frame):
        try:
            self.descriptor.compute(frame)
            descriptors = self.descriptor.descriptors
            dest_matches = np.zeros((self.nfeatures, self.descriptor.desc_size))
            for i in range(min(len(descriptors), len(dest_matches))):
                dest_matches[i, :] = descriptors[i, :]
            pure_data = dest_matches.ravel() / 256
            return self.model.predict(np.expand_dims(pure_data, axis=0))
        except:
            return 0

    def process_video(self, path_to_video, output_size, fps=30):
        cap = cv2.VideoCapture(path_to_video)
        images = []

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                if self.predict(frame):
                    text_to_show = "I see"
                else:
                    text_to_show = "I can't see"
                cv2.putText(
                    frame,
                    text_to_show,
                    (20, 120),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=5,
                    color=(255, 255, 255),
                )

                output_img = cv2.resize(frame, output_size)
                images.append(output_img)
            else:
                break

        fourcc = cv2.VideoWriter_fourcc(*"FMP4")
        out = cv2.VideoWriter(
            self.descriptor.name + "_out.avi", fourcc, fps, output_size
        )
        for frame in images:
            out.write(frame)
        cap.release()
        out.release()
        cv2.destroyAllWindows()
