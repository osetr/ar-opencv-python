from abc import ABC, abstractmethod
import cv2


class Descriptor:
    def __init__(self):
        self.algorithm = None
        self._d = None
        self._k = None

    @abstractmethod
    def compute(self, img):
        pass

    @property
    def points(self):
        return self._k

    @property
    def descriptors(self):
        return self._d

    @property
    @abstractmethod
    def name(self):
        pass


class ORB_Descriptor(Descriptor):
    def __init__(self, nfeatures=1000):
        self.desc_size = 32
        self.algorithm = cv2.ORB_create(nfeatures=nfeatures)
        self.nfeatures = nfeatures

    def compute(self, img):
        self._k, self._d = self.algorithm.detectAndCompute(img, None)

    @property
    def name(self):
        return "orb"


class SIFT_Descriptor(Descriptor):
    def __init__(self, nfeatures=1000):
        self.desc_size = 128
        self.algorithm = cv2.SIFT_create()
        self.nfeatures = nfeatures

    def compute(self, img):
        self._k, self._d = self.algorithm.detectAndCompute(img, None)

    @property
    def name(self):
        return "sift"


if __name__ == "__main__":
    smpl = ORB_Descriptor()
    # smpl = SIFT_Descriptor()
    img = cv2.imread("frames/frame1.jpg")
    smpl.compute(img)
    print(smpl.descriptors)
