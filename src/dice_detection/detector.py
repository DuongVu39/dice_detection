from abc import ABC, abstractmethod
import cv2
import numpy as np
from sklearn import cluster
from typing import Iterable


class BaseDetector(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def detect(self, frame):
        raise NotImplementedError



class SimpleDotDetector(BaseDetector):

    def __init__(self):
        super(SimpleDotDetector, self).__init__()
        self.params = cv2.SimpleBlobDetector_Params()

        # What does this do?
        self.params.minInertiaRatio = 0.6
        self._detector = cv2.SimpleBlobDetector_create(self.params)

    def detect(self, frame):
        blobs = self._get_blobs(frame)
        dice = self._get_dice_from_blobs(blobs)

        return dice

    def _get_blobs(self, frame) -> Iterable:
        """
        Blobs are 'regions of interest' in a frame

        Args:
            frame (ndarray): image height x width x channels array of numbers representing a frame

        Returns:
            (Iterable[Blob])  Each of the elements in the returned iterable represents one of the regions of interest.
        """
        frame_blurred = cv2.medianBlur(frame, 7)
        frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
        blobs = self._detector.detect(frame_gray)

        return blobs

    def _get_dice_from_blobs(self, blobs):
        """
        Uses DBSCAN to

        Args:
            blobs ():

        Returns:

        """
        # get centroids of all blobs
        X = []
        for b in blobs:
            pos = b.pt

            if pos != None:
                X.append(pos)
        X = np.asarray(X)

        if len(X) > 0:
            # Important to set min_sample to 0, as a dice may only have one dot
            clustering = cluster.DBSCAN(eps=40, min_samples=1).fit(X)

            # Find the largest label assigned + 1, to get the number of dice found
            num_dice = max(clustering.labels_) + 1

            dice = []

            # Calculate centroid of each dice, the average between all dice's dots
            for i in range(num_dice):
                X_dice = X[clustering.labels_ == i]
                centroid_dice = np.mean(X_dice, axis=0)
                dice.append([len(X_dice), *centroid_dice])

            return dice
        else:
            return []


class SimpleNumberDetector(BaseDetector):
    def __init__(self):
        super(SimpleNumberDetector, self).__init__()

    def detect(self, frame):
        raise NotImplementedError

    def localize_number(self):
        """Determines location of dice in frame"""
        gray = self._to_gray(img)
        mser = cv2.MSER(_delta=1)
        regions = mser.detect(gray, None)
        bounding_boxes = self._get_boxes(regions)
        regions = Regions(img, bounding_boxes)
        return regions

    @staticmethod
    def _to_gray(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image

    @staticmethod
    def _get_boxes(regions):
        bbs = []
        for i, region in enumerate(regions):
            (x, y, w, h) = cv2.boundingRect(region.reshape(-1, 1, 2))
            bbs.append((y, y + h, x, x + w))

        return np.array(bbs)

    def identify_number(self):
        """Determines type of dice"""
        raise NotImplementedError



