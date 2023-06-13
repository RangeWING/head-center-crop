from typing import Any
import cv2
import numpy as np
from functools import partial
import face_recognition
from anime_face_detector import create_detector


class HeadDetector:
    def __init__(self):
        raise Exception("Abstract class -- cannot instantiate")
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.detect(image)

    def detect(self, image: np.ndarray) -> np.ndarray: #xyxy
        im = image.copy()
        im = self._preprrocess(im)
        preds = self._detect(im)
        return self._postprocess(preds)

    def _preprrocess(self, image: np.ndarray) -> np.ndarray:
        if image.shape[2] > 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image

    def _detect(self, image: np.ndarray) -> np.ndarray:
        preds = self.detector(image)
        if len(preds) < 1:
            return np.array([])
        return preds
    
    def _postprocess(self, preds: np.ndarray) -> np.ndarray:
        return preds


class AnimeHeadDetector(HeadDetector):
    def __init__(self):
        self.detector = create_detector('yolov3')

    def _postprocess(self, preds: np.ndarray) -> np.ndarray:
        keypoints = preds[0]['keypoints']
        pt1 = keypoints.min(axis=0)[:2]
        pt2 = keypoints.max(axis=0)[:2]
        return np.concatenate([pt1, pt2])


class HumanHeadDetector(HeadDetector):
    def __init__(self):
        self.detector = partial(face_recognition.face_locations, model='hog')

    def _postprocess(self, preds: np.ndarray) -> np.ndarray:
        top, right, bottom, left = preds[0]
        return np.array([left, top, right, bottom])
    