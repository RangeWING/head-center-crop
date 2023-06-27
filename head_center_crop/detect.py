from typing import Tuple
import cv2
import numpy as np
from functools import partial
import face_recognition
from anime_face_detector import create_detector
from .utils import expand_box


class HeadDetector:
    def __init__(self, detector, margin_x: Tuple[int, int]=(0,0), margin_y: Tuple[int, int]=(0,0)):
        self.detector = detector
        self.margins = np.stack([margin_x, margin_y]).transpose().ravel()
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.detect(image)

    def detect(self, image: np.ndarray) -> np.ndarray: #xyxy
        im = image.copy()
        im = self._preprrocess(im)
        preds = self._detect(im)
        
        if len(preds) < 1:
            return None
        
        result = self._postprocess(preds)
        result = expand_box(result, self.margins)
        return result

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
    def __init__(self, margin_x: Tuple[int, int]=(0,0), margin_y: Tuple[int, int]=(0,0)):
        super().__init__(detector=create_detector('yolov3'), 
                         margin_x=margin_x, margin_y=margin_y)

    def _postprocess(self, preds: np.ndarray) -> np.ndarray:
        if len(preds) < 1:
            return None

        keypoints = preds[0]['keypoints']
        pt1 = keypoints.min(axis=0)[:2]
        pt2 = keypoints.max(axis=0)[:2]
        box = np.concatenate([pt1, pt2])
        return box


class HumanHeadDetector(HeadDetector):
    def __init__(self, margin_x: Tuple[int, int]=(0,0), margin_y: Tuple[int, int]=(0,0)):
        super().__init__(detector=partial(face_recognition.face_locations, model='hog'), 
                         margin_x=margin_x, margin_y=margin_y)

    def _postprocess(self, preds: np.ndarray) -> np.ndarray:
        if len(preds) < 1:
            return None
        top, right, bottom, left = preds[0]
        return np.array([left, top, right, bottom])
    