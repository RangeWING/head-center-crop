import cv2
import numpy as np
from typing import Tuple
import os

from detect import AnimeHeadDetector, HumanHeadDetector, HeadDetector

# box: np.ndarray[x, y, w, h]
def xywh2xyxy(box: np.ndarray):
    result = np.zeros(4)
    result[:2] = box[:2] - box[2:]/2
    result[2:] = box[:2] + box[2:]/2
    return result


def xyxy2xywh(box: np.ndarray):
    result = np.zeros(4) #x1 y1 x2 y2
    result[:2] = (box[:2] + box[2:]) / 2
    result[2:] = (box[2:] - box[:2]) / 2
    return result


class HeadCenterCrop:
    def __init__(self, 
                 detector: HeadDetector, 
                 target_size=(0.5, 0.5), 
                 margin_x=(0.5, 0.5), 
                 margin_y=(1, 0)):
        """
            Align head to center
        """
        self.detector = detector
        self.margins = np.stack([margin_x, margin_y]).transpose().ravel() #x1 y1 x2 y2 [x-left y-top x-right y-bottom]
        self.target_size = target_size

    def _expand_box(self, box: np.ndarray, image_size: Tuple[int, int]=(-1, -1), input_type='xyxy'):
        if input_type == 'xywh':
            box = xywh2xyxy(box)

        wh = xyxy2xywh(box)[2:]
        W, H = image_size

        result = np.zeros(4) #xyxy 
        result[:2] = box[:2] - wh * self.margins[:2]
        result[2:] = box[2:] + wh * self.margins[2:]

        if W > 0: 
            result[[0, 2]] = np.clip(result[[0, 2]], 0, W)
        
        if H > 0:  
            result[[1, 3]] = np.clip(result[[1, 3]], 0, H)

        return result #xyxy
    
    def _get_area(self, image: np.ndarray, use_bgr: bool = False) -> np.ndarray:
        H, W = image.shape[:2]
        if use_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        box = self.detector(image)
        box = self._expand_box(box, image_size=(W, H))
        box = xyxy2xywh(box)
        box[[0, 2]] /= W
        box[[1, 3]] /= H
        
        return box

    
    def resize(self, image: np.ndarray, manual_area: np.ndarray = None, use_bgr: bool = False) -> np.ndarray:
        H, W = image.shape[:2]
        
        x, y, w, h = self._get_area(image, use_bgr) if manual_area is None else manual_area

        ratio = min(self.target_size[0]/w, self.target_size[1]/h)
        #print(x, y, w, h, ratio)

        resized = cv2.resize(image.copy(), (0, 0), fx=ratio, fy=ratio)
        result = np.ones_like(image, dtype=np.uint8) * 255

        RH, RW = resized.shape[:2]
        #resized = cv2.rectangle(resized, (int((x-w/2)*RW), int((y-h/2)*RH)), (int((x+w/2)*RW), int((y+h/2)*RH)), (0, 255, 0))
        
        x1, y1 = int(W/2 - x*RW), int(H/2 - y*RH)
        if x1 < 0:
            resized = resized[:,-x1:]
            RW += x1
            x1 = 0
        if x1 + RW > W:
            resized = resized[:,:W-x1]
            RW = W - x1
        if y1 < 0:
            resized = resized[-y1:,:]
            RH += y1
            y1 = 0
        if y1 + RH > H:
            resized = resized[:H-y1,:]
            RH = H - y1

        result[y1:y1+RH, x1:x1+RW] = resized

        return result
    
    def crop_image(self, file: str, save_path: str) -> bool:
        image = cv2.imread(file)
        resized = self.resize(image, use_bgr=True)
        return cv2.imwrite(save_path, resized)
    
    def crop_video(self, file: str, save_path: str, detect_frame_index: int=0) -> bool:
        if not os.path.isfile(file):
            raise Exception(f"File {file} not exist")
        cap = cv2.VideoCapture(file)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        cap.set(cv2.CAP_PROP_POS_FRAMES, detect_frame_index)

        writer = cv2.VideoWriter(save_path,
                                 cv2.VideoWriter_fourcc(*'XVID'),
                                 fps,
                                 (width, height))

        ret, first_frame = cap.read()
        if not ret:
            if cap.isOpened():
                cap.release()
            writer.release()
            return False
        
        area = self._get_area(first_frame, use_bgr=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = self.resize(frame, area)
            writer.write(resized)
            
        if cap.isOpened():
            cap.release()
        writer.release()

        return True
        

    

class AnimeHeadCenterCrop(HeadCenterCrop):
    def __init__(self, **kwargs):
        super().__init__(AnimeHeadDetector(), **kwargs)


class HumanHeadCenterCrop(HeadCenterCrop):
    def __init__(self, **kwargs):
        super().__init__(HumanHeadDetector(), **kwargs)

