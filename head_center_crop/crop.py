import cv2
import numpy as np
from typing import Tuple
import os

from .detect import AnimeHeadDetector, HumanHeadDetector, HeadDetector
from .utils import xywh2xyxy, xyxy2xywh


class HeadCenterCrop:
    def __init__(self, 
                 detector: HeadDetector, 
                 output_size: Tuple[int, int]=(256, 256), #w, h
                 target_size: Tuple[float, float]=(0.5, 0.5)):
        """
            Align head to center
        """
        self.detector = detector
        self.output_size = np.array(output_size).transpose()
        self.target_size = np.array(target_size) * self.output_size

    
    def _get_area(self, image: np.ndarray, use_bgr: bool = False) -> np.ndarray:
        C = image.shape[2]
        im = image.copy()
        if use_bgr:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB if C < 4 else cv2.COLOR_BGRA2RGB)

        box = self.detector(im)
        box = xyxy2xywh(box)
        
        return box

    
    def resize(self, image: np.ndarray, manual_area: np.ndarray = None, use_bgr: bool = False) -> np.ndarray:
        num_channels = image.shape[-1]
        if num_channels <= 3:
            result = np.ones((*self.output_size, num_channels), dtype=np.uint8) * 255
        else:        
            result = np.zeros((*self.output_size, num_channels), dtype=np.uint8)
        
        x, y, w, h = self._get_area(image, use_bgr) if manual_area is None else manual_area

        # resize image
        ratio = min(self.target_size[0]/w, self.target_size[1]/h)
        resized = cv2.resize(image.copy(), (0, 0), fx=ratio, fy=ratio)


        # translate image (x, y) -> (ratio*x, ratio*y) -> (output_size[0]/2, output_size[1]/2)
        RH, RW = resized.shape[:2]
        H, W = self.output_size
        x, y = ratio * x, ratio * y

        x1, y1 = int(W/2 - x), int(H/2 - y)

        print(f'{resized.shape=}, {self.output_size=}, {x=}, {y=}, {x1=}, {y1=}')

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

        print(f'{RW=}, {RH=}, {resized.shape=}, {x1=}, {y1=}')

        result[y1:y1+RH, x1:x1+RW] = resized

        return result
    
    def crop_image(self, file: str, save_path: str) -> bool:
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
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
    def __init__(self,
                 margin_x: Tuple[int, int]=(0.5, 0.5), 
                 margin_y: Tuple[int, int]=(1, 0),
                **kwargs):
        super().__init__(AnimeHeadDetector(margin_x, margin_y), **kwargs)


class HumanHeadCenterCrop(HeadCenterCrop):
    def __init__(self,
                 margin_x=(0.2, 0.2), 
                 margin_y=(0.5, 0),
                 **kwargs):
        super().__init__(HumanHeadDetector(margin_x, margin_y), **kwargs)

