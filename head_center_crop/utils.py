# input module: image or video
# head detector + cropper
# output module

import numpy as np

# box: np.ndarray[x, y, w, h]
def xywh2xyxy(box: np.ndarray) -> np.ndarray:
    result = np.zeros(4)
    result[:2] = box[:2] - box[2:]/2
    result[2:] = box[:2] + box[2:]/2
    return result


def xyxy2xywh(box: np.ndarray) -> np.ndarray:
    result = np.zeros(4) #x1 y1 x2 y2
    result[:2] = (box[:2] + box[2:]) / 2
    result[2:] = (box[2:] - box[:2])
    return result


def expand_box(box: np.ndarray, margins: np.ndarray) -> np.ndarray:
    """
        expand bbox

        *param box* xyxy
        *param margins* x1 y1 x2 y2 [x-left y-top x-right y-bottom]
    """
    wh = xyxy2xywh(box)[2:]

    result = np.zeros(4) #xyxy 
    result[:2] = box[:2] - wh * margins[:2]
    result[2:] = box[2:] + wh * margins[2:]

    return result #xyxy