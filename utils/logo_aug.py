from typing import Tuple, Dict, Any, Callable

import albumentations as A
import glob
import cv2
import random

import numpy as np


def get_logos(path):
    img_files = [file for file in glob.glob(f"{path}/*.*")]
    images = []
    for file in img_files:
        img = cv2.imread(file)
        images.append(img)
    return images


def convert_to_absolut_box(box, img_shape):
    x_min, y_min, x_max, y_max = box[:4]
    h, w = img_shape[:2]
    return (
        int(x_min * w),
        int(y_min * h),
        int(x_max * w),
        int(y_max * h),
    )


def paste_and_clip(image, r_logo, new_box):
    try:
        x_min, y_min, x_max, y_max = new_box
        x1 = max(x_min, 0)
        y1 = max(y_min, 0)
        x2 = min(image.shape[1], x_max)
        y2 = min(image.shape[0], y_max)
        image[y1:y2, x1:x2] = r_logo[(y1 - y_min):(y2 - y_min), (x1 - x_min):(x2 - x_min)]
    except Exception as e:
        print(e)
        raise e
    return image, (x1, y1, x2, y2)


class LogoAug(A.BasicTransform):
    def apply(self, img, **params) -> np.ndarray:
        pass

    @property
    def targets(self) -> Dict[str, Callable]:
        pass

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        pass

    def __init__(self, always_apply: bool = True, p: float = 1.0, logo_path: str = '/home/ilia_kiselev/Documents/IdeaProjects/yolov5/logos'):
        super(LogoAug, self).__init__(always_apply=always_apply, p=p)
        self.logo_cache = get_logos(logo_path)

    def __call__(self, *args, **kwargs):
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        image = kwargs['image']
        boxes = kwargs['bboxes']
        bb = []
        try:
            for box in boxes:
                tail = box[4:]
                a_box = convert_to_absolut_box(box, image.shape)
                if any([a_box[2] - a_box[0] < 10, a_box[3] - a_box[1] < 10]):
                    continue
                logo = random.choice(self.logo_cache)
                # scale in order to save aspect ratio
                scale = max(logo.shape[1] / (a_box[2] - a_box[0]), logo.shape[0] / (a_box[3] - a_box[1]))
                r_logo = cv2.resize(logo, (int(logo.shape[1] / scale), int(logo.shape[0] / scale)))
                x_c, y_c = int((a_box[0] + a_box[2]) / 2), int((a_box[1] + a_box[3]) / 2)
                h, w = r_logo.shape[:2]
                new_box = (
                    x_c - int(w / 2),
                    y_c - int(h / 2),
                    x_c - int(w / 2) + w,
                    y_c - int(h / 2) + h
                )
                image, new_box = paste_and_clip(image, r_logo, new_box)
                new_box = (
                    new_box[0] / image.shape[1],
                    new_box[1] / image.shape[0],
                    new_box[2] / image.shape[1],
                    new_box[3] / image.shape[0],
                ) + tail
                bb.append(new_box)
        except Exception as e:
            print(e)
            return kwargs
        return {
            "image": image,
            "bboxes": bb
        }
