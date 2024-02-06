import copy
from pathlib import Path
import time
from typing import List

import cv2
import numpy as np
from dlib import rectangle, shape_predictor  # type: ignore
from utils.file_utils import append_filename, load_single_img_and_bbox
from utils.base_manager import BaseManager


class DlibManager(BaseManager):
    def __init__(self, model_weights_path: Path) -> None:
        BaseManager.__init__(self)
        self.shape_predictor = shape_predictor(str(model_weights_path))

    def infer(self, im: np.ndarray, bbox_list: List[List[int]]):
        """
        dlib infers on RGB-order image
        """
        start = time.time()
        landmarks_for_one_image = []
        for bbox in bbox_list:
            x0, y0, w, h = [int(ele) for ele in bbox]
            rect = rectangle(x0, y0, x0 + w, y0 + h)
            shape = self.shape_predictor(im, rect)
            coords = self.shape_to_np(shape)  # shape = (68, 2)
            landmarks_for_one_image.append(coords)
        end = time.time()
        self.insert_time(end - start)
        return landmarks_for_one_image

    def manual_infer(
        self, img_paths: List[Path], bbox_json_paths: List[Path], need_save: bool
    ):
        """
        Inference on human-prepared images. You can save the results.
        """
        assert len(img_paths) == len(bbox_json_paths), f"Check input file number."
        for img_path, bbox_json_path in zip(img_paths, bbox_json_paths):
            img, bbox_list = load_single_img_and_bbox(img_path, bbox_json_path)
            landmarks_for_one_image = self.infer(img, bbox_list)

            if need_save:
                canvas = copy.deepcopy(img)
                for landmark, bbox in zip(landmarks_for_one_image, bbox_list):
                    x0, y0, w, h = [int(ele) for ele in bbox]
                    self.plot_dlib_landmark(canvas, landmark)
                    cv2.rectangle(
                        canvas,
                        (x0, y0),
                        (x0 + w, y0 + h),
                        (255, 0, 0),
                        2,
                        lineType=cv2.LINE_AA,
                    )

                # save figure
                cv2.imwrite(
                    str(append_filename(img_path, "dlib")),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                )
        print("DLIB is done")

    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def plot_dlib_landmark(self, img, landmark_array) -> None:
        # 顎 (Jaw: 17 points) 1 ~ 17
        jaw = landmark_array[0:17]
        # 左眉 (Left eyebrow: 5 points)  18 ~ 22
        left_eyebrow = landmark_array[17:22]
        # 右眉 (Right eyebrow: 5 points)  23 ~ 27
        right_eyebrow = landmark_array[22:27]
        # 鼻子 (Nose: 9 points) 28 ~ 31 , 32 ~ 36
        vertical_nose = landmark_array[27:31]
        horizontal_nose = landmark_array[31:36]
        # 左眼 (Left eye: 6 points)  37 ~ 42
        left_eye = landmark_array[36:42]
        # 右眼 (Right eye: 6 points)  43 ~ 48
        right_eye = landmark_array[42:48]
        # 口 (Mouth: 20 points) 49 ~ 68
        mouth = landmark_array[48:68]

        # plot
        for i in range(landmark_array.shape[0]):
            (x, y) = landmark_array[i, :]
            img = cv2.circle(img, (x, y), 0, (50, 255, 50), 5)
