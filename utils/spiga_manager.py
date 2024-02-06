import copy
import warnings
from pathlib import Path
import time
from typing import List

import cv2
import numpy as np
from utils.file_utils import append_filename, load_single_img_and_bbox

from spiga.demo.visualize.plotter import Plotter
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from utils.base_manager import BaseManager

warnings.filterwarnings("ignore")


class SpigaManager(BaseManager):
    def __init__(self, dataset="wflw") -> None:
        BaseManager.__init__(self)
        self.processor = SPIGAFramework(ModelConfig(dataset))
        self._init_dataset = dataset

    def infer(self, image: np.ndarray, bbox_list: List[List[int]]):
        """
        SPIGA infers on RGB-order image
        """
        start = time.time()
        features = self.processor.inference(image, bbox_list)
        end = time.time()
        self.insert_time(end - start)
        return features["landmarks"], features["headpose"]

    def manual_infer(
        self, img_paths: List[Path], bbox_json_paths: List[Path], need_save: bool
    ) -> None:
        """
        Inference on human-prepared images. You can save the results.
        """
        assert len(img_paths) == len(bbox_json_paths), f"Check input file number."
        for img_path, bbox_json_path in zip(img_paths, bbox_json_paths):
            img, bbox_list = load_single_img_and_bbox(img_path, bbox_json_path)
            landmarks, headposes = self.infer(img, bbox_list)

            if need_save:
                canvas = copy.deepcopy(img)
                plotter = Plotter()
                for i, bbox in enumerate(bbox_list):
                    x0, y0, w, h = [int(ele) for ele in bbox]
                    landmark = np.array(landmarks[i]) # shape = (N, 2)
                    headpose = np.array(headposes[i])

                    # Plot features
                    canvas = plotter.landmarks.draw_landmarks(canvas, landmark)
                    canvas = plotter.hpose.draw_headpose(
                        canvas,
                        [x0, y0, x0 + w, y0 + h],
                        headpose[:3],
                        headpose[3:],
                        euler=True,
                    )
                    # Plot Box
                    canvas = cv2.rectangle(
                        canvas,
                        (x0, y0),
                        (x0 + w, y0 + h),
                        (255, 0, 0),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                # save figure
                cv2.imwrite(
                    str(append_filename(img_path, '_'.join(["spiga", self._init_dataset]))),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                )

        print("SPIGA is done.")
