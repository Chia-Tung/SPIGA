import cv2
import numpy as np
import tensorflow as tf
import time
from typing import Dict, Optional

from utils.mimic_landmarks import MimicLandmarks
from utils.base_manager import BaseManager
from utils.mp_lm_indices import lips, lefteye, righteye


class MpLandmarkWrapper(BaseManager):
    def __init__(self, model_path: str, thld=5.0, num_workers=1) -> None:
        """
        initializer for MP_FACE_MESH
        """
        BaseManager.__init__(self)
        self.interpreter = tf.lite.Interpreter(model_path, num_threads=num_workers)
        self.interpreter.allocate_tensors()

        # [0] for batch size, [1] for width, [2] for height, [3] for channel
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.thld = thld

    def process(
        self, image: np.ndarray, driver_bbox: Optional[Dict], is_padding: bool = True
    ):
        """
        get image and return face landmarks
        include pre-processing of an image, may need to change due to different model

        Args:
            image: 'np.ndarray', required
                RGB image from cv2 as input
            driver_bbox: 'Dict' or None, required
                a dictionary consist of 'bounding_box', 'class_label', and 'score'
                value of key'bounding box' here is already denormalized to image coordinate and sanitized
            is_padding: bool
                if the driver_bbox is out of the image size, then padding with black pixels

        Returns:
            results: 'MIMIC_LANDMARKS' or None
                MIMIC_LANDMARKS or None
        """
        start = time.time()
        img_height, img_width, _ = image.shape
        if driver_bbox:
            ymin, xmin, ymax, xmax = driver_bbox["bounding_box"].get_cv2_position()
            target_height = int(ymax - ymin)
            target_width = int(xmax - xmin)
            if ymin < 0 or xmin < 0 or ymax > img_height or xmax > img_width:
                if not is_padding:
                    return None, -10
                else:
                    pad_top = int(max(-ymin, 0))
                    pad_left = int(max(-xmin, 0))
                    pad_bot = int(max(ymax - img_height, 0))
                    pad_right = int(max(xmax - img_width, 0))

                    image = cv2.copyMakeBorder(
                        image,
                        top=pad_top,
                        bottom=pad_bot,
                        left=pad_left,
                        right=pad_right,
                        borderType=cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )

                    ymin += pad_top
                    xmin += pad_left
            target_face = image[
                ymin : ymin + target_height, xmin : xmin + target_width, :
            ]
            img = self.preprocess_image(target_face)

            self.interpreter.set_tensor(self.input_details[0]["index"], img)
            self.interpreter.invoke()

            output_face_landmarks = np.array(
                self.interpreter.get_tensor(self.output_details[3]["index"])[0]
            ).reshape(-1, 3)
            output_face_flag = np.array(
                self.interpreter.get_tensor(self.output_details[2]["index"])[0]
            ).reshape(-1)[0]
            output_left_eye_landmarks = np.array(
                self.interpreter.get_tensor(self.output_details[0]["index"])[0]
            ).reshape(-1, 2)
            output_right_eye_landmarks = np.array(
                self.interpreter.get_tensor(self.output_details[1]["index"])[0]
            ).reshape(-1, 2)
            output_left_iris_landmarks = np.array(
                self.interpreter.get_tensor(self.output_details[4]["index"])[0]
            ).reshape(-1, 2)
            output_right_iris_landmarks = np.array(
                self.interpreter.get_tensor(self.output_details[6]["index"])[0]
            ).reshape(-1, 2)
            output_lips_landmarks = np.array(
                self.interpreter.get_tensor(self.output_details[5]["index"])[0]
            ).reshape(-1, 2)

            output_face_landmarks[lefteye, :2] = output_left_eye_landmarks
            output_face_landmarks[righteye, :2] = output_right_eye_landmarks
            output_face_landmarks[lips, :2] = output_lips_landmarks

            output_face_landmarks = np.concatenate(
                (
                    output_face_landmarks,
                    np.pad(output_left_iris_landmarks, ((0, 0), (0, 1))),
                )
            )
            output_face_landmarks = np.concatenate(
                (
                    output_face_landmarks,
                    np.pad(output_right_iris_landmarks, ((0, 0), (0, 1))),
                )
            )

            # scale back to image scale
            output_face_landmarks[:, 0:1] = (xmax - xmin) * (
                (output_face_landmarks[:, 0:1] / self.input_details[0]["shape"][2])
            )
            output_face_landmarks[:, 1:2] = (ymax - ymin) * (
                (output_face_landmarks[:, 1:2] / self.input_details[0]["shape"][1])
            )
            output_face_landmarks[:, 2:3] = (xmax - xmin) * (
                (output_face_landmarks[:, 2:3] / self.input_details[0]["shape"][2])
            )

            output_face_landmarks[:, 0:1] = output_face_landmarks[:, 0:1] + xmin
            output_face_landmarks[:, 1:2] = output_face_landmarks[:, 1:2] + ymin

            end = time.time()
            self.insert_time(end - start)
            return MimicLandmarks(output_face_landmarks), float(output_face_flag)
        return None, -10

    def preprocess_image(self, image):
        resize_to = (
            self.input_details[0]["shape"][1],
            self.input_details[0]["shape"][2],
        )
        # padding
        # img = np.zeros([(int)(image.shape[0]*1.5),
        #                int(image.shape[1]*1.5), image.shape[2]], np.float32)
        # img[int(image.shape[0]*0.4):int(image.shape[0]*0.4)+image.shape[0],
        #     int(image.shape[1]*0.4):int(image.shape[1]*0.4)+image.shape[1], :] = image/255.
        img = image / 255.0
        img = cv2.resize(img, resize_to)
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        return img
