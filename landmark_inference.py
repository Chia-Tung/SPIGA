import copy
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import deeplake
import numpy as np
from tqdm import tqdm

from utils.spiga_manager import SpigaManager
from utils.dlib_manager import DlibManager
from utils.mp_landmark_wrapper import MpLandmarkWrapper
from utils.bounding_box import Boundingbox
from utils.mp_lm_indices import CRITICAL_POINT_INDICES


def load_300W_dataset():
    """
    image is in RGB order
    """
    ds = deeplake.load("hub://activeloop/300w", verbose=False)
    images = ds.images
    keypoints = ds.keypoints

    boxes = []
    for i in range(images.shape[0]):
        sample_keypoints = keypoints[i].numpy().squeeze().reshape(-1, 3)  # (68, 3)
        x_list = []
        y_list = []
        for idx in range(sample_keypoints.shape[0]):
            x, y, v = sample_keypoints[idx]
            if v == 0:  # keypoint not in image
                continue
            x_list.append(x)
            y_list.append(y)
        # (x0, y0, w, h)
        bbox = [
            min(x_list),
            min(y_list),
            max(x_list) - min(x_list),
            max(y_list) - min(y_list),
        ]
        boxes.append(bbox)
    return images, keypoints, boxes


def main(executor, model_name):
    batch_nme = []
    # for i in tqdm(range(images.shape[0])):
    for i in tqdm([137, 159, 180]):
        sample_image = images[i].numpy()  # (H, W, 3)
        sample_box = boxes[i]  # (x0, y0, w, h)
        sample_keypoints = keypoints[i].numpy().squeeze().reshape(-1, 3)  # (68, 3)

        if sample_image.shape[2] == 1:  # gray scale
            sample_image = np.concatenate(
                [sample_image, sample_image, sample_image], axis=2
            )

        if model_name == "spiga":
            lm, _ = executor.infer(sample_image, [sample_box])
            lm = np.array(lm[0])  # only 1 box/image for 300W, shape is [N, 2]
        elif model_name == "dlib":
            lm = executor.infer(sample_image, [sample_box])
            lm = lm[0]  # only 1 box/image for 300W, shape is [68, 2]
        elif model_name == "mediapipe":
            x0, y0, w, h = sample_box
            bbox = Boundingbox([x0, y0, x0 + w, y0 + h])
            bbox.scale(1.5)
            driver_bbox = dict(
                [("bounding_box", bbox), ("class_label", "face"), ("score", 1)]
            )
            mimic_landmarks, _ = executor.process(sample_image, driver_bbox)
            mp_lm = mimic_landmarks.parse_to_client()
            lm = []
            for index in CRITICAL_POINT_INDICES:
                lm.append((mp_lm[index][0], mp_lm[index][1]))
            lm = np.array(lm)  # [68, 2]

        plot_landmarks(sample_image, lm, f"300W_id-{i:03d}_{model_name}")

        nme = cal_nme(sample_keypoints, lm)
        batch_nme.append(nme)

    executor.show_infer_time(clear=True)
    print(f"[{model_name}] NME: {np.mean(batch_nme):.3f}")


def cal_nme(landmark_gt, landmark_pred):
    """
    landmark_gt: np.ndarray, (N, 3), representing x, y, v where v is visibility
    landmark_pred: np.ndarray, (N, 2)
    """
    med = np.mean(np.linalg.norm(landmark_pred - landmark_gt[:, :2], axis=1))
    interocular_distance = np.linalg.norm(landmark_gt[37, :2] - landmark_gt[46, :2])
    nme = med / interocular_distance
    return nme


def plot_landmarks(img, landmarks, img_name=None, suffix=".png"):
    """
    img: np.ndarray, (H, W, 3), in RGB-order
    landmarks: np.ndarray, (N, 2), N=68 for 300W dataset
    """
    canvas = copy.deepcopy(img)
    for i in range(landmarks.shape[0]):
        x, y = [int(m) for m in landmarks[i]]
        # cv2.putText(canvas, str(i + 1), (x, y), 0, 0.2, (0, 255, 255), 1, cv2.LINE_AA) # plot text
        cv2.circle(canvas, (int(x), int(y)), 0, (0, 255, 255), 5)
    if img_name:
        cv2.imwrite(f"./gallery/{img_name}{suffix}", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    # image path
    dir_name = Path("./assets/ir_people/")
    img_path = next(dir_name.glob("*.jpg"))
    bbox_json_path = next(dir_name.glob("*.json"))

    images, keypoints, boxes = load_300W_dataset()

    # SPIGA model
    spiga_manager = SpigaManager("cofw68")
    # spiga_manager.manual_infer([img_path], [bbox_json_path], True)

    # dlib model
    dlib_weight_path = Path(
        "./spiga/models/weights/shape_predictor_68_face_landmarks.dat"
    )
    dlib_manager = DlibManager(dlib_weight_path)
    # dlib_manager.manual_infer([img_path], [bbox_json_path], True)

    # mediapipe model
    mediapipe_landmark_path = Path(
        "./spiga/models/weights/face_landmark_with_attention_test.tflite"
    )
    mp_landmark_manager = MpLandmarkWrapper(str(mediapipe_landmark_path))

    main(mp_landmark_manager, "mediapipe")
