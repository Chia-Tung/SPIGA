import copy
import json
import warnings
from pathlib import Path

import cv2
import deeplake
import dlib
import numpy as np

warnings.filterwarnings("ignore")
from spiga.demo.visualize.plotter import Plotter
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework


def load_img(img_path, bbox_json):
    """
    return np.ndarray of the img in order of BGR
    """
    # Load image and bbox
    image = cv2.imread(img_path)  # shape = (H, W, C)
    with open(bbox_json) as jsonfile:
        bbox = json.load(jsonfile)["bbox"]
    print(f"Shape of input image: {image.shape}")
    print(f"Number of input faces: {len(bbox)}")
    return image, bbox


def spiga_infer(img_path, bbox_json, img_save_path):
    """
    SPIGA infers on BGR-order image
    """
    img, bbox_list = load_img(img_path, bbox_json)

    # Process image
    dataset = "wflw"
    canvas = copy.deepcopy(img)
    processor = SPIGAFramework(ModelConfig(dataset))
    features = processor.inference(img, bbox_list)

    for i, bbox in enumerate(bbox_list):
        x0, y0, w, h = [int(ele) for ele in bbox]
        landmarks = np.array(features["landmarks"][i])
        headpose = np.array(features["headpose"][i])

        # Plot features
        plotter = Plotter()
        canvas = plotter.landmarks.draw_landmarks(canvas, landmarks)
        canvas = plotter.hpose.draw_headpose(
            canvas, [x0, y0, x0 + w, y0 + h], headpose[:3], headpose[3:], euler=True
        )
        canvas = cv2.rectangle(
            canvas, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 2, lineType=cv2.LINE_AA
        )

    # save figure
    cv2.imwrite(img_save_path, canvas)
    print("all done")


def dlib_infer(img_path, bbox_json, predictor_path, img_save_path):
    """
    dlib infers on RGB-order image
    """
    shape_predictor = dlib.shape_predictor(predictor_path)
    img, bbox_list = load_img(img_path, bbox_json)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    canvas = copy.deepcopy(img)

    for i, bbox in enumerate(bbox_list):
        x0, y0, w, h = [int(ele) for ele in bbox]
        rect = dlib.rectangle(x0, y0, x0 + w, y0 + h)
        landmarks = get_landmarks(img, rect, shape_predictor)
        plot_dlib_landmark(canvas, landmarks)
        canvas = cv2.rectangle(
            canvas, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 2, lineType=cv2.LINE_AA
        )

    # save figure
    cv2.imwrite(img_save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print("all done")


def get_landmarks(im, rect, shape_predictor):
    def shape_to_np(shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    shape = shape_predictor(im, rect)
    coords = shape_to_np(shape)

    return coords


def plot_dlib_landmark(img, landmark_array) -> None:
    """sperate points"""
    jaw = landmark_array[0:17]  # 顎 (Jaw: 17 points) 1 ~ 17
    left_eyebrow = landmark_array[17:22]  # 左眉 (Left eyebrow: 5 points)  18 ~ 22
    right_eyebrow = landmark_array[22:27]  # 右眉 (Right eyebrow: 5 points)  23 ~ 27
    vertical_nose = landmark_array[27:31]  # 鼻子 (Nose: 9 points) 28 ~ 31 , 32 ~ 36
    horizontal_nose = landmark_array[31:36]
    left_eye = landmark_array[36:42]  # 左眼 (Left eye: 6 points)  37 ~ 42
    right_eye = landmark_array[42:48]  # 右眼 (Right eye: 6 points)  43 ~ 48
    mouth = landmark_array[48:68]  # 口 (Mouth: 20 points) 49 ~ 68

    """plot"""
    for i in range(landmark_array.shape[0]):
        (x, y) = landmark_array[i, :]
        img = cv2.circle(img, (x, y), 0, (50, 255, 50), 5)


def load_300W_dataset():
    """
    image is in RGB order
    """
    ds = deeplake.load("hub://activeloop/300w", verbose=False)
    images = ds.images
    keypoints = ds.keypoints

    sample_image = images[27].numpy()  # (H, W, 3)
    sample_keypoints = keypoints[27].numpy().squeeze().reshape(-1, 3)  # (68, 3)
    canvas = copy.deepcopy(sample_image)

    for idx in range(sample_keypoints.shape[0]):
        x, y, v = sample_keypoints[idx]
        if v == 0:  # keypoint not in image
            continue

        canvas = cv2.circle(canvas, (x, y), 0, (50, 255, 50), 5)

    cv2.imwrite(img_save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    dir_name = Path("./assets/trump/")

    img_path = next(dir_name.glob("*.jpg"))
    bbox_json_path = next(dir_name.glob("*.json"))
    # bbox_json_path = dir_name / "face_0.json"

    dlib_weight_path = "./spiga/models/weights/shape_predictor_68_face_landmarks.dat"
    img_save_path = "./sample_image5.png"

    # spiga_infer(str(img_path), str(bbox_json_path), img_save_path)
    # dlib_infer(str(img_path), str(bbox_json_path), dlib_weight_path, img_save_path)
    load_300W_dataset()
