import copy
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import deeplake
import numpy as np
from tqdm import tqdm

from utils.spiga_manager import SpigaManager
from utils.dlib_manager import DlibManager


def load_300W_dataset(spiga_manager, dlib_manager):
    """
    image is in RGB order
    """
    ds = deeplake.load("hub://activeloop/300w", verbose=False)
    images = ds.images
    keypoints = ds.keypoints

    ans = []
    for i in tqdm(range(images.shape[0])):
        sample_image = images[i].numpy()  # (H, W, 3)
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

        ##### SPIGA Inference #####
        # lm_spiga, _ = spiga_manager.infer(sample_image, [bbox])
        # lm_spiga = np.array(lm_spiga[0])  # only 1 box/image for 300W, shape is [N, 2]
        # calculation
        # med = np.mean(np.linalg.norm(lm_spiga - sample_keypoints[:, :2], axis=1))
        # interocular_distance = np.linalg.norm(sample_keypoints[37, :2] - sample_keypoints[46, :2])
        # nme = med / interocular_distance

        # canvas = copy.deepcopy(sample_image)
        # for i in range(lm_spiga.shape[0]):
        #     x, y = [int(m) for m in lm_spiga[i]]
        #     cv2.putText(canvas, str(i+1), (x, y), 0, 0.2, (0, 255, 255), 1, cv2.LINE_AA)
        # cv2.imwrite("./sample_spiga_merlrav.png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        ##### DLIB Inference #####
        lm_dlib = dlib_manager.infer(sample_image, [bbox])
        lm_dlib = lm_dlib[0]  # only 1 box/image for 300W, shape is [68, 2]
        # calculation
        med = np.mean(np.linalg.norm(lm_dlib - sample_keypoints[:, :2], axis=1))
        interocular_distance = np.linalg.norm(sample_keypoints[37, :2] - sample_keypoints[46, :2])
        nme = med / interocular_distance

        # canvas = copy.deepcopy(sample_image)
        # for i in range(lm_dlib.shape[0]):
        #     x, y = [int(m) for m in lm_dlib[i]]
        #     cv2.putText(canvas, str(i+1), (x, y), 0, 0.2, (0, 255, 255), 1, cv2.LINE_AA)
        # cv2.imwrite("./sample_dlib.png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        ##### 300W annotations #####
        # canvas = copy.deepcopy(sample_image)
        # for i in range(sample_keypoints.shape[0]):
        #     x, y, v = sample_keypoints[i]
        #     cv2.putText(canvas, str(i+1), (x, y), 0, 0.2, (0, 255, 255), 1, cv2.LINE_AA)
        # cv2.imwrite("./sample_300W.png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        ans.append(nme)
    print(f"NME: {np.mean(ans):.3f}")
    dlib_manager.show_infer_time(clear=True)


if __name__ == "__main__":
    dir_name = Path("./assets/ir_people/")

    img_path = next(dir_name.glob("*.jpg"))
    # bbox_json_path = next(dir_name.glob("*.json"))
    bbox_json_path = dir_name / "face_0.json"

    dlib_weight_path = Path(
        "./spiga/models/weights/shape_predictor_68_face_landmarks.dat"
    )

    spiga_manager = SpigaManager("300wprivate")
    # spiga_manager.manual_infer([img_path], [bbox_json_path], True)
    dlib_manager = DlibManager(dlib_weight_path)
    # dlib_manager.manual_infer([img_path], [bbox_json_path], True)
    load_300W_dataset(spiga_manager, dlib_manager)
