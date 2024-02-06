import json
from pathlib import Path

import cv2


def load_single_img_and_bbox(img_path: Path, bbox_json_path: Path):
    """
    Returns:
        np.ndarray, image in the order of RGB
        list[list[int]], list of bounding box coordiante
    """
    image = cv2.imread(str(img_path))  # shape = (H, W, C)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(str(bbox_json_path)) as jsonfile:
        bbox = json.load(jsonfile)["bbox"]

    print(f"Shape of input image: {image.shape}")
    print(f"Number of input faces: {len(bbox)}")
    return image, bbox


def append_filename(original_path: Path, postfix: str) -> Path:
    filename = original_path.name  # image_sportsfan.jpg
    name = filename.split(".")[0]  # image_sportsfan
    fmt = filename.split(".")[1]  # jpg

    return original_path.parent / f"{name}_{postfix}.png"
