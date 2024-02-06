from __future__ import annotations

from typing import List

import numpy as np


class Boundingbox:
    def __init__(self, position: List[float]):
        """
        Args:
            position: '2D np.array', in the absoluate [xmin, ymin, xmax, ymax] format.
        """
        assert position[0] <= position[2], "Invalid value, xmin must <= xmax"
        assert position[1] <= position[3], "Invalid value, ymin must <= ymax"
        self.xmin = position[0]
        self.ymin = position[1]
        self.xmax = position[2]
        self.ymax = position[3]
        self.width = self.get_width()
        self.height = self.get_height()
        self.aspect_ratio = self.get_aspect_ratio()

    def __repr__(self) -> str:
        return f"Boundingbox(xmin:{self.xmin} ymin:{self.ymin} xmax:{self.xmax} ymax:{self.ymax})"

    def get_position(self) -> List[float]:
        """
        In left, top, right, bottom format
        """
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def get_cv2_position(self) -> list[int]:
        """
        In "top, left, bottom, right" format
        """
        return [int(self.ymin), int(self.xmin), int(self.ymax), int(self.xmax)]

    def get_width(self) -> float:
        w = np.maximum((self.xmax - self.xmin), 0)
        return w

    def get_height(self) -> float:
        h = np.maximum((self.ymax - self.ymin), 0)
        return h

    def get_centerX(self) -> float:
        return (self.xmin + self.xmax) / 2

    def get_centerY(self) -> float:
        return (self.ymin + self.ymax) / 2

    def get_center_point(self) -> List[float]:
        return [self.get_centerX(), self.get_centerY()]

    def get_box_area(self) -> float:
        return self.height * self.width

    def get_aspect_ratio(self) -> float:
        return self.width / self.height

    def intersection_area(self, box: Boundingbox) -> float:
        if self.ymin >= box.ymax or self.ymax <= box.ymin:
            return 0
        if self.xmin >= box.xmax or self.xmax <= box.xmin:
            return 0
        inter_xmin = np.maximum(self.xmin, box.xmin)
        inter_ymin = np.maximum(self.ymin, box.ymin)
        inter_xmax = np.minimum(self.xmax, box.xmax)
        inter_ymax = np.minimum(self.ymax, box.ymax)
        return (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

    def IOU(self, box: Boundingbox) -> float:
        intersection_area = self.intersection_area(box)
        union_area = self.get_box_area() + box.get_box_area() - intersection_area
        return intersection_area / union_area

    def is_center_inside(self, box: Boundingbox) -> bool:
        return (self.xmin <= box.get_centerX() <= self.xmax) and (
            self.ymin <= box.get_centerY() <= self.ymax
        )

    def is_overlap(self, box: Boundingbox) -> bool:
        return self.intersection_area(box) > 0

    def scale(self, factor: float, is_shrink: bool = False) -> None:
        IMAGE_HEIGHT = 720
        IMAGE_WIDTH = 1280
        ALLOW_H = True
        ALLOW_V = True

        horizontal_diff = self.width * (factor - 1) / 2
        vertiacal_diff = self.height * (factor - 1) / 2

        if horizontal_diff > self.xmin or horizontal_diff > (IMAGE_WIDTH - self.xmax):
            ALLOW_H = False

        if vertiacal_diff > self.ymin or vertiacal_diff > (IMAGE_HEIGHT - self.ymax):
            ALLOW_V = False

        if (not ALLOW_H or not ALLOW_V) and is_shrink:
            vertiacal_diff = min(self.ymin, IMAGE_HEIGHT - self.ymax)
            horizontal_diff = min(self.xmin, IMAGE_WIDTH - self.xmax)

        self.xmin = self.xmin - horizontal_diff
        self.ymin = self.ymin - vertiacal_diff
        self.xmax = self.xmax + horizontal_diff
        self.ymax = self.ymax + vertiacal_diff
