import numpy as np


class BaseManager:
    def __init__(self) -> None:
        self.__infer_time_store = []

    def insert_time(self, ele: float):
        assert type(ele) == float
        self.__infer_time_store.append(ele)

    def show_infer_time(self, clear: bool = False):
        print(
            f"average inference speed = {np.mean(self.__infer_time_store):.3f} sec/iter"
        )

        if clear:
            self.clear_infer_time()

    def clear_infer_time(self):
        self.__infer_time_store.clear()
