from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        with gzip.open(filename=image_filename, mode="rb") as image_file:
            image_file_content = image_file.read()
        image_num = np.frombuffer(buffer=image_file_content, dtype=">u4", count=1, offset=4)[0]
        image_size = np.frombuffer(buffer=image_file_content, dtype=">u4", count=2, offset=8)
        image_data = np.frombuffer(buffer=image_file_content, dtype=np.uint8, offset=16).reshape(-1, image_size[0], image_size[1], 1)
        image_data = image_data.astype(np.float32) / 255.
        assert image_num == image_data.shape[0]

        with gzip.open(filename=label_filename, mode="rb") as lable_file:
            lable_file_content = lable_file.read()
        lable_num = np.frombuffer(buffer=lable_file_content, dtype=">u4", count=1, offset=4)[0]
        lable_data = np.frombuffer(buffer=lable_file_content, dtype=np.uint8, offset=8)
        assert lable_num == lable_data.shape[0]
        assert image_num == lable_num
        
        self.images, self.labels = image_data, lable_data
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image, label = self.images[index], self.labels[index]
        return self.apply_transforms(image), label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION