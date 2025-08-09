import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset
import pickle

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        data_files = []
        if train:
            data_files = [
                os.path.join(base_folder, 'data_batch_1'),
                os.path.join(base_folder, 'data_batch_2'),
                os.path.join(base_folder, 'data_batch_3'),
                os.path.join(base_folder, 'data_batch_4'),
                os.path.join(base_folder, 'data_batch_5')
            ]
        else:
            data_files = [os.path.join(base_folder, 'test_batch')]

        image_data, lable_data, size = [], [], 0
        for file in data_files:
            with open(file, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
            
            features = batch['data'].reshape((len(batch['data']), 3, 32, 32)) / 255.
            labels = np.array(batch['labels'])

            assert features.shape[0] == labels.shape[0]

            image_data.append(features)
            lable_data.append(labels)
            size += labels.shape[0]
        
        self.images, self.labels = np.array(image_data).reshape(size, 3, 32, 32), np.array(lable_data).reshape(size)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        image, label = self.images[index], self.labels[index]
        return self.apply_transforms(image), label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION
