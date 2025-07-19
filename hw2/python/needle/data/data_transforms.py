import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        assert len(img.shape) == 3

        if not flip_img: 
            return img

        return np.flip(img, axis=1)
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        assert len(img.shape) == 3

        H, W = img.shape[0], img.shape[1]

        img_padded = np.zeros_like(img)

        if abs(shift_x) >= H or abs(shift_y) >= W:
            return img_padded

        img_padded[max(0, -shift_x):min(H - shift_x, H), max(0, -shift_y):min(W - shift_y, W), :] = img[max(0, shift_x):min(H + shift_x, H), max(0, shift_y):min(W + shift_y, W), :]
        return img_padded
        ### END YOUR SOLUTION
