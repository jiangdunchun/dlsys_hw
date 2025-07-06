"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(filename=image_filename, mode="rb") as image_file:
        image_file_content = image_file.read()
    image_num = np.frombuffer(buffer=image_file_content, dtype=">u4", count=1, offset=4)[0]
    image_size = np.frombuffer(buffer=image_file_content, dtype=">u4", count=2, offset=8)
    image_data = np.frombuffer(buffer=image_file_content, dtype=np.uint8, offset=16).reshape(-1, image_size[0]*image_size[1])
    assert image_num == image_data.shape[0]
    image_data = image_data.astype(np.float32) / 255.
    #print(f"{image_filename} {image_num} {image_size[0]}*{image_size[1]} {image_data.shape}")

    with gzip.open(filename=label_filename, mode="rb") as lable_file:
        lable_file_content = lable_file.read()
    lable_num = np.frombuffer(buffer=lable_file_content, dtype=">u4", count=1, offset=4)[0]
    lable_data = np.frombuffer(buffer=lable_file_content, dtype=np.uint8, offset=8)
    assert lable_num == lable_data.shape[0]
    #print(f"{label_filename} {image_num} {lable_data.shape}")

    assert image_num == lable_num
    
    return image_data, lable_data
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size, num_classes = Z.shape[0], Z.shape[1]

    e_Z = ndl.exp(Z)
    sum_Z = ndl.reshape(ndl.summation(e_Z, 1), (batch_size,1))
    e_Z = ndl.divide(e_Z, ndl.broadcast_to(sum_Z, e_Z.shape))

    losses = ndl.mul_scalar(ndl.log(ndl.summation(ndl.multiply(e_Z, y_one_hot), 1)), -1.)

    return ndl.divide_scalar(ndl.summation(losses), e_Z.shape[0])
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_samples, input_dim = X.shape[0], X.shape[1]
    for i in range(0, num_samples, batch):
        X_, y_ = ndl.Tensor(X[i:i+batch], requires_grad=False), y[i:i+batch]

        y_hat = ndl.matmul(ndl.relu(ndl.matmul(X_, W1)), W2)

        y_one_hot = np.zeros(y_hat.shape)
        y_one_hot[np.arange(batch), y_] = 1
        y_one_hot = ndl.Tensor(y_one_hot)

        loss = softmax_loss(y_hat, y_one_hot)
        loss.backward()

        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
