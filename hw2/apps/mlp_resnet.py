import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
            )
        ),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Flatten(), # flatten the mnist data (shape is n*h*w*c) firstly
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()

    if opt:
        model.train()
    else:
        model.eval()

    total_loss, total_err = 0., 0.          
    for X, y in dataloader:
        if opt:
            opt.reset_grad()

        y_hat = model(X)
        loss = loss_func(y_hat, y)

        total_loss += loss.numpy() * X.shape[0]
        total_err += (y.numpy() != y_hat.numpy().argmax(1)).sum()

        if opt:
            loss.backward()
            opt.step()

        if not opt:
            print(f"Train Loss: {total_loss:.4f} | Train Error: {total_err}")

    return total_err / len(dataloader.dataset), total_loss / len(dataloader.dataset)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_data = ndl.data.MNISTDataset(
        data_dir + '/train-images-idx3-ubyte.gz',
        data_dir + '/train-labels-idx1-ubyte.gz'
    )
    test_data = ndl.data.MNISTDataset(
        data_dir + '/t10k-images-idx3-ubyte.gz',
        data_dir + '/t10k-labels-idx1-ubyte.gz',
    )
    train_loader = ndl.data.DataLoader(train_data, batch_size, True)
    test_loader = ndl.data.DataLoader(test_data, batch_size)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr, weight_decay)

    print("\nstart training------------>")
    for i in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt)
        print(f"Epoch {i}/{epochs} | Train Loss: {train_loss:.4f} | Train Error: {train_err:.4f}")

    test_err, test_loss = epoch(test_loader, model)
    print(f"end training------------>Test Loss: {test_loss:.4f} | Test Error: {test_err:.4f}")

    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
