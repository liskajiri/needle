import gzip
from pathlib import Path

import numpy as np
import numpy.typing as npt
from needle.data.datasets.mnist import MNISTPaths


def parse_mnist(
    image_filename: Path, label_filename: Path
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns
    -------
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.

    """
    # Parse images file
    with gzip.open(image_filename, "rb") as img_file:
        magic_number = int.from_bytes(img_file.read(4), "big")
        if magic_number != 2051:
            msg = "Invalid magic number in image file."
            raise ValueError(msg)

        num_images = int.from_bytes(img_file.read(4), "big")
        rows = int.from_bytes(img_file.read(4), "big")
        cols = int.from_bytes(img_file.read(4), "big")

        image_data = img_file.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(
            (num_images, rows * cols)
        )
        X = images.astype(np.float32) / 255.0

    # Parse labels file
    with gzip.open(label_filename, "rb") as lbl_file:
        magic_number = int.from_bytes(lbl_file.read(4), "big")
        if magic_number != 2049:
            msg = "Invalid magic number in label file."
            raise ValueError(msg)

        num_labels = int.from_bytes(lbl_file.read(4), "big")
        label_data = lbl_file.read()
        y = np.frombuffer(label_data, dtype=np.uint8)

    if num_images != num_labels:
        msg = "Mismatch between number of images and labels."
        raise ValueError(msg)

    return X, y


def softmax(X: np.ndarray) -> np.ndarray:
    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)


def softmax_loss(Z: np.ndarray[np.float32], y: np.ndarray[np.int8]):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns
    -------
        Average softmax loss over the sample.

    """
    ### BEGIN YOUR CODE
    # # numerical stabilization
    # Z_ = Z - np.max(Z, axis=1, keepdims=True)
    # # shape: (60k, 784)
    # probs = softmax(Z_)
    # # exp / sum of exps in row
    # range_ = np.arange(Z.shape[0])
    # # get the probability of each correct sample
    # y_probs = probs[range_, y]

    # ce_loss = np.mean(-np.log(y_probs))
    # return ce_loss
    #     # Compute log-sum-exp
    log_sum_exp = np.log(np.sum(np.exp(Z), axis=1))

    # Compute correct class log probabilities
    correct_class_log_probs = Z[np.arange(Z.shape[0]), y]

    # Compute loss for each sample
    losses = log_sum_exp - correct_class_log_probs

    # Return average loss
    return np.mean(losses)
    ### END YOUR CODE


def softmax_regression_epoch(
    X: np.ndarray[np.float32],
    y: np.ndarray[np.uint8],
    theta: np.ndarray[np.float32],
    lr: float = 0.1,
    batch: int = 100,
) -> None:
    """Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns
    -------
    None

    """
    n_batches = X.shape[0] // batch

    for n in range(n_batches):
        X_batch = X[n * batch : (n + 1) * batch]
        y_batch = y[n * batch : (n + 1) * batch]

        k = theta.shape[1]
        Z = softmax(X_batch @ theta)

        I_y = np.zeros((batch, k))
        I_y[np.arange(batch), y_batch] = 1
        grad = X_batch.T @ (Z - I_y) / batch

        theta -= lr * grad


def loss_err(h, y):
    """Helper function to compute both loss and error."""
    return softmax_loss(h, y), np.mean(h.argmax(axis=1) != y)


def nn_epoch(
    X: np.ndarray[np.float32],
    y: np.ndarray[np.uint8],
    W1: np.ndarray[np.float32],
    W2: np.ndarray[np.float32],
    lr: float = 0.1,
    batch: int = 100,
) -> None:
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns
    -------
        None

    """

    def relu(X):
        return np.maximum(0, X)

    n_batches = X.shape[0] // batch

    for n in range(n_batches):
        X_batch = X[n * batch : (n + 1) * batch]
        y_batch = y[n * batch : (n + 1) * batch]

        k = W2.shape[1]

        I_y = np.zeros((batch, k))
        I_y[np.arange(batch), y_batch] = 1

        Z1 = relu(X_batch @ W1)
        G2 = softmax(Z1 @ W2) - I_y

        bin_matrix = np.zeros_like(Z1)
        bin_matrix[Z1 > 0] = 1
        G1 = np.multiply(bin_matrix, G2 @ W2.T)
        W1_grad = X_batch.T @ G1 / batch
        W2_grad = Z1.T @ G2 / batch

        W1 -= lr * W1_grad
        W2 -= lr * W2_grad


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100, cpp=False):
    """Example function to fully train a softmax regression classifier."""
    theta = np.zeros((X_tr.shape[1], y_tr.max() + 1), dtype=np.float32)
    for _epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=500, epochs=10, lr=0.5, batch=100):
    """Example function to train two layer neural network."""
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    for _epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr @ W1, 0) @ W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te @ W1, 0) @ W2, y_te)


if __name__ == "__main__":
    X_tr, y_tr = parse_mnist(MNISTPaths.TRAIN_IMAGES, MNISTPaths.TRAIN_LABELS)
    X_te, y_te = parse_mnist(MNISTPaths.TEST_IMAGES, MNISTPaths.TEST_LABELS)

    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1)

    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr=0.2)
