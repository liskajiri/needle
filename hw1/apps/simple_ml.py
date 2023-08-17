import numpy as np
import numpy.typing as npt
import needle as ndl

import sys

sys.path.append("python/")


def parse_mnist(
    image_filename, label_filename
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
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
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    from torchvision.datasets import MNIST

    train_set = MNIST("./data", train=True, download=False)

    def normalize_values(data: np.ndarray) -> np.ndarray:
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    data = train_set.data.numpy().astype("float32").reshape((-1, 784))
    y = train_set.targets.numpy().astype("uint8")

    data = normalize_values(data)

    return data, y


def softmax_loss(Z: ndl.Tensor, y: ndl.Tensor) -> ndl.Tensor:
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logits predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    batch_size = Z.shape[0]

    diff = (Z * y).sum(axes=1)

    log_Z = ndl.log(ndl.exp(Z).sum(axes=1))
    total = log_Z - diff
    mean = total.sum() / batch_size

    return mean


def nn_epoch(
    X: np.ndarray,
    y: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    lr: float = 0.1,
    batch: int = 100,
) -> tuple[ndl.Tensor, ndl.Tensor]:
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
    n_batches = X.shape[0] // batch

    for n in range(n_batches):
        X_batch = X[n * batch : (n + 1) * batch]
        y_batch = y[n * batch : (n + 1) * batch]

        X_batch = ndl.Tensor(X_batch)

        k = W2.shape[1]
        # creates one hot vector by permuting the eye matrix
        I_y = np.eye(k)[y_batch]

        y_batch = ndl.Tensor(I_y)

        # forward pass
        Z = ndl.relu(X_batch @ W1)
        loss = softmax_loss((Z @ W2), y_batch)

        loss.backward()

        W1 -= lr * W1.grad.numpy()
        W2 -= lr * W2.grad.numpy()

        W1 = ndl.Tensor(W1)
        W2 = ndl.Tensor(W2)

    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
