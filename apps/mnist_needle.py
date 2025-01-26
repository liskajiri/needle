import needle as ndl
import numpy as np


def loss_err(h, y):
    """Helper function to compute both loss and error."""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)


def softmax(X: np.ndarray) -> np.ndarray:
    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)


def softmax_loss(Z: ndl.Tensor, y: ndl.Tensor) -> ndl.Tensor:
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logits predictions for
            each class.
        y (Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns
    -------
        Average softmax loss over the sample. (Tensor[np.float32])

    """
    batch_size = Z.shape[0]

    diff = (Z * y).sum(axes=1)

    log_Z = ndl.log(ndl.exp(Z).sum(axes=1))
    total = log_Z - diff
    return total.sum() / batch_size


def nn_epoch(
    X: np.ndarray,
    y: np.ndarray,
    W1: ndl.Tensor,
    W2: ndl.Tensor,
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
        W1 (Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns
    -------
        Tuple: (W1, W2)
            W1: Tensor[np.float32]
            W2: Tensor[np.float32]

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
