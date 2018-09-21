from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    out = np.dot(x.reshape(x.shape[0], -1), w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    num_train = x.shape[0]
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    db = np.sum(dout, axis = 0)
    dx = np.dot(dout, w.T).reshape(*x.shape) #to match dimensions of x
    dw = np.dot(x.reshape(num_train, -1).T, dout)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x) #ReLU functions
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.zeros(x.shape)
    dx[x>0] = 1
    dx = dx * dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def sigmoid_forward(x):
    """
    Computes the forward pass for a layer of sigmoid units.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: out itself
    """
    out = None
    ###########################################################################
    # TODO: Implement the Sigmoid forward pass.                               #
    ###########################################################################
    out = 1 / (1.0 + np.exp(-x))
    cache = out
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def sigmoid_backward(dout, cache):
    """
    Computes the backward pass for a layer of sigmoid units .

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: output of sigmoid, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, out = None, cache
    ###########################################################################
    # TODO: Implement the sigmoid backward pass.                              #
    ###########################################################################
    dx = out * (1 - out)
    dx = dx * dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def tanh_forward(x):
    """
    Computes the forward pass for a layer of tanh units.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: out itself
    """
    out = None
    ###########################################################################
    # TODO: Implement the tanh forward pass.                                  #
    ###########################################################################
    out = np.tanh(x)
    cache = out
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def tanh_backward(dout, cache):
    """
    Computes the backward pass for a layer of tanh units .

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: output of tanh, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, out = None, cache
    ###########################################################################
    # TODO: Implement the tanh backward pass.                                 #
    ###########################################################################
    dx = 1.0 - (out**2)
    dx = dx * dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def leakyrelu_forward(x):
    """
    Computes the forward pass for a layer of LeakyReLU units.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: out itself
    """
    out = None
    ###########################################################################
    # TODO: Implement the LeakyReLU forward pass.                             #
    ###########################################################################
    pos_out = np.maximum(0, x) #get the positive output
    neg_out = -np.maximum(-0.01*x, 0) #the negative one fixing alpha to 0.01
    out = pos_out + neg_out
    cache = x
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def leakyrelu_backward(dout, cache):
    """
    Computes the backward pass for a layer of leakyReLU units .

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: output of tanh, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, out = None, cache
    ###########################################################################
    # TODO: Implement the LeakyReLU backward pass.                            #
    ###########################################################################
    dx_pos = np.zeros_like(out)
    dx_pos[out>0] = 1  #for all the positive ones gradient will be one

    dx_neg = np.zeros_like(out)
    dx_neg[out<0] = 0.01 #I fixed the alpha in the leakyrelu, can be sent variable

    dx = dx_pos + dx_neg #positive 1, negative 0.01 and for zero 0

    dx = dx * dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
