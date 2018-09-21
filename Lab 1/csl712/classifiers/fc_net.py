from builtins import range
from builtins import object
import numpy as np

from csl712.layers import *
from csl712.layer_utils import *

class LogisticRegression(object):
    """
    In this Logisitc Regression is implemented with softmax loss. We assume an input 
    dimension of D and perform classification over C classes.

    The architecure should be affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, num_classes=10, weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the model. Weights should be  #
        # initialized from a Gaussian with standard deviation equal to weight_scale#
        # and biases should be initialized to zero. All weights and biases should  #
        # be stored in the dictionary self.params, with keys 'W1' and 'b1'         #
        ############################################################################
        self.params["W1"] = np.random.normal(scale = weight_scale, size = (input_dim, num_classes))
        self.params["b1"] = np.zeros(num_classes, )
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for logisitc regression, computing the  #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        afn_scores,afn_cache = affine_forward(X, self.params["W1"], self.params["b1"])
        scores, sgmd_cache = sigmoid_forward(afn_scores)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for logistic regression. Store the loss#
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        dscores = sigmoid_backward(dscores, sgmd_cache)
        _, grads["W1"], grads["b1"] = affine_backward(dscores, afn_cache)

        loss += 0.5 * self.reg * np.sum(self.params["W1"] * self.params["W1"])
        
        grads["W1"] += self.reg * self.params["W1"]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0, activation_fn = 0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - activation_fn: which function to use default is 0 - ReLU
        				 1 - LeakyReLU, 2 - Tanh, 3 - Sigmoid
        """
        self.params = {}
        self.reg = reg
        self.activation_fn = activation_fn
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params["W1"] = np.random.normal(scale = weight_scale, size = (input_dim, hidden_dim))
        self.params["b1"] = np.zeros(hidden_dim, )

        self.params["W2"] = np.random.normal(scale = weight_scale, size = (hidden_dim, num_classes))
        self.params["b2"] = np.zeros(num_classes, )
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        affine1_output, layer1_cache = affine_forward(X, self.params["W1"], self.params["b1"])

        if (self.activation_fn == 0):
        	layer1_output, actvn_cache  = relu_forward(affine1_output)

        elif (self.activation_fn == 1):
        	layer1_output, actvn_cache  = leakyrelu_forward(affine1_output)

        elif (self.activation_fn == 2):
        	layer1_output, actvn_cache  = tanh_forward(affine1_output)

        elif (self.activation_fn == 3):
        	layer1_output, actvn_cache  = sigmoid_forward(affine1_output)

        else:
        	print ("Please give proper activation function")
        	return None

        scores, layer2_cache = affine_forward(layer1_output, self.params["W2"], self.params["b2"])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        dlayer1, grads["W2"], grads["b2"] = affine_backward(dscores, layer2_cache)

        if (self.activation_fn == 0):
        	dlayer1 = relu_backward(dlayer1, actvn_cache)

        elif (self.activation_fn == 1):
        	dlayer1 = leakyrelu_backward(dlayer1, actvn_cache)

        elif (self.activation_fn == 2):
        	dlayer1 = tanh_backward(dlayer1, actvn_cache)

        elif (self.activation_fn == 3):
        	dlayer1 = sigmoid_backward(dlayer1, actvn_cache)

        else:
        	print ("Please give proper activation function")
        	return None

        dx, grads["W1"], grads["b1"] = affine_backward(dlayer1, layer1_cache)

        loss += 0.5 * self.reg * np.sum(self.params["W1"] * self.params["W1"]) 
        loss += 0.5 * self.reg * np.sum(self.params["W2"] *  self.params["W2"])

        grads["W1"] += self.reg * self.params["W1"]
        grads["W2"] += self.reg * self.params["W2"]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads