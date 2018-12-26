import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)                 # 1 x 10 
    correct_class_score = scores[y[i]]
    exp_sum = 0
    for j in range(num_classes):
      exp_sum += np.exp(scores[j])
    softmax_activation = np.exp(correct_class_score) / exp_sum
    loss += -1*np.log(softmax_activation)

    for j in range(num_classes):
      if j == y[i]:
        dW[:, y[i]] += X[i, :] * (np.exp(scores[y[i]])/exp_sum - 1)
      else:
        dW[:, j] += X[i, :] * (np.exp(scores[j])/exp_sum)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW = dW / num_train + 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)

  # Exponential scores
  # Normalize the scores beforehand with max as zero to avoid
  # numerical problems with the exponential
  exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
  # Softmax activation
  probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True) # N x C 
  # Log loss of the correct class of each of our samples
  correct_logprobs = -np.log(probs[np.arange(X.shape[0]), y])

  # Compute the average loss
  loss = np.sum(correct_logprobs)/X.shape[0]

  # Add regularization using the L2 norm
  # reg is a hyperparameter and controls the strength of regularization
  reg_loss = reg*np.sum(W*W)
  loss += reg_loss


  # For Gradient (dW)
  
  dscores = probs # N x C
  dscores[np.arange(X.shape[0]), y] -= 1
  dscores /= X.shape[0]

  # dscores = X * W
  dW = X.T.dot(dscores)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

