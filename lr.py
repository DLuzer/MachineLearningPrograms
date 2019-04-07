#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100

# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    example = [int(x) for x in p.split(l.strip())]
    x = example[0:-1]
    y = example[-1]
    data.append( (x,y) )
  return (data, varnames)

def sigmoid(z):
  sig = 1/(1 + exp(-z))
  return sig

def magnitude(w_vector):
  answer = 0
  for item in w_vector:
    answer += item**2
  return answer

# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
  numvars = len(data[0][0])
  w = [0.0] * numvars
  b = 0.0

  for i in range(MAX_ITERS):
    b_grad = 0
    w_grad = [0.0] * numvars

    for (x,y) in data:
      wTx = 0
      for j in range(numvars):
        wTx += (w[j] * x[j])
      wTx += b

      b_grad -= eta * (y/(1 + exp(y*wTx)))

      for k in range(numvars):
        w_grad[k] -= eta * ((y*x[k]) / (1 + exp(y*wTx)))

    for l in range(numvars):
      w_grad[l] += eta*(l2_reg_weight*w[l])

    if sqrt(magnitude(w_grad) + b_grad**2) < .0001:
      return (w,b)
    else:
      b -= b_grad
      for m in range(numvars):
        w[m] -= w_grad[m]

  return (w,b)

# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
  (w,b) = model

  activation = 0
  for i in range(len(w)):
    activation += (w[i] * x[i])
  activation += b

  return sigmoid(activation)


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 5):
    print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  eta = float(argv[2])
  lam = float(argv[3])
  modelfile = argv[4]

  # Train model
  (w,b) = train_lr(train, eta, lam)

  # Write model file
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    prob = predict_lr( (w,b), x )
    #print(prob)
    if (prob - 0.5) * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])
