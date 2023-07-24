import pasp
import torchvision
import pandas as pd
import numpy as np

def mnist_labels():
  "Return the first and second digit values of the test set."
  Y = pd.read_csv("data/test_labels_small.csv", header = None)
  return np.array([Y.iloc[idx, 1]*2 for idx in range(len(Y))])

P = pasp.parse("double_digit_execution/double_digit_rule.plp")                      # Parse the sum of digits program.
R = P(quiet=True)                        # Run program and capture output.
print("output of pasp file",P)
# Y = np.argmax(R, axis=1).reshape(len(R)) # Retrieve what sum is likeliest.
# D = mnist_labels()                       # D contains the digits of the test set.                       # The ground-truth in the test set.
# accuracy = np.sum(Y == D)/len(D)         # Digit sum accuracy.
# print(f"Accuracy: {100*accuracy}%")

