from PIL import Image
import numpy as np
import math
from scipy import signal

# returns a box filter of size n by n. You should check that n is odd, 
# checking and signaling an error with an assert statement. 
# The filter should be a Numpy array. For example, your function should work as follows:

# >>> boxfilter(5)
# array([[ 0.04,  0.04,  0.04,  0.04,  0.04],
#        [ 0.04,  0.04,  0.04,  0.04,  0.04],
#        [ 0.04,  0.04,  0.04,  0.04,  0.04],
#        [ 0.04,  0.04,  0.04,  0.04,  0.04],
#        [ 0.04,  0.04,  0.04,  0.04,  0.04]])

# >>> boxfilter(4)
# Traceback (most recent call last):
#   ...
# AssertionError: Dimension must be odd
# HINT: The generation of the filter can be done as a simple one-line expression. Of course, checking that n is odd requires a bit more work.

# Show the results of your boxfilter(n) function for the cases n=3, n=4, and n=5.
def boxfilter(n):

    # ensure that n is always odd
    assert n % 2 == 1, "AssertionError: boxfilter dimension must be odd"

    # create an empty n by n numpy array
    filter = np.empty([n, n])

    # fill each entry with 1/n^2 
    filter.fill(float(1) / (n * n))

    return filter

# Write a Python function, gauss1d(sigma), that returns a 1D Gaussian 
# filter for a given value of sigma. The filter should be a 1D array with 
# length 6 times sigma rounded up to the next odd integer. Each value of 
# the filter can be computed from the Gaussian function, exp(- x^2 / (2*sigma^2)), 
# where x is the distance of an array value from the center. This formula for 
# the Gaussian ignores the constant factor. Therefore, you should normalize 
# the values in the filter so that they sum to 1.

# HINTS: For efficiency and compactness, it is best to avoid for loops in Python.
# One way to do this is to first generate a 1D array of values for x, for example
#  [-3 -2 -1 0 1 2 3] for a sigma of 1.0. These can then be used in a single
# Numpy expression to calculate the Gaussian value corresponding to each element.

# Show the filter values produced for sigma values of 0.3, 0.5, 1, and 2.
def gauss1d(sigma):

    # length should be 6 times sigma rounded up to the next odd integer
    length = math.ceil(sigma * 6)

    # increment if length is even
    if length % 2 == 0:
        length = length + 1;

    return length;

def runMe():
    print boxfilter(5)
    print boxfilter(3)
    print boxfilter(7)

    print gauss1d(0.3)
    # print boxfilter(4)

runMe()