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



def runMe():
    print boxfilter(5)
    print boxfilter(3)
    print boxfilter(7)
    # print boxfilter(4)

runMe()