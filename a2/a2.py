from PIL import Image
import numpy as np
import math
from scipy import signal

# Suppresses the ComplexWarning for convolve2d
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

def boxfilter(n):

    ''' 
    Returns a box filter of size n by n (a numpy array)
    Throws an error if n is even.
     '''

    # ensure that n is always odd
    assert n % 2 == 1, "AssertionError: boxfilter dimension must be odd"

    # create an empty n by n numpy array
    filter = np.empty([n, n])

    # fill each entry with 1/n^2 
    filter.fill(float(1) / (n * n))

    return filter


def gauss1d(sigma):

    '''
    Returns a 1D Gaussian filter with length ceil(sigma * 6), 
    rounded up to the next odd integer.

    Each value of the filter is computed with the Gaussian function, exp(- x^2 / (2*sigma^2)),
    and the values of the filter sum to 1.
    '''

    assert sigma > 0, 'gauss1d: sigma cannot be less than or equal to zero'

    sigma = float(sigma)

    # length should be 6 times sigma rounded up to the next odd integer
    length = math.ceil(sigma * 6)

    # increment if length is even
    if length % 2 == 0:
        length = length + 1;

    # construct initial 1D x values
    maxx = int(length)/2 
    arange = np.arange(-maxx, maxx + 1)
    
    # apply the formula to each value in our array
    twoSigmaSqr = 2 * sigma * sigma
    gaussFilter = np.exp(-arange ** 2 / twoSigmaSqr)
    
    # normalize (ensure sum of matrix values is 1)
    gaussFilter /= np.sum(gaussFilter)
    
    return gaussFilter

def gauss2d(sigma):

    '''
    Returns a 2D Gaussian filter for a given sigma.
    '''

    gauss = gauss1d(sigma)[np.newaxis]
    gaussTranspose = gauss1d(sigma)[np.newaxis].transpose()

    convolved = signal.convolve2d(gauss, gaussTranspose)
    return convolved

# Write a function gaussconvolve2d(image,sigma) that 
# applies Gaussian convolution to a 2D image for the
# given value of sigma. Do this by first generating 
# a filter with your gauss2d, and then applying it 
# to the image with signal.convolve2d(image,filter,'same').
# The same option makes the result the same size as the 
# image.

# The Scipy Signal Processing toolbox also has a function
# signal.correlate2d. Applying the filter gauss2d to 
# the image with signal.correlate2d(image,filter,'same')
# produces the same result as with 
# signal.convolve2d(image,filter,'same'). Why does Scipy 
# have separate functions signal.convolve2d and 
# signal.correlate2d? HINT: Think of a situation in which
# signal.convolve2d and signal.correlate2d (with 
# identical arguments) produce different results.

# Try downloading an image of your choice from the web
# (right-click on an image in your browser and choose saveas).
# Load this image into Python, convert it to a greyscale, 
# Numpy array and run your gaussconvolve2d on it with 
# a sigma of 3.

# Show both the original and filtered images.

def gaussconvolve2d(image_array, sigma):

    '''
    Applies 2D Gaussian convolution to the given image array.
    '''

    filtered_array = signal.convolve2d(image_array, gauss2d(sigma), 'same')
    return filtered_array

def imageAsGrayscaleNumpyArray(image_str):

    '''
    Opens the image at the given location, converts it to 
    a grayscale image, and returns the numpy array.
    '''

    im = Image.open(image_str)
    im = im.convert('L')

    im_array = np.asarray(im)
    return im_array

def runMe():
    
    print 'boxfiler(3)'
    print boxfilter(3)
    print 'boxfilter(4)'
    print boxfilter(4)
    print 'boxfilter(5)'
    print boxfilter(5)

    print 'gauss1d(0.3)'
    print gauss1d(0.3)
    print 'gauss1d(0.5)'
    print gauss1d(0.5)
    print 'gauss1d(1)'
    print gauss1d(1)
    print 'gauss1d(2)'
    print gauss1d(2)

    print 'gauss2d(0.5)'
    print gauss2d(0.5)
    print np.sum(gauss2d(0.5))
    print 'gauss2d(1)'
    print gauss2d(1)

    fileName = 'waves'

    orig_arr = imageAsGrayscaleNumpyArray(fileName + '.jpg')
    orig_arr = orig_arr.astype('uint8')
    orig_im = Image.fromarray(orig_arr)
    orig_im.save('img/' + fileName + '-gray.png', 'PNG')

    for i in range(1, 10):
        im_arr = gaussconvolve2d(orig_arr, i/10.0)
        im = Image.fromarray(im_arr.astype('uint8'))
        im.save('img/' + fileName + '-blur' + str(i) + '.png', 'PNG')

        orig_copy = orig_arr.copy()
        orig_copy *= 2
        im_arr = orig_arr - im_arr
        im = Image.fromarray(im_arr.astype('uint8'))
        im.save('img/' + fileName + '-sharp' + str(i) + '.png', 'PNG')

runMe()