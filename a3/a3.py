from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc

def MakePyramid(image, minsize):
    '''
    Creates a pyramid for an image.

    Returns a list including the original PIL image followed 
    by all the PIL images of reduced size, using a scale factor 
    of 0.75 from one level to the next.

    The pyramid will stop when any further reduction in size will 
    make the dimension of the image smaller than minsize.
    '''
    pyramid = []
    im = image
    while (im.size[0] >= minsize) and (im.size[1]) >= minsize:
        pyramid.append(im);
        im = im.resize((int(im.size[0] * 0.75), int(im.size[1] * 0.75)), Image.BICUBIC)

    return pyramid

def ShowPyramid(pyramid):
    '''
    Joins the images in the list into a single horizontal image 
    and displays them with im.show()
    '''

    width = 0
    height = 0

    for img in pyramid:
        width = width + img.size[0]
        height = max(height, img.size[1])

    image = Image.new("RGB", (width, height), 0xFFFFFF)

    offsetX = 0
    offsetY = 0

    for img in pyramid:
        image.paste(img, (offsetX, offsetY))
        offsetX = offsetX + img.size[0]
        offsetY = 0

    image.show()

def FindTemplate(pyramid, template, threshold):
    '''
    Finds and marks all locations in pyramid at which the normalized 
    cross correlation of the template with the image is above the threshold
    '''

    goalWidth = 15
    # resize template
    ratio = template.size[0] / goalWidth
    template = template.resize((goalWidth, template.size[1] / ratio), Image.BICUBIC)

    points = []
    for i in range(len(pyramid)):
        image = pyramid[i]
        nccResult = ncc.normxcorr2D(image, template)
        aboveThreshold = np.where(nccResult > threshold)

        points.append(zip(aboveThreshold[1] / (0.75**i), aboveThreshold[0] / (0.75**i)))

    convert = pyramid[0].convert('RGB')
    for i in range(len(points)):
        pointList = points[i]
        scaleFactor = 0.75 ** i

        for pt in pointList:
            x1 = pt[0] - template.size[0]//(2 * scaleFactor)
            y1 = pt[1] - template.size[1]//(2 * scaleFactor)
            x2 = pt[0] + template.size[0]//(2 * scaleFactor)
            y2 = pt[1] + template.size[1]//(2 * scaleFactor)
            draw = ImageDraw.Draw(convert)
            draw.rectangle([x1,y1,x2,y2], outline="red")
            del draw

    convert.show()

def runMe():
    imgLocs = ['faces/judybats.jpg']

    for imgLoc in imgLocs:
        img = Image.open(imgLoc)
        img = img.convert('L')

        pyramid = MakePyramid(img, 20)
        ShowPyramid(pyramid)

        templateLoc = 'faces/template.jpg'
        template = Image.open(templateLoc)
        template = template.convert('L')

        FindTemplate(pyramid, template, 0.55)
    
runMe()