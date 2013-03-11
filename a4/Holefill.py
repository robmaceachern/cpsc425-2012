from PIL import Image, ImageDraw
import numpy as np
import random
import os.path
import pickle

##############################################################################
#                        Functions for you to complete                       #
##############################################################################

def ComputeSSD(TODOPatch, TODOMask, textureIm, patchL):
    '''
    ComputeSSD

    Computes the sum of squares difference between the patch and texture image  
    for each valid location of the patch within the texture image.

    TODOPatch - The image patch, of size [2 * patchL + 1, 2 * patchL + 1, 3]

    TODOMask - Specifies the elements of which elements of TODOPatch are empty and
                waiting to be filled in. A 1 represents an empty pixel.
                The first two dimensions are the same as TODOPatch, and there is no 
                third dimension.

    textureIm - The texture image. Size is [texImRows, texImCols, 3]

    patchL - The length used to define the patch size.
    '''
    patch_rows, patch_cols, patch_bands = np.shape(TODOPatch)
    tex_rows, tex_cols, tex_bands = np.shape(textureIm)
    ssd_rows = tex_rows - 2 * patchL
    ssd_cols = tex_cols - 2 * patchL
    SSD = np.zeros((ssd_rows,ssd_cols))

    for r in range(ssd_rows):
        for c in range(ssd_cols):
            # Compute sum square difference between textureIm and TODOPatch
            # for all pixels where TODOMask = 0, and store the result in SSD
            #
            # ADD YOUR CODE HERE
            #
            for patchRow in range(patch_rows):
                for patchCol in range(patch_cols):
                    if TODOMask[patchRow][patchCol] == 0:
                        patchArr = TODOPatch[patchRow][patchCol]
                        textureArr = textureIm[r + patchRow][c + patchCol]
                        SSD[r][c] += ((patchArr[0] - textureArr[0] * 1.0) ** 2)
                        SSD[r][c] += ((patchArr[1] - textureArr[1] * 1.0) ** 2)
                        SSD[r][c] += ((patchArr[2] - textureArr[2] * 1.0) ** 2)
            pass
        pass
    return SSD

def CopyPatch(imHole,TODOMask,textureIm,iPatchCenter,jPatchCenter,iMatchCenter,jMatchCenter,patchL):
    '''
    CopyPatch

    Copies the patch into the destination image at the specified location.

    imHole - The image we are copying the patch into.

    TODOMask - Specifies the elements of which elements of TODOPatch are empty and
                waiting to be filled in. A 1 represents an empty pixel.
                The first two dimensions are the same as TODOPatch, and there is no 
                third dimension.

    textureIm - The texture image. Size is [texImRows, texImCols, 3]

    iPatchCenter - The row index of the patch center in the destination image

    jPatchCenter - The column index of the patch center in the destination image

    iMatchCenter - The row index of the match center in the texture image.

    jMatchCenter - The column index of the patch center in the texture image.

    patchL - The length used to define the patch size.

    '''
    patchSize = 2 * patchL + 1
    for i in range(patchSize):
        for j in range(patchSize):
            # Copy the selected patch selectPatch into the image containing
            # the hole imHole for each pixel where TODOMask = 1.
            # The patch is centred on iPatchCenter, jPatchCenter in the image imHole
            #
            # ADD YOUR CODE HERE
            #
            if TODOMask[i][j] == 1:
                holeLoc = imHole[iPatchCenter - patchL + i][jPatchCenter - patchL + j];
                textureLoc = textureIm[iMatchCenter - patchL + i][jMatchCenter - patchL + j];
                holeLoc[0] = textureLoc[0];
                holeLoc[1] = textureLoc[1];
                holeLoc[2] = textureLoc[2];
            pass
        pass
    return imHole

##############################################################################
#                            Some helper functions                           #
##############################################################################

def UserDefinedRegion(imRows, imCols):
    print "A polygon consists of straight lines between the given coordinates,"
    print "plus a straight line between the last and the first coordinate."
    print "Enter row, column coordinates, comma separated, one point per line."
    print "End with -1,-1"
    fillPolyPoints = []
    Done = False
    while not Done:
        user_input = raw_input()
        coords = [int(x) for x in user_input.split(",")]
        assert (len(coords) == 2 and
                -1 <= coords[0] < imRows and
                -1 <= coords[1] < imCols), "Bad point coordinates"
        if coords[0] == -1 and coords[1] == -1:
            Done = True
        if not Done:
            fillPolyPoints.append(coords[1])
            fillPolyPoints.append(coords[0])
    assert len(fillPolyPoints) >= 6, "A polygon requires at least 3 points"
    img = Image.new('L', (imCols, imRows), 0)
    ImageDraw.Draw(img).polygon(fillPolyPoints, outline=1, fill=1)
    fillRegion = np.array(img, dtype=np.uint8)
    return fillRegion

def DrawBox(im,x1,y1,x2,y2):
    draw = ImageDraw.Draw(im)
    draw.line((x1,y1,x1,y2),fill="white",width=1)
    draw.line((x1,y1,x2,y1),fill="white",width=1)
    draw.line((x2,y2,x1,y2),fill="white",width=1)
    draw.line((x2,y2,x2,y1),fill="white",width=1)
    del draw
    return im

def Find_Edge(hole_mask):
    [cols, rows] = np.shape(hole_mask)
    edge_mask = np.zeros(np.shape(hole_mask))
    for y in range(rows):
        for x in range(cols):
            if (hole_mask[x,y] == 1):
                if (hole_mask[x-1,y] == 0 or
                    hole_mask[x+1,y] == 0 or
                    hole_mask[x,y-1] == 0 or
                    hole_mask[x,y+1] == 0):
                    edge_mask[x,y] = 1
    return edge_mask

##############################################################################
#                           Main script starts here                          #
##############################################################################

#
# Constants
#

# Change patchL to change the patch size used (patch size is 2 *patchL + 1)
patchL = 10
patchSize = 2*patchL+1

# Standard deviation for random patch selection
randomPatchSD = 1

# Display results interactively
showResults = True

#
# Read input image
#

im = Image.open('donkey.jpg').convert('RGB')
im_array = np.asarray(im, dtype=np.uint8)
imRows, imCols, imBands = np.shape(im_array)

#
# Define hole and texture regions.  This will use regions.pkl if it exists,
#   but otherwise will allow the user to select the regions.

if os.path.isfile('regions.pkl'):
    regions_file = open('regions.pkl', 'rb')
    fillRegion = pickle.load(regions_file)
    textureRegion = pickle.load(regions_file)
    regions_file.close()
else:
    # let user define fill region
    print "Specify the fill region by entering coordinates of the bounding polygon"
    fillRegion = UserDefinedRegion(imRows,imCols)
    # let user define texture region
    print "Specify the region to use as the texture sample"
    print "Note: This region will made rectangular"
    textureRegion = UserDefinedRegion(imRows,imCols)
    # Note: we should save these results to avoid user retyping
    regions_file = open('regions.pkl', 'wb')
    pickle.dump(fillRegion, regions_file, -1)
    pickle.dump(textureRegion, regions_file, -1)
    regions_file.close()

#
# Get coordinates for hole and texture regions
#

fill_indices = fillRegion.nonzero()
nFill = len(fill_indices[0])                # number of pixels to be filled
iFillMax = max(fill_indices[0])
iFillMin = min(fill_indices[0])
jFillMax = max(fill_indices[1])
jFillMin = min(fill_indices[1])
assert((iFillMin >= patchL) and
       (iFillMax < imRows - patchL) and
       (jFillMin >= patchL) and
       (jFillMax < imCols - patchL)) , "Hole is too close to edge of image for this patch size"

texture_indices = textureRegion.nonzero()
iTextureMax = max(texture_indices[0])
iTextureMin = min(texture_indices[0])
jTextureMax = max(texture_indices[1])
jTextureMin = min(texture_indices[1])
textureIm   = im_array[iTextureMin:iTextureMax+1, jTextureMin:jTextureMax+1, :]
texImRows, texImCols, texImBands = np.shape(textureIm)
assert((texImRows > patchSize) and
       (texImCols > patchSize)) , "Texture image is smaller than patch size"

#
# Initialize imHole for texture synthesis (i.e., set fill pixels to 0)
#

imHole = im_array.copy()
imHole[fill_indices] = 0

#
# Is the user happy with fillRegion and textureIm?
#
if showResults == True:
    # original
    im.show()
    # convert to a PIL image, show fillRegion and draw a box around textureIm
    im1 = Image.fromarray(imHole).convert('RGB')
    im1 = DrawBox(im1,jTextureMin,iTextureMin,jTextureMax,iTextureMax)
    im1.show()
    print "Are you happy with this choice of fillRegion and textureIm?"
    Yes_or_No = False
    while not Yes_or_No:
        answer = raw_input("Yes or No: ")
        if answer == "Yes" or answer == "No":
            Yes_or_No = True
    assert answer == "Yes", "You must be happy. Please try again."

#
# Perform the hole filling
#

while (nFill > 0):
    print "Number of pixels remaining = " , nFill

    # Set TODORegion to pixels on the boundary of the current fillRegion
    TODORegion = Find_Edge(fillRegion)
    edge_pixels = TODORegion.nonzero()
    nTODO = len(edge_pixels[0])

    while(nTODO > 0):

        # Pick a random pixel from the TODORegion
        index = np.random.randint(0,nTODO)
        iPatchCenter = edge_pixels[0][index]
        jPatchCenter = edge_pixels[1][index]

        # Define the coordinates for the TODOPatch
        TODOPatch = imHole[iPatchCenter-patchL:iPatchCenter+patchL+1,jPatchCenter-patchL:jPatchCenter+patchL+1,:]
        TODOMask = fillRegion[iPatchCenter-patchL:iPatchCenter+patchL+1,jPatchCenter-patchL:jPatchCenter+patchL+1]

        #
        # Compute masked SSD of TODOPatch and textureIm
        #
        ssdIm = ComputeSSD(TODOPatch, TODOMask, textureIm, patchL)

        # Randomized selection of one of the best texture patches
        ssdIm1 = np.sort(np.copy(ssdIm),axis=None)
        ssdValue = ssdIm1[min(round(abs(random.gauss(0,randomPatchSD))),np.size(ssdIm1)-1)]
        ssdIndex = np.nonzero(ssdIm==ssdValue)
        iSelectCenter = ssdIndex[0][0]
        jSelectCenter = ssdIndex[1][0]

        # adjust i, j coordinates relative to textureIm
        iSelectCenter = iSelectCenter + patchL
        jSelectCenter = jSelectCenter + patchL
        selectPatch = textureIm[iSelectCenter-patchL:iSelectCenter+patchL+1,jSelectCenter-patchL:jSelectCenter+patchL+1,:]

        #
        # Copy patch into hole
        #
        imHole = CopyPatch(imHole,TODOMask,textureIm,iPatchCenter,jPatchCenter,iSelectCenter,jSelectCenter,patchL)

        # Update TODORegion and fillRegion by removing locations that overlapped the patch
        TODORegion[iPatchCenter-patchL:iPatchCenter+patchL+1,jPatchCenter-patchL:jPatchCenter+patchL+1] = 0
        fillRegion[iPatchCenter-patchL:iPatchCenter+patchL+1,jPatchCenter-patchL:jPatchCenter+patchL+1] = 0

        edge_pixels = TODORegion.nonzero()
        nTODO = len(edge_pixels[0])

    fill_indices = fillRegion.nonzero()
    nFill = len(fill_indices[0])

#
# Output results
#
if showResults == True:
    Image.fromarray(imHole).convert('RGB').show()
Image.fromarray(imHole).convert('RGB').save('results.jpg')
