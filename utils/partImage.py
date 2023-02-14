import math
import numpy as np

def img2patches(img, imgPatchSizeX, imgPatchSizeY, stepSizeX, stepSizeY):
    results =  []
    for x in range(0, img.shape[0], stepSizeX):
        for y in range(0, img.shape[1], stepSizeY):
            
            xOffset = x + imgPatchSizeX
            if xOffset > img.shape[0]:
                x = img.shape[0] - imgPatchSizeX
                xOffset = img.shape[0]
            yOffset = y + imgPatchSizeY
            if yOffset > img.shape[1]:
                y = img.shape[1] - imgPatchSizeY
                yOffset = img.shape[1]
            patch = img[x : xOffset, y : yOffset, ...] 
            results.append(patch)
    return np.array(results)

# Assume channel last order
def patches2img(patches, imgSizeX, imgSizeY, imgPatchSizeX, imgPatchSizeY):
    _, patchSizeC = patches.shape
    img = np.zeros([imgSizeX, imgSizeY, patchSizeC], dtype=np.float32)    
    imgPerRow = np.ceil(imgSizeY / imgPatchSizeY)
    for idxX, x in enumerate(range(0, imgSizeX, imgPatchSizeX)):
        for idxY, y in enumerate(range(0, imgSizeY, imgPatchSizeY)):
            xOffset = x + imgPatchSizeX
            if xOffset > imgSizeX:
                x = imgSizeX - imgPatchSizeX
                xOffset = imgSizeX
            yOffset = y + imgPatchSizeY
            if yOffset > imgSizeY:
                y = imgSizeY - imgPatchSizeY
                yOffset = imgSizeY
            img[x : xOffset, y : yOffset, ...] = patches[int(idxX * imgPerRow + idxY), ...]
    
    return img

# def patchLabels2labelMat(patchEstimations, imgSize, imgPatchSize):
#      labelMat = np.zeros(imgSize)
#      rows = math.ceil(imgSize[0] / imgPatchSize)
#      cols = math.ceil(imgSize[1] / imgPatchSize)
#      for i in range(rows):
#          for j in range(cols):
#              labelMat[i * imgPatchSize: imgPatchSize + i * imgPatchSize, j * imgPatchSize : imgPatchSize + j * imgPatchSize] = patchEstimations[i * cols + j]
#      return labelMat