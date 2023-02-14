import cv2
import os
import numpy as np
import math
from datetime import datetime
import ast
import glob

DEFAULT_IMG_PATCH_SIZE = 128
DEFAULT_GAUSS_FILTER_SIZE = 71
DEFAULT_MAX_STD_DEV = 100.0
PATCH_SIZE_X = 16
PATCH_SIZE_Y = 8

def image2Blocks(img):
    try:
        h, w = img.shape
    
        psize_x = min(PATCH_SIZE_X, w)
        psize_y = min(PATCH_SIZE_Y, h)
        shift_factor = 1
    
        rangex = range(0, w, psize_x)
        rangey = range(0, h, psize_y)
    
        # resize input
        allStdDevs = []
        imgBlocks = []
        minStdDev = DEFAULT_MAX_STD_DEV
        for start_x in rangex:
            for start_y in rangey:
    
                end_x = start_x + psize_x
                end_y = start_y + psize_y
                if end_x > w:
                    end_x = w
                    end_x = shift_factor * ((end_x) // shift_factor)
                    start_x = end_x - psize_x
                if end_y > h:
                    end_y = h
                    end_y = shift_factor * ((end_y) // shift_factor)
                    start_y = end_y - psize_y
                        
                imgBlock = img[start_y : end_y, start_x : end_x]
                imgBlockStd = np.std(imgBlock)
                allStdDevs.append(imgBlockStd)
                imgBlocks.append(imgBlock)
                if imgBlockStd < minStdDev:
                    minStdDev = imgBlockStd
        return minStdDev, allStdDevs, imgBlocks
    except:
        return None
    
def computeHomogeneousBlocks(minStdDev, allStdDevs, imgBlocks):
    homogBlocks = []
    for idx in range(len(allStdDevs)):
        stdDev = allStdDevs[idx]
        imgBlock = imgBlocks[idx]
        
        if abs(math.floor(minStdDev) - round(minStdDev)) < 1.0:
            if abs(math.floor(stdDev) - math.floor(minStdDev)) < 2.0:
                homogBlocks.append(imgBlock)
        else:
            if abs(round(stdDev) - round(minStdDev)) < 2.0:
                homogBlocks.append(imgBlock)
    
    return homogBlocks

def addGaussianBlur(img, sigma, filterSize=DEFAULT_GAUSS_FILTER_SIZE, borderType=cv2.BORDER_DEFAULT):
    blurredImg = cv2.GaussianBlur(img, (filterSize, filterSize), sigma, borderType=borderType)
    return blurredImg

def blurBlocks(homogBlocks, minStdDev):
    blurredBlocks = []
    for homogBlock in homogBlocks:
        kernelSize = int(math.ceil(minStdDev * float(3.0)))
        kernelSize = kernelSize+1 if kernelSize % 2 == 0 else kernelSize
        blurredBlock = addGaussianBlur(homogBlock, minStdDev, filterSize=kernelSize)
        blurredBlocks.append(blurredBlock)
    return blurredBlocks
    
def stdDevOfBlockDiffs(homogBlocks, blurredHomogBlocks):
    diffs = []
    n, h, w = homogBlocks.shape
    for idx in range(len(homogBlocks)):
        homogBlock = homogBlocks[idx]
        blurredHomogBlock = blurredHomogBlocks[idx]
        diff = homogBlock.astype("float32") - blurredHomogBlock.astype("float32")
        diffs.append(diff)
        
    return np.std(diffs)
    
def run(imgPath, dirOut, patchSize=DEFAULT_IMG_PATCH_SIZE, saveResults=False):
    """
    Estimates the standard deviation of (additive white gaussian) noise of image patches.
    The noise is estimated patch by patch.
    Based on: "Block Based Noise Estimation Using Adaptive Gaussian Filtering" (2005)
    :param imgPath: Path to the input image.
    :param dirOut: Directory where to save the noise estimation results.
    :param patchSize: Image patch size. 
    :param saveResults: Whether to save the estimation results or not.
    :return: None
    """
    # Load image
    img = np.array(cv2.imread(imgPath))   
    runtimes = []
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    psize = min(min(patchSize, h), w)
    psize -= psize % 2
    patch_step = psize
    shift_factor = 2

    estimatedNoiseMap = np.zeros([h, w], dtype=np.int8)
    rangex = range(0, w, patch_step)
    rangey = range(0, h, patch_step)
    # Iterate the image patches
    for start_x in rangex:
        for start_y in rangey:

            end_x = start_x + psize
            end_y = start_y + psize
            if end_x > w:
                end_x = w
                end_x = shift_factor * ((end_x) // shift_factor)
                start_x = end_x - psize
            if end_y > h:
                end_y = h
                end_y = shift_factor * ((end_y) // shift_factor)
                start_y = end_y - psize

            patch = img[start_y:end_y, start_x:end_x]
            h_, w_ = patch.shape
            
            start = datetime.now()
            # Split the patch into small sub-patches (blocks) and determine their standard deviations.
            minStdDev, allStdDevs, imgBlocks = image2Blocks(patch)
            # Find the homog. blocks according to the min. standard deviation.
            homogBlocks = computeHomogeneousBlocks(minStdDev, allStdDevs, imgBlocks)
            # Blur the homogeneous blocks with the min. standard deviation as filter kernel size.
            blurredHomogBlocks = blurBlocks(homogBlocks, minStdDev)
            # Calculate overall std. dev. of intensities from homog. blocks minus filtered blocks.
            sigma = stdDevOfBlockDiffs(np.array(homogBlocks), np.array(blurredHomogBlocks))
            end = datetime.now()
            # Apply std. dev. as noise estimation for the respectivei mage patch.
            estimatedNoiseMap[start_y :start_y + h_, start_x : start_x + w_] = sigma
            
            runtime = (end - start).total_seconds()
            if runtime > 0.0:
                runtimes.append(runtime)
                
    if saveResults:
        if dirOut is not None:
            imgName = imgPath.split(os.sep)[-1].split(".")[0]
            dirOut = os.path.join(dirOut)
            if not os.path.exists(dirOut):
                os.makedirs(dirOut)
                
            noiseMapPath = os.path.join(dirOut, imgName + ".npz")
            if not os.path.exists(noiseMapPath):
                np.savez_compressed(noiseMapPath, estimatedNoiseMap) 
            
    runtime = np.mean(runtimes)
            
    return estimatedNoiseMap, runtime
    
def testNoiseEstimationWithMetadata(dirIn, dirOut, metadataPathIn, imgFileEnding):
    """
    Test the noise estimator against ground truth noise levels.
    :param dirIn: path to directory with noised input images to test.
    :param dirOut: path to directory to save test results.
    :param imgFileEnding: string to specify file ending for test images.
    :return: -
    """
    
    if not os.path.exists(dirOut):
        os.makedirs(dirOut)
        
    metadataFile = open(metadataPathIn, 'r')
    lines = metadataFile.readlines()
    runtimes = []
    with open(os.path.join(dirOut, "log.txt"), "w+") as log:
        for line in lines:
            results = ast.literal_eval(line.strip())
            imgName = results["imgName"]
            imgPath = os.path.join(dirIn, imgName + imgFileEnding)
            
            estimation, runtime = run(imgPath, dirOut, patchSize=128, saveResults=True)
            estimation = estimation[..., np.newaxis]
            estimation = np.concatenate((np.zeros((estimation.shape[0], estimation.shape[1], 3)), estimation), axis=2)
            
            if runtime > 0.0:
                runtimes.append(runtime)
            
            gtNoiseValues = [results["photonShotNoiseLevel"], results["darkCurrentShotNoiseLevel"], results["readoutNoiseLevel"], results["residualNoiseLevel"]]
            gtNoiseValue = np.sqrt(np.power(gtNoiseValues[0], 2.0) + np.power(gtNoiseValues[1], 2.0) + np.power(gtNoiseValues[2], 2.0) + np.sign(gtNoiseValues[3]) * np.power(gtNoiseValues[3], 2.0))
            gtNoiseArray = [0.0, 0.0, 0.0, gtNoiseValue]
            
            log.write(str(imgName) + "\n")
            log.write("GT: " + str(gtNoiseArray) + "\n")
            log.write("Est.:" + str(np.mean(estimation, axis=(0,1))) + "\n")
            log.write("-------------------" + "\n")
        
    print("Overall average time per image patch:", np.mean(runtimes) * 1000.0, "ms.")
            
def testNoiseEstimationWithoutMetadata(dirIn, dirOut, imgFileEnding):
    """
    Test the noise estimator without having ground truth noise levels.
    :param dirIn: path to directory with noised input images to test.
    :param dirOut: path to directory to save test results.
    :param imgFileEnding: string to specify file ending for test images.
    :return: -
    """
    
    if not os.path.exists(dirOut):
        os.makedirs(dirOut)
    with open(os.path.join(dirOut, "log.txt"), "w+") as log:
        for imgPath in glob.glob(os.path.join(dirIn, "*" + imgFileEnding)):
            imgName = imgPath.split(os.sep)[-1].split(".")[0]
            estimation, _ = run(imgPath, dirOut, patchSize=128, saveResults=True)
            estimation = estimation[..., np.newaxis]
            estimation = np.concatenate((np.zeros((estimation.shape[0], estimation.shape[1], 3)), estimation), axis=2)
            
            gtNoiseArray = [0.0, 0.0, 0.0, 0.0]
            log.write(str(imgName) + "\n")
            log.write("GT: " + str(gtNoiseArray) + "\n")
            log.write("Est.:" + str(np.mean(estimation, axis=(0,1))) + "\n")
            log.write("-------------------" + "\n")

# Uncomment the following code to test the B+F noise estimation 
# if __name__ == '__main__':        
#         baseDirIn = "./../../../data/udacity"
#         dirIn = os.path.join(baseDirIn, "randomSensor")
#         dirOut = os.path.join(baseDirIn, "results", "randomSensor", "B+F")
#         imgFileEnding = ".jpg"
#         metadataPathIn = os.path.join(dirIn, "metadata.txt")

#         # testNoiseEstimationWithoutMetadata(dirIn, dirOut, imgFileEnding)
#         testNoiseEstimationWithMetadata(dirIn, dirOut, metadataPathIn, imgFileEnding)


