# Original source code:
#     @misc{chen2015efficientGitHub,
# 	  author = {Zongsheng Yue},
# 	  title = {Noise Level Estimation for Signal Image},
# 	  year = {2019},
# 	  publisher = {GitHub},
# 	  journal = {GitHub repository},
# 	  howpublished = {\url{https://github.com/zsyOAOA/noise_est_ICCV2015}},
#  	 commit = {a53b4dd}
# 	}

import numpy as np
import cv2
import sys
import os
import ast
import glob
from datetime import datetime

MAX_IMG_INTENSITY = 255.0
IMG_PATCH_SIZE = 128
DEFAULT_IMG_BLOCK_SIZE = 8

def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')


    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))

def noise_estimate(im, pch_size=DEFAULT_IMG_BLOCK_SIZE):
    '''
    Implement of noise level estimation of the following paper:
    Chen G , Zhu F , Heng P A . An Efficient Statistical Method for Image Noise Level Estimation[C]// 2015 IEEE International Conference
    on Computer Vision (ICCV). IEEE Computer Society, 2015.
    Input:
        im: the noise image, H x W x 3 or H x W numpy tensor, range [0,1]
        pch_size: patch_size
    Output:
        noise_level: the estimated noise level
    '''

    if im.ndim == 3:
        im = im.transpose((2, 0, 1))
    else:
        im = np.expand_dims(im, axis=0)

    # image to patch
    pch = im2patch(im, pch_size, 3)  # C x pch_size x pch_size x num_pch tensor
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))  # d x num_pch matrix
    d = pch.shape[0]

    mu = pch.mean(axis=1, keepdims=True)  # d x 1
    X = pch - mu
    sigma_X = np.matmul(X, X.transpose()) / num_pch
    sig_value, _ = np.linalg.eigh(sigma_X)
    sig_value.sort()

    for ii in range(-1, -d-1, -1):
        tau = np.mean(sig_value[:ii])
        if np.sum(sig_value[:ii]>tau) == np.sum(sig_value[:ii] < tau):
            result = np.sqrt(tau)
            result = result if not np.isnan(result) else 0.0
            return result
        
def run(imgPath, patchSize, internalNumPatches, dirOut, saveResults=True):
    """
    Estimates the standard deviation of (additive white gaussian) noise of image patches.
    The noise is estimated patch by patch.
    Based on: "An Efficient Statistical Method for Image Noise Level Estimation" (2015)
    :param imgPath: Path to the input image.
    :param patchSize: Image patch size. 
    :param internalNumPatches: Internal number of sub-image-patches.
    :param dirOut: Directory where to save the noise estimation results.
    :param saveResults: Whether to save the estimation results or not.
    :return: None
    """
    # Load image
    img = np.array(cv2.imread(imgPath))
    runtimes = []
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img / MAX_IMG_INTENSITY
    
    h, w = img.shape
    psize = min(min(patchSize, h), w)
    psize -= psize % 2
    patch_step = psize
    shift_factor = 2

    # Result array
    estimatedNoiseMap = np.zeros([h, w], dtype=np.int8)
    rangex = range(0, w, patch_step)
    rangey = range(0, h, patch_step)
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

            tileM = img[start_y:end_y, start_x:end_x]
            h_, w_ = tileM.shape
            start = datetime.now()
            sigma = noise_estimate(tileM, internalNumPatches) * MAX_IMG_INTENSITY
            end = datetime.now()
            estimatedNoiseMap[start_y :start_y + h_, start_x : start_x + w_] = sigma  
            # print(sigma)
            
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
            
            estimation, runtime = run(imgPath, IMG_PATCH_SIZE, DEFAULT_IMG_BLOCK_SIZE, dirOut, saveResults=True)
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
            estimation, _ = run(imgPath, IMG_PATCH_SIZE, DEFAULT_IMG_BLOCK_SIZE, dirOut, saveResults=True)
            estimation = estimation[..., np.newaxis]
            estimation = np.concatenate((np.zeros((estimation.shape[0], estimation.shape[1], 3)), estimation), axis=2)
            
            gtNoiseArray = [0.0, 0.0, 0.0, 0.0]
            log.write(str(imgName) + "\n")
            log.write("GT: " + str(gtNoiseArray) + "\n")
            log.write("Est.:" + str(np.mean(estimation, axis=(0,1))) + "\n")
            log.write("-------------------" + "\n")

# Example
# Uncomment the following code to test the PCA noise estimation 
# if __name__ == '__main__':        
#         baseDirIn = "./../../../data/benchmarking/udacity"
#         dirIn = os.path.join(baseDirIn, "randomSensor")
#         dirOut = os.path.join(baseDirIn, "results", "randomSensor", "PCA")
#         imgFileEnding = ".jpg"
#         metadataPathIn = os.path.join(dirIn, "metadata.txt")

#         testNoiseEstimationWithMetadata(dirIn, dirOut, metadataPathIn, imgFileEnding)
#        # testNoiseEstimationWithoutMetadata(dirIn, dirOut, imgFileEnding)
