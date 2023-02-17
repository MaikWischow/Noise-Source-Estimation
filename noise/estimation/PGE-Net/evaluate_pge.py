import torch
import numpy as np
import random
from datetime import datetime
import os
import cv2
import ast

from unet import est_UNet

MAX_IMG_INTENSITY = 255.0
IMG_PATH_SIZE = 128

# control the randomness
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=True
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    
def loadModel(modelPath):
    num_output_channel = 2
    pge_model=est_UNet(num_output_channel, depth=3)
    pge_model.load_state_dict(torch.load(modelPath))
    pge_model.cuda()
    return pge_model

def estimateNoise(model, imgGT, imgNoised, dirOut):
    with torch.no_grad():
        imgNoised = imgNoised[..., np.newaxis]
        img_cuda = torch.from_numpy(imgNoised.reshape(1, 1, imgNoised.shape[0], imgNoised.shape[1])).float().cuda()
        
        start = datetime.now()
        est_param = model(img_cuda)
        end = datetime.now()
        
        h, w = imgGT.shape
        numImgPatches = h // IMG_PATH_SIZE * w // IMG_PATH_SIZE
        runtime = (end - start).total_seconds() / numImgPatches
        
        original_alpha=est_param[:,0]
        original_beta=est_param[:,1] 
        alphas=original_alpha.cpu().numpy()[0,...]
        sigmas=original_beta.cpu().numpy()[0,...]
        
        return alphas, sigmas, runtime    
    
def test(model, dirInGT, dirInNoised, dirOut, imgFileEnding, metadataPathIn):
    """
    Test the PGE-Net noise estimators.
    :param model: loaded PGe-Net model.
    :param dirInGT: path to directory with uncorrupted input images.
    :param dirInNoised: path to directory with noised input images to test.
    :param dirOut: path to directory to save test results.
    :param imgFileEnding: string to specify file ending for test images.
    :param metadataPathIn: path to file that contains ground truth noise estimations and metadata for each image in dirInNoised
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
            imgPathGT = os.path.join(dirInGT, imgName + imgFileEnding)
            imgPathNoised = os.path.join(dirInNoised, imgName + imgFileEnding)
            
            # Assure that unnoised and noise images are grayscale and intensities in [0,1]
            imgGT = cv2.imread(imgPathGT)
            if len(imgGT.shape) == 3:
                imgGT = cv2.cvtColor(imgGT, cv2.COLOR_RGB2GRAY)
            imgGT = imgGT / MAX_IMG_INTENSITY
            imgNoised = cv2.imread(imgPathNoised)
            if len(imgNoised.shape) == 3:
                imgNoised = imgNoised[..., 0]
            imgNoised = imgNoised / MAX_IMG_INTENSITY
            hGT, wGT = imgGT.shape
            hN, wN = imgNoised.shape

            # Scale uncorrupted image size to size of noise image.
            if hGT > hN or wGT > wN:
                hScalingFactor = float(hN) / hGT
                wScalingFactor = float(wN) / wN
                scalingFactor = min(hScalingFactor, wScalingFactor)
                imgGT = cv2.resize(imgGT, None, fx=scalingFactor, fy=scalingFactor, interpolation=cv2.INTER_AREA)
            
            # Run the estimator and measure its inference time per image patch
            alphas, sigmas, runtime = estimateNoise(model, imgGT, imgNoised, dirOut)
            if runtime > 0.0:
                runtimes.append(runtime)
            
            # Calculate estimated noise levels according to the original paper ("Fbi-denoiser: Fast blind image denoiser for poisson-gaussian noise")
            totalNoiseLevels = np.sqrt(alphas**2 * imgGT + sigmas**2) * MAX_IMG_INTENSITY
            gaussNoiseLevels = sigmas * MAX_IMG_INTENSITY
            poissonNoiseLevels = np.sqrt(totalNoiseLevels**2 - gaussNoiseLevels**2)
            estimation = [np.nanmean(poissonNoiseLevels), 0.0, 0.0, np.mean(totalNoiseLevels)] 
            gtNoiseValues = [results["photonShotNoiseLevel"], results["darkCurrentShotNoiseLevel"], results["readoutNoiseLevel"], results["residualNoiseLevel"]]
            
            # Log the results
            log.write(str(imgName) + "\n")
            log.write("GT: " + str(gtNoiseValues) + "\n")
            log.write("Est.:" + str(estimation) + "\n")
            log.write("-------------------" + "\n")
            
            # print(str(imgName) + "\n")
            # print("GT: " + str(gtNoiseValues) + "\n")
            # print("Est.:" + str(estimation) + "\n")
            # print("-------------------" + "\n")
        
    print("Overall average time per image patch:", np.mean(runtimes) * 1000.0, "ms.")

# Uncomment the following code to test the PGE-Net
# if __name__ == '__main__':

#     baseDirIn = "./../../../data/benchmarking/udacity"
#     dirInGT = os.path.join(baseDirIn, "GT")
#     dirInNoised = os.path.join(baseDirIn, "randomSensor")
#     dirOut = os.path.join(baseDirIn, "results", "randomSensor", "PGE")
#     imgFileEnding = ".jpg"
#     metadataPathIn = os.path.join(dirInNoised, "metadata.txt")
#     modelPath = r'.\weights\synthetic_noise/211127_PGE_Net_RawRGB_random_noise_cropsize_200.w'

#     model = loadModel(modelPath)
#     test(model, dirInGT, dirInNoised, dirOut, imgFileEnding, metadataPathIn)