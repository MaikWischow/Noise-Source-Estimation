import cv2
import math
import os
import numpy as np
import random
import glob
import copy
from cameraParameters import *
from ast import literal_eval

def logParams(dirOut, p):
    with open(os.path.join(dirOut,'parameters.txt'), 'w') as f:
        print(p, file=f)

def kelvin2celsius(kelvin):
    return kelvin - k2c

def celsius2kelvin(celsius):
    return celsius + k2c

def gainDB2Factor(gainDb):
    return pow(10, gainDb/20.0)

def getRandImg(imgDir, fileEnding):
    # get random image
    imgPaths = glob.glob(os.path.join(imgDir, "*" + fileEnding))
    randNum = random.randint(0, len(imgPaths) - 1)
    imgPath = imgPaths[randNum]
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    imgName = imgPath.split(os.sep)[-1].split(".")[0]
    return img, imgName

def sampleRandGauss(mean, std):
    U, V, W = np.random.uniform(smallNum, 1.0, mean.shape), np.random.uniform(0.0, 1.0, mean.shape), np.random.uniform(0.0, 1.0, mean.shape)
    return np.where(
                W < 0.5, 
                (np.sqrt(-2.0 * np.log(U)) * np.sin(2.0 * math.pi * V)) * std + mean, 
                (np.sqrt(-2.0 * np.log(U)) * np.cos(2.0 * math.pi * V)) * std + mean
            )

def sampleRandPoisson(mean):
    return np.random.poisson(mean).astype("float32")

def estimatePhotonNumber(imgIn, offset, gain, fullWellSize):
    normalizedOffset = offset / NmaxIn
    return (imgIn - normalizedOffset) / gain * fullWellSize

def calculateSourceFollNoise(fClock, fClockStepSize, senseNodeGain, sourceFollGain, corrDoubleSampStoSTime, \
                             corrDoubleSampTimeFact, thermalWhiteNoise, flickerNoiseCornerFreq, sensorType, sourceFollowerCurrMod):
    fClockRange = range(1, int(fClock)+1, fClockStepSize)
    sfFreqSum = fClockStepSize * np.sum(np.array([sourceFollPowerSpec(f, thermalWhiteNoise, flickerNoiseCornerFreq, sensorType, corrDoubleSampTimeFact, corrDoubleSampStoSTime, sourceFollowerCurrMod) * corrDoubleSampTrans(f, corrDoubleSampTimeFact, corrDoubleSampStoSTime) for f in fClockRange]))
    sourceFollStdDev = np.sqrt(sfFreqSum) / (senseNodeGain * sourceFollGain * (1.0 - np.exp(-corrDoubleSampStoSTime / (corrDoubleSampTimeFact * corrDoubleSampStoSTime))))
    return sourceFollStdDev

def sourceFollPowerSpec(freq, thermalWhiteNoise, flickerNoiseCornerFreq, sensorType, corrDoubleSampTimeFact, corrDoubleSampStoSTime, sourceFollowerCurrMod):
	return math.pow(thermalWhiteNoise, 2.0) * (1.0 + flickerNoiseCornerFreq / freq) + randTelNoisePowerSpec(freq, sensorType, corrDoubleSampTimeFact, corrDoubleSampStoSTime, sourceFollowerCurrMod)

def randTelNoisePowerSpec(freq, sensorType, corrDoubleSampTimeFact, corrDoubleSampStoSTime, sourceFollowerCurrMod):
    if sensorType.lower() == 'cmos':
        randTelNoiseTimeConst = 0.1 * corrDoubleSampTimeFact * corrDoubleSampStoSTime; # in s
        termA = 2.0 * math.pow(sourceFollowerCurrMod, 2.0) * randTelNoiseTimeConst;
        termB = 4.0 + math.pow(2.0 * math.pi * freq * randTelNoiseTimeConst, 2.0);
        return termA / termB;
    else:
        return 0.0
    
def corrDoubleSampTrans(freq, corrDoubleSampTimeFact, corrDoubleSampStoSTime):
	termA = 1.0 / (1.0 + math.pow(2.0 * math.pi * freq * corrDoubleSampTimeFact * corrDoubleSampStoSTime, 2.0));
	termB = 2.0 - 2.0 * math.cos(2.0 * math.pi * freq * corrDoubleSampStoSTime);
	return termA * termB;

def calculateDarkSignal(temperature, texp, pixelSize, darkSignalFoM):
    eGap = siEGap0 - ((siAlpha * np.power(temperature, 2.0)) / (temperature + siBeta));
    dsTemp = np.power(temperature, 1.5) * np.exp(-1.0 * eGap / (2.0 * boltzmannConstEV * temperature));
    darkSignal =  2.55 * 1e15 * texp * np.power(pixelSize, 2.0) * darkSignalFoM * dsTemp;
    return darkSignal
              
def applyHighLevelNoise(imgIn, params, returnNoiseLevel=True):
    """
    Simulate noise and add it to a given input image.
    :param imgIn: input image with three dimensional shape (width, height, channels).
    :param paramsIn: dictionary with configuration parameters overriding the default parameters.
    :return: noised input image.
    """
     
    # Check if image has three dimensions.
    if len(imgIn.shape) == 3:
        height, width, channels = imgIn.shape
    else:
        raise Exception("Expected input image to have 3 dimensions. Input image shape:", imgIn.shape)
        
    if len(params) == 0:
        params = getDefaultCameraMetadata()
         
     # Initialize uniform matrices.
    zeroes = np.full((height, width, channels), 0.0)
    ones = np.full((height, width, channels), 1.0)
    
    # Estimate interacting photons from input image (assuming quantum yield of 1)
    
    # Add photon noise: 1) sample noise, 2) amplify noise, 3) use amplified noise
    if(params["applyPhotonNoise"]):
        photons = estimatePhotonNumber(imgIn, params["offset"], params["cameraGainFactor"], params["fullWellSize"])
        electrons = sampleRandPoisson(photons)
        photonShotNoise = photons.astype("float32") - electrons.astype("float32")
        electrons = photons * params["cameraGainFactor"] + photonShotNoise
        electrons = np.clip(electrons, 0, params["fullWellSize"])
    else:
        electrons = imgIn * params["fullWellSize"]
        
    # Add dark current and dark current shot noise
    temperature = params["temperature"] * ones
    if(params["applyDarkCurrent"]):
        darkSignal = calculateDarkSignal(temperature, params["exposureTime"], params["sensorPixelSize"], params["darkSignalFoM"])
        darkSignalWithDarkNoise = sampleRandPoisson(darkSignal);
        if params["darksignalCorrection"]:
            darkSignalWithDarkNoise -= darkSignal
        electrons = electrons + darkSignalWithDarkNoise
        
    # Source follower noise
    if(params["applySourceFollwerNoise"]):
        sourceFollStdDev = calculateSourceFollNoise(params["fClock"], fClockStepSize, params["senseNodeGain"], params["sourceFollGain"], \
                                                    params["corrDoubleSampStoSTime"], params["corrDoubleSampTimeFact"], params["thermalWhiteNoise"], \
                                                    params["flickerNoiseCornerFreq"], params["sensorType"], params["sourceFollowerCurrMod"])
        
        sourceFollStdDevMat = np.full((height, width, channels), sourceFollStdDev)
        electrons += np.round(sampleRandGauss(zeroes, sourceFollStdDevMat))
    
    # Truncate to full well size and round to full electrons
    electrons = np.floor(np.clip(electrons, 0.0, params["fullWellSize"]))
                
    # Convert charge to voltage
    chargeNodeRefVoltMat = np.full((height, width, channels), params["chargeNodeRefVolt"])
    senseNodeCap = q / params["senseNodeGain"]
    if params["sensorType"].lower() == 'cmos':
        # Add kTc noise
        if (params["applyKtcNoise"]):
            ktcNoiseStdDev = np.sqrt((boltzmannConstJK * temperature) / senseNodeCap) # in V
            ktcNoiseStdDevMat = np.full((height, width, channels), ktcNoiseStdDev)
            ktcNoise = np.exp(sampleRandGauss(zeroes, ktcNoiseStdDevMat)) - 1.0
            voltage = (chargeNodeRefVoltMat + params["senseNodeResetFactor"] * ktcNoise) - (electrons * params["senseNodeGain"])
        else:
            voltage = chargeNodeRefVoltMat - (electrons * params["senseNodeGain"])
    elif params["sensorType"].lower() == 'ccd':
        voltage = chargeNodeRefVoltMat - (electrons * params["senseNodeGain"])
    else:
        raise Exception("Unsupported sensor type:", params["sensorType"])
       
    # Apply source follower gain
    voltage *=  params["sourceFollGain"]
    
    # Add influence of correlated double sampling
    voltage *= params["corrDoubleSampGain"];
    
    # Convert voltage to digital numbers
    Vmin = q * params["senseNodeGain"] / senseNodeCap
    Vmax = params["fullWellSize"] * q / senseNodeCap
    adGain = NmaxOut /(Vmax - Vmin)
    dn = params["offset"] + adGain * (chargeNodeRefVoltMat - voltage)
    
    # Extract and amplify noise
    imgIn = imgIn * NmaxOut
    noiseMap = dn.astype("float32") - imgIn.astype("float32")
    noiseMap = noiseMap * params["cameraGainFactor"]
    dn = imgIn + noiseMap
    dn = np.clip(dn, 0.0, NmaxOut)
    noiseMap = dn.astype("float32") - imgIn.astype("float32")
    
    if returnNoiseLevel:
        noise = np.std(noiseMap)
    else:
        noise = noiseMap
    
    return dn, noise
 
def prepareImage(img):
    """
    Convert image to three dimensions (widht, height, channels) and normalize intensity values to range [0,1].
    :param img: Input gray-scale image with dimensions (widht, height) or (width, height, channels) with #(channels) in {1,3}.
    :return: Input image of dimension (width, height, channels=1) with intensities in range [0,1].
    """
    if len(img.shape) == 3 and img.shape[2] == 1:
        pass
    elif len(img.shape) == 3 and img.shape[2] != 1:
        img = img[..., 1]
        img = np.expand_dims(img, axis=2)
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    else:
        raise Exception("Unexpected image dimension. Expected: (widht, height) or (width, height, channels) with #(channels) in {1,3}. Given:", img.shape)
    
    img = img.astype("float32") 
    if np.max(img) > 1.0:
        img = img / (NmaxIn)
    
    return img

def applyPhotonNoise(img, paramsIn={}, returnNoiseLevel=True):
    """
    Apply only photon noise with desired mean noise level to a given input image.
    :param img: Input gray-scale image.
    :param noiseLevel: Noise level in digital numbers (e.g., between 1 and 30).
    :return: Photon noised input image.
    """
    paramsOut = paramsIn.copy()
    paramsOut["applyPhotonNoise"] = True
    paramsOut["applyDarkCurrent"] = False
    paramsOut["darksignalCorrection"] = False
    paramsOut["applySourceFollwerNoise"] = False
    paramsOut["applyKtcNoise"] = False
    
    img = prepareImage(img)
    img, noise = applyHighLevelNoise(img, paramsOut, returnNoiseLevel=returnNoiseLevel)
    return img[..., 0], noise

def applyDarkCurrentShotNoise(img, paramsIn={}, returnNoiseLevel=True):
    """
    Apply dark current shot noise with desired mean noise level to a given input image.
    :param img: Input gray-scale image.
    :param noiseLevel: Noise level in digital numbers (e.g., between 1 and 30).
    :return: Dark current shot noised input image.
    """
    paramsOut = copy.deepcopy(paramsIn)
    paramsOut["applyPhotonNoise"] = False
    paramsOut["applyDarkCurrent"] = True
    paramsOut["darksignalCorrection"] = True
    paramsOut["applySourceFollwerNoise"] = False
    paramsOut["applyKtcNoise"] = False
        
    img = prepareImage(img)
    img, noise = applyHighLevelNoise(img, paramsOut, returnNoiseLevel=returnNoiseLevel) 
    return img[..., 0], noise

def applyReadoutNoise(img, paramsIn={}, returnNoiseLevel=True):
    """
    Apply read noise with desired mean noise level to a given input image.
    :param img: Input gray-scale image.
    :param noiseLevel: Noise level in digital numbers (e.g., between 1 and 30).
    :return: Read noised input image.
    """
    paramsOut = copy.deepcopy(paramsIn)
    paramsOut["applyPhotonNoise"] = False
    paramsOut["applyDarkCurrent"] = False
    paramsOut["darksignalCorrection"] = False
    paramsOut["applySourceFollwerNoise"] = True
    paramsOut["applyKtcNoise"] = True
    
    img = prepareImage(img)
    img, noise = applyHighLevelNoise(img, paramsOut, returnNoiseLevel=returnNoiseLevel)
    return img[..., 0], noise

def applyPhotonDarkReadNoise(img, paramsIn={}, returnNoiseLevel=True):
    """
    Apply photon shot noise, dark current shot noise, and readout noise to a given input image.
    :param img: input gray-scale image.
    :return:noised input image.
    """
    paramsOut = copy.deepcopy(paramsIn)
    paramsOut["applyPhotonNoise"] = True
    paramsOut["applyDarkCurrent"] = True
    paramsOut["darksignalCorrection"] = True
    paramsOut["applySourceFollwerNoise"] = True
    paramsOut["applyKtcNoise"] = True
    
    img = prepareImage(img)
    img, noise = applyHighLevelNoise(img, paramsOut, returnNoiseLevel=returnNoiseLevel)
    return img[..., 0], noise

def generateSimulatedTestImgsWithMetadata(dirIn, dirOut, imgFileEnding, metadata, useRandomMetadata=False):
    """
     Add synthetic noise to some image dataset.
    :param dirIn: path to directory of unnoised image dataset.
    :param dirOut: path to directory to save resulting noised dataset.
    :param imgFileEnding: file ending of images in provided image dataset.
    :param metadata: (optional) metadata of a camera; None: random metadata is generated (i.e., useRandomMetadata is set to True)
    :param useRandomMetadata: True: generate random camera metadata for each image; False: use provided (fixed) metadata.
    :return:noised input image.
    """

    if not os.path.exists(dirOut):
        os.makedirs(dirOut)
    
    imgPaths = glob.glob(os.path.join(dirIn, "*" + imgFileEnding))
    with open(os.path.join(dirOut, 'metadata.txt'), 'w') as f:
        for imgPath in imgPaths:
            imgName = imgPath.split(os.sep)[-1].split(".")[0]
            img = cv2.imread(imgPath)[...,0].astype("float32") / (NmaxIn)
            h, w = img.shape
            
            useRandomMetadata = True if metadata is None else False
            if useRandomMetadata:
                metadata = randomSensor()
            metadata = randomEnvCondition(metadata)
                
            # Apply noise sources
            img, photonShotNoiseLevel = applyPhotonNoise(img, metadata)
            img, readoutNoiseLevel = applyReadoutNoise(img, metadata)
            img, dcsnLevel = applyDarkCurrentShotNoise(img, metadata)
            
            # Normalize Metadata to ranges [0,1]
            normalizedMetadata = normalizeCameraMetadata(metadata)
            
            # Prepare results log
            results = {}
            results["imgName"] = imgName
            results["photonShotNoiseLevel"] = photonShotNoiseLevel
            results["darkCurrentShotNoiseLevel"] = dcsnLevel
            results["readoutNoiseLevel"] = readoutNoiseLevel
            results["residualNoiseLevel"] = 0.0
            results["cameraGainFactor"] = normalizedMetadata["cameraGainFactor"]
            results["temperature"] = normalizedMetadata["temperature"]
            results["exposureTime"] = normalizedMetadata["exposureTime"]
            results["fullWellSize"] = normalizedMetadata["fullWellSize"]
            results["sensorType"] = normalizedMetadata["sensorType"]
            results["sensorPixelSize"] = normalizedMetadata["sensorPixelSize"]
            results["darkSignalFoM"] = normalizedMetadata["darkSignalFoM"]
            results["fClock"] = normalizedMetadata["fClock"]
            results["senseNodeGain"] = normalizedMetadata["senseNodeGain"]
            results["thermalWhiteNoise"] = normalizedMetadata["thermalWhiteNoise"]
            results["senseNodeResetFactor"] = normalizedMetadata["senseNodeResetFactor"]
                
            # Write image and log
            cv2.imwrite(os.path.join(dirOut, imgName + imgFileEnding), img.astype("uint8"))    
            f.write(str(results) + "\n")
            
def generateModelNoise(dirIn, dirOut, fileNameOut, imgFileEnding, metadataFileName):
    """
     Estimate noise levels for the single noise sources on metadata (the image only provides the intensity for photon noise estimation).
    :param dirIn: path to directory of images and metadata file.
    :param dirOut: path to directory to save resulting log file.
    :param imgFileEnding: file ending of images in provided image dataset.
    :param metadataFileName: log file name that includes the resulting calculated noise levels.
    :return:noised input image.
    """
    with open(os.path.join(dirOut, fileNameOut), 'w+') as resultFile:
        with open(os.path.join(dirIn, metadataFileName), 'r') as metadataFile:
            lines = metadataFile.readlines()
            for imgPath in glob.glob(os.path.join(dirIn, "*" + imgFileEnding)):
                imgName = imgPath.split(os.sep)[-1].split(".")[0]
                for line in lines:
                    metadata = literal_eval(line.rstrip())
                    if metadata["imgName"] == imgName:
                      metadata = denormalizeCameraMetadata(metadata)
                      metadata = addFixedMetadata(metadata)
                      break
                    
                img = cv2.imread(imgPath)
                _, pNoiseLevel = applyPhotonNoise(img, metadata)
                _, dcsNoiseLevel = applyDarkCurrentShotNoise(img, metadata)
                _, rNoiseLevel = applyReadoutNoise(img, metadata)
                
                # Log calculated noise levels
                resultDict = {imgName : [pNoiseLevel, dcsNoiseLevel, rNoiseLevel]}
                resultFile.write(str(resultDict) + "\n")
                
def generateRealTestImgsWithMetadata(dirIn, dirOut, pathInNoiseData, metadata, imgFileEnding, noiseImgFileEnding):
    """
     Add real-world noise to some image dataset.
     Attention: Noise images are not included in the repository.
    :param dirIn: path to directory of unnoised image dataset.
    :param dirOut: path to directory to save resulting noised dataset.
    :param pathInNoiseData: path to .npz file that includes the path and statistics to real-world noise images.
    :param metadata: metadata of a camera
    :param imgFileEnding: file ending of images in provided image dataset.
    :param noiseImgFileEnding: file ending of real-world noise images.
    :return: -
    """
    
    if not os.path.exists(dirOut):
        os.makedirs(dirOut)
    
    # Load images and real-world noise data
    imgPaths = glob.glob(os.path.join(dirIn, "*" + imgFileEnding))
    cleanNoiseData = np.load(pathInNoiseData)["arr_0"]
    with open(os.path.join(dirOut, 'metadata.txt'), 'w') as f:
        for imgPath in imgPaths:
            imgName = imgPath.split(os.sep)[-1].split(".")[0]
            img = cv2.imread(imgPath)[...,0].astype("float32") / (NmaxIn)
            hImg, wImg = img.shape
            
            # Load calculated std. devs. (noise level) for real noise images
            # Attention: Noise images are not included in the repository.
            randNum = random.randint(0, len(cleanNoiseData) - 1)
            rnDirOut, dcsnDirOut, temp, gain, expTime = cleanNoiseData[randNum]
            sessionNr = int(dcsnDirOut.split(os.sep)[-2])
            rnStdDevs = np.load(os.path.join(rnDirOut, "noiseLevelsByImgName.npz"))["arr_0"]
            rnStdDevs = {i[0] : float(i[1]) for i in rnStdDevs}
            dcsnStdDevs = np.load(os.path.join(dcsnDirOut, "noiseLevelsByImgName.npz"))["arr_0"]
            dcsnStdDevs = {i[0] : float(i[1]) for i in dcsnStdDevs}
            
            # Load corresponding metadata
            metadata["temperature"] = celsius2kelvin(float(temp))
            metadata["cameraGainFactor"] = gainDB2Factor(float(gain))
            metadata["exposureTime"] = float(expTime)
            
            # Load img patches
            dcsnDataEntryImg, dcsnImgName = getRandImg(dcsnDirOut, noiseImgFileEnding)
            hDCSN, wDCSN = dcsnDataEntryImg.shape
            readNoiseDataEntryImg, rnImgName = getRandImg(rnDirOut, noiseImgFileEnding)
            hRN, wRN = readNoiseDataEntryImg.shape
            
            # Scale uncorrupted image size to size of noise images.
            hMin = min(hDCSN, hRN)
            wMin = min(wDCSN, wRN)
            if hImg > hMin or wImg > wMin:
                hScalingFactor = float(hMin) / hImg
                wScalingFactor = float(wMin) / wImg
                scalingFactor = min(hScalingFactor, wScalingFactor)
                img = cv2.resize(img, None, fx=scalingFactor, fy=scalingFactor, interpolation=cv2.INTER_AREA)
                hImg, wImg = img.shape
                
            # Apply synthetic photon shot noise
            img, photonShotNoiseLevel = applyPhotonNoise(img, metadata)
            img = normalizeRange(img, 0.0, NmaxOut)
            
            # Apply real-world readout noise
            readNoiseDataEntryImg = readNoiseDataEntryImg[:hImg, :wImg]
            readoutNoiseLevel = rnStdDevs[rnImgName]
            img += readNoiseDataEntryImg
            img = np.clip(img, 0.0, 1.0)
            
            # Apply real-world dark current shot noise
            dcsnDataEntryImg = dcsnDataEntryImg[:hImg, :wImg]
            dcsnNoiseLevel = dcsnStdDevs[dcsnImgName]
            img += dcsnDataEntryImg
            img = np.clip(img, 0.0, 1.0)
            
            normalizedMetadata = normalizeCameraMetadata(metadata)
            
            results = {}
            results["imgName"] = imgName
            results["photonShotNoiseLevel"] = photonShotNoiseLevel
            results["darkCurrentShotNoiseLevel"] = dcsnNoiseLevel * NmaxIn
            results["readoutNoiseLevel"] = readoutNoiseLevel * NmaxIn
            results["residualNoiseLevel"] = residualNoiseLevel * NmaxIn
            results["cameraGainFactor"] = normalizedMetadata["cameraGainFactor"]
            results["temperature"] = normalizedMetadata["temperature"]
            results["exposureTime"] = normalizedMetadata["exposureTime"]
            results["fullWellSize"] = normalizedMetadata["fullWellSize"]
            results["sensorType"] = normalizedMetadata["sensorType"]
            results["sensorPixelSize"] = normalizedMetadata["sensorPixelSize"]
            results["darkSignalFoM"] = normalizedMetadata["darkSignalFoM"]
            results["fClock"] = normalizedMetadata["fClock"]
            results["senseNodeGain"] = normalizedMetadata["senseNodeGain"]
            results["thermalWhiteNoise"] = normalizedMetadata["thermalWhiteNoise"]
            results["senseNodeResetFactor"] = normalizedMetadata["senseNodeResetFactor"]
                
            img = img * NmaxOut
            img = img.astype("uint8")
            cv2.imwrite(os.path.join(dirOut, imgName + imgFileEnding), img)    
            f.write(str(results) + "\n")
            
def performParameterSensitivityAnalysis(pathInImg):
        img = cv2.imread(pathInImg).astype("float32") / NmaxIn
        metadata = getDefaultCameraMetadata()
        metadata = addMaxValMetadata(metadata)
        # metadata["sensorType"] = 'ccd'
        numParamSamples = 10
        
        labels = ["cameraGainFactor", "exposureTime", "temperature", "darkSignalFoM", "fullWellSize", "fClock", "senseNodeGain", "senseNodeResetFactor", \
                "sensorPixelSize", "thermalWhiteNoise"]
        minVals = [MIN_CAMERA_GAIN_FACTOR, MIN_EXPOSURE_TIME, MIN_TEMPERATURE, MIN_DARK_SIGNAL_FOM, MIN_FULL_WELL_SIZE, MIN_PIXEL_CLOCK_RATE, \
                    MIN_SENSE_NODE_GAIN, MIN_SENSE_NODE_RESET_FACTOR, MIN_SENSOR_PIXEL_SIZE, MIN_THERMAL_WHITE_NOISE]
        maxVals = [MAX_CAMERA_GAIN_FACTOR, MAX_EXPOSURE_TIME, MAX_TEMPERATURE, MAX_DARK_SIGNAL_FOM, MAX_FULL_WELL_SIZE, MAX_PIXEL_CLOCK_RATE, \
                    MAX_SENSE_NODE_GAIN, MAX_SENSE_NODE_RESET_FACTOR, MAX_SENSOR_PIXEL_SIZE, MAX_THERMAL_WHITE_NOISE]
        
        for idx, (label, minVal, maxVal) in enumerate(zip(labels, minVals, maxVals)):
            print("---------" + label + "----------")
            for paramVal in np.linspace(start=minVal, stop=maxVal, num=numParamSamples):
                metadata[label] = paramVal
                
                if idx in [0, 4]:
                    _, noiseLevel = applyPhotonDarkReadNoise(img, metadata)
                elif idx in [1, 2, 3, 8]:
                    _, noiseLevel = applyDarkCurrentShotNoise(img, metadata)
                elif idx in [5, 6, 7, 9]:
                    _, noiseLevel = applyReadoutNoise(img, metadata)
                print("Parameter value: " + str(paramVal) + ", generated noise level: " + str(noiseLevel))

# Example to generate some noised images or to repeat the parameter sensitivity analysis
# if __name__ == "__main__":  
    # baseDirIn = "./../../data/udacity"
    # dirIn = os.path.join(baseDirIn, "GT")
    # dirOut = os.path.join(baseDirIn, "results", "randomSensor")
    # imgFileEnding = ".jpg"
    
    # metadata = randomSensor()
    # generateSimulatedTestImgsWithMetadata(dirIn, dirOut, imgFileEnding, metadata)

    # pathInImg = os.path.join(dirIn, "1478019952686311006.jpg")
    # performParameterSensitivityAnalysis(pathInImg)
    