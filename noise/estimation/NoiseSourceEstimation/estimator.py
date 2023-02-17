import os
import cv2
import numpy as np
from numpy import random
import ast
import glob
import sys
from datetime import datetime

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger

from createModel import loadModel
from prepareDataset import DatabaseCreator

sys.path.append(r"./../../simulation/")
from generateHighLevelNoise import applyPhotonNoise, applyDarkCurrentShotNoise, applyReadoutNoise, normalizeCameraMetadata
from cameraParameters import randomSensor, randomEnvCondition, randomCameraGain
sys.path.append(r"./../../../utils/")
from partImage import img2patches, patches2img

# Define image parameters
NOISE_IMG_FILE_ENDING = ".tiff"
MAX_IMG_INTENSITY = 255.0
IMG_PATCH_SHAPE = 128
MAX_IMG_INTENSITY_AUGMENTATION = 20.0

# Define training parameters
BATCH_SIZE = 4
VAL_DATA_SPLIT = 0.10
MAX_TRAINING_EPOCHS = 100

def randBool():
    return random.choice([True, False])

def normalizeZeroOne(npArray, maxIntensity):
    return npArray.astype("float32") / maxIntensity

def normalizeMinusOnePlusOne(npArray, minIntensity, maxIntensity):
    return 2.0 * ((npArray.astype("float32") - minIntensity) / (maxIntensity - minIntensity)) - 1

def denormalize(npArray, maxIntensity):
    return npArray.astype("float32") * maxIntensity 

def estimateNoise(modelType, model, imgPath, dirOut, normalizedParams):
    """
    Estimate noise of an image (patch-wise).
    :param modelType: string to specify the used model type.
    :param model: loaded CNN model with trained weights.
    :param imgPath: path to the image to assess.
    :param dirOut: path to directory to save results. dirOut!=None: noise estimation results will be saved in this directory; otherwise: no results will be saved.
    :return noiseEstimation: Noise estimation image of size [IMG_PATCH_SHAPE, IMG_PATCH_SHAPE].
    :return runtime: determined mean inference time per image patch.
    """
    
    # Load and convert image to range [0,1]
    img = np.array(cv2.imread(imgPath))
    img = normalizeZeroOne(img, MAX_IMG_INTENSITY)
        
    # Check input image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape[:2]
    
    # Split image into image patches
    patches = img2patches(img, IMG_PATCH_SHAPE, IMG_PATCH_SHAPE, IMG_PATCH_SHAPE, IMG_PATCH_SHAPE)
    
    # Prepare input data for the neural network
    if modelType == "fullMetadata":
        metadata = np.array([[
                         normalizedParams["cameraGainFactor"],
                         normalizedParams["temperature"],
                         normalizedParams["exposureTime"],
                         normalizedParams["sensorType"],
                         normalizedParams["fullWellSize"],
                         normalizedParams["sensorPixelSize"],
                         normalizedParams["darkSignalFoM"],
                         normalizedParams["fClock"],
                         normalizedParams["senseNodeGain"],
                         normalizedParams["thermalWhiteNoise"],
                         normalizedParams["senseNodeResetFactor"]
                    ]])
        metadata = np.repeat(metadata, len(patches), axis=0)
        inputData = {"x": patches, \
                     "gain" : metadata[:, 0], \
                     "temp" : metadata[:, 1], \
                     "texp" : metadata[:, 2], \
                     "sensorType" : metadata[:, 3], \
                    "fullWellSize" : metadata[:, 4], \
                    "sensorPixelSize" : metadata[:, 5], \
                    "darkSignalFoM" : metadata[:, 6], \
                    "pixelClock" : metadata[:, 7], \
                    "senseNodeGain" : metadata[:, 8], \
                    "thermalWhiteNoise" : metadata[:, 9], \
                    "senseNodeResetFactor" : metadata[:, 10]
                    }
    elif modelType == "minimalMetadata":
        metadata = np.array([[
                         normalizedParams["cameraGainFactor"],
                         normalizedParams["temperature"],
                         normalizedParams["exposureTime"]
                    ]])
        metadata = np.repeat(metadata, len(patches), axis=0)
        inputData = {"x": patches, \
                     "gain" : metadata[:, 0], \
                     "temp" : metadata[:, 1], \
                     "texp" : metadata[:, 2]
                    }
    elif modelType == "withoutMetadata" or modelType == "baseline":
        inputData = {"x":patches}
    else:
        raise Exception("Unknown model type: " + modelType  + ". Please specifiy a model type from the following ones: withoutMetadata, minimalMetadata or fullMetadata.")      
    
    # Perform estimation
    startTime = datetime.now()
    results = model.predict_on_batch(inputData)
    endTime = datetime.now()
    runtime = (endTime - startTime).total_seconds() / len(patches)
    # print("Time per image patch:", runtime, "s.")
    
    # De-Mosaic patch-wise estimations to size of input image
    if modelType != "baseline":
        results = np.array(results)[..., 0]
        results = np.swapaxes(results, 0, 1)
    noiseEstimation =  patches2img(results, h, w, IMG_PATCH_SHAPE, IMG_PATCH_SHAPE)
    
    # Clip estimations to range [0, MAX_IMG_INTENSITY] for the noise sources and to range [-MAX_IMG_INTENSITY, MAX_IMG_INTENSITY] for the rest noise component
    if modelType != "baseline":
        noiseEstimation[...,0:3][noiseEstimation[...,0:3] < 0.0] = 0.0
        noiseEstimation[...,0:3][noiseEstimation[...,0:3] > 1.0] = 1.0
        noiseEstimation[...,3][noiseEstimation[...,3] < -1.0] = -1.0
        noiseEstimation[...,3][noiseEstimation[...,3] > 1.0] = 1.0
    else:
        noiseEstimation[noiseEstimation < 0.0] = 0.0
        noiseEstimation[noiseEstimation > 1.0] = 1.0
    noiseEstimation = denormalize(noiseEstimation, MAX_IMG_INTENSITY)
    
    # Save the result as .npz file (but do not override existing files)
    if dirOut is not None:
        if not os.path.exists(dirOut):
            os.makedirs(dirOut)
        imgName = imgPath.split(os.sep)[-1].split(".")[0]
        noiseMapPath = os.path.join(dirOut, imgName + ".npz")
        if not os.path.exists(noiseMapPath):
            np.savez_compressed(noiseMapPath, noiseEstimation)
        else:
            print("The noise estimation result file", noiseMapPath, "already exists. Did not override the result.")

    return noiseEstimation, runtime

def train(modelType, model, checkpointPath, trainDatasetPath):
        """
        Train a noise source estimator model.
        :param modelType: string to specify the used model type.
        :param model: compiled (untrained) model.
        :param checkpointPath: path to save model weight checkpoints.
        :param trainDatasetPath: path to training dataset
        :return: -
        """
                
        # Load the test and validation dataset (preserved compatibility to datasets of previous version)
        dc = DatabaseCreator()
        trainData = dc.load_hdf5_v1(trainDatasetPath, 'gray')
        trainData = np.array(trainData).astype(np.float32)
        print("Train dataset size:", len(trainData))
        
        # Loop to create training data
        dataX = []
        dataMeta = []
        labels = []
        for imgPatch in trainData:
            
            # Randomly augment intensity of images
            randomOffset = random.uniform(-MAX_IMG_INTENSITY_AUGMENTATION, MAX_IMG_INTENSITY_AUGMENTATION)
            imgPatch = imgPatch + randomOffset
            imgPatch = np.clip(imgPatch, 0.0, MAX_IMG_INTENSITY)
                    
            # Randomly generate camera metadata
            camMetadata = randomSensor()
            camMetadata = randomEnvCondition(camMetadata)
            
            # Apply photon shot noise that may be randomly corrupted
            corruptedPhotonShotNoiseLevel = 0.0
            isPNCorrupted = randBool()
            if isPNCorrupted:
                oldCamGainFactor = camMetadata["cameraGainFactor"]
                newCamGainFactor = randomCameraGain()
                camMetadata["cameraGainFactor"] = newCamGainFactor
            tmpImgPatch, photonShotNoiseLevel = applyPhotonNoise(imgPatch, camMetadata)
            if isPNCorrupted:
                camMetadata["cameraGainFactor"] = oldCamGainFactor
                _, oldPhotonShotNoiseLevel = applyPhotonNoise(imgPatch, camMetadata)
                photonShotNoiseLevel = oldPhotonShotNoiseLevel
            corruptedPhotonShotNoiseLevel = photonShotNoiseLevel
            imgPatch = tmpImgPatch
            imgPatch = normalizeZeroOne(imgPatch, MAX_IMG_INTENSITY)
            normedPSNLevel = normalizeZeroOne(photonShotNoiseLevel, MAX_IMG_INTENSITY)
            
            # Apply readout noise that may be randomly corrupted
            corruptedReadoutNoiseLevel = 0.0
            isRNCorrupted = randBool()
            if isRNCorrupted:
                oldCamGainFactor = camMetadata["cameraGainFactor"]
                newCamGainFactor = randomCameraGain()
                camMetadata["cameraGainFactor"] = newCamGainFactor
            tmpImgPatch, readoutNoiseLevel = applyReadoutNoise(imgPatch, camMetadata)
            if isRNCorrupted:
                camMetadata["cameraGainFactor"] = oldCamGainFactor
                _, oldReadoutNoiseLevel = applyReadoutNoise(imgPatch, camMetadata)
                readoutNoiseLevel = oldReadoutNoiseLevel
            corruptedReadoutNoiseLevel = readoutNoiseLevel
            imgPatch = tmpImgPatch
            imgPatch = normalizeZeroOne(imgPatch, MAX_IMG_INTENSITY)
            normedRNLevel = normalizeZeroOne(readoutNoiseLevel, MAX_IMG_INTENSITY)
                
            # Apply DCSN that may be randomly corrupted
            corruptedDcsnLevel = 0.0
            isDCSNCorrupted = randBool()
            if isDCSNCorrupted:
                oldCamGainFactor = camMetadata["cameraGainFactor"]
                newCamGainFactor = randomCameraGain()
                camMetadata["cameraGainFactor"] = newCamGainFactor
            tmpImgPatch, dcsnLevel = applyDarkCurrentShotNoise(imgPatch, camMetadata)
            if isDCSNCorrupted:
                camMetadata["cameraGainFactor"] = oldCamGainFactor
                _, oldDcsnLevel = applyDarkCurrentShotNoise(imgPatch, camMetadata)
                dcsnLevel = oldDcsnLevel
            corruptedDcsnLevel = dcsnLevel
            imgPatch = tmpImgPatch
            imgPatch = normalizeZeroOne(imgPatch, MAX_IMG_INTENSITY)
            normedDCSNLevel = normalizeZeroOne(dcsnLevel, MAX_IMG_INTENSITY)   
           
            normedNoiseLevelError = 0.0
            if isPNCorrupted or isRNCorrupted or isDCSNCorrupted:
                corruptedNoiseLevel = np.sqrt(corruptedPhotonShotNoiseLevel**2 + corruptedReadoutNoiseLevel**2 + corruptedDcsnLevel**2)
                trueNoiseLevel = np.sqrt(photonShotNoiseLevel**2 + readoutNoiseLevel**2 + dcsnLevel**2)
                noiseLevelError = trueNoiseLevel - corruptedNoiseLevel
                normedNoiseLevelError = normalizeMinusOnePlusOne(noiseLevelError, -MAX_IMG_INTENSITY, MAX_IMG_INTENSITY)
                
            # Normalize metadata and combine them as input for the neural network
            normedCamMetadata = normalizeCameraMetadata(camMetadata)   
            if modelType == "fullMetadata":
                metaData = [normedCamMetadata["cameraGainFactor"], normedCamMetadata["temperature"], normedCamMetadata["exposureTime"], \
                                 normedCamMetadata["sensorType"], normedCamMetadata["fullWellSize"], normedCamMetadata["sensorPixelSize"], \
                                 normedCamMetadata["darkSignalFoM"], normedCamMetadata["fClock"], normedCamMetadata["senseNodeGain"], \
                                 normedCamMetadata["thermalWhiteNoise"], normedCamMetadata["senseNodeResetFactor"]]
            elif modelType == "minimalMetadata":    
                metaData = [normedCamMetadata["cameraGainFactor"], normedCamMetadata["temperature"], normedCamMetadata["exposureTime"]]
            elif modelType == "withoutMetadata" or modelType == "baseline":
                metaData = None
            else:
                raise Exception("Unknown model type: " + modelType  + ". Please specifiy a model type from the following ones: withoutMetadata, minimalMetadata or fullMetadata.")
                
            dataX.append(imgPatch)
            dataMeta.append(metaData)
            
            if modelType == "baseline":
                gtNoiseValue = np.sqrt(np.power(normedPSNLevel, 2.0) + np.power(normedDCSNLevel, 2.0) + np.power(normedRNLevel, 2.0))
                labels.append(gtNoiseValue)
            else:
                labels.append([normedPSNLevel, normedDCSNLevel, normedRNLevel, normedNoiseLevelError])
        
        # Convert to numpy arrays
        dataX = np.array(dataX) 
        dataMeta = np.array(dataMeta)
        labels = np.array(labels)
        
        # Save model+weights every 2 epochs
        modelCheckpointCallback = ModelCheckpoint(
            filepath=checkpointPath + "/weights-improvement-{epoch:02d}",
            save_weights_only=False,
            monitor='val_loss',
            save_best_only=False, 
            save_freq='epoch',
            period=1
        )
        
        # Log training and validation losses during training
        csvLogger = CSVLogger(os.path.join(checkpointPath, "log.csv"), append=True, separator=';')
        
        # Prepare target data dictionary for neural network
        targetData = {"photonNoiseOutput": labels[:,0], "dcsnOutput": labels[:,1], "readoutNoiseOutput": labels[:,2], \
                      "residualNoiseOutput": labels[:,3]
                      }
            
        # Prepare input data dictionary for neural network
        if modelType == "fullMetadata":
            inputData = {"x":dataX, "gain":dataMeta[:,0], "temp":dataMeta[:,1], "texp":dataMeta[:,2], "sensorType":dataMeta[:,3], \
                           "fullWellSize":dataMeta[:,4], "sensorPixelSize":dataMeta[:,5], "darkSignalFoM":dataMeta[:,6], \
                           "pixelClock":dataMeta[:,7], "senseNodeGain":dataMeta[:,8], "thermalWhiteNoise":dataMeta[:,9], "senseNodeResetFactor":dataMeta[:,10]
                           }
        elif modelType == "minimalMetadata":
            inputData = {"x":dataX, "gain":dataMeta[:,0], "temp":dataMeta[:,1], "texp":dataMeta[:,2]}
        elif modelType == "withoutMetadata":
            inputData = {"x":dataX}
        elif modelType == "baseline":
            inputData = {"x":dataX}
            targetData = labels
        else:
            raise Exception("Unknown model type: " + modelType  + ". Please specifiy a model type from the following ones: baseline, withoutMetadata, minimalMetadata or fullMetadata.")
            
        # Train the model. You may also use separated train and val data sets.
        model.fit(
            x=inputData, 
            y=targetData, 
            epochs=MAX_TRAINING_EPOCHS, verbose=1, validation_split=VAL_DATA_SPLIT, shuffle=True, 
            callbacks=[modelCheckpointCallback, csvLogger], batch_size=BATCH_SIZE, initial_epoch=0)
            
        # Save the model after training
        model.save_weights(checkpointPath)
        
def test(modelType, model, dirIn, dirOut, metadataPathIn, imgFileEnding):
    """
    Test a noise source estimator model.
    :param modelType: string to specify the used model type.
    :param model: loaded model.
    :param dirIn: path to directory with input images to test.
    :param dirIn: path to directory to save test results.
    :param metadataPathIn: path to metadata file for testdataset.
    :param imgFileEnding: string to specify file ending for test images.
    :return: -
    """
        
    if not os.path.exists(dirOut):
        os.makedirs(dirOut)
        
    metadataFile = open(metadataPathIn, 'r')
    lines = metadataFile.readlines()
    runtimes = []
    with open(os.path.join(dirOut, "log.txt"), "w+") as log:
        # Iterate metadata file that contains images with corresponding metadata.
        for line in lines:
            gtResults = ast.literal_eval(line.strip())
            imgName = gtResults["imgName"]
            imgPath = os.path.join(dirIn, imgName + imgFileEnding)            
            # try:
            estimation, runtime = estimateNoise(modelType, model, imgPath, dirOut, gtResults)
            if runtime > 0.0:
                runtimes.append(runtime)
            # except:
                # print("Unexpected error for testing image: " + str(imgName))
            
            # Get GT noise values
            gtNoiseValues = [gtResults["photonShotNoiseLevel"], gtResults["darkCurrentShotNoiseLevel"], gtResults["readoutNoiseLevel"], gtResults["residualNoiseLevel"]]
            if modelType == "baseline":
                estimation = np.concatenate((np.zeros((estimation.shape[0], estimation.shape[1], 3)), estimation), axis=2)
                gtTotalNoise = np.sqrt(np.power(gtNoiseValues[0], 2.0) + np.power(gtNoiseValues[1], 2.0) + np.power(gtNoiseValues[2], 2.0) + np.power(gtNoiseValues[3], 2.0))
                gtNoiseValues = [0.0, 0.0, 0.0, gtTotalNoise]
                
            log.write(str(imgName) + "\n")
            log.write("GT: " + str(gtNoiseValues) + "\n")
            log.write("Est.:" + str(np.mean(estimation, axis=(0,1))) + "\n")
            log.write("-------------------" + "\n")
     
    # Print the overall mean runtime per image patch
    print("Overall average time per image patch:", np.mean(runtimes) * 1000.0, "ms.")
    
def testWithoutMetadata(modelType, model, dirIn, dirOut, imgFileEnding):
    """
    Test the baseline noise estimators (without metadata).
    :param modelType: string to specify the used model type (for model baseline only).
    :param model: loaded model.
    :param dirIn: path to directory with input images to test.
    :param dirIn: path to directory to save test results.
    :param imgFileEnding: string to specify file ending for test images.
    :return: -
    """
    
    if modelType != "baseline":
        raise Exception("This method does not include metadata and is thus only applicable for the model type 'baseline'.")
        
    if not os.path.exists(dirOut):
        os.makedirs(dirOut)
        
    runtimes = []
    with open(os.path.join(dirOut, "log.txt"), "w+") as log:
       for imgPath in glob.glob(os.path.join(dirIn, "*" + imgFileEnding)):
            imgName = imgPath.split(os.sep)[-1].split(".")[0]
            try:
                estimation, runtime = estimateNoise(modelType, model, imgPath, dirOut, None)
                runtimes.append(runtime)
            except:
                pass
            estimation = np.concatenate((np.zeros((estimation.shape[0], estimation.shape[1], 3)), estimation), axis=2)
            gtNoiseArray = [0.0, 0.0, 0.0, 0.0]
            log.write(str(imgName) + "\n")
            log.write("GT: " + str(gtNoiseArray) + "\n")
            log.write("Est.:" + str(np.mean(estimation, axis=(0,1))) + "\n")
            log.write("-------------------" + "\n")
                
    print("Overall average time per image patch:", np.mean(runtimes), "s.")

# # Example to test or train the noise (source) estimators
# if __name__ == '__main__':
    
#     # Set GPU and allow dynamic memory growth
#     tf.keras.backend.clear_session()
#     physical_devices = tf.config.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     tf.keras.backend.set_image_data_format('channels_last')

#     # Train the model
#     if False:
#         modelType = "fullMetadata" # "baseline", "withoutMetadata", "minimalMetadata", "fullMetadata"
#         trainDatasetPath = "./../../../data/training/train.h5"
#         modelDir = os.path.join("./", "weights", "noiseSourceEstimator-" + modelType)
#         if not os.path.exists(modelDir):
#             os.makedirs(modelDir)
            
#         model = loadModel(modelDir, modelType)
#         train(modelType, model, modelDir, trainDatasetPath)
       
#     # Test the model
#     if True:
#         baseDirIn = "./../../../data/benchmarking"
#         dataset = "udacity"
#         sensorType = "randomSensor"
#         imgFileEnding = ".jpg"
#         modelType = "withoutMetadata" #  "baseline", "withoutMetadata", "minimalMetadata", "fullMetadata"
#         dirIn = os.path.join(baseDirIn, dataset, sensorType)
#         dirOut = os.path.join(baseDirIn, dataset, "results", sensorType, modelType)
#         metadataPathIn = os.path.join(dirIn, "metadata.txt")
        
#         modelDir = os.path.join(".", "weights", "noiseSourceEstimator-" + modelType, "weights", "variables", "variables")
#         model = loadModel(modelDir, modelType)
        
#         test(modelType, model, dirIn, dirOut, metadataPathIn, imgFileEnding)
#         # testWithoutMetadata(modelType, model, dirIn, dirOut, imgFileEnding)
