import os
import numpy as np
import json

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)}, suppress=True)

def parseResultsLogLine(line):
        # return " ".join( \
        #             line.split(":")[1] \
        #             .replace("0.", "0.0").split()) \
        #             .replace(".]", ".0]") \
        #             .replace(" ]", "]") \
        #             .replace("[ ", "[")
        return " ".join( \
                    line.split(":")[1] \
                    .replace("0.", "0.0").split()) \
                    .replace(".]", ".0]") \
                    .replace(" ]", "]") \
                    .replace("[ ", "[") \
                    .replace(" ", ",")
            
def plotBiasStdRMS(resultsDir, method, isTotalNoise=False, isNoiseSourceEstimator=False):
    """
    Evaluate noise estimations in terms of bias (accuracy), std (rebostness) and RMS (overall performance)
    :param dirIn: path to directory with noised input images to test.
    :param dirOut: path to directory to save test results.
    :param imgFileEnding: string to specify file ending for test images.
    :return: -
    """
    logFileName = "noiseSourcesLog.txt" if not isTotalNoise else "overallNoiseLog.txt"
    with open(os.path.join(resultsDir, logFileName), "w+") as log:
        gtNoiseLevel = []
        estNoiseLevel = []
        # Parse and collect noise estimation results from results log file
        with open(os.path.join(resultsDir, "log.txt"), "r") as resultsLog:
            gtVals = []
            estVals = []
            gtValFound = False
            for idx, line in enumerate(resultsLog.readlines()):
                line = line.rstrip()
                if "GT" in line:
                    gtVals = json.loads(line.split(":")[1])
                    gtValFound = True
                elif "Est" in line and gtValFound:
                    estVals = parseResultsLogLine(line)
                    estVals = json.loads(estVals)
                    gtNoiseLevel.append(gtVals)
                    estNoiseLevel.append(estVals)
                    gtValFound = False
                    
        # In case the total noise is evaluated, differentiate between noise estimators and noise source estimators
        if isTotalNoise:
            if isNoiseSourceEstimator:
                if "PGE" in method:
                    estNoiseLevel = [x[3]  for x in estNoiseLevel[:]]
                    gtNoiseLevel = [np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) - x[3]  for x in gtNoiseLevel[:]]
                else:
                    estNoiseLevel = [np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) - x[3]  for x in estNoiseLevel[:]]
                    gtNoiseLevel = [np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) - x[3]  for x in gtNoiseLevel[:]]
            else:
                estNoiseLevel = [x[3]  for x in estNoiseLevel[:]]
                gtNoiseLevel = [x[3]  for x in gtNoiseLevel[:]]
        
        # To numpy arrays
        gtNoiseLevel = np.array(gtNoiseLevel)
        estNoiseLevel = np.array(estNoiseLevel)
        
        # Calculate result statistics
        mse = np.square(np.subtract(estNoiseLevel, gtNoiseLevel)).mean(axis=0)
        bias = np.abs(np.mean(gtNoiseLevel - estNoiseLevel, axis=(0)))
        std = np.std(gtNoiseLevel - estNoiseLevel, axis=0)
        
        # Log and print results
        print(method)
        print("--------------")
        print("Bias: ", bias)
        print("Std: ", std)
        print("RMS: ", np.sqrt(mse))
        print("")
        # log.write(str(method) + "\n")
        # log.write("--------------" + "\n")
        # log.write("Bias: " +  str(bias) + "\n")
        # log.write("Std: " +  str(std) + "\n")
        # log.write("RMS: " +  str(np.sqrt(mse)) + "\n")
        # log.write("\n")
    
# # Example to calculate metrics for noise estimations
# if __name__ == '__main__':
#     baseDirIn = "./../../data/benchmarking"
    
#     # noiseTypes => 0: "Photon~Shot~Noise", 1: "Dark~Current~Shot~Noise", 2: "Readout~Noise", 3: "Residual~Noise"}
#     noiseTypeIdx = 1
#     dataset = "Udacity"
#     sensorType = "randomSensor" # "randomSensor", "sonyICX285", "e2vEV76C661"
#     isTotalNoise = True
#     # # "----------------"
#     if isTotalNoise:
#         label = "PCA"
#         dirOut = os.path.join(baseDirIn, dataset, "results", sensorType, label)
#         plotBiasStdRMS(dirOut, label, isTotalNoise=isTotalNoise)
    
        # label = "B+F"
        # dirOut = os.path.join(baseDirIn, dataset, "results", sensorType, label)
        # plotBiasStdRMS(dirOut, label, isTotalNoise=isTotalNoise)
        
        # label = "baseline"
        # dirOut = os.path.join(baseDirIn, dataset, "results", sensorType, label)
        # plotBiasStdRMS(dirOut, label, isTotalNoise=isTotalNoise)
        
        # label = "PGE-Net"
        # dirOut = os.path.join(baseDirIn, dataset, "results", sensorType, label)
        # plotBiasStdRMS(dirOut, label, isTotalNoise=isTotalNoise)
        
    # label = "PGE-Net"
    # dirOut = os.path.join(baseDirIn, dataset, "results", sensorType, label)
    # plotBiasStdRMS(dirOut, label, isTotalNoise=isTotalNoise)
    # "----------------"
    # label = "withoutMetadata"
    # dirOut = os.path.join(baseDirIn, dataset, "results", sensorType, label)
    # plotBiasStdRMS(dirOut, label, isTotalNoise=isTotalNoise, isNoiseSourceEstimator=True)
    # # #"----------------"
    # label = "minimalMetadata"
    # dirOut = os.path.join(baseDirIn, dataset, "results", sensorType, label, )
    # plotBiasStdRMS(dirOut, label, isTotalNoise=isTotalNoise, isNoiseSourceEstimator=True)
    # # # # "----------------"
    # label = "fullMetadata"
    # dirOut = os.path.join(baseDirIn, dataset, "results", sensorType, label)
    # plotBiasStdRMS(dirOut, label, isTotalNoise=isTotalNoise, isNoiseSourceEstimator=True)      