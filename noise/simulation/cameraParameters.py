import copy
import random
import numpy as np

# Custom variables
fClockStepSize = 10000
smallNum = 1e-20

# Constants
NmaxIn = 255.0
NmaxOut = 255.0
siEGap0 = 1.1557 # in eV for silicon
siAlpha = 7.021*1e-4 # in eV/K for silicon
siBeta = 1108.0 # in K for for silicon
boltzmannConstEV = 8.617343 * 1e-5 # in eV/K
boltzmannConstJK = 1.3806504 * 1e-23 # in J/K
k1 = 1.0909e-14
q = 1.602176487 * 1e-19
chargePerCoulomb = 6.2415e18
k2c = 273.15

# Metadata ranges
MIN_CAMERA_GAIN_FACTOR = 1.0 # just the factor, not in db!
MAX_CAMERA_GAIN_FACTOR = 16.0 # just the factor, not in db!
MIN_TEMPERATURE = k2c # in K
MAX_TEMPERATURE = k2c + 80.0 # in K
MIN_EXPOSURE_TIME = 0.001 # in s
MAX_EXPOSURE_TIME = 0.2 # in s
MIN_SENSOR_PIXEL_SIZE = 0.00009 # in cm
MAX_SENSOR_PIXEL_SIZE = 0.001 # in cm
MIN_FULL_WELL_SIZE = 2000.0 # in e-
MAX_FULL_WELL_SIZE = 100000.0 # in e-
MIN_SENSE_NODE_GAIN = 1.0 * 1e-6 # in V/e-
MAX_SENSE_NODE_GAIN = 5.0 * 1e-6 # in V/e-
MIN_DARK_SIGNAL_FOM = 0.0
MAX_DARK_SIGNAL_FOM = 1.0
MIN_PIXEL_CLOCK_RATE = 8.0 * 1e6 # in Hz
MAX_PIXEL_CLOCK_RATE = 150.0 * 1e6 # in Hz
MIN_THERMAL_WHITE_NOISE = 1.0 * 1e-9
MAX_THERMAL_WHITE_NOISE = 60.0 * 1e-9
MIN_SENSE_NODE_RESET_FACTOR = 0.0
MAX_SENSE_NODE_RESET_FACTOR = 1.0

# Default camera metadata
def getDefaultCameraMetadata():
    params = {}

    # Flags
    params["applyPhotonNoise"] = False
    params["applyDarkCurrent"] = False
    params["darksignalCorrection"] = False
    params["applySourceFollwerNoise"] = False
    params["applyKtcNoise"] = False

    # General
    params["temperature"] = k2c + 20.0 # in K
    params["exposureTime"] = 0.00014 # in s
    params["sensorPixelSize"] =  0.000645 # in cm # IPS: 6.45 um
    params["sensorType"] = 'ccd'
    params["cameraGainFactor"] = 1.0

    # Photons to electrons
    params["fullWellSize"] = 14000.0 # in e-

    # Electrons to charge
    params["darkSignalFoM"] = 0.008890825831876017 #0.082 # nA/cm^2
    params["darkSignalFPNFactor"] = 0.04
    params["senseNodeGain"] =  5.0 * 1e-6 #5.0 * 1e-6 # in V/e- # in range [1,5] * 1e6 according to original paper
    params["corrDoubleSampStoSTime"] = 1e-6 # in s

    # Charge to voltage
    params["sourceFollGain"] = 1.0 # in V/V
    params["neighborCorrFactor"] = 0.0005
    params["flickerNoiseCornerFreq"] = 1e6 # in Hz
    params["corrDoubleSampTimeFact"] = 0.5
    params["sourceFollowerCurrMod"] = 1e-8; # in A, CMOS only
    params["fClock"] = 28.64 * 1e6 # in Hz
    params["thermalWhiteNoise"] = 30.8 * 1e-9 # in V/(Hz)^0.5
    params["senseNodeResetFactor"] = 0.0 # 0.0 = fully compensated by CDS; 1.0 = no compensation
    params["sourceFollNonLinRatio"] = 1.05
    params["corrDoubleSampGain"] = 1.0 # in V/V; Correlated Double Sampling gain, lower means amplifying the noise.

    # Voltage to digital numbers
    params["chargeNodeRefVolt"] = 5.0 # in V
    params["adcNonLinRatio"] = 1.04
    params["offset"] = 0.0 # in DN
    
    return params

# Sample metadata of a random camera sensor
def addMaxValMetadata(paramsIn):    
    paramsOut = copy.deepcopy(paramsIn)
    paramsOut["sensorType"] =  'cmos'
    paramsOut["cameraGainFactor"] = MAX_CAMERA_GAIN_FACTOR
    paramsOut["exposureTime"] = MAX_EXPOSURE_TIME
    paramsOut["temperature"] =  MAX_TEMPERATURE
    paramsOut["fullWellSize"] = MAX_FULL_WELL_SIZE
    paramsOut["sensorPixelSize"] =  MAX_SENSOR_PIXEL_SIZE
    paramsOut["darkSignalFoM"] = MAX_DARK_SIGNAL_FOM
    paramsOut["fClock"] = MAX_PIXEL_CLOCK_RATE
    paramsOut["senseNodeGain"] = MAX_SENSE_NODE_GAIN
    paramsOut["thermalWhiteNoise"] = MAX_THERMAL_WHITE_NOISE
    paramsOut["senseNodeResetFactor"] = MAX_SENSE_NODE_RESET_FACTOR
    return paramsOut

# Sample metadata of a random camera sensor
def randomSensor():    
    paramsOut = getDefaultCameraMetadata()
    paramsOut["sensorType"] = np.random.choice(['ccd', 'cmos'], 1)[0]
    paramsOut["fullWellSize"] = np.random.randint(MIN_FULL_WELL_SIZE, MAX_FULL_WELL_SIZE + 1) 
    paramsOut["sensorPixelSize"] = random.uniform(MIN_SENSOR_PIXEL_SIZE, MAX_SENSOR_PIXEL_SIZE + smallNum)
    paramsOut["darkSignalFoM"] = random.uniform(MIN_DARK_SIGNAL_FOM, MAX_DARK_SIGNAL_FOM + smallNum)
    paramsOut["fClock"] = np.random.randint(MIN_PIXEL_CLOCK_RATE, MAX_PIXEL_CLOCK_RATE + 1) 
    paramsOut["senseNodeGain"] = random.uniform(MIN_SENSE_NODE_GAIN, MAX_SENSE_NODE_GAIN + smallNum)
    paramsOut["thermalWhiteNoise"] = random.uniform(MIN_THERMAL_WHITE_NOISE, MAX_THERMAL_WHITE_NOISE + smallNum)
    paramsOut["senseNodeResetFactor"] = random.uniform(MIN_SENSE_NODE_RESET_FACTOR, MAX_SENSE_NODE_RESET_FACTOR + smallNum)
    return paramsOut

def addFixedMetadata(paramsIn):
    paramsOut = copy.deepcopy(paramsIn)
    defaultMetadata = getDefaultCameraMetadata()
    paramsOut["corrDoubleSampStoSTime"] = defaultMetadata["corrDoubleSampStoSTime"]
    paramsOut["sourceFollGain"] = defaultMetadata["sourceFollGain"]
    paramsOut["neighborCorrFactor"] = defaultMetadata["neighborCorrFactor"]
    paramsOut["flickerNoiseCornerFreq"] = defaultMetadata["flickerNoiseCornerFreq"] 
    paramsOut["corrDoubleSampTimeFact"] = defaultMetadata["corrDoubleSampTimeFact"]
    paramsOut["sourceFollowerCurrMod"] = defaultMetadata["sourceFollowerCurrMod"]
    paramsOut["sourceFollNonLinRatio"] = defaultMetadata["sourceFollNonLinRatio"]
    paramsOut["corrDoubleSampGain"] = defaultMetadata["corrDoubleSampGain"]
    paramsOut["chargeNodeRefVolt"] = defaultMetadata["chargeNodeRefVolt"]
    paramsOut["offset"] = defaultMetadata["offset"]
    return paramsOut
    

# Sony ICX285 sensor (CCD) metadata
# Source: https://www.1stvision.com/cameras/sensor_specs/ICX285.pdf
# Read Noise std. dev. from experimental data = 12.526631 e-
def sonyICX285():    
    params = getDefaultCameraMetadata()
    return params

# E2V EV76C661 sensor (CMOS) metadata
# Source: https://www.ximea.com/en/products/cameras-filtered-by-sensor-sizes/e2v-ev76c661-usb3-nir-industrial-camera
# Source: https://www.1stvision.com/cameras/sensor_specs/EV76C661ABT.pdf
# Dark Signal = 38 LBS10/s => 38 / 1024DN * 8400e- = 312 e-/s
# Read Noise from experimental data = 24.01 e-
def e2vEV76C661():
    params = getDefaultCameraMetadata()
    params["sensorType"] = 'cmos'
    params["fullWellSize"] = 8400
    params["sensorPixelSize"] = 0.00053
    params["chargeNodeRefVolt"] = 3.3 # in V
    params["fClock"] = 90 * 1e6 # in Hz
    params["darkSignalFoM"] = 0.9569181675700699
    params["thermalWhiteNoise"] = 59.0 * 1e-9
    params["senseNodeResetFactor"] = 0.0
    return params

# Normalize some metadata to range [0,1]
def normalizeRange(x, minX, maxX):
    return (x - minX) / (maxX - minX)

# Denormalize some metadata from range [0,1]
def denormalizeRange(normalizedX, minX, maxX):
    return normalizedX * (maxX - minX) + minX
    
# Normalize all metadata to range [0,1]
def normalizeCameraMetadata(paramsIn):
    paramsOut = copy.deepcopy(paramsIn)
    paramsOut["cameraGainFactor"] =  normalizeRange(paramsOut["cameraGainFactor"], MIN_CAMERA_GAIN_FACTOR, MAX_CAMERA_GAIN_FACTOR)
    paramsOut["temperature"] = normalizeRange(paramsOut["temperature"], MIN_TEMPERATURE, MAX_TEMPERATURE)
    paramsOut["exposureTime"] = normalizeRange(paramsOut["exposureTime"], MIN_EXPOSURE_TIME, MAX_EXPOSURE_TIME)
    paramsOut["fullWellSize"] = normalizeRange(paramsOut["fullWellSize"], MIN_FULL_WELL_SIZE, MAX_FULL_WELL_SIZE)
    paramsOut["sensorType"] = (1 if paramsOut["sensorType"]=='ccd' else 0)
    paramsOut["sensorPixelSize"] = normalizeRange(paramsOut["sensorPixelSize"], MIN_SENSOR_PIXEL_SIZE, MAX_SENSOR_PIXEL_SIZE)
    paramsOut["darkSignalFoM"] = normalizeRange(paramsOut["darkSignalFoM"], MIN_DARK_SIGNAL_FOM, MAX_DARK_SIGNAL_FOM)
    paramsOut["fClock"] = normalizeRange(paramsOut["fClock"], MIN_PIXEL_CLOCK_RATE, MAX_PIXEL_CLOCK_RATE)
    paramsOut["senseNodeGain"] = normalizeRange(paramsOut["senseNodeGain"], MIN_SENSE_NODE_GAIN, MAX_SENSE_NODE_GAIN)  
    paramsOut["thermalWhiteNoise"] = normalizeRange(paramsOut["thermalWhiteNoise"], MIN_THERMAL_WHITE_NOISE, MAX_THERMAL_WHITE_NOISE)  
    paramsOut["senseNodeResetFactor"] = normalizeRange(paramsOut["senseNodeResetFactor"], MIN_SENSE_NODE_RESET_FACTOR, MAX_SENSE_NODE_RESET_FACTOR)  
    return paramsOut

# Denormalize all metadata from range [0,1]
def denormalizeCameraMetadata(paramsIn):
    paramsOut = copy.deepcopy(paramsIn)
    paramsOut["cameraGainFactor"] =  denormalizeRange(paramsOut["cameraGainFactor"], MIN_CAMERA_GAIN_FACTOR, MAX_CAMERA_GAIN_FACTOR)
    paramsOut["temperature"] = denormalizeRange(paramsOut["temperature"], MIN_TEMPERATURE, MAX_TEMPERATURE)
    paramsOut["exposureTime"] = denormalizeRange(paramsOut["exposureTime"], MIN_EXPOSURE_TIME, MAX_EXPOSURE_TIME)
    paramsOut["fullWellSize"] = denormalizeRange(paramsOut["fullWellSize"], MIN_FULL_WELL_SIZE, MAX_FULL_WELL_SIZE)
    paramsOut["sensorType"] = ("ccd" if paramsOut["sensorType"]== 1 else 'cmos')
    paramsOut["sensorPixelSize"] = denormalizeRange(paramsOut["sensorPixelSize"], MIN_SENSOR_PIXEL_SIZE, MAX_SENSOR_PIXEL_SIZE)
    paramsOut["darkSignalFoM"] = denormalizeRange(paramsOut["darkSignalFoM"], MIN_DARK_SIGNAL_FOM, MAX_DARK_SIGNAL_FOM)
    paramsOut["fClock"] = denormalizeRange(paramsOut["fClock"], MIN_PIXEL_CLOCK_RATE, MAX_PIXEL_CLOCK_RATE)
    paramsOut["senseNodeGain"] = denormalizeRange(paramsOut["senseNodeGain"], MIN_SENSE_NODE_GAIN, MAX_SENSE_NODE_GAIN)  
    paramsOut["thermalWhiteNoise"] = denormalizeRange(paramsOut["thermalWhiteNoise"], MIN_THERMAL_WHITE_NOISE, MAX_THERMAL_WHITE_NOISE)  
    paramsOut["senseNodeResetFactor"] = denormalizeRange(paramsOut["senseNodeResetFactor"], MIN_SENSE_NODE_RESET_FACTOR, MAX_SENSE_NODE_RESET_FACTOR)  
    return paramsOut

# Sample random minimal camera metadata (gain, exposure time, sensor temperature)
def randomEnvCondition(params):
    # tmpGain = np.random.exponential(2) + 1
    params["cameraGainFactor"] = random.uniform(MIN_CAMERA_GAIN_FACTOR, MAX_CAMERA_GAIN_FACTOR)
    params["exposureTime"] = random.uniform(MIN_EXPOSURE_TIME, MAX_EXPOSURE_TIME) # in s
    params["temperature"] =  random.uniform(MIN_TEMPERATURE, MAX_TEMPERATURE)
    return params

# Sample random camera gain
def randomCameraGain():
    return random.uniform(MIN_CAMERA_GAIN_FACTOR, MAX_CAMERA_GAIN_FACTOR)

# Calculate dark signal figure of merit @ 300K according to James Janesick "Photon Transfer"
# => Temperature should be as close as possible to 300K
# darkSignal [e-/s] at temperature [K], sensorPixelSize[cm]
def darkSignalFoM(darkSignal, temperature, sensorPixelSize):
    eGap = siEGap0 - ((siAlpha * np.power(temperature, 2.0)) / (temperature + siBeta));
    dsTemp = np.power(temperature, 1.5) * np.exp(-1.0 * eGap / (2.0 * boltzmannConstEV * temperature))
    darkSignalFoM = darkSignal / (2.55 * 1e15 * np.power(sensorPixelSize, 2.0) * dsTemp)
    return darkSignalFoM