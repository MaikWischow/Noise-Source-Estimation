import cv2
from scipy.stats import norm
import numpy as np

DEFAULT_HIST_BIN_SIZE = 4096

def correctNoiseImg(rnImgPath, dcsnImgPath, binSize=DEFAULT_HIST_BIN_SIZE):
    
    rnImg = cv2.imread(rnImgPath, cv2.IMREAD_UNCHANGED)
    dcsnImg = cv2.imread(dcsnImgPath, cv2.IMREAD_UNCHANGED)
    
    ## RN img first...
    # 1) Find maximum in histogram
    h, w = rnImg.shape
    vals = rnImg.flatten()
    
    # Delete zero values
    vals = [i for i in vals if i > 0]
    bins = range(binSize)
    binResults = np.digitize(vals, bins)
    maxX = np.argmax(np.bincount(binResults))
    # print("Uncorrected Std.:", np.std(vals, ddof=1))
    # print("Max x val:", maxX )
    
    # 2) Mirror values for x < 0
    tempVals = np.concatenate((vals, [2*maxX-i for i in vals if i >= 2*maxX]))
    
    # 3) Fit normal distribution
    (rnMu, rnSigma) = norm.fit(tempVals)
    # print("Mean/Std.:", (rnMu, rnSigma))
    
    # 4) Sample new values
    rand = norm.rvs(rnMu, rnSigma, size=h*w)
    newRnImg = np.reshape(rand, (h, w)).astype("float32")
    
    # # Bonus: Check new distribution
    # (mu, sigma) = norm.fit(rand)
    # b, bins, patches = plt.hist(rand, binSize, density = True,  facecolor='blue', alpha=0.50)
    # print("Mean/Std. (Reconstructed):", (mu, sigma))
    
    ## Now the DCSN img...
    h, w = dcsnImg.shape
    vals = dcsnImg.flatten()
    vals = [i for i in vals if i > 0]
    binResults = np.digitize(vals, bins)
    maxX = np.argmax(np.bincount(binResults))
    tempVals = np.concatenate((vals, [2*maxX-i for i in vals if i >= 2*maxX]))
    (dcsnMu, dcsnSigma) = norm.fit(tempVals)
    
    # Correct dcsn distribution by removing rn distribution
    dcsnMu = dcsnMu - rnMu
    dcsnSigma = np.sqrt(dcsnSigma**2 - rnSigma**2)
    dcsnSigma = 0.0 if np.isnan(dcsnSigma) or dcsnSigma < 0.0 else dcsnSigma
    
    rand = norm.rvs(dcsnMu, dcsnSigma, size=h*w)
    newDcsnImg = np.reshape(rand, (h, w)).astype("float32")
    
    return newRnImg, rnSigma, newDcsnImg, dcsnSigma
