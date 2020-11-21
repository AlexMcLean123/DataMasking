import pandas as p
import numpy as np
import random
from math import sqrt
import string
from sklearn import preprocessing
import matplotlib.pyplot as plt

def main():
    data = p.read_csv(r"CASCrefmicrodata.csv")
    data = NormaliseData(data)
    risk = []
    infoLoss = []
    variances = []
    index = 0
    for i in np.arange(0.0, 2.1, 0.1):
        maskedData = MultiplicativeNoise(i, data)     
        risk.append(dbrl(data, maskedData))
        infoLoss.append(CalculateInfoLoss(data, maskedData))
        plt.scatter([infoLoss[index]],[risk[index]])
        print(i)
        index += 1
        variances.append(i)

    plt.figure(1)
    plt.title("Gaussian Noise Individual Ranking dRisk(k)")
    plt.ylabel("dRisk")
    plt.xlabel("dUtility")
    plt.show()

    plt.figure(2)
    plt.scatter([infoLoss],[variances])
    plt.title("Information Loss vs Increased Variance (Multiplicative Noise)")
    plt.ylabel("Information Loss")
    plt.xlabel("Variance")
    plt.show()

    plt.figure(3)
    plt.scatter([infoLoss],[risk])
    plt.title("Disclosure Risk vs Increased Variance (Multiplicative Noise)")
    plt.ylabel("Disclosure Risk")
    plt.xlabel("Variance")
    plt.show()

def NormaliseData(data):
    names = data.columns
    scaler = preprocessing.StandardScaler()
    scaled_original = scaler.fit_transform(data)
    scaled_original = p.DataFrame(scaled_original, columns=names)

    return scaled_original

def AddNoise(variance, data):
    mean = 0
    gaussNoise = np.random.normal(mean, variance, data.shape)
    return data + gaussNoise

def MultiplicativeNoise(variance, data):
    mean = 1
    gaussNoise = np.random.normal(mean, variance, data.shape)
    return data * gaussNoise

def Distance(original, signal):
    return sqrt(sum((original-signal)**2))

def dbrl(data, masked):
    data = data.to_numpy() 
    masked = masked.to_numpy()
    i = 1
    reindentified = 0
    while i < len(data):
        j = 1
        minDist = 100000
        minRecord = -1
        while j < len(masked):
            if Distance(data[i,], masked[j,]) < minDist:
                minDist = Distance(data[i,], masked[j,])
                minRecord = j
            j += 1
        if minRecord == i:
            reindentified += 1
        i += 1
    return reindentified / 1080 * 100

def CalculateInfoLoss(data, masked):
    data = data.to_numpy()
    masked = masked.to_numpy()
    return np.square(np.subtract(data, masked)).mean()

if __name__ == '__main__':
    main()