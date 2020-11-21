import random
import math
import statistics
import openpyxl
import numpy as np
import string
import matplotlib
import matplotlib.pyplot as plt

wb = openpyxl.load_workbook('CASCrefmicrodata.xlsx')
sheet = wb['Census']
total = 14040
infoLossD=[]
dbrlD=[]

kParams =[0.01, 0.04, 0.07, 0.1, 0.2, 0.3, 0.6, 0.9]


def normalize(numbers):
    myMax = max(numbers)
    myMin = min(numbers)
    for x in range(len(numbers)):
        numbers[x] = ((numbers[x] - myMin)/(myMax - myMin))


def maskAddiNoise(numbers, parameter, maskedArray):
    var = statistics.variance(numbers)
    arr = []
    for x in numbers:
        masked = x + random.gauss(0, math.sqrt(var*parameter))
        arr.append(masked)
    maskedArray.append(arr)


def maskMultiNoise(numbers, parameter, maskedArray):
    var = statistics.variance(numbers)
    arr = []
    for x in numbers:
        masked = x * random.gauss(1, math.sqrt(var*parameter))
        arr.append(masked)
    maskedArray.append(arr)


def distance(record1, record2):
    sum = 0
    sum += math.sqrt((record1 - record2) * (record1 - record2))
    return sum

def dbrl(original, masked):
    i = 0
    reidentified = 0
    while(i < len(original)):
        j = 0
        minDist = 100000
        minRecord = -1
        while(j < len(masked)):
            if(distance(original[i], masked[j]) < minDist):
                minDist = distance(original[i], masked[j])
                minRecord = j
            j = j + 1
        if(minRecord == i):
            reidentified = reidentified + 1
        i = i+1
    return reidentified


def computeCol(original, masked):
    sum = 0
    mean = statistics.mean(original)
    for x in range(len(original)):
        sum = sum + ((original[x] - masked[x])**2 / mean)
    return sum


def infoLoss(array1, array2):
    outerSum = 0
    for x in range(len(array1)):
        outerSum = outerSum + computeCol(array1[x], array2[x])
    return outerSum / 13



for i in kParams:
    mySheet = []
    myMaskedSheet = []
    sumReidentified = 0
    def fillArray(column, array):
        first_column = sheet[column]
        arr = []
        for x in range(1, len(first_column)):
            arr.append(first_column[x].value)
        array.append(arr)

    for x, y in zip(range(1, sheet.max_column + 1), string.ascii_uppercase):
        fillArray(y, mySheet)

    for x in range(len(mySheet)):
        normalize(mySheet[x])
        maskAddiNoise(mySheet[x], i, myMaskedSheet)
        sumReidentified = sumReidentified + dbrl(mySheet[x], myMaskedSheet[x])

    infoLossD.append(infoLoss(mySheet, myMaskedSheet))
    dbrlD.append((sumReidentified/total)*100)
    print(i, "Information loss value:", infoLoss(mySheet, myMaskedSheet),
            "Disclosure Risk: ", (sumReidentified/total)*100)


plt.scatter(infoLossD, dbrlD)
plt.title("Disclosure Risk vs Information Loss for Additive Noise")
plt.ylabel("Disclosure Risk")
plt.xlabel("Information Loss")
plt.show()
plt.scatter(kParams, dbrlD)
plt.title("K Parameter vs Disclosure Risk for Additive Noise")
plt.ylabel("Disclosure Risk")
plt.xlabel("K Parameter")
plt.show()
plt.scatter(kParams, infoLossD)
plt.title("K Parameter vs Information Loss for Additive Noise")
plt.ylabel("Info Loss")
plt.xlabel("K Parameter")
plt.show()