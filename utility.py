#-*- coding:utf-8
import numpy as np

"""
   generateIndex from hiddenState or ovservationsState

   Attributes
   ----------
   states : hiddenStates = ("Healthy","Fever")

   Returns
   ----------
   labelIndex : {'Healthy': 0, 'Fever': 1}

"""
def generateStatesIndex(lables):
    labelIndex = {}
    i = 0
    for l in lables:
        labelIndex[l] = i
        i += 1
    return labelIndex

"""
   generate matrix from map

   Attributes
   ----------
   map : A = {
    "Healthy":{"Healthy":0.7,"Fever":0.3},
    "Fever":{"Healthy": 0.4, "Fever": 0.6}
}

   Returns
   ----------
   matrixA : [[ 0.7  0.3]
             [ 0.4  0.6]]

"""
def generateMatrix(map, labelIndex1, labelIndex2):
    m = np.zeros((len(labelIndex1),len(labelIndex2)),dtype=float)
    for row in map:
        for col in map[row]:
            m[labelIndex1[row]][labelIndex2[col]] = map[row][col]
    return m

"""
   generate matrix from map

   Attributes
   ----------
   map : pi = {"Healthy":0.6,"Fever":0.4}
}

   Returns
   ----------
   list : pi = [0.6,0.4]

"""
def generatePiVector(map,labelIndex):
    pi = np.zeros((len(labelIndex)),dtype=float)
    for key in map:
        pi[labelIndex[key]] = map[key]
    return pi