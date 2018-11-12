import pickle
import gzip
from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



def preprocessingMNISTdataset(filepath):
    #filename = filepath
    filename = 'mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    trainingDataMatrix = training_data[0]
    trainingTargetMatrix = training_data[1]
    testDataMatrix = test_data[0]
    testTargetMatrix = test_data[1]
    return trainingDataMatrix,trainingTargetMatrix,testDataMatrix,testTargetMatrix



svclassifier = SVC(kernel='linear')
svclassifier.fit(trainingDataMatrix, trainingTargetMatrix)
y_pred = svclassifier.predict(testDataMatrix)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(testTargetMatrix,y_pred))
print(classification_report(testTargetMatrix,y_pred))
print (accuracy_score(testTargetMatrix,y_pred))

USPSMat = []
USPSTar = []
curPath = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)

y_pred_USPS = svclassifier.predict(USPSMat)
print (accuracy_score(USPSTar,y_pred_USPS))
