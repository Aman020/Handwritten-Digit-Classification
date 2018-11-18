import pickle
import gzip
from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def preprocessingMNISTdataset(filepath):
    filename = filepath
    #filename = 'mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    trainingDataMatrix = training_data[0]
    print(type(trainingDataMatrix))
    trainingTargetMatrix = training_data[1]

    testDataMatrix = test_data[0]
    testTargetMatrix = test_data[1]
    print(trainingTargetMatrix[0].shape)
    image = trainingDataMatrix[0].reshape(28,28)
    plt.imsave('image.png',image,cmap= 'binary')
    plt.show()
    return trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix


def SVM(dataset, kernel):
    trainingDataMatrix,trainingTargetMatrix,testDataMatrix,testTargetMatrix = preprocessingMNISTdataset(dataset)
    svc = SVC(kernel=kernel, gamma=1)
    svc.fit(trainingDataMatrix, trainingTargetMatrix)
    y_pred = svc.predict(testDataMatrix)
    print(confusion_matrix(testTargetMatrix,y_pred))
    print(classification_report(testTargetMatrix,y_pred))
    print (accuracy_score(testTargetMatrix,y_pred))

def RandomForest(dataset):
    trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix = preprocessingMNISTdataset(dataset)
    rfClassifier =RandomForestClassifier(n_estimators=10)
    rfClassifier.fit(trainingDataMatrix, trainingTargetMatrix)
    y_pred = rfClassifier.predict(testDataMatrix)
    print(confusion_matrix(testTargetMatrix,y_pred))
    print(classification_report(testTargetMatrix,y_pred))
    print (accuracy_score(testTargetMatrix,y_pred))

def NeuralNetwork(dataset):
    trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix = preprocessingMNISTdataset(dataset)
    num_classes = 10
    image_vector_size = 784
    x_train = trainingDataMatrix.reshape(trainingDataMatrix.shape[0], image_vector_size)
    x_test = testDataMatrix.reshape(testDataMatrix.shape[0], image_vector_size)
    y_train = keras.utils.to_categorical(trainingTargetMatrix, num_classes)
    y_test = keras.utils.to_categorical(testTargetMatrix, num_classes)
    image_size = 784
    model = Sequential()
    model.add(Dense(100  , activation='relu', input_shape=(image_size,)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=7, mode='min')
    history = model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=False, validation_split=.1, callbacks=[ earlystopping_cb])
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print(accuracy)
    df = pd.DataFrame(history.history)
    df.plot(subplots=True, grid=True, figsize=(10, 15))
    plt.savefig('graph.png')

def preprocessingUSPSdataset():

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

    #y_pred_USPS = svclassifier.predict(USPSMat)
    #print (accuracy_score(USPSTar,y_pred_USPS))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(hypo, target):
    lossValue = (-target * np.log(hypo) - (1 - target) * np.log(1 - hypo)).mean()
    return lossValue

def preprocessingMNISTdataset(filepath):
    filename = filepath
    #filename = 'mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    trainingDataMatrix = training_data[0]
    trainingTargetMatrix = training_data[1]
    testDataMatrix = test_data[0]
    testTargetMatrix = test_data[1]
    return trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix




def logisticRegression(dataset):
    rawDataTraining,rawTargetTraining, rawDataTesting, rawTargetTesting =  preprocessingMNISTdataset(dataset)
    numFeatures = rawDataTraining.shape[1]
    numLabels = 10
    classifiers = np.zeros(shape=(numLabels, numFeatures))
    for i in range(0, numLabels):
        label = (rawTargetTraining == i).astype(int)
        theta = np.zeros(rawDataTraining.shape[1])
        learning_rate = 0.01
        lossFunction = []
        for j in range(0, 500):
            z_train = np.dot(rawDataTraining, theta)
            h_train = sigmoid(z_train)
            temp = h_train - label
            gradient = np.dot(rawDataTraining.T, temp) / len(label)
            theta = theta - learning_rate * gradient
            lossFunction.append(loss(h_train, np.array(label)))
        classifiers[i, :] = theta
    z_test = np.dot(rawDataTesting, classifiers.transpose())
    h_test = sigmoid(z_test)
    predictions = h_test.argmax(axis=1)
    print("Testing accuracy:", str(100 * np.mean(predictions == rawTargetTesting)) + "%")


if __name__ == '__main__':
    SVM('mnist.pkl.gz','linear')
    #logisticRegression('mnist.pkl.gz')
    #SVM('mnist.pkl.gz', 'rbf')
    #NeuralNetwork('mnist.pkl.gz')
    #RandomForest('mnist.pkl.gz')
