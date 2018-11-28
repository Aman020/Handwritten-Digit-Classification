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
import datetime
from scipy import stats


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

#Function to train using SVM
def SVM( trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix,USPSDataMatrix, USPSTarget ,kernel, gamma = 'auto_deprecated'):
    print('-------------------------------SVM-------------------------------------------------------------------------')
    print('SVM Started ' + str(datetime.datetime.now()))
    svc = SVC(kernel=kernel, gamma=gamma)
    svc.fit(trainingDataMatrix, trainingTargetMatrix)
    y_predMNIST = svc.predict(testDataMatrix)
    print('--------- MNIST DATA ANALYSIS ------')
    print( 'Confusion Matrix- MNIST')
    print(ConfusionMatrix(testTargetMatrix,y_predMNIST))
    print(classification_report(testTargetMatrix,y_predMNIST))
    print (" Accuracy when tested on MNIST Dataset = " + str(accuracy_score(testTargetMatrix,y_predMNIST)))
    print('')
    print('---------- USPS DATA ANALYSIS ------')
    y_predUSPS = svc.predict(USPSDataMatrix)
    print('Confusion Matrix- USPS')
    print(ConfusionMatrix(USPSTarget, y_predUSPS)) # THe funciton is defined below
    #print(classification_report(USPSTarget, y_predUSPS))
    print(" Accuracy when tested on USPS Dataset = "  + str (accuracy_score(USPSTarget, y_predUSPS)))
    print('SVM ended ' + str(datetime.datetime.now()))
    return y_predMNIST,y_predUSPS


def RandomForest( trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix,USPSDataMatrix, USPSTarget):
    print('--------------------------------RANDOM FOAREST-------------------------------------------------------------')
    print('Random Forest started-' + str(datetime.datetime.now()))
    rfClassifier =RandomForestClassifier(n_estimators=10)
    rfClassifier.fit(trainingDataMatrix, trainingTargetMatrix)
    y_predMNIST = rfClassifier.predict(testDataMatrix)
    y_predUSPS = rfClassifier.predict(USPSDataMatrix)

    print('-------- MNIST DATA ANALYSIS --------')
    print('Confusion Matrix- MNIST')
    print( ConfusionMatrix(testTargetMatrix,y_predMNIST)) #Below is the definition of confusion matrix
    #print(classification_report(testTargetMatrix,y_predMNIST))
    print ("Accuracy when tested on MNIST dataset = " + str(accuracy_score(testTargetMatrix,y_predMNIST)))
    print('-------- USPS DATA ANALYSIS --------')
    print('Confusion Matrix - USPS')
    print(ConfusionMatrix(USPSTarget, y_predUSPS))  # Below is the definition of confusion matrix
    #print(classification_report(USPSTarget, y_predUSPS))
    print("Accuracy when tested on USPS dataset = " + str(accuracy_score(USPSTarget, y_predUSPS)))
    print('RF Ended -' + str(datetime.datetime.now()))
    return y_predMNIST,y_predUSPS

def NeuralNetwork( trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix,USPSDataMatrix, USPSTarget):
    print('---------------------------------NEURAL NETWORK------------------------------------------------------------')
    print( 'NN started-' + str(datetime.datetime.now()))
    num_classes = 10;
    image_vector_size = 784
    x_train = trainingDataMatrix.reshape(trainingDataMatrix.shape[0], image_vector_size)
    x_test = testDataMatrix.reshape(testDataMatrix.shape[0], image_vector_size)
    y_train = keras.utils.to_categorical(trainingTargetMatrix, num_classes)
    y_test = keras.utils.to_categorical(testTargetMatrix, num_classes)
    y_test_USPS = keras.utils.to_categorical(USPSTarget, num_classes)
    image_size = 784
    model = Sequential()
    model.add(Dense(100  , activation='relu', input_shape=(image_size,)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=7, mode='min')
    history = model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=False, validation_split=.1, callbacks=[ earlystopping_cb])
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    yPredict = model.predict(x_test)
    y_predMNIST = yPredict.argmax(axis=1)
    yPredUSPS = model.predict(np.array(USPSDataMatrix))
    y_predUSPS = yPredUSPS.argmax(axis=1)
    print('-------- MNIST DATA ANALYSIS----------')
    print('Confusion Matrix-MNIST')
    print(ConfusionMatrix(y_test.argmax(axis=1), y_predMNIST)) #THe confusion matrix function is defined below
    print( " Accuracy- MNIST = " + str(accuracy))
    print('-------- USPS DATA ANALYSIS-----------')
    print('Confusion Matrix - USPS')
    print(ConfusionMatrix(y_test_USPS.argmax(axis=1),y_predUSPS))
    print("Accuracy- USPS =" + str(accuracy_score(y_test_USPS.argmax(axis=1),y_predUSPS)))
    df = pd.DataFrame(history.history)
    df.plot(subplots=True, grid=True, figsize=(10, 15))
    plt.savefig('graph.png')
    print('NN edded-' + str(datetime.datetime.now()))
    return y_predMNIST,y_predUSPS

#Function created tpo
def ConfusionMatrix(actualTargetVector , predictedTargetVector):
    numLabels = len(np.unique(actualTargetVector))
    confusion_matrix = np.zeros(shape = (numLabels,numLabels), dtype=int)
    for i in range(len(actualTargetVector)):
        confusion_matrix[actualTargetVector[i]][predictedTargetVector[i]] +=1
    return confusion_matrix



#Function for preprocessing USPS dataset
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
    return USPSMat,USPSTar

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



# A basic logistic regression function which uses sigmoid function
def logisticRegression(rawDataTraining,rawTargetTraining, rawDataTesting, rawTargetTesting, USPSDataMatrix, USPSTarget):
    numFeatures = rawDataTraining.shape[1]
    numLabels = 10
    classifiers = np.zeros(shape=(numLabels, numFeatures))
    minLoss = {}
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
        minLoss[i] = np.round(min(lossFunction),5)
    z_test = np.dot(rawDataTesting, classifiers.transpose())
    h_test = sigmoid(z_test)
    z_testUSPS = np.dot(USPSDataMatrix, classifiers.transpose())
    h_testUSPS = sigmoid(z_testUSPS)
    LogisticPredections = h_test.argmax(axis=1)
    LogisticPredectionsUSPS = h_testUSPS.argmax(axis=1)
    print(minLoss)
    print("Testing accuracy- MNIST:", str(100 * np.mean(LogisticPredections == rawTargetTesting)) + "%")
    print("Testing accuracy- USPS:", str(100 * np.mean(LogisticPredectionsUSPS == USPSTarget)) + "%")
    return LogisticPredections, LogisticPredectionsUSPS

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm



def LogisticRegressionWithSoftmax( rawDataTraining, rawTargetTraining, rawDataTesting, rawTargetTesting,USPSDataMatrix, USPSTarget):

    print('-----------------------------------------LOGISTIC REGRESSION WIH SOFTMAX-------------------------------------')
    print( 'Logistic with Softmax Started -' +  str(datetime.datetime.now()))
    w = np.zeros([rawDataTraining.shape[1], len(np.unique(rawTargetTraining))])
    iterations = 600
    learningRate = 0.10
    m = rawDataTraining.shape[0]
    losses = []
    y_mat= keras.utils.to_categorical(rawTargetTraining, 10)
    for i in range(0, iterations):
        scores = np.dot(rawDataTraining, w)
        prob = softmax(scores)
        loss = (-1 / m) * np.sum(y_mat * np.log(prob))
        grad = (-1 / m) * np.dot(rawDataTraining.T, (y_mat - prob))
        losses.append(loss)
        w = w - (learningRate * grad)
    probabilities = softmax(np.dot(rawDataTesting, w))
    predections = np.argmax(probabilities, axis=1)
    print("Testing accuracy- MNIST:", str(100 * np.mean(predections == rawTargetTesting)) + "%")
    print('Confusion Matrix- MNIST ')
    print(ConfusionMatrix(rawTargetTesting,predections))
    probsUSPS = softmax(np.dot(USPSDataMatrix,w))
    predectionsUSPS = np.argmax(probsUSPS, axis=1)
    print('Confusion Matrix- USPS ')
    print(ConfusionMatrix(USPSTarget,predectionsUSPS))
    print("Testing accuracy- USPS:", str(100 * np.mean(predectionsUSPS == USPSTarget)) + "%")
    print('Logistic Ended' + str(datetime.datetime.now()))
    return predections,predectionsUSPS

#Funciton to find the mode of the predicted values by all the classifiers
def MajorityVotingPrediction(NNPredict, RFPredict, SVMPredict, LogisticPredict, testTarget):
    allPred =  np.zeros(len(testTarget), dtype= int)
    for i in range(len(testTarget)):
        allPred[i] =  stats.mode([ NNPredict[i] , RFPredict[i], SVMPredict[i], LogisticPredict[i] ])[0]
    print(ConfusionMatrix(testTarget,allPred))
    return accuracy_score(testTarget, allPred)




if __name__ == '__main__':
    # All the functions are called below. We can comment the functions in order to run a specific one.
    print(' Main Started at---' + str(datetime.datetime.now()))
    trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix = preprocessingMNISTdataset('mnist.pkl.gz')
    USPSDataMatrix, USPSTarget = preprocessingUSPSdataset()
    SVMLinearMNIST,SVMLinearUSPS= SVM( trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix,USPSDataMatrix, USPSTarget ,'linear')
    LogisticPredictMNIST,LogisticPredictUSPS = logisticRegression(trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix,USPSDataMatrix, USPSTarget)
    SVMPredictRbfMNIST,SVMPredictRbfUSPS =SVM( trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix,USPSDataMatrix, USPSTarget ,'rbf')
    SVMPredictRbfGamma = SVM( trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix,USPSDataMatrix, USPSTarget ,'rbf',1)
    NNPredictMNIST,NNPredictUSPS = NeuralNetwork( trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix,USPSDataMatrix, USPSTarget)
    LogisticPredictSoftmaxMNIST, LogisticPredictSoftmaxUSPS= LogisticRegressionWithSoftmax( trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix,USPSDataMatrix, USPSTarget)
    RFPredictMNIST,RFPredictUSPS = RandomForest(trainingDataMatrix, trainingTargetMatrix, testDataMatrix, testTargetMatrix,USPSDataMatrix, USPSTarget)
    print('-------- MAJORITY VOTING----------')

    MNIST_finalAccuracy=  MajorityVotingPrediction(NNPredictMNIST, RFPredictMNIST, SVMLinearMNIST,LogisticPredictSoftmaxMNIST, testTargetMatrix )
    USPS_finalAccuracy = MajorityVotingPrediction(NNPredictUSPS,RFPredictUSPS, SVMLinearUSPS, LogisticPredictSoftmaxUSPS, USPSTarget)
    print('Accuracy after performing majority voting - MNIST ' + str(MNIST_finalAccuracy))
    print('Accuracy after performing majority voting - USPS ' + str(USPS_finalAccuracy))
    print('Main Ended at ' + str(datetime.datetime.now()))


