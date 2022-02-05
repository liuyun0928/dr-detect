#!/usr/bin/env python3

from datetime import datetime
from lib2to3.pgen2.token import LBRACE
from statistics import mode
from matplotlib import pyplot
import numpy as np
import pandas as pd

import os
import random
import sys
import cv2
import matplotlib
import csv

from subprocess import check_output

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import adam_v2
from keras.utils.vis_utils import model_to_dot
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from IPython.display import SVG

class DiabeticRetinopathyDetect:
    def __init__(self, imagePath, csvFilePath):
        self.imagePath = imagePath
        self.csvFilePath = csvFilePath
        self.numClasses = 5
        self.epochs = 15
        self.initLR = 1e-3
        self.imgWidth = 128
        self.imgHeight = 128
        self.imgDepth = 3
        self.inputShape = (self.imgWidth, self.imgHeight, self.imgDepth)
        self.imageNameDataMap = {}
        self.uniquePatientIdList = {}
        self.patientIdList = []
        
    def classes_to_int(self, label):
        label = label.strip()
        if label == "No": return 0
        if label == "Mild": return 1
        if label == "Moderate": return 2
        if label == "Severe": return 3
        if label == "Proliferative": return 4
        return 5

    def int_to_classes(self, i):
        if i == 0: return "No"
        if i == 1: return "Mild"
        if i == 2: return "Moderate"
        if i == 3: return "Severe"
        if i == 4: return "Proliferative"
        return "Invalid"

    def readTrainData(self, path, patientIdList, suffix = ".png"):
        imageNameDataMap = {}

        for patientId in patientIdList:
            for eye in ["_left", "_right"]:
                imageName = str(patientId) + eye
                imagePath = os.path.join(os.path.sep, path, imageName + suffix)
                image = load_img(imagePath)

                # convert image to array
                originImageArray = img_to_array(image)
                # reshape array to defined ${inputShap}
                resizedImageArray = cv2.resize(originImageArray, (self.imgWidth, self.imgHeight))
                # convert to np arr
                npArray = np.array(resizedImageArray, dtype="float") / 255.0 

                print(imageName)
                imageNameDataMap[imageName] = np.array(npArray)
            
        return imageNameDataMap
            
    def readTrainCsv(self, filePath):
        rawData = pd.read_csv(filePath, sep=",")
        
        patientIdList = []
        for idx, row in rawData.iterrows():
            patientId = str(row[0]).replace("_right", "").replace("_left", "")
            patientIdList.append(patientId)
            rawData.at[idx, "PatientID"] = patientId
            
        return rawData, pd.unique(patientIdList).tolist()
    
    def mergeCsvAndImageData(self, csvData, imageNameDataMap):
        imageNameArr = []
        dataArr = []
        for idx, row in csvData.iterrows():
            imageNameArr.append(str(row[0]))
            dataArr.append(np.array(imageNameDataMap[str(row[0])]))
            
        imageData = pd.DataFrame({"image": imageNameArr, "data": dataArr})
        
        # 校验csv data 与 image data行是否匹配
        for idx in range(0, len(imageData)):
            if imageData.loc[imageData.index[idx], 'image'] != csvData.loc[csvData.index[idx], 'image']:
                print("Error image: {} ".format(imageData.loc[imageData[idx], 'image']))
                return None
        
        return pd.merge(imageData, csvData, left_on='image', right_on='image', how='outer')   
    
    def printSample(self, mergedData):
        sample0 = mergedData.loc[mergedData.index[0], 'data']
        print(sample0)
        print(type(sample0)) # <class 'numpy.ndarray'>
        print(sample0.shape) # 128,128,3
        pyplot.imshow(sample0, interpolation="nearest")
        pyplot.show()        

    def createModel(self):
        model = Sequential()
        # first set of CONV => RELU => MAX POOL layers
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.inputShape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.numClasses, activation='softmax'))
        # returns our fully constructed deep learning + Keras image classifier 
        opt = adam_v2.Adam(lr=self.initLR, decay=self.initLR / self.epochs)
        # use binary_crossentropy if there are two classes
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model
    
    def process(self, dataCount):
        # 读训练 label
        csvData, patientIdList = self.readTrainCsv(self.csvFilePath)
        # 随机选择 dataCount 个病例数据进行训练
        trainPatientIdList = random.sample(patientIdList, dataCount)
        # 读取图像数据
        imageNameDataMap = self.readTrainData(self.imagePath, trainPatientIdList)
        # 组装dataframe
        mergedData = self.mergeCsvAndImageData(csvData.loc[csvData.PatientID.isin(trainPatientIdList)], imageNameDataMap)

        self.printSample(mergedData)
        if (mergedData is None):
            print("prepare data error")
            return
        trainIdList, valIdList = train_test_split(patientIdList, test_size=0.25, random_state=10)

        trainData = mergedData[mergedData.PatientID.isin(trainIdList)].reset_index(drop=True)
        validateData = mergedData[~mergedData.PatientID.isin(trainIdList)].reset_index(drop=True)

        trainX = trainData['data']
        trainY = trainData['level']
        
        validateX = validateData['data'] 
        validateY = validateData['level']

        print("to categorical")
        trainY = to_categorical(trainY, num_classes=self.numClasses)
        validateY = to_categorical(validateY, num_classes=self.numClasses)
        
        print("Reshaping trainX at..."+ str(datetime.now()))
        print(trainX.shape) # (750,)
        Xtrain = np.zeros([trainX.shape[0], self.imgWidth, self.imgHeight, self.imgDepth])
        for i in range(trainX.shape[0]): # 0 to traindf Size -1
            Xtrain[i] = trainX[i]
            
        print("xvalidate")
        Xval = np.zeros([validateX.shape[0], self.imgWidth, self.imgHeight, self.imgDepth])
        for i in range(validateX.shape[0]): # 0 to traindf Size -1
            Xval[i] = validateX[i]
        
        print("create model")
        model = self.createModel()

        print("summary")
        model.summary()
        # SVG(model_to_dot(model).create(prog='dot', format='svg'))
        
        
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
        H = model.fit_generator(aug.flow(Xtrain, trainY, batch_size=32), validation_data=(Xval, validateY), steps_per_epoch=len(trainX) // 32, epochs=self.epochs, verbose=1)
        print("save model")
        model.save("/home/liuyun/model/dr-test")
        
        print("show")
        matplotlib.use("Agg")
        matplotlib.pyplot.style.use("ggplot")
        matplotlib.pyplot.figure()
        N = self.epochs
        matplotlib.pyplot.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        matplotlib.pyplot.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        matplotlib.pyplot.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        matplotlib.pyplot.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        matplotlib.pyplot.title("Training Loss and Accuracy on diabetic retinopathy detection")
        matplotlib.pyplot.xlabel("Epoch #")
        matplotlib.pyplot.ylabel("Loss/Accuracy")
        matplotlib.pyplot.legend(loc="lower left")
        matplotlib.pyplot.savefig("plot.png")

if __name__ == '__main__':
    drd = DiabeticRetinopathyDetect("/home/liuyun/data", "/home/liuyun/code/data/trainLabels.csv") 
    drd.process(100)
