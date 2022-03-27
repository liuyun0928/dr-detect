#!/usr/bin/env python3

import os
import cv2

class DRLesionDetect:
    def __init__(self, dataSet, dataDir):
        self.dataSet = dataSet
        self.dataDir = dataDir
        self.healthyImages = {}
        self.maImages = {}
        self.maAnnotationImages = {}
        
    def load(self):
        # load image data
        if self.dataSet == 'EOphtha':
            self.loadEOphthaData()

    # read e-ophtha dataset: https://www.adcis.net/en/third-party/e-ophtha/
    def loadEOphthaData(self):
        # read healthy image
        healthyPath = os.path.join(self.dataDir, "healthy")
        for retinalDirName in os.listdir(healthyPath):
            retinalDir = os.path.join(healthyPath, retinalDirName)
            for healthyImageFileName in os.listdir(retinalDir):
                if healthyImageFileName.lower().endswith(".jpg"):
                    healthyImageName = healthyImageFileName.split(".")[0]
                    self.healthyImages[healthyImageName] = cv2.imread(os.path.join(retinalDir, healthyImageFileName))
        
        # read ma data
        maPath = os.path.join(self.dataDir, "MA")
        for retinalDirName in os.listdir(maPath):
            retinalDir = os.path.join(maPath, retinalDirName)
            for maImageFileName in os.listdir(retinalDir):
                if maImageFileName.lower().endswith(".jpg"):
                    maImageName = maImageFileName.split(",")[0]
                    self.maImages[maImageName] = cv2.imread(os.path.join(retinalDir, maImageFileName))
                    
        # read annotations data
        maAnnotationsPath = os.path.join(self.dataDir, "Annotation_MA")
        for retinalDirName in os.listdir(maAnnotationsPath):
            retinalDir = os.path.join(maAnnotationsPath, retinalDirName)
            for maAnnotationsFileName in os.listdir(retinalDir):
                if maAnnotationsFileName.lower().endswith(".png"):
                    maAnnotationsName = maAnnotationsFileName.split(",")[0]
                    self.maAnnotationImages[maAnnotationsName] = cv2.imread(os.path.join(retinalDir, maAnnotationsFileName))
                    
        # print data summary
        print("healthy images: %d, ma images: %d, ma annotation images: %d" % (len(self.healthyImages), len(self.maImages), len(self.maAnnotationImages)))
    
    def preProcess(self):
        return None
    
    def patchGenerate(self):
        return None
    
    def trainModel(self):
        return None

    def train(self):
        # load data
        self.load()
        
        # pre-process
        self.preProcess()
        
        # patch generation
        self.patchGenerate()
        
        # train model
        self.trainModel() 
                
        return None
    
    def detect(self, imagePath):
        return None
            
        
if __name__ == '__main__':
    lesionDetect = DRLesionDetect(dataSet='EOphtha', dataDir='/home/liuyun/data/e_optha_MA')
    lesionDetect.train()