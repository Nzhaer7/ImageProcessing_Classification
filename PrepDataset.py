import numpy as np
import os
import cv2
import random
import pickle
class Preparation():
    def prepData(self,dirpath,category,imgSize):
        self.img_size=imgSize
        self.training_data=[]
        self.datadir = dirpath
        self.categories = category
        for self.Category in self.categories:
            self.path=os.path.join(self.datadir, self.Category)
            self.class_num=self.categories.index(self.Category)
            for img in os.listdir(self.path):
                try:
                    self.img_array=cv2.imread(os.path.join(self.path,img),cv2.IMREAD_GRAYSCALE)
                    self.resized_array = cv2.resize(self.img_array, (self.img_size, self.img_size))
                    self.training_data.append([self.resized_array,self.class_num])
                except Exception as e:
                    print("data error!!")
        random.shuffle(self.training_data)
        return self.training_data

    def createTraniningData(self,training_data):
        self.x=[]
        self.y=[]
        self.Training_Data=training_data
        for self.features, self.label in self.Training_Data:
            self.x.append(self.features)
            self.y.append(self.label)
        self.X=np.array(self.x).reshape(-1,self.img_size,self.img_size,1)
        self.dumpX=open("X.pickle","wb")
        pickle.dump(self.X,self.dumpX)
        self.dumpX.close()
        self.dumpy=open("Y.pickle","wb")
        pickle.dump(self.y,self.dumpy)
        self.dumpy.close()

