import numpy as np
import os
import cv2
import random
import pickle
import pandas as pd
class Preparation():

    def prepData(self, dataType,path,category,imgSize, ClssNms):
        if dataType=="image":
            self.img_size=imgSize
            self.image_training_data=[]
            self.datadir = path
            self.categories = category
            for self.Category in self.categories:
                self.path=os.path.join(self.datadir, self.Category)
                self.class_num=self.categories.index(self.Category)
                for img in os.listdir(self.path):
                    try:
                        self.img_array=cv2.imread(os.path.join(self.path,img),cv2.IMREAD_GRAYSCALE)
                        self.resized_array = cv2.resize(self.img_array, (self.img_size, self.img_size))
                        self.image_training_data.append([self.resized_array,self.class_num])
                    except Exception as e:
                        print("data error!!")
            random.shuffle(self.image_training_data)
            return self.image_training_data
        if dataType=="text":
            self.dataPath=path
            self.classNames=[]
            self.classNums=0
            self.text_training_data=pd.read_csv(self.dataPath, encoding="latin-1")
            for name in self.text_training_data.columns:
                if name!="message" or name!="class":
                    self.text_training_data.drop(["name"], axis=1, inplace=True)
            for clsNames in ClssNms:
                self.classNames.append(clsNames)
                self.classNums+=1
            if self.classNums==2:
                self.text_training_data["class"] = self.text_training_data["class"].map({self.classNames[0]: 0, self.classNames[1]: 1})
            if self.classNums==3:
                self.text_training_data["class"] = self.text_training_data["class"].map({self.classNames[0]: 0, self.classNames[1]: 1, self.classNames[2]: 2})
            if self.classNums==4:
                self.text_training_data["class"] = self.text_training_data["class"].map({self.classNames[0]: 0, self.classNames[1]: 1, self.classNames[2]: 2, self.classNames[3]: 3})
            if self.classNums==5:
                self.text_training_data["class"] = self.text_training_data["class"].map({self.classNames[0]: 0, self.classNames[1]: 1, self.classNames[2]: 2, self.classNames[3]: 3, self.classNames[4]: 4})
            return self.text_training_data
    def createTraniningData(self,training_data, dataType):
        if dataType=="image":
            self.x=[]
            self.y=[]
            self.Training_Data=training_data
            for self.features, self.label in self.Training_Data:
                self.x.append(self.features)
                self.y.append(self.label)
            self.X=np.array(self.x).reshape(-1,self.img_size,self.img_size, 1)
            self.dumpX=open("image_X.pickle","wb")
            pickle.dump(self.X,self.dumpX)
            self.dumpX.close()
            self.dumpy=open("image_Y.pickle","wb")
            pickle.dump(self.y,self.dumpy)
            self.dumpy.close()
        if dataType=="text":
            self.x=training_data["message"]
            self.y=training_data["class"]

            self.dumpX = open("text_X.pickle", "wb")
            pickle.dump(self.x, self.dumpX)
            self.dumpX.close()
            self.dumpy = open("text_Y.pickle", "wb")
            pickle.dump(self.y, self.dumpy)
            self.dumpy.close()


