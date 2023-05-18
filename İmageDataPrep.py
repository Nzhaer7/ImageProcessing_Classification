import numpy as np
import os
from cv2 import cv2
import random
import pickle


class Preparation():
    def prepData(self, path, category, imgSize):
        self.img_size=imgSize
        self.image_training_data=[]
        self.datadir = path
        self.categories = category

        for Category in self.categories:
            self.path=os.path.join(self.datadir, Category)
            self.class_num=self.categories.index(Category)

            for img in os.listdir(self.path):
                try:
                    self.img_array=cv2.imread(os.path.join(self.path,img),cv2.IMREAD_GRAYSCALE)
                    self.resized_array = cv2.resize(self.img_array, (self.img_size, self.img_size))
                    self.image_training_data.append([self.resized_array,self.class_num])

                except Exception as e:
                    print("data error!!")

        random.shuffle(self.image_training_data)

        return self.image_training_data

    def createTraniningData(self,training_data, imgSize):
        self.x=[]
        self.y=[]
        self.img_size=imgSize
        self.Training_Data=training_data

        for features, label in self.Training_Data:
            self.x.append(features)
            self.y.append(label)

        self.X=np.array(self.x).reshape(-1, self.img_size, self.img_size, 1)
        self.dumpX=open("Image_X.pickle","wb")
        pickle.dump(self.X,self.dumpX)
        self.dumpX.close()
        self.dumpy=open("Image_Y.pickle","wb")
        pickle.dump(self.y,self.dumpy)
        self.dumpy.close()