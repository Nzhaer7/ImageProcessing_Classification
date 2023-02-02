import tensorflow as tf
import numpy as np
import string
import os
import cv2
import random
import pickle
import pandas as pd
class Preparation():

    def prepData(self, dataType, dataPrpose, path, category, imgSize, ClssNms):

        if dataType=="image":
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

        if dataType=="text":
            if dataPrpose=="classification":
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
                    self.text_training_data["class"] = self.text_training_data["class"].map({self.classNames[0]: 0,
                                                                                             self.classNames[1]: 1})
                if self.classNums==3:
                    self.text_training_data["class"] = self.text_training_data["class"].map({self.classNames[0]: 0,
                                                                                             self.classNames[1]: 1,
                                                                                             self.classNames[2]: 2})
                if self.classNums==4:
                    self.text_training_data["class"] = self.text_training_data["class"].map({self.classNames[0]: 0,
                                                                                             self.classNames[1]: 1,
                                                                                             self.classNames[2]: 2,
                                                                                             self.classNames[3]: 3})
                if self.classNums==5:
                    self.text_training_data["class"] = self.text_training_data["class"].map({self.classNames[0]: 0,
                                                                                             self.classNames[1]: 1,
                                                                                             self.classNames[2]: 2,
                                                                                             self.classNames[3]: 3,
                                                                                             self.classNames[4]: 4})
                return self.text_training_data

            elif dataPrpose=="text learning":
                self.dataPath=path
                self.dataFile=open(path, "r", encoding="utf8")
                self.lines=[]
                self.sequences=[]
                self.data=""
                self.temp=[]

                for i in self.dataFile:
                    self.lines.append(i)

                for i in self.lines:
                    self.data=' '.join(self.lines)

                self.data=self.data.replace('\n', '').replace('\r', '').replace('\ufeff', '')
                self.translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
                self.new_data=self.data.translate(self.translator)

                for i in self.data.split():
                    if i not in self.temp:
                        self.temp.append(i)

                self.data=' '.join(self.temp)
                self.tokenizer=tf.keras.preprocessing.text.Tokenizer()
                self.tokenizer.fit_on_texts([self.data])
                pickle.dump(self.tokenizer, open("tokenizer_1.pkl", "wb"))
                self.sequence_data=self.tokenizer.texts_to_sequences([self.data])[0]
                self.vocab_size=len(self.tokenizer.word_index)+1

                for i in range(1, len(self.sequence_data)):
                    self.words=self.sequence_data[i-1:i+1]
                    self.sequences.append(self.words)

                self.sequences=np.array(self.sequences)

                return self.sequences, self.vocab_size
    def createTraniningData(self,training_data, dataType, dataPrpose, imgSize, vcbSize):
        if dataType=="image":
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
        if dataType=="text":
            if dataPrpose=="classification":
                self.x=training_data["message"]
                self.y=training_data["class"]

                self.dumpX = open("textClftn_X.pickle", "wb")
                pickle.dump(self.x, self.dumpX)
                self.dumpX.close()
                self.dumpy = open("textClftn_Y.pickle", "wb")
                pickle.dump(self.y, self.dumpy)
                self.dumpy.close()
            elif dataPrpose=="text learning":
                self.x=[]
                self.y=[]

                for i in training_data:
                    self.x.append(i[0])
                    self.y.append(i[1])

                self.x=np.array(self.x)
                self.y=np.array(self.y)
                self.y=tf.keras.utils.to_categorical(self.y, num_classes=vcbSize)

                self.dumpX = open("textLrnng_X.pickle", "wb")
                pickle.dump(self.x, self.dumpX)
                self.dumpX.close()
                self.dumpy = open("textLrnng_Y.pickle", "wb")
                pickle.dump(self.y, self.dumpy)
                self.dumpy.close()




