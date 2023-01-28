import PrepDataset as prd
import ClassificationModel as cm
import pickle as pkl
import tensorflow as tf

path=""
category=[""]
img_size=0

training_data=prd.Preparation.prepData(path,category,img_size)
prd.Preparation.createTraniningData(training_data)

X=pkl.open("X.pickle", "rb")
Y=pkl.open("Y.pickle", "rb")

actFunc=tf.nn.relu
fLayActFunc=tf.nn.softmax
catNum=0
optim="adam"
lssFunc="sparse_categorical_crossentrophy"
mtrc=["accuracy"]
epch=0

cm.TrainingModel.Sequential(X, Y, actFunc, fLayActFunc, catNum, optim, lssFunc, mtrc, epch)